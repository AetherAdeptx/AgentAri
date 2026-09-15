#!/usr/bin/env python3
"""Extract clean, line-oriented English records from OpenAssistant JSONL.

The C++ learner consumes one independent text record per line.  This adapter
removes JSON metadata, filters deleted/non-English records, preserves the
speaker role, and folds embedded newlines into spaces.  It writes to stdout by
default so a caller can stream a bounded slice without creating a second copy
of the corpus.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import TextIO


def open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OpenAssistant .jsonl or .jsonl.gz file")
    parser.add_argument("--output", type=Path, help="Output text file; defaults to stdout")
    parser.add_argument("--skip", type=int, default=0, help="Filtered records to skip")
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to emit; 0 means all")
    args = parser.parse_args()
    if args.skip < 0 or args.limit < 0:
        parser.error("--skip and --limit must be non-negative")

    output: TextIO
    close_output = False
    if args.output is None:
        output = sys.stdout
    else:
        output = args.output.open("w", encoding="utf-8")
        close_output = True

    filtered = emitted = 0
    try:
        with open_input(args.input) as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("lang") != "en"
                    or record.get("role") not in {"prompter", "assistant"}
                    or record.get("deleted", False)
                ):
                    continue
                text = " ".join(str(record.get("text", "")).split())
                if not text:
                    continue
                if filtered < args.skip:
                    filtered += 1
                    continue
                role = "user" if record["role"] == "prompter" else "assistant"
                print(role, text, file=output)
                emitted += 1
                if args.limit and emitted >= args.limit:
                    break
    finally:
        if close_output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
