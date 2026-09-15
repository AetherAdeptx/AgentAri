#!/usr/bin/env python3
"""Extract speaker utterances from a TalkBank/CABNC .cha transcript."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO


UTTERANCE = re.compile(r"^\*([^:]+):\s*(.*)$")


def input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(path.rglob("*.cha"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .cha file or directory of .cha files")
    parser.add_argument("--output", type=Path, help="Output text file; defaults to stdout")
    parser.add_argument("--limit", type=int, default=0, help="Maximum utterances; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    output: TextIO
    close_output = False
    if args.output is None:
        output = sys.stdout
    else:
        output = args.output.open("w", encoding="utf-8")
        close_output = True

    emitted = 0
    try:
        for path in input_files(args.input):
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    match = UTTERANCE.match(line.rstrip("\n"))
                    if match is None:
                        continue
                    text = " ".join(match.group(2).split())
                    if not text:
                        continue
                    print("speaker", match.group(1), text, file=output)
                    emitted += 1
                    if args.limit and emitted >= args.limit:
                        return 0
    finally:
        if close_output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
