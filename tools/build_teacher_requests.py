#!/usr/bin/env python3
"""Build prompt/candidate records for an external frozen-model judge.

This adapter does not train a language model. It extracts English
OpenAssistant prompt/assistant pairs and writes the stable JSONL input schema
consumed by qwen_teacher_judge.py. The candidate is the original assistant
reply; a later AgentAri run can replace it with its own generated candidate.
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


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OpenAssistant .jsonl or .jsonl.gz file")
    parser.add_argument("--output", type=Path, required=True, help="JSONL request output")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum prompt/candidate pairs; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    # OpenAssistant exports normally place a parent before its children. A
    # bounded in-memory index also handles trees that are interleaved while
    # keeping the source corpus untouched.
    messages: dict[str, tuple[str, str]] = {}
    emitted = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_input(args.input) as source, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("lang") != "en" or record.get("deleted", False):
                continue
            message_id = str(record.get("message_id", record.get("id", "")))
            if not message_id:
                continue
            role = str(record.get("role", ""))
            text = clean_text(record.get("text", ""))
            if role not in {"prompter", "assistant"} or not text:
                continue

            parent_id = str(record.get("parent_id", ""))
            parent = messages.get(parent_id)
            if role == "assistant" and parent is not None and parent[0] == "prompter":
                request = {
                    "id": message_id,
                    "prompt": parent[1],
                    "candidate": text,
                    "source_line": line_number,
                }
                print(json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                      file=output)
                emitted += 1
                if args.limit and emitted >= args.limit:
                    break

            messages[message_id] = (role, text)

    if emitted == 0:
        print("warning: no prompt/candidate pairs found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
