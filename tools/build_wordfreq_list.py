#!/usr/bin/env python3
"""Generate a reproducible, cleaned English vocabulary from wordfreq."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from wordfreq import top_n_list


WORD_PATTERN = re.compile(r"^[a-z]+(?:['-][a-z]+)*$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--candidates", type=int, default=200_000)
    arguments = parser.parse_args()

    if arguments.count <= 0 or arguments.candidates < arguments.count:
        raise SystemExit("count must be positive and candidates must be at least count")

    arguments.output.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_token in top_n_list("en", arguments.candidates, wordlist="best"):
        token = raw_token.casefold()
        if not WORD_PATTERN.fullmatch(token) or token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) == arguments.count:
            break

    if len(selected) != arguments.count:
        raise SystemExit(
            f"Only found {len(selected)} cleaned English words; increase --candidates."
        )

    vocabulary_path = arguments.output / f"wordfreq-en-common-{len(selected)}.tsv"
    with vocabulary_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("rank\tword\n")
        for rank, token in enumerate(selected, start=1):
            output.write(f"{rank}\t{token}\n")

    package_version = importlib.metadata.version("wordfreq")
    metadata = {
        "format": "firstagent-ranked-word-list-v1",
        "language": "English",
        "word_count": len(selected),
        "selection": "Frequency-ranked wordfreq English list, filtered to lowercase ASCII words with optional internal apostrophes or hyphens.",
        "wordfreq_version": package_version,
        "generator": "tools/build_wordfreq_list.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "wordfreq code is Apache-2.0. Its included language data has CC-BY-SA-4.0 redistribution terms; retain attribution and share-alike obligations for derived data.",
        "source": "https://pypi.org/project/wordfreq/",
    }
    (arguments.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (arguments.output / "ATTRIBUTION.md").write_text(
        "# Attribution and license\n\n"
        "This ranked English word list was generated with `wordfreq`.\n\n"
        "- Project: https://github.com/rspeer/wordfreq\n"
        "- Package: https://pypi.org/project/wordfreq/\n"
        "- Code license: Apache-2.0\n"
        "- Included language-frequency data: CC-BY-SA-4.0\n\n"
        "Keep this file with the derived list. Redistribution of the list must preserve appropriate attribution and comply with the CC-BY-SA-4.0 share-alike terms.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected):,} ranked English words to {vocabulary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
