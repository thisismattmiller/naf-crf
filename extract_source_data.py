#!/usr/bin/env python3
"""Extract marcKey and authoritativeLabel from a line-delimited MADS/RDF XML file.

Each line in the input file is expected to be a complete XML document. We pull
the first <bflc:marcKey> and first <madsrdf:authoritativeLabel> from each line
and write them tab-separated to the output file.
"""

import html
import re
import sys
from pathlib import Path

INPUT_PATH = Path("/Volumes/ImNotGlum/lc_bibs/names.madsrdf.xml")
OUTPUT_PATH = Path("/Volumes/ImNotGlum/naf-crf/source_data.txt")

MARC_KEY_RE = re.compile(r"<bflc:marcKey\b[^>]*>(.*?)</bflc:marcKey>", re.DOTALL)
AUTH_LABEL_RE = re.compile(
    r"<madsrdf:authoritativeLabel\b[^>]*>(.*?)</madsrdf:authoritativeLabel>",
    re.DOTALL,
)


def clean(value: str) -> str:
    # Decode XML entities and collapse any embedded tabs/newlines so the TSV
    # output stays one record per line.
    return html.unescape(value).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with INPUT_PATH.open("r", encoding="utf-8", errors="replace") as src, \
         OUTPUT_PATH.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            if not line.strip():
                continue

            marc_match = MARC_KEY_RE.search(line)
            label_match = AUTH_LABEL_RE.search(line)

            if not marc_match or not label_match:
                skipped += 1
                continue

            marc_key = clean(marc_match.group(1))
            label = clean(label_match.group(1))

            if not marc_key or not label:
                skipped += 1
                continue

            dst.write(f"{marc_key}\t{label}\n")
            written += 1

            if total % 100_000 == 0:
                print(f"processed {total:,} lines, wrote {written:,}", file=sys.stderr)

    print(
        f"done: read {total:,} lines, wrote {written:,}, skipped {skipped:,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
