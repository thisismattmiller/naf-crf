#!/usr/bin/env python3
"""Align 1XX MARC rows in source_data.txt to BIO-tagged training examples.

For each input row of the form:

    <marc>\t<authoritativeLabel>

we:
  1. Parse the MARC into (tag, ind1, ind2) and an ordered list of (code, value)
     subfields.
  2. Keep only 1XX records (100, 110, 111, 130, 150, 151, 181) — see
     pilot results: 4XX/430/451 rows are mispaired with the authorized label.
  3. Tokenize the clean label and each subfield value with the same tokenizer
     (words + standalone punctuation/symbol characters).
  4. Walk the subfields in MARC order, finding each subfield's token sequence
     as a contiguous span in the clean-label token stream. Spans must appear
     in order and every clean-label token must be covered by some subfield.
  5. Emit one JSONL record per aligned row:
        {"header": "100|1|#", "tokens": [...], "tags": ["B-a","I-a", ...]}

Subfields whose code is in CONTROL_SUBFIELDS (e.g. $w, $0, $1, $2, $5) are
ignored when aligning since their values do not appear in the clean label.

Tags follow BIO: B-<code> for the first token of a subfield span, I-<code>
for the rest. We do NOT emit O tags since every token belongs to some subfield.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("/Volumes/ImNotGlum/naf-crf/source_data.txt")
DEFAULT_OUTPUT_PATH = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
DEFAULT_SKIPS_PATH = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx_skips.txt")

KEEP_TAGS = {"100", "110", "111", "130", "150", "151", "181"}

# Subfields whose values are part of the displayed label.
# Includes geographic ($z) and chronological ($y) subdivisions used by 15X/18X,
# plus form ($m, $o, $r, $s) subfields that occasionally appear in 1XX headings,
# and $h (medium / general material designation, e.g. "[sound recording]").
DISPLAY_SUBFIELDS = set("abcdefghjklmnopqrstuvxyz")
# Subfields that are control / metadata and should be ignored during alignment.
CONTROL_SUBFIELDS = set("w012345678")

SUBFIELD_SPLIT_RE = re.compile(r"\$([a-z0-9])")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_marc(marc: str):
    """Parse a MARC string of the form `TTTII...` or `TTTI ...` or `TTT ...`
    where TTT is the 3-digit tag and I/space are indicators (trailing blanks
    may have been trimmed). Indicators below come from the chars at positions
    3 and 4; if missing they are taken as blank (space)."""
    if len(marc) < 3 or not marc[:3].isdigit():
        return None
    tag = marc[:3]

    # Find the start of the subfield section (first '$').
    dollar = marc.find("$")
    if dollar == -1:
        return None
    # The chars between position 3 and the first '$' are: ind1, ind2, optional
    # space separator. Trailing spaces may have been trimmed, so missing chars
    # default to blank.
    header_block = marc[3:dollar]
    # Drop the single space that precedes "$" when present (a separator the
    # source data inserts between the header and the first subfield).
    if header_block.endswith(" "):
        header_block = header_block[:-1]
    if len(header_block) == 0:
        ind1, ind2 = " ", " "
    elif len(header_block) == 1:
        ind1, ind2 = header_block, " "
    else:
        ind1, ind2 = header_block[0], header_block[1]
        # Anything beyond two indicator chars is unexpected; reject.
        if len(header_block) > 2:
            return None

    rest = marc[dollar:]

    parts = SUBFIELD_SPLIT_RE.split(rest)
    subfields: list[tuple[str, str]] = []
    if parts[0]:
        # Text before the first $ is rare for 1XX but fall back to $a.
        subfields.append(("a", parts[0]))
    for i in range(1, len(parts), 2):
        code = parts[i]
        value = parts[i + 1] if i + 1 < len(parts) else ""
        subfields.append((code, value))
    return (tag, ind1, ind2), subfields


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def find_span(haystack: list[str], needle: list[str], start: int) -> int:
    """Return the index in `haystack` (>= start) where `needle` matches as a
    contiguous subsequence, or -1 if no match."""
    if not needle:
        return start
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def align(label_tokens: list[str], subfields: list[tuple[str, str]]):
    """Try to align subfield values onto label tokens in MARC order.

    Returns (tags, ok, reason). On success, tags has the same length as
    label_tokens and ok=True. On failure, tags is None and reason describes why.
    """
    tags = [None] * len(label_tokens)
    cursor = 0
    last_end = 0  # index just past the last subfield's span

    for code, value in subfields:
        if code in CONTROL_SUBFIELDS:
            continue
        if code not in DISPLAY_SUBFIELDS:
            return None, False, f"unknown_subfield_${code}"

        toks = tokenize(value)
        if not toks:
            continue

        idx = find_span(label_tokens, toks, cursor)
        if idx == -1:
            return None, False, f"subfield_${code}_not_found"

        # Tokens between last_end and idx are not covered by any subfield.
        # That happens when MARC has separators (commas, periods) that the
        # tokenizer attaches to a subfield value, but the clean label adds an
        # extra separator between subfields. We allow gaps only if every gap
        # token is pure punctuation — otherwise alignment is rejected.
        for j in range(last_end, idx):
            if re.match(r"\w", label_tokens[j], re.UNICODE):
                return None, False, "uncovered_word_between_subfields"

        # Assign tags for this span.
        for k, j in enumerate(range(idx, idx + len(toks))):
            tags[j] = ("B-" if k == 0 else "I-") + code

        cursor = idx + len(toks)
        last_end = cursor

    # Trailing gap: any tokens after the last subfield must be pure punctuation.
    for j in range(last_end, len(label_tokens)):
        if re.match(r"\w", label_tokens[j], re.UNICODE):
            return None, False, "uncovered_word_after_last_subfield"

    # Fill leading-gap and inter-subfield-gap punctuation tokens with O tags.
    # We choose O for these because they're separators inserted by the label
    # renderer, not part of any subfield value.
    for j in range(len(tags)):
        if tags[j] is None:
            tags[j] = "O"

    return tags, True, "ok"


def header_label(tag: str, ind1: str, ind2: str) -> str:
    def disp(c: str) -> str:
        return "#" if c == " " else c
    return f"{tag}|{disp(ind1)}|{disp(ind2)}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(INPUT_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    p.add_argument("--skips", default=str(DEFAULT_SKIPS_PATH),
                   help="path to write skip examples (first N per reason)")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only process this many input lines (for testing)")
    p.add_argument("--skip-examples-per-reason", type=int, default=20)
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    skips_path = Path(args.skips)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parse_fail = 0
    not_1xx = 0
    aligned = 0
    skipped = 0
    skip_reasons = Counter()
    header_counts = Counter()
    skip_examples: dict[str, list[str]] = {}

    with in_path.open("r", encoding="utf-8", errors="replace") as src, \
         out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            if args.limit and total > args.limit:
                break
            line = line.rstrip("\n")
            if "\t" not in line:
                parse_fail += 1
                continue
            marc, label = line.split("\t", 1)

            parsed = parse_marc(marc)
            if parsed is None:
                parse_fail += 1
                continue
            (tag, ind1, ind2), subfields = parsed

            if tag not in KEEP_TAGS:
                not_1xx += 1
                continue

            label_tokens = tokenize(label)
            tags, ok, reason = align(label_tokens, subfields)

            if not ok:
                skipped += 1
                skip_reasons[reason] += 1
                bucket = skip_examples.setdefault(reason, [])
                if len(bucket) < args.skip_examples_per_reason:
                    bucket.append(f"{marc}\t{label}")
                continue

            hdr = header_label(tag, ind1, ind2)
            header_counts[hdr] += 1
            aligned += 1

            dst.write(json.dumps({
                "header": hdr,
                "tokens": label_tokens,
                "tags": tags,
            }, ensure_ascii=False))
            dst.write("\n")

            if total % 500_000 == 0:
                print(f"  ...{total:,} read, {aligned:,} aligned, {skipped:,} skipped",
                      file=sys.stderr)

    with skips_path.open("w", encoding="utf-8") as sf:
        for reason, count in skip_reasons.most_common():
            sf.write(f"=== {reason}: {count} ===\n")
            for ex in skip_examples.get(reason, []):
                sf.write(ex + "\n")
            sf.write("\n")

    print()
    print("=== summary ===")
    print(f"total lines:    {total:,}")
    print(f"parse failures: {parse_fail:,}")
    print(f"not 1XX:        {not_1xx:,}")
    print(f"aligned:        {aligned:,}")
    print(f"skipped (1XX):  {skipped:,}")
    if aligned + skipped:
        print(f"alignment rate within 1XX: {100*aligned/(aligned+skipped):.2f}%")
    print()
    print("skip reasons:")
    for reason, count in skip_reasons.most_common():
        print(f"  {reason}: {count:,}")
    print()
    print("aligned headers (top 15):")
    for hdr, c in header_counts.most_common(15):
        print(f"  {hdr}: {c:,}")
    print()
    print(f"output: {out_path}")
    print(f"skip examples: {skips_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
