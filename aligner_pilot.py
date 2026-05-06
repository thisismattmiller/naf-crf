#!/usr/bin/env python3
"""Pilot aligner: sample rows from source_data.txt, parse the MARC side into
(header, subfields), and check how often the authoritative-label side is a
straightforward derivation of the subfield values.

Goal of this script: measure data quality and surface failure modes BEFORE we
invest in a full aligner. Output is a report, not training data.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("/Volumes/ImNotGlum/naf-crf/source_data.txt")
SAMPLE_SIZE = 5000
SEED = 1


HEADER_RE = re.compile(r"^(\d{3})(.)(.)(?=\$|$)")
SUBFIELD_SPLIT_RE = re.compile(r"\$([a-z0-9])")

# Subfields that carry user-visible label content. Anything else is metadata/control
# (e.g. $w linking, $0 authority number, $1 RWO URI, $2 source, $4 relator code,
# $5 institution, $6 linkage, $7 provenance, $8 sequence).
DISPLAY_SUBFIELDS = set("abcdefgjklnpqtuv")


def parse_marc(marc: str):
    """Split the MARC string into ((tag, ind1, ind2), [(code, value), ...]).

    Returns None if the header doesn't parse.
    """
    m = HEADER_RE.match(marc)
    if not m:
        return None
    tag, ind1, ind2 = m.group(1), m.group(2), m.group(3)

    # Skip past header + optional space before first $
    rest = marc[m.end():]
    if rest.startswith(" "):
        rest = rest[1:]

    parts = SUBFIELD_SPLIT_RE.split(rest)
    # parts[0] is text before the first $ (should be empty for well-formed rows)
    subfields = []
    if parts[0]:
        # Some rows might lack a leading $ — treat as $a
        subfields.append(("a", parts[0]))
    for i in range(1, len(parts), 2):
        code = parts[i]
        value = parts[i + 1] if i + 1 < len(parts) else ""
        subfields.append((code, value))
    return (tag, ind1, ind2), subfields


def derive_label(subfields):
    """Concatenate display-subfield values the way LCNAF authoritativeLabel
    typically renders: values joined by a single space, with the marker characters
    stripped. This is a heuristic baseline."""
    values = [v for code, v in subfields if code in DISPLAY_SUBFIELDS]
    return " ".join(v.strip() for v in values).strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def loose_equal(a: str, b: str) -> bool:
    """Equality after collapsing whitespace and stripping a few cosmetic chars."""
    a2 = normalize(a)
    b2 = normalize(b)
    return a2 == b2


def reservoir_sample(path: Path, k: int, seed: int):
    rng = random.Random(seed)
    sample = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if i < k:
                sample.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = line
    return sample


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"missing input: {INPUT_PATH}", file=sys.stderr)
        return 1

    print(f"reservoir-sampling {SAMPLE_SIZE} rows ...", file=sys.stderr)
    sample = reservoir_sample(INPUT_PATH, SAMPLE_SIZE, SEED)
    print(f"got {len(sample)} rows", file=sys.stderr)

    header_counter = Counter()
    parse_failures = 0
    exact_match = 0
    prefix_match = 0
    suffix_match = 0
    contains_match = 0
    no_match = 0
    no_match_examples = []
    subfield_codes_seen = Counter()

    for line in sample:
        if "\t" not in line:
            parse_failures += 1
            continue
        marc, label = line.split("\t", 1)
        parsed = parse_marc(marc)
        if parsed is None:
            parse_failures += 1
            continue
        header, subfields = parsed
        header_counter[header] += 1
        for code, _ in subfields:
            subfield_codes_seen[code] += 1

        derived = derive_label(subfields)
        d, l = normalize(derived), normalize(label)
        if loose_equal(d, l):
            exact_match += 1
        elif d.startswith(l) or l.startswith(d):
            prefix_match += 1
        elif d.endswith(l) or l.endswith(d):
            suffix_match += 1
        elif l in d or d in l:
            contains_match += 1
        else:
            no_match += 1
            if len(no_match_examples) < 25:
                no_match_examples.append((header, marc, label, derived))

    print()
    print("=== summary ===")
    n = len(sample) - parse_failures
    print(f"parsed rows: {n}")
    print(f"parse failures: {parse_failures}")
    print(f"exact match (derive == label): {exact_match} ({100*exact_match/n:.1f}%)")
    print(f"prefix match: {prefix_match} ({100*prefix_match/n:.1f}%)")
    print(f"suffix match: {suffix_match} ({100*suffix_match/n:.1f}%)")
    print(f"contains: {contains_match} ({100*contains_match/n:.1f}%)")
    print(f"no match: {no_match} ({100*no_match/n:.1f}%)")
    print()

    print("=== top headers in sample ===")
    for hdr, c in header_counter.most_common(20):
        tag, i1, i2 = hdr
        i1d = "#" if i1 == " " else i1
        i2d = "#" if i2 == " " else i2
        print(f"  {tag} {i1d}{i2d}  {c}")
    print()

    print("=== subfield codes in sample ===")
    for code, c in subfield_codes_seen.most_common():
        print(f"  ${code}: {c}")
    print()

    print("=== no-match examples (first 25) ===")
    for hdr, marc, label, derived in no_match_examples:
        print(f"  header={hdr}")
        print(f"    marc:    {marc}")
        print(f"    label:   {label}")
        print(f"    derived: {derived}")
        print()

    # By-header match-rate breakdown
    print("=== exact-match rate by header (top 10) ===")
    by_header_total = Counter()
    by_header_exact = Counter()
    for line in sample:
        if "\t" not in line:
            continue
        marc, label = line.split("\t", 1)
        parsed = parse_marc(marc)
        if parsed is None:
            continue
        header, subfields = parsed
        by_header_total[header] += 1
        derived = derive_label(subfields)
        if loose_equal(derived, label):
            by_header_exact[header] += 1
    for hdr, total in by_header_total.most_common(10):
        ex = by_header_exact[hdr]
        tag, i1, i2 = hdr
        i1d = "#" if i1 == " " else i1
        i2d = "#" if i2 == " " else i2
        print(f"  {tag} {i1d}{i2d}  exact {ex}/{total}  ({100*ex/total:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
