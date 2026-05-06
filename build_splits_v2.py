#!/usr/bin/env python3
"""Pattern-aware stratified split builder.

Strategy:
  1. First pass — count occurrences of every (header, subfield-pattern) pair,
     where the subfield-pattern is the ordered concatenation of subfield codes
     in the row (e.g. for $a$d -> "a|d", for $a$c$d$t -> "a|c|d|t").
  2. Second pass — for each (header, pattern), reservoir-sample at most
     PER_PATTERN_CAP rows. Patterns with < PER_PATTERN_FLOOR total occurrences
     are still kept (we just take all of them).
  3. Apply a per-header cap of PER_HEADER_CAP after collecting patterns, then
     split each header's pool into train/dev/test.

This keeps rare subfield combinations represented while preventing common
patterns (e.g. plain "a|d" personal names) from dominating.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ALIGNED = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
OUT_DIR = Path("/Volumes/ImNotGlum/naf-crf/splits")
SEED = 7

PER_PATTERN_CAP = 1500
PER_PATTERN_FLOOR = 50  # informational only — patterns with fewer rows still get all of them
PER_HEADER_CAP = 18_000  # 15k train + 1.5k dev + 1.5k test

DEV_FRAC = 1.0 / 12.0  # 1500 of 18000
TEST_FRAC = 1.0 / 12.0


def pattern_from_tags(tags: list[str]) -> str:
    """Reconstruct the subfield-code pattern from BIO tags. Each B-<code> marks
    the start of a subfield in label-token order; joined with '|' for hashing."""
    codes = []
    for t in tags:
        if t.startswith("B-"):
            codes.append(t[2:])
    return "|".join(codes)


# Tags whose ind2 is "Undefined" per MARC. For these we collapse ind2 to "#"
# regardless of what the source data carried, because the surface label
# contains no signal that distinguishes ind2='#' from ind2='0' here.
IND2_UNDEFINED_TAGS = {"100", "110", "111"}


def collapse_header(hdr: str) -> str:
    """Normalize header labels by collapsing meaningless ind2 distinctions."""
    parts = hdr.split("|")
    if len(parts) != 3:
        return hdr
    tag, i1, i2 = parts
    if tag in IND2_UNDEFINED_TAGS and i2 != "#":
        i2 = "#"
    return f"{tag}|{i1}|{i2}"


def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: count (header, pattern) ----
    print("pass 1: counting (header, pattern) ...", file=sys.stderr)
    pattern_counts: Counter[tuple[str, str]] = Counter()
    with ALIGNED.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            hdr = collapse_header(rec["header"])
            pat = pattern_from_tags(rec["tags"])
            pattern_counts[(hdr, pat)] += 1
            if (i + 1) % 1_000_000 == 0:
                print(f"  scanned {i+1:,}", file=sys.stderr)

    n_patterns = len(pattern_counts)
    print(f"distinct (header, pattern) pairs: {n_patterns:,}", file=sys.stderr)

    # ---- Pass 2: per-(header,pattern) reservoir sampling ----
    print("pass 2: reservoir-sampling per (header, pattern) ...", file=sys.stderr)
    reservoirs: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: Counter[tuple[str, str]] = Counter()

    with ALIGNED.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            hdr = collapse_header(rec["header"])
            # Mutate the record so the line we sample carries the collapsed
            # label (used downstream in train.py without further processing).
            if hdr != rec["header"]:
                rec["header"] = hdr
                line = json.dumps(rec, ensure_ascii=False) + "\n"
            pat = pattern_from_tags(rec["tags"])
            key = (hdr, pat)

            seen[key] += 1
            res = reservoirs[key]
            if len(res) < PER_PATTERN_CAP:
                res.append(line)
            else:
                j = rng.randint(0, seen[key] - 1)
                if j < PER_PATTERN_CAP:
                    res[j] = line

            if (i + 1) % 1_000_000 == 0:
                print(f"  scanned {i+1:,}", file=sys.stderr)

    # ---- Group sampled rows by header, then cap per header ----
    by_header: dict[str, list[str]] = defaultdict(list)
    by_header_pattern_count: dict[str, int] = defaultdict(int)
    for (hdr, pat), rows in reservoirs.items():
        by_header[hdr].extend(rows)
        by_header_pattern_count[hdr] += 1

    # Optional per-header cap. Shuffle within each header so we don't just take
    # patterns in the order we saw them.
    train_lines: list[str] = []
    dev_lines: list[str] = []
    test_lines: list[str] = []
    print()
    print("=== per-header sampling summary ===")
    print(f"{'header':12s} {'distinct_pats':>13s} {'pool_before':>12s} {'pool_after':>11s} {'train':>7s} {'dev':>5s} {'test':>5s}")
    for hdr in sorted(by_header, key=lambda h: -sum(seen[(h, p)] for (h2, p) in pattern_counts if h2 == h)):
        pool = by_header[hdr]
        rng.shuffle(pool)
        pool_before = len(pool)
        if len(pool) > PER_HEADER_CAP:
            pool = pool[:PER_HEADER_CAP]

        n = len(pool)
        n_dev = max(1, int(n * DEV_FRAC)) if n > 1 else 0
        n_test = max(1, int(n * TEST_FRAC)) if n > 2 else 0
        n_train = n - n_dev - n_test
        if n_train < 0:
            n_train = n
            n_dev = 0
            n_test = 0

        train_lines += pool[:n_train]
        dev_lines += pool[n_train : n_train + n_dev]
        test_lines += pool[n_train + n_dev : n_train + n_dev + n_test]

        print(f"{hdr:12s} {by_header_pattern_count[hdr]:>13d} {pool_before:>12d} {n:>11d} {n_train:>7d} {n_dev:>5d} {n_test:>5d}")

    rng.shuffle(train_lines)
    rng.shuffle(dev_lines)
    rng.shuffle(test_lines)

    for name, lines in [("train", train_lines), ("dev", dev_lines), ("test", test_lines)]:
        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"wrote {out}: {len(lines):,} rows")

    # Summary stats: how many patterns are at floor / cap?
    n_at_cap = sum(1 for c in pattern_counts.values() if c >= PER_PATTERN_CAP)
    n_below_floor = sum(1 for c in pattern_counts.values() if c < PER_PATTERN_FLOOR)
    print()
    print(f"patterns with >= {PER_PATTERN_CAP} occurrences (capped):   {n_at_cap:,}")
    print(f"patterns with <  {PER_PATTERN_FLOOR} occurrences (all kept): {n_below_floor:,}")
    print(f"total distinct patterns:                                 {n_patterns:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
