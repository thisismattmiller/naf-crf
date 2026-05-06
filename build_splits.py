#!/usr/bin/env python3
"""Build stratified train/dev/test splits from aligned_1xx.jsonl.

Stratification is by header label so rare headers (e.g. 100|3|#, 151|#|0)
are represented in all splits. We cap per-header counts to keep the train set
balanced enough that common labels don't drown out rare ones.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ALIGNED = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
OUT_DIR = Path("/Volumes/ImNotGlum/naf-crf/splits")
SEED = 7

# Per-header caps. With ~15 distinct headers, capping at 15k per header keeps
# the train set ~225k max (much smaller in practice for rare headers).
TRAIN_PER_HEADER = 15_000
DEV_PER_HEADER = 1_500
TEST_PER_HEADER = 1_500


def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reservoir-sample TRAIN_PER_HEADER + DEV_PER_HEADER + TEST_PER_HEADER per header.
    # We do per-header reservoir sampling in a single pass over the file.
    cap_per_hdr = TRAIN_PER_HEADER + DEV_PER_HEADER + TEST_PER_HEADER
    reservoirs: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    print("scanning aligned file ...", file=sys.stderr)
    with ALIGNED.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            # Cheap header parse without full json.loads.
            # Format: {"header": "100|1|#", ...}
            try:
                hdr_start = line.index('"header":') + len('"header":')
                # find first quote after that
                q1 = line.index('"', hdr_start) + 1
                q2 = line.index('"', q1)
                hdr = line[q1:q2]
            except ValueError:
                continue

            seen[hdr] += 1
            res = reservoirs[hdr]
            if len(res) < cap_per_hdr:
                res.append(line)
            else:
                j = rng.randint(0, seen[hdr] - 1)
                if j < cap_per_hdr:
                    res[j] = line

            if (i + 1) % 1_000_000 == 0:
                print(f"  scanned {i+1:,}", file=sys.stderr)

    print()
    print("=== per-header counts (total / sampled) ===")
    for hdr in sorted(seen, key=lambda h: -seen[h]):
        print(f"  {hdr}: {seen[hdr]:,} -> {len(reservoirs[hdr]):,}")

    # Split each header's reservoir into train/dev/test.
    train_lines: list[str] = []
    dev_lines: list[str] = []
    test_lines: list[str] = []
    for hdr, res in reservoirs.items():
        rng.shuffle(res)
        train_lines += res[:TRAIN_PER_HEADER]
        dev_lines += res[TRAIN_PER_HEADER : TRAIN_PER_HEADER + DEV_PER_HEADER]
        test_lines += res[TRAIN_PER_HEADER + DEV_PER_HEADER : TRAIN_PER_HEADER + DEV_PER_HEADER + TEST_PER_HEADER]

    # Final shuffle so examples in each file aren't grouped by header.
    rng.shuffle(train_lines)
    rng.shuffle(dev_lines)
    rng.shuffle(test_lines)

    for name, lines in [("train", train_lines), ("dev", dev_lines), ("test", test_lines)]:
        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"wrote {out}: {len(lines):,} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
