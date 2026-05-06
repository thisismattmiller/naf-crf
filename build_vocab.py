#!/usr/bin/env python3
"""Build a vocabulary of high-frequency tokens for the header classifier.

We scan the train split (NOT the full corpus, to avoid leakage from test/dev),
count how many rows each lowercased token appears in (document frequency),
and keep tokens with df >= MIN_DF.

Why document frequency rather than total count?
  Tokens like "," appear thousands of times in a single label sometimes —
  what we care about is "does this token's PRESENCE help predict the header"
  which is a per-row signal.

Output: a JSON file with the sorted vocab list, used by train.py and the JS runtime.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

TRAIN = Path("/Volumes/ImNotGlum/naf-crf/splits/train.jsonl")
OUT = Path("/Volumes/ImNotGlum/naf-crf/splits/header_vocab.json")
MIN_DF = 50  # token must appear in >= 50 rows


def main():
    df = Counter()
    n_rows = 0
    with TRAIN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_rows += 1
            seen = set()
            for t in rec["tokens"]:
                low = t.lower()
                if low in seen:
                    continue
                seen.add(low)
                df[low] += 1

    vocab = sorted(t for t, c in df.items() if c >= MIN_DF)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"min_df": MIN_DF, "n_train_rows": n_rows, "vocab": vocab}, f,
                  ensure_ascii=False)
    print(f"train rows: {n_rows:,}")
    print(f"distinct tokens: {len(df):,}")
    print(f"vocab (df>={MIN_DF}): {len(vocab):,}")
    print(f"wrote {OUT}")
    # Show a few high-DF tokens to sanity check.
    top = df.most_common(40)
    print()
    print("=== top 40 by df ===")
    for t, c in top:
        print(f"  {c:>7d}  {t!r}")


if __name__ == "__main__":
    main()
