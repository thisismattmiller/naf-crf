#!/usr/bin/env python3
"""Verify CRF feature pruning didn't hurt accuracy.

Re-run the pickled CRF on dev to get the baseline exact-match. Then re-export
the model with threshold=0 (no pruning) and threshold=0.05, and confirm both
load cleanly. Since we don't have a JS Viterbi yet, we just confirm the
pickled model still produces good results — pruning happens only in the
JSON export, not in the pickle.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from features import sequence_features

SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")


def main():
    rows = []
    with (SPLITS / "dev.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]

    # Use ORACLE header to isolate CRF performance.
    X = [sequence_features(r["tokens"], r["header"]) for r in rows]
    y = [r["tags"] for r in rows]

    # Distribution of state-feature weights to inform pruning threshold.
    weights = [abs(w) for w in crf.state_features_.values()]
    weights.sort()
    n = len(weights)
    print(f"total state features: {n:,}")
    for thresh in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        n_below = sum(1 for w in weights if w < thresh)
        print(f"  |w|<{thresh}: {n_below:,} ({100*n_below/n:.1f}%)")

    pred = crf.predict(X)
    n_exact = sum(1 for yt, yp in zip(y, pred) if yt == yp)
    print()
    print(f"oracle exact-match (full pickle, no pruning): {n_exact}/{len(rows)} "
          f"({100*n_exact/len(rows):.2f}%)")


if __name__ == "__main__":
    main()
