#!/usr/bin/env python3
"""Run the Python CRF on the dev set and write predictions to JSON, so we can
check JS predictions against Python's exactly. This catches subtle featurizer
divergences that wouldn't show up in 'both produce the right answer'."""

import json
import pickle
from pathlib import Path

from features import sequence_features

SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")
OUT = SPLITS / "dev_python_pred.jsonl"

with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
    crf = pickle.load(f)["crf"]

rows = []
with (SPLITS / "dev.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Use ORACLE header so we measure CRF agreement, not header LR agreement.
X = [sequence_features(r["tokens"], r["header"]) for r in rows]
pred = crf.predict(X)

with OUT.open("w", encoding="utf-8") as f:
    for r, yp in zip(rows, pred):
        f.write(json.dumps({"tokens": r["tokens"], "header": r["header"], "py_pred": yp}, ensure_ascii=False) + "\n")
print(f"wrote {OUT}: {len(rows)} rows")
