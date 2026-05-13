#!/usr/bin/env python3
"""Run Python CRF + rules on dev set, dump tags so JS can cross-check parity."""
import json
import pickle
from pathlib import Path

from features import sequence_features
from rules import apply_rules

SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")
OUT = SPLITS / "dev_python_with_rules.jsonl"

with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
    crf = pickle.load(f)["crf"]

rows = []
with (SPLITS / "dev.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

X = [sequence_features(r["tokens"], r["header"]) for r in rows]
pred = crf.predict(X)

with OUT.open("w", encoding="utf-8") as f:
    for r, yp in zip(rows, pred):
        after, fired = apply_rules(r["header"], r["tokens"], yp)
        f.write(json.dumps({
            "header": r["header"],
            "tokens": r["tokens"],
            "py_crf": yp,
            "py_after": after,
            "fired": fired,
        }, ensure_ascii=False) + "\n")
print(f"wrote {OUT}: {len(rows)} rows")
