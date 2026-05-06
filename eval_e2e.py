#!/usr/bin/env python3
"""End-to-end evaluation: predict header with the LR, feed it to the CRF,
measure full-pipeline exact-sequence match. This is the realistic metric for
how well the system works in deployment.

Also prints a confusion table for the header classifier to help us see which
header pairs are getting confused.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report

from features import sequence_features
from train import header_doc_features  # reuse exactly the same featurizer

SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")


def load_split(name):
    rows = []
    with (SPLITS / f"{name}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    with (MODEL_DIR / "header_clf.pkl").open("rb") as f:
        bundle = pickle.load(f)
    clf = bundle["clf"]
    vec = bundle["vec"]
    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]

    rows = load_split("dev")
    print(f"dev rows: {len(rows)}")

    # Predict headers.
    Xh = vec.transform([header_doc_features(r["tokens"]) for r in rows])
    pred_headers = clf.predict(Xh)
    true_headers = [r["header"] for r in rows]

    n_hdr_correct = sum(1 for a, b in zip(pred_headers, true_headers) if a == b)
    print(f"header accuracy: {n_hdr_correct}/{len(rows)} ({100*n_hdr_correct/len(rows):.2f}%)")
    print()

    # Confusion: top off-diagonal.
    confusion = Counter()
    for t, p in zip(true_headers, pred_headers):
        if t != p:
            confusion[(t, p)] += 1
    print("=== top 15 header confusions (true -> pred) ===")
    for (t, p), c in confusion.most_common(15):
        print(f"  {t:10s} -> {p:10s}  : {c}")
    print()

    # End-to-end CRF using PREDICTED headers.
    X_pred = [sequence_features(r["tokens"], h) for r, h in zip(rows, pred_headers)]
    pred_tags = crf.predict(X_pred)
    n_exact_pred = sum(1 for r, yp in zip(rows, pred_tags) if r["tags"] == yp)
    print(f"end-to-end exact match (predicted header):  {n_exact_pred}/{len(rows)} ({100*n_exact_pred/len(rows):.2f}%)")

    # Oracle CRF using TRUE header.
    X_true = [sequence_features(r["tokens"], h) for r, h in zip(rows, true_headers)]
    pred_tags_true = crf.predict(X_true)
    n_exact_true = sum(1 for r, yp in zip(rows, pred_tags_true) if r["tags"] == yp)
    print(f"oracle exact match (true header):           {n_exact_true}/{len(rows)} ({100*n_exact_true/len(rows):.2f}%)")
    print()

    # Among rows where header was wrong, how often did CRF still tag correctly?
    n_hdr_wrong = 0
    n_hdr_wrong_tag_correct = 0
    for r, h_pred, yp in zip(rows, pred_headers, pred_tags):
        if h_pred != r["header"]:
            n_hdr_wrong += 1
            if yp == r["tags"]:
                n_hdr_wrong_tag_correct += 1
    if n_hdr_wrong:
        print(f"of {n_hdr_wrong} rows where header was wrong, "
              f"{n_hdr_wrong_tag_correct} ({100*n_hdr_wrong_tag_correct/n_hdr_wrong:.1f}%) "
              f"still got tags exactly right")


if __name__ == "__main__":
    main()
