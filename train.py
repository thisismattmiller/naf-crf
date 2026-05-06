#!/usr/bin/env python3
"""Train the header classifier and the per-token CRF tag model.

Header classifier:
  - Input: the token sequence (the clean label).
  - Output: a single header label like "100|1|#".
  - Model: scikit-learn LogisticRegression over a hashed bag-of-features:
      * unigram lowercased tokens
      * unigram shape buckets
      * has-comma / has-period / has-paren / has-digit / has-year-like / etc.
    We use a small hand-built feature set (no n-grams) so the JS port stays tiny.

Tag CRF:
  - Input: per-token features (see features.py), with header passed as a feature.
  - Output: BIO-tagged sequence with labels in {B-a, I-a, B-d, I-d, ..., O}.
  - Model: sklearn-crfsuite CRF, L-BFGS, light L1+L2.

We pickle both models to disk, then run eval on the dev set and report
token-F1 plus exact-sequence accuracy.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from features import sequence_features, shape, is_punct, has_digit


SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")

with (SPLITS / "header_vocab.json").open("r", encoding="utf-8") as _vf:
    HEADER_VOCAB: set[str] = set(json.load(_vf)["vocab"])


def load_split(name: str) -> list[dict]:
    rows = []
    with (SPLITS / f"{name}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --- Header-classifier features ---------------------------------------------

def header_doc_features(tokens: list[str]) -> dict[str, float]:
    """Compact structural feature representation of a label.

    We deliberately exclude per-token identity features (`has_w=schmidt`, etc.)
    because (a) they balloon the model and (b) they don't generalize: header
    label depends on the *structure* of the heading, not which specific people
    or institutions appear in it.
    """
    feats: dict[str, float] = {"bias": 1.0}
    n = len(tokens)
    # Sequence length buckets.
    if n <= 2:
        feats["nlen=1-2"] = 1.0
    elif n <= 4:
        feats["nlen=3-4"] = 1.0
    elif n <= 8:
        feats["nlen=5-8"] = 1.0
    elif n <= 16:
        feats["nlen=9-16"] = 1.0
    else:
        feats["nlen=17+"] = 1.0

    # First/last shape (no identity).
    first = tokens[0]
    last = tokens[-1]
    feats[f"first_shape={shape(first)}"] = 1.0
    feats[f"last_shape={shape(last)}"] = 1.0
    if is_punct(first):
        feats["first_punct"] = 1.0
    if is_punct(last):
        feats["last_punct"] = 1.0

    # Bag-of-shapes (no token identity).
    shape_counts = Counter(shape(t) for t in tokens)
    for sh, c in shape_counts.items():
        feats[f"has_shape={sh}"] = 1.0

    # Bag-of-tokens, but ONLY for tokens in our high-DF vocabulary. This keeps
    # signal words ("Conference", "Saint", "Inc", ...) while dropping rare
    # personal names that would just memorize.
    seen_lower = set()
    for t in tokens:
        low = t.lower()
        if low in seen_lower:
            continue
        seen_lower.add(low)
        if low in HEADER_VOCAB:
            feats[f"has_w={low}"] = 1.0
    # First/last token identity if in vocab.
    first_low = tokens[0].lower()
    last_low = tokens[-1].lower()
    if first_low in HEADER_VOCAB:
        feats[f"first_w={first_low}"] = 1.0
    if last_low in HEADER_VOCAB:
        feats[f"last_w={last_low}"] = 1.0

    # Structural signals — these are the actual discriminators between
    # personal/corporate/meeting/uniform-title headings.
    n_comma = sum(1 for t in tokens if t == ",")
    n_period = sum(1 for t in tokens if t == ".")
    n_paren_open = sum(1 for t in tokens if t == "(")
    n_paren_close = sum(1 for t in tokens if t == ")")
    n_colon = sum(1 for t in tokens if t == ":")
    n_dash = sum(1 for t in tokens if t == "-")
    n_year = sum(1 for t in tokens if t.isdigit() and len(t) == 4)
    n_digit_token = sum(1 for t in tokens if t.isdigit())
    n_punct = sum(1 for t in tokens if is_punct(t))
    n_upper = sum(1 for t in tokens if t and t[0].isupper())
    n_lower = sum(1 for t in tokens if t and t[0].islower())

    if n_comma: feats["has_comma"] = 1.0
    if n_period: feats["has_period"] = 1.0
    if n_paren_open: feats["has_paren"] = 1.0
    if n_colon: feats["has_colon"] = 1.0
    if n_dash: feats["has_dash"] = 1.0
    if n_year: feats["has_year"] = 1.0

    # Bucketed counts (so the LR can distinguish "one comma" from "many commas").
    def bucket(name, c):
        if c == 0: return
        if c == 1: feats[f"{name}=1"] = 1.0
        elif c == 2: feats[f"{name}=2"] = 1.0
        else: feats[f"{name}=3+"] = 1.0
    bucket("n_comma", n_comma)
    bucket("n_period", n_period)
    bucket("n_paren", n_paren_open)
    bucket("n_year", n_year)

    feats["punct_frac"] = n_punct / max(1, n)
    feats["upper_frac"] = n_upper / max(1, n)
    feats["lower_frac"] = n_lower / max(1, n)
    feats["digit_frac"] = n_digit_token / max(1, n)

    # Position of first comma / first paren — strong signal for personal-name
    # vs corporate-name vs meeting headings.
    for i, t in enumerate(tokens):
        if t == ",":
            feats["first_comma_pos"] = i / max(1, n)
            break
    for i, t in enumerate(tokens):
        if t == "(":
            feats["first_paren_pos"] = i / max(1, n)
            break
    for i, t in enumerate(tokens):
        if t == ".":
            feats["first_period_pos"] = i / max(1, n)
            break

    return feats


def train_header_classifier(train_rows, dev_rows):
    print(">>> training header classifier ...", file=sys.stderr)
    X_train_dicts = [header_doc_features(r["tokens"]) for r in train_rows]
    y_train = [r["header"] for r in train_rows]
    X_dev_dicts = [header_doc_features(r["tokens"]) for r in dev_rows]
    y_dev = [r["header"] for r in dev_rows]

    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_dicts)
    X_dev = vec.transform(X_dev_dicts)

    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(X_train, y_train)
    print(f"  fit took {time.time()-t0:.1f}s", file=sys.stderr)

    pred = clf.predict(X_dev)
    acc = (pred == np.array(y_dev)).mean()
    print(f"  dev accuracy: {acc*100:.2f}% ({len(dev_rows)} examples)", file=sys.stderr)
    print(file=sys.stderr)
    return clf, vec


# --- CRF tag model ----------------------------------------------------------

def build_crf_inputs(rows):
    X = [sequence_features(r["tokens"], r["header"]) for r in rows]
    y = [r["tags"] for r in rows]
    return X, y


def train_crf(train_rows, dev_rows):
    print(">>> training CRF tag model ...", file=sys.stderr)
    X_train, y_train = build_crf_inputs(train_rows)
    X_dev, y_dev = build_crf_inputs(dev_rows)
    print(f"  {len(X_train)} training sequences, {len(X_dev)} dev sequences",
          file=sys.stderr)

    t0 = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False,
    )
    crf.fit(X_train, y_train)
    print(f"  fit took {time.time()-t0:.1f}s", file=sys.stderr)

    pred = crf.predict(X_dev)

    # Token-level metrics.
    flat_true: list[str] = []
    flat_pred: list[str] = []
    for yt, yp in zip(y_dev, pred):
        flat_true += yt
        flat_pred += yp
    print(file=sys.stderr)
    print("=== token-level dev metrics ===", file=sys.stderr)
    labels_present = sorted(set(flat_true))
    print(classification_report(flat_true, flat_pred, labels=labels_present,
                                digits=4, zero_division=0))

    # Sequence-level exact match.
    n_exact = sum(1 for yt, yp in zip(y_dev, pred) if yt == yp)
    print(f"sequence exact match: {n_exact}/{len(y_dev)} "
          f"({100*n_exact/len(y_dev):.2f}%)")

    return crf


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=0,
                   help="optional cap on train rows for quick experiments")
    p.add_argument("--skip-crf", action="store_true",
                   help="only retrain the header classifier")
    p.add_argument("--skip-header", action="store_true",
                   help="only retrain the CRF")
    args = p.parse_args(argv)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_split("train")
    dev_rows = load_split("dev")
    if args.max_train and args.max_train < len(train_rows):
        train_rows = train_rows[: args.max_train]
        print(f"capped train to {len(train_rows)} rows", file=sys.stderr)

    print(f"train={len(train_rows)}, dev={len(dev_rows)}", file=sys.stderr)
    print(file=sys.stderr)

    # Header label distribution sanity check.
    cnt = Counter(r["header"] for r in train_rows)
    print("train header distribution:", dict(cnt.most_common()), file=sys.stderr)
    print(file=sys.stderr)

    # 1. header classifier
    if not args.skip_header:
        clf, vec = train_header_classifier(train_rows, dev_rows)
        with (MODEL_DIR / "header_clf.pkl").open("wb") as f:
            pickle.dump({"clf": clf, "vec": vec}, f)
        print(f"saved header classifier -> {MODEL_DIR/'header_clf.pkl'}", file=sys.stderr)

    # 2. CRF
    if not args.skip_crf:
        crf = train_crf(train_rows, dev_rows)
        with (MODEL_DIR / "tag_crf.pkl").open("wb") as f:
            pickle.dump({"crf": crf}, f)
        print(f"saved CRF -> {MODEL_DIR/'tag_crf.pkl'}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
