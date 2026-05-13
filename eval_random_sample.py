#!/usr/bin/env python3
"""Evaluate header LR + CRF on a random sample of the FULL aligned corpus.

This avoids the test-set bias problem: our dev/test splits were drawn from a
pattern-stratified pool, so they over-represent rare subfield combinations.
A uniform random sample tells us what the model actually does in production.

Steps:
  1. Reservoir-sample N rows from aligned_1xx.jsonl.
  2. Filter out any rows that appear in the training set (token+tag identity).
  3. Apply the same ind2-collapse rule used during training.
  4. Run header LR + CRF, report metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

from features import sequence_features
from train import header_doc_features
from build_splits_v2 import collapse_header

ALIGNED = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")


def reservoir_sample(path: Path, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    with path.open("r", encoding="utf-8") as f:
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
            if (i + 1) % 1_000_000 == 0:
                print(f"  scanned {i+1:,}", file=sys.stderr)
    return sample


def row_signature(rec: dict) -> tuple:
    """Stable identity for de-duplication against the train split."""
    return (rec["header"], tuple(rec["tokens"]), tuple(rec["tags"]))


def load_train_signatures() -> set[tuple]:
    sigs: set[tuple] = set()
    with (SPLITS / "train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sigs.add(row_signature(rec))
    return sigs


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000, help="random sample size")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--examples", type=int, default=10,
                   help="number of failure examples to print")
    args = p.parse_args(argv)

    print(f"reservoir-sampling {args.n:,} rows from {ALIGNED} ...", file=sys.stderr)
    raw = reservoir_sample(ALIGNED, args.n, args.seed)
    rows = [json.loads(l) for l in raw]
    print(f"  got {len(rows):,} rows", file=sys.stderr)

    # Apply ind2 collapse so labels match how the model was trained.
    for r in rows:
        r["header"] = collapse_header(r["header"])

    # De-duplicate against the training set.
    print("loading train-set signatures for dedup ...", file=sys.stderr)
    train_sigs = load_train_signatures()
    n_overlap = sum(1 for r in rows if row_signature(r) in train_sigs)
    rows = [r for r in rows if row_signature(r) not in train_sigs]
    print(f"  removed {n_overlap:,} rows that overlapped with train; "
          f"{len(rows):,} rows remain", file=sys.stderr)

    # Load models.
    with (MODEL_DIR / "header_clf.pkl").open("rb") as f:
        bundle = pickle.load(f)
    clf, vec = bundle["clf"], bundle["vec"]
    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]

    # 1. Header prediction.
    Xh = vec.transform([header_doc_features(r["tokens"]) for r in rows])
    pred_headers = clf.predict(Xh)
    true_headers = [r["header"] for r in rows]

    n_hdr_correct = sum(1 for a, b in zip(pred_headers, true_headers) if a == b)
    print()
    print("=== results on uniform random sample ===")
    print(f"sample size:       {len(rows):,}")
    print(f"header accuracy:   {n_hdr_correct}/{len(rows)} ({100*n_hdr_correct/len(rows):.2f}%)")

    # 2. Oracle CRF (true header → tags).
    X_true = [sequence_features(r["tokens"], h) for r, h in zip(rows, true_headers)]
    pred_tags_true = crf.predict(X_true)
    n_oracle = sum(1 for r, yp in zip(rows, pred_tags_true) if r["tags"] == yp)
    print(f"CRF oracle exact:  {n_oracle}/{len(rows)} ({100*n_oracle/len(rows):.2f}%)")

    # 3. End-to-end (predicted header → tags).
    X_pred = [sequence_features(r["tokens"], h) for r, h in zip(rows, pred_headers)]
    pred_tags = crf.predict(X_pred)
    n_e2e = sum(1 for r, yp in zip(rows, pred_tags) if r["tags"] == yp)
    print(f"end-to-end exact:  {n_e2e}/{len(rows)} ({100*n_e2e/len(rows):.2f}%)")

    # Token-level for sanity.
    n_tok = 0
    n_tok_ok = 0
    for r, yp in zip(rows, pred_tags):
        for t, p in zip(r["tags"], yp):
            n_tok += 1
            if t == p:
                n_tok_ok += 1
    print(f"end-to-end token:  {n_tok_ok:,}/{n_tok:,} ({100*n_tok_ok/n_tok:.2f}%)")
    print()

    # Header confusions.
    confusion = Counter()
    for t, p in zip(true_headers, pred_headers):
        if t != p:
            confusion[(t, p)] += 1
    print("=== top header confusions (true -> pred) ===")
    for (t, p), c in confusion.most_common(10):
        print(f"  {t:10s} -> {p:10s}  : {c:,}")
    print()

    # Per-header end-to-end accuracy.
    print("=== end-to-end accuracy by header (top 12 by support) ===")
    by_hdr_total = Counter()
    by_hdr_ok = Counter()
    for r, h_pred, yp in zip(rows, pred_headers, pred_tags):
        h = r["header"]
        by_hdr_total[h] += 1
        if h == h_pred and r["tags"] == yp:
            by_hdr_ok[h] += 1
    for h, total in by_hdr_total.most_common(12):
        ok = by_hdr_ok[h]
        print(f"  {h:10s}  {ok:>6d}/{total:>6d}  ({100*ok/total:.2f}%)")
    print()

    # Show some failure examples.
    if args.examples:
        print(f"=== {args.examples} end-to-end failure examples ===")
        shown = 0
        for r, h_pred, yp in zip(rows, pred_headers, pred_tags):
            if r["tags"] == yp and h_pred == r["header"]:
                continue
            print(f"  tokens: {' / '.join(r['tokens'])}")
            print(f"  true h: {r['header']}    pred h: {h_pred}")
            print(f"  gold:   {' '.join(r['tags'])}")
            print(f"  pred:   {' '.join(yp)}")
            print()
            shown += 1
            if shown >= args.examples:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
