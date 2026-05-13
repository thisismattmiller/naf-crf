#!/usr/bin/env python3
"""Measure each rule's impact on a held-out random sample.

For each row in the sample:
  1. Predict with CRF using oracle header.
  2. Run rules sequentially. For each rule, record:
       - rows where (CRF wrong) and (rule changed something) and (after-rule == gold)  -> fix
       - rows where (CRF right) and (rule changed something) and (after-rule != gold)  -> break
       - rows where the rule fired but neither fixed nor broke (e.g. partial fix)      -> neutral
  3. Report fix-count, break-count, and the net impact per rule.

Critically, we evaluate rules INDEPENDENTLY (each rule applied to the raw CRF
output) and also CUMULATIVELY (the full pipeline). The first tells us each
rule's marginal value; the second tells us what they do together.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from features import sequence_features
from rules import RULES, apply_rules
from build_splits_v2 import collapse_header

ALIGNED = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")
SPLITS = Path("/Volumes/ImNotGlum/naf-crf/splits")


def reservoir_sample(path: Path, k: int, seed: int):
    rng = random.Random(seed)
    sample = []
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


def load_train_signatures():
    sigs = set()
    with (SPLITS / "train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sigs.add((rec["header"], tuple(rec["tokens"]), tuple(rec["tags"])))
    return sigs


def main(argv):
    p = argparse.ArgumentParser()
    # Use a different seed than the failure miner so we measure on fresh data.
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--examples", type=int, default=5)
    args = p.parse_args(argv)

    print(f"reservoir-sampling {args.n:,} rows ...", file=sys.stderr)
    raw = reservoir_sample(ALIGNED, args.n, args.seed)
    rows = [json.loads(l) for l in raw]
    for r in rows:
        r["header"] = collapse_header(r["header"])

    # Optional dedup against train (small overlap; we keep it simple).
    train_sigs = load_train_signatures()
    rows = [r for r in rows
            if (r["header"], tuple(r["tokens"]), tuple(r["tags"])) not in train_sigs]
    print(f"  {len(rows):,} rows after train-dedup", file=sys.stderr)

    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]

    # CRF prediction with ORACLE header.
    print("running CRF ...", file=sys.stderr)
    X = [sequence_features(r["tokens"], r["header"]) for r in rows]
    pred = crf.predict(X)

    n = len(rows)
    crf_correct = sum(1 for r, yp in zip(rows, pred) if r["tags"] == yp)
    print()
    print(f"baseline CRF exact-match: {crf_correct}/{n} ({100*crf_correct/n:.3f}%)")
    print()

    # ---- 1) Each rule applied INDEPENDENTLY ----
    print("=== per-rule independent measurement ===")
    print(f"  {'rule':40s}  {'fired':>7s}  {'fix':>5s}  {'break':>5s}  {'neutral':>7s}  {'net':>5s}")
    per_rule_examples: dict[str, dict[str, list]] = defaultdict(lambda: {"fix": [], "break": []})

    for name, rule in RULES:
        n_fired = 0
        n_fix = 0
        n_break = 0
        n_neutral = 0
        for r, yp in zip(rows, pred):
            applied = rule(r["header"], r["tokens"], yp)
            if applied == yp:
                continue
            n_fired += 1
            was_correct = (yp == r["tags"])
            now_correct = (applied == r["tags"])
            if not was_correct and now_correct:
                n_fix += 1
                if len(per_rule_examples[name]["fix"]) < args.examples:
                    per_rule_examples[name]["fix"].append((r, yp, applied))
            elif was_correct and not now_correct:
                n_break += 1
                if len(per_rule_examples[name]["break"]) < args.examples:
                    per_rule_examples[name]["break"].append((r, yp, applied))
            else:
                n_neutral += 1
        net = n_fix - n_break
        print(f"  {name:40s}  {n_fired:>7d}  {n_fix:>5d}  {n_break:>5d}  {n_neutral:>7d}  {net:>+5d}")
    print()

    # ---- 2) Rules applied CUMULATIVELY (in registry order) ----
    print("=== cumulative pipeline ===")
    cumulative_pred = []
    fire_counts: Counter[str] = Counter()
    for r, yp in zip(rows, pred):
        final, fired = apply_rules(r["header"], r["tokens"], yp)
        cumulative_pred.append(final)
        for nm in fired:
            fire_counts[nm] += 1

    new_correct = sum(1 for r, yp in zip(rows, cumulative_pred) if r["tags"] == yp)
    delta = new_correct - crf_correct
    print(f"  CRF baseline:    {crf_correct}/{n} ({100*crf_correct/n:.3f}%)")
    print(f"  with rules:      {new_correct}/{n} ({100*new_correct/n:.3f}%)  Δ {delta:+d}")
    print()
    print("  rule fire counts under pipeline:")
    for nm, c in fire_counts.most_common():
        print(f"    {nm}: {c}")
    print()

    # Show some fix/break examples per rule.
    for name, _ in RULES:
        ex = per_rule_examples[name]
        if not ex["fix"] and not ex["break"]:
            continue
        print(f"--- {name} examples ---")
        for kind in ("fix", "break"):
            for r, yp, applied in ex[kind][:args.examples]:
                print(f"  [{kind}] header={r['header']}")
                print(f"    tokens: {' '.join(r['tokens'])}")
                print(f"    gold:   {' '.join(r['tags'])}")
                print(f"    crf:    {' '.join(yp)}")
                print(f"    after:  {' '.join(applied)}")
                print()


if __name__ == "__main__":
    main(sys.argv[1:])
