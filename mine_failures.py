#!/usr/bin/env python3
"""Failure miner: run the CRF over a large random sample of aligned 1XX rows,
diff predictions against gold, and cluster the per-token disagreements by
signature so we can spot recurring patterns suitable for regex rules.

Outputs:
  - failures.jsonl  (one line per failed row, with full context)
  - failure_signatures.txt  (top signatures, ranked by count)
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from features import sequence_features
from rules import apply_rules
from build_splits_v2 import collapse_header

ALIGNED = Path("/Volumes/ImNotGlum/naf-crf/aligned_1xx.jsonl")
MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")
OUT_DIR = Path("/Volumes/ImNotGlum/naf-crf/analysis")


def shape(token: str) -> str:
    """Same shape buckets as features.py — coarsen punct/letter/digit runs."""
    out = []
    last = ""
    for ch in token:
        cat = unicodedata.category(ch)
        if cat.startswith("Lu"):
            c = "A"
        elif cat.startswith("L"):
            c = "a"
        elif cat.startswith("N"):
            c = "9"
        elif cat.startswith("P") or cat.startswith("S"):
            c = "p"
        else:
            c = "?"
        if c != last:
            out.append(c)
            last = c
    return "".join(out)


def tag_code(tag: str) -> str:
    """B-a -> a, I-a -> a, O -> O"""
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1]


def span_diffs(tokens, gold, pred):
    """Yield (gold_code, pred_code, span_token_indices) for each maximal
    contiguous run where the *subfield code* (ignoring B/I) differs."""
    n = len(tokens)
    i = 0
    while i < n:
        gc = tag_code(gold[i])
        pc = tag_code(pred[i])
        if gc == pc:
            i += 1
            continue
        # Extend the disagreement run while the pair (gc, pc) stays the same.
        j = i
        while j + 1 < n:
            ngc = tag_code(gold[j + 1])
            npc = tag_code(pred[j + 1])
            if ngc == gc and npc == pc:
                j += 1
            else:
                break
        yield (gc, pc, range(i, j + 1))
        i = j + 1


def context_token(tokens, i):
    """Compact context token: shape if word/digit, literal if punct."""
    if i < 0 or i >= len(tokens):
        return "·"
    t = tokens[i]
    s = shape(t)
    if s == "p":
        return repr(t)
    return s


def signature(tokens, gold, pred, span_range, gold_code, pred_code, header):
    """Build a clustering signature for one failure span.

    Components:
      - header
      - direction: gold_code -> pred_code
      - span content sketch: sequence of shapes (compressed)
      - left context: shape of token immediately before span (or '·' if BOS)
      - right context: shape of token immediately after span (or '·' if EOS)
      - tail-of-span literal: the LAST token of the span if it's a word, lowercased
    """
    span_list = list(span_range)
    left = context_token(tokens, span_list[0] - 1)
    right = context_token(tokens, span_list[-1] + 1)
    shapes_in_span = "_".join(context_token(tokens, k) for k in span_list)
    # Tail word (often the distinguishing literal for $l "English", $k "Selections", etc.)
    last_tok = tokens[span_list[-1]]
    last_shape = shape(last_tok)
    if last_shape in ("Aa", "A", "a"):  # word-like
        tail_word = f"tail={last_tok.lower()}"
    else:
        tail_word = f"tail_shape={last_shape}"
    return (f"H={header} {gold_code}->{pred_code} | "
            f"L={left} R={right} | span={shapes_in_span} | {tail_word}")


def reservoir_sample(path: Path, k: int, seed: int):
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


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500_000, help="sample size")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--top", type=int, default=60, help="how many signatures to print")
    p.add_argument("--examples-per-sig", type=int, default=3)
    p.add_argument("--batch", type=int, default=2000, help="CRF inference batch size")
    p.add_argument("--apply-rules", action="store_true", default=True,
                   help="apply current post-CRF rules before counting failures "
                        "(default true so we surface only NEW patterns)")
    p.add_argument("--no-apply-rules", dest="apply_rules", action="store_false")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures_path = OUT_DIR / "failures.jsonl"
    sigs_path = OUT_DIR / "failure_signatures.txt"

    print(f"reservoir-sampling {args.n:,} rows ...", file=sys.stderr)
    raw = reservoir_sample(ALIGNED, args.n, args.seed)
    print(f"  got {len(raw):,} rows", file=sys.stderr)

    print("loading CRF ...", file=sys.stderr)
    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]

    sig_counts: Counter[str] = Counter()
    sig_examples: dict[str, list[tuple]] = defaultdict(list)
    n_total = 0
    n_failed_rows = 0
    n_failed_spans = 0

    # Per-direction counter (gold_code -> pred_code) for a higher-level view.
    direction_counts: Counter[tuple[str, str]] = Counter()

    with failures_path.open("w", encoding="utf-8") as ff:
        # Process in batches for CRF efficiency.
        for start in range(0, len(raw), args.batch):
            batch_lines = raw[start : start + args.batch]
            batch = [json.loads(l) for l in batch_lines]
            # Apply the same ind2-collapse the production pipeline uses, so
            # the CRF receives the same header it would in real use.
            for rec in batch:
                rec["header"] = collapse_header(rec["header"])
            X = [sequence_features(r["tokens"], r["header"]) for r in batch]
            pred = crf.predict(X)

            for rec, yp_raw in zip(batch, pred):
                n_total += 1
                yp = yp_raw
                if args.apply_rules:
                    yp, _ = apply_rules(rec["header"], rec["tokens"], yp_raw)
                if rec["tags"] == yp:
                    continue
                n_failed_rows += 1
                rec_failures = []
                for gc, pc, span in span_diffs(rec["tokens"], rec["tags"], yp):
                    n_failed_spans += 1
                    direction_counts[(gc, pc)] += 1
                    sig = signature(rec["tokens"], rec["tags"], yp, span,
                                    gc, pc, rec["header"])
                    sig_counts[sig] += 1
                    if len(sig_examples[sig]) < args.examples_per_sig:
                        span_list = list(span)
                        sig_examples[sig].append((
                            rec["tokens"], rec["tags"], yp, span_list, rec["header"],
                        ))
                    rec_failures.append({
                        "gold_code": gc,
                        "pred_code": pc,
                        "span": list(span),
                    })
                ff.write(json.dumps({
                    "header": rec["header"],
                    "tokens": rec["tokens"],
                    "gold": rec["tags"],
                    "pred": yp,
                    "failures": rec_failures,
                }, ensure_ascii=False))
                ff.write("\n")

            if (start + args.batch) % 50_000 == 0 or start + args.batch >= len(raw):
                print(f"  processed {min(start+args.batch, len(raw)):,} "
                      f"failed_rows={n_failed_rows:,} "
                      f"failed_spans={n_failed_spans:,}",
                      file=sys.stderr)

    print()
    print(f"total rows processed: {n_total:,}")
    print(f"rows with at least one failure: {n_failed_rows:,} "
          f"({100*n_failed_rows/n_total:.2f}%)")
    print(f"distinct failure spans: {n_failed_spans:,}")
    print(f"distinct signatures: {len(sig_counts):,}")
    print()

    print("=== top gold -> pred directions (per failure span) ===")
    for (gc, pc), c in direction_counts.most_common(20):
        print(f"  {gc:>3s} -> {pc:<3s}  : {c:,}")
    print()

    # Write detailed signature report.
    with sigs_path.open("w", encoding="utf-8") as sf:
        sf.write(f"# failure signatures (top {args.top} of {len(sig_counts):,})\n")
        sf.write(f"# sample size: {n_total:,} rows | failed rows: {n_failed_rows:,}\n\n")
        for sig, c in sig_counts.most_common(args.top):
            sf.write(f"### {c:>6d} ## {sig}\n")
            for tokens, gold, pred, span_list, header in sig_examples[sig]:
                marked_tokens = []
                for i, t in enumerate(tokens):
                    if i in span_list:
                        marked_tokens.append(f"[{t}]")
                    else:
                        marked_tokens.append(t)
                sf.write(f"    tokens : {' '.join(marked_tokens)}\n")
                gold_anno = " ".join(
                    f"*{g}*" if i in span_list else g for i, g in enumerate(gold)
                )
                pred_anno = " ".join(
                    f"*{p}*" if i in span_list else p for i, p in enumerate(pred)
                )
                sf.write(f"    gold   : {gold_anno}\n")
                sf.write(f"    pred   : {pred_anno}\n")
                sf.write(f"    header : {header}\n")
                sf.write("\n")
            sf.write("\n")
    print(f"wrote {failures_path}")
    print(f"wrote {sigs_path}")
    print()
    print(f"=== top {min(args.top, 20)} signatures ===")
    for sig, c in sig_counts.most_common(min(args.top, 20)):
        print(f"  {c:>5d}  {sig[:140]}")


if __name__ == "__main__":
    main(sys.argv[1:])
