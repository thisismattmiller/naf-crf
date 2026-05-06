#!/usr/bin/env python3
"""Export the trained CRF + header LR to a single JSON file for the JS runtime.

The JSON contains everything the browser needs to replicate inference:

{
  "labels": [...],                 # CRF tag labels in fixed order
  "transitions": {                 # transition[from][to] = weight
     "B-a": {"I-a": 1.23, ...},
     ...
  },
  "state_features": {              # state_features[label][feat] = weight
     "B-a": {"w=smith": 0.4, "shape=Aa": 0.2, ...},
     ...
  },
  "header_classifier": {
     "labels": [...],              # header label classes
     "weights": {                   # weights[class][feat] = weight
        "100|1|#": {"has_comma": 0.5, ...},
        ...
     },
     "intercepts": {"100|1|#": 0.1, ...}
  }
}

The JS runtime computes:
  state_score(label, token_features) = sum_f (state_features[label][f] * f_value)
  Viterbi over tokens with transition_score(prev, cur) = transitions[prev][cur].

Header inference: argmax over (intercept[c] + sum_f (weights[c][f] * doc[f])).
(Since LR uses softmax, argmax of the unnormalized logit is correct.)
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

MODEL_DIR = Path("/Volumes/ImNotGlum/naf-crf/models")
OUT_JSON = Path("/Volumes/ImNotGlum/naf-crf/models/model.json")
OUT_GZ = Path("/Volumes/ImNotGlum/naf-crf/models/model.json.gz")


def export_crf(crf, weight_threshold: float = 0.01):
    """sklearn-crfsuite wraps python-crfsuite; we read transition_features_
    and state_features_ which are exposed dicts.

    State features with |weight| < threshold are pruned. Transitions are kept
    in full (only ~1k of them). The threshold is small enough that pruning
    has near-zero impact on Viterbi outputs but cuts file size dramatically.
    """
    transitions: dict[str, dict[str, float]] = {}
    for (frm, to), w in crf.transition_features_.items():
        transitions.setdefault(frm, {})[to] = w

    state_features: dict[str, dict[str, float]] = {}
    pruned = 0
    kept = 0
    for (feat, label), w in crf.state_features_.items():
        if abs(w) < weight_threshold:
            pruned += 1
            continue
        state_features.setdefault(label, {})[feat] = w
        kept += 1
    print(f"  CRF state features: kept {kept:,}, pruned {pruned:,} "
          f"(threshold |w|<{weight_threshold})")

    return {
        "labels": list(crf.classes_),
        "transitions": transitions,
        "state_features": state_features,
    }


def load_header_vocab():
    p = Path("/Volumes/ImNotGlum/naf-crf/splits/header_vocab.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)["vocab"]


def export_header_classifier(clf, vec):
    feature_names = list(vec.get_feature_names_out())
    classes = list(clf.classes_)

    weights: dict[str, dict[str, float]] = {}
    intercepts: dict[str, float] = {}

    coefs = clf.coef_  # (n_classes, n_features) — for binary, shape is (1, n_features)
    if coefs.shape[0] == 1 and len(classes) == 2:
        # Binary case: coef_[0] gives weights for class 1 vs class 0.
        positive = classes[1]
        negative = classes[0]
        weights[positive] = {}
        weights[negative] = {}
        for j, fname in enumerate(feature_names):
            w = float(coefs[0, j])
            if w != 0:
                weights[positive][fname] = w
                weights[negative][fname] = -w
        intercepts[positive] = float(clf.intercept_[0])
        intercepts[negative] = -float(clf.intercept_[0])
    else:
        for ci, cls in enumerate(classes):
            row = coefs[ci]
            wd = {}
            for j, fname in enumerate(feature_names):
                w = float(row[j])
                if w != 0:
                    wd[fname] = w
            weights[cls] = wd
            intercepts[cls] = float(clf.intercept_[ci])

    return {
        "labels": classes,
        "weights": weights,
        "intercepts": intercepts,
        "vocab": load_header_vocab(),
    }


def main():
    with (MODEL_DIR / "tag_crf.pkl").open("rb") as f:
        crf = pickle.load(f)["crf"]
    with (MODEL_DIR / "header_clf.pkl").open("rb") as f:
        bundle = pickle.load(f)
        clf, vec = bundle["clf"], bundle["vec"]

    payload = {
        "version": 1,
        "tags": export_crf(crf),
        "header_classifier": export_header_classifier(clf, vec),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with gzip.open(OUT_GZ, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False)

    sz_json = OUT_JSON.stat().st_size
    sz_gz = OUT_GZ.stat().st_size
    print(f"wrote {OUT_JSON} ({sz_json/1024:.1f} KB)")
    print(f"wrote {OUT_GZ} ({sz_gz/1024:.1f} KB gzip)")

    # Print some quick stats.
    n_state = sum(len(d) for d in payload["tags"]["state_features"].values())
    n_trans = sum(len(d) for d in payload["tags"]["transitions"].values())
    n_hdr_w = sum(len(d) for d in payload["header_classifier"]["weights"].values())
    print(f"  CRF labels:         {len(payload['tags']['labels'])}")
    print(f"  CRF transitions:    {n_trans:,}")
    print(f"  CRF state features: {n_state:,}")
    print(f"  header LR classes:  {len(payload['header_classifier']['labels'])}")
    print(f"  header LR weights:  {n_hdr_w:,}")


if __name__ == "__main__":
    main()
