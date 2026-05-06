#!/usr/bin/env python3
"""Token-level feature extraction for the CRF.

Design constraint: every feature here must be reproducible in plain JavaScript
with no external libraries, since the trained model will run in-browser. So we
stick to: lowercase comparison, character classification (letter/digit/punct),
prefix/suffix slicing, exact-token match, and a small fixed inventory of
"shape" buckets.

The header is supplied as a context-level string and is emitted as a feature
on every token (so the CRF can condition tag transitions on header).
"""

from __future__ import annotations

import re
import unicodedata


def shape(token: str) -> str:
    """Bucket a token into a coarse character-class string.

    Examples:
      "Smith"   -> "Aa"
      "JFK"     -> "A"
      "1962"    -> "9"
      "1962-"   -> "9p"
      ","       -> "p"
      "M."      -> "Ap"
      "él"      -> "Aa"   (treats accented letters as letters)
    """
    out: list[str] = []
    last = ""
    for ch in token:
        cat = unicodedata.category(ch)
        if cat.startswith("Lu"):
            c = "A"
        elif cat.startswith("L"):  # Ll, Lt, Lo, Lm
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


def is_punct(token: str) -> bool:
    if not token:
        return False
    cat = unicodedata.category(token[0])
    return cat.startswith("P") or cat.startswith("S")


def is_digit(token: str) -> bool:
    return token.isdigit()


def has_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def token_features(tokens: list[str], i: int, header: str) -> dict[str, float]:
    """Generate features for token i in a sequence."""
    t = tokens[i]
    feats: dict[str, float] = {
        # Bias term (always on).
        "bias": 1.0,
        # The header is a global feature on every token.
        f"H={header}": 1.0,
        # Token identity, lowercased.
        f"w={t.lower()}": 1.0,
        # Shape.
        f"shape={shape(t)}": 1.0,
    }

    # Length bucket (helpful for digit-year vs digit-page distinctions).
    if len(t) == 1:
        feats["len=1"] = 1.0
    elif len(t) <= 3:
        feats["len=2-3"] = 1.0
    elif len(t) <= 6:
        feats["len=4-6"] = 1.0
    else:
        feats["len=7+"] = 1.0

    # Character-class flags.
    if is_punct(t):
        feats["punct"] = 1.0
        feats[f"punct={t}"] = 1.0
    if is_digit(t):
        feats["digit"] = 1.0
        if len(t) == 4:
            feats["year_like"] = 1.0
    elif has_digit(t):
        feats["alnum"] = 1.0
    if t and t[0].isupper():
        feats["upper_first"] = 1.0
    if t and t.isupper() and len(t) > 1:
        feats["all_upper"] = 1.0

    # Prefix / suffix (lowercased) — useful for spotting tokens like "ed.", "Mrs"
    low = t.lower()
    if len(low) >= 2:
        feats[f"pre2={low[:2]}"] = 1.0
        feats[f"suf2={low[-2:]}"] = 1.0
    if len(low) >= 3:
        feats[f"pre3={low[:3]}"] = 1.0
        feats[f"suf3={low[-3:]}"] = 1.0

    # Position features — start/end of sequence.
    if i == 0:
        feats["BOS"] = 1.0
    if i == len(tokens) - 1:
        feats["EOS"] = 1.0

    # Window features (prev / next tokens, lowercased + shape).
    if i > 0:
        prev = tokens[i - 1]
        feats[f"-1w={prev.lower()}"] = 1.0
        feats[f"-1shape={shape(prev)}"] = 1.0
    else:
        feats["-1=BOS"] = 1.0
    if i + 1 < len(tokens):
        nxt = tokens[i + 1]
        feats[f"+1w={nxt.lower()}"] = 1.0
        feats[f"+1shape={shape(nxt)}"] = 1.0
    else:
        feats["+1=EOS"] = 1.0

    return feats


def sequence_features(tokens: list[str], header: str) -> list[dict[str, float]]:
    return [token_features(tokens, i, header) for i in range(len(tokens))]
