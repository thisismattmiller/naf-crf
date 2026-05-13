#!/usr/bin/env python3
"""Post-processing rules that fire on (header, tokens, crf_predicted_tags) and
produce corrected tags.

Each rule is a function `(header, tokens, tags) -> tags` that returns a NEW
tag list (does not mutate). It should be a no-op when its preconditions
aren't met.

Conventions:
  - tags are BIO with subfield codes: B-a, I-a, B-d, I-d, ..., O.
  - We never change a span's start position; we only change WHICH subfield a
    span belongs to. This keeps rules narrow and easy to reason about.
  - Each rule has a name used in the measurement runner so we can report
    fix-count vs break-count per rule.

Convention for span identification: a "span" is a maximal run of tags whose
*code* (post B-/I- prefix) is the same. Rules typically retag a whole span.
"""

from __future__ import annotations

import re
from typing import Callable


def tag_code(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1]


def retag_span(tags: list[str], start: int, end_exclusive: int, code: str) -> list[str]:
    """Return a new tag list with positions [start, end_exclusive) tagged
    B-<code> I-<code> ... I-<code>."""
    new = list(tags)
    for i in range(start, end_exclusive):
        new[i] = ("B-" if i == start else "I-") + code
    return new


def find_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Yield (start, end_exclusive, code) for each maximal same-code span."""
    out: list[tuple[int, int, str]] = []
    n = len(tags)
    i = 0
    while i < n:
        code = tag_code(tags[i])
        j = i + 1
        while j < n and tag_code(tags[j]) == code:
            j += 1
        out.append((i, j, code))
        i = j
    return out


# --- helpers for tokens ---------------------------------------------------

INITIAL_RE = re.compile(r"^[A-ZÀ-ÝĀ-ŪŴ-Ɏ]{1,3}$")  # short capitalized letter run


def is_word(token: str) -> bool:
    return bool(re.fullmatch(r"[\w]+", token, flags=re.UNICODE)) and not token.isdigit()


def is_capitalized_word(token: str) -> bool:
    # A token whose first char is uppercase letter and whose rest is letters.
    if not token:
        return False
    return token[0].isupper() and token.isalpha()


def is_initial(token: str) -> bool:
    # A short uppercase letter run, typically 1-3 chars: "F", "V", "Es", "Rā".
    if not token:
        return False
    if not token[0].isupper():
        return False
    if not token.isalpha():
        return False
    return 1 <= len(token) <= 3


def looks_like_paren_group(tokens: list[str], start: int) -> tuple[int, int] | None:
    """If tokens[start] is '(' and there's a matching ')' nearby (within a few
    tokens), return (start, end_exclusive) covering the paren group. Else None.
    """
    if start >= len(tokens) or tokens[start] != "(":
        return None
    for j in range(start + 1, min(len(tokens), start + 20)):
        if tokens[j] == ")":
            return (start, j + 1)
    return None


# --- rules ----------------------------------------------------------------


def _share_first_letter(a: str, b: str) -> bool:
    """True if a and b start with the same letter (case-insensitive,
    accent-insensitive — 'E' matches 'É')."""
    if not a or not b:
        return False
    import unicodedata as _u
    def _normalize(c: str) -> str:
        # Remove combining accents.
        nfd = _u.normalize("NFD", c)
        for ch in nfd:
            if _u.category(ch) != "Mn":
                return ch.lower()
        return ""
    return _normalize(a[0]) == _normalize(b[0])


def rule_q_fuller_form_after_initial(header: str, tokens: list[str],
                                     tags: list[str]) -> list[str]:
    """Pattern 2: 'Coombs, F. (Frederick)' — single-initial followed by a
    parenthesized capitalized word *that starts with the same letter* is $q
    (fuller form), not $s.

    The first-letter check is the key disambiguator vs. occupation/role:
      Baker, L. (Pharmacist)  -> L vs P, no match, leave alone (gold says $c)
      Coombs, F. (Frederick)  -> F vs F, match, fix to $q
    """
    if not header.startswith("100"):
        return tags
    n = len(tokens)
    new = tags
    for k in range(2, n):
        if tokens[k] != "(":
            continue
        if not (k - 2 >= 0 and is_initial(tokens[k - 2]) and tokens[k - 1] == "."):
            continue
        paren = looks_like_paren_group(tokens, k)
        if paren is None:
            continue
        ps, pe = paren
        inside = tokens[ps + 1 : pe - 1]
        if len(inside) != 1 or not is_capitalized_word(inside[0]):
            continue
        # CRITICAL: only fire when the fuller form shares first letter with
        # the preceding initial.
        if not _share_first_letter(tokens[k - 2], inside[0]):
            continue
        if all(tag_code(new[i]) == "q" for i in range(ps, pe)):
            continue
        new = retag_span(new, ps, pe, "q")
    return new


def rule_c_paren_role_after_full_name(header: str, tokens: list[str],
                                      tags: list[str]) -> list[str]:
    """Pattern 3: 'Crosse, Thomas (Goldsmith)' — parenthesized single capitalized
    word after a full given name is $c, but ONLY when the inside word does NOT
    share its first letter with the preceding name (otherwise it's likely a
    fuller form: 'Roberts, Ken (Kenneth)' -> Ken matches Kenneth).
    """
    if not header.startswith("100|"):
        return tags
    n = len(tokens)
    new = tags
    for k in range(1, n):
        if tokens[k] != "(":
            continue
        paren = looks_like_paren_group(tokens, k)
        if paren is None:
            continue
        ps, pe = paren
        inside = tokens[ps + 1 : pe - 1]
        if len(inside) != 1 or not is_capitalized_word(inside[0]):
            continue
        prev = tokens[k - 1] if k - 1 >= 0 else ""
        if not is_capitalized_word(prev) or len(prev) <= 1:
            continue
        # CRITICAL: skip if inside word looks like a fuller form (shares first
        # letter with the preceding word).
        if _share_first_letter(prev, inside[0]):
            continue
        if all(tag_code(new[i]) == "c" for i in range(ps, pe)):
            continue
        if not all(tag_code(new[i]) == "q" for i in range(ps, pe)):
            continue
        new = retag_span(new, ps, pe, "c")
    return new


def rule_a_trailing_initials_in_personal_name(header: str, tokens: list[str],
                                              tags: list[str]) -> list[str]:
    """Pattern 4: 'Flimankov, V. I.' — trailing initial+period(s) after an
    initial+period inside a personal-name $a stay in $a; the CRF sometimes
    splits them into $b/$t/$k.

    Trigger:
      - header begins with '100'
      - sequence ends with: ... <initial> '.' <initial> '.'
      - the last <initial> '.' was tagged with a different subfield from the
        preceding initial+period
    """
    if not header.startswith("100"):
        return tags
    n = len(tokens)
    if n < 4:
        return tags
    # Walk from the end: find pairs of [initial '.'] at the tail.
    # We need at least two such pairs and the last pair's code != the prior pair's code.
    # Most failure examples are exactly: ... A . A .  with the trailing A . in a new subfield.
    # We collapse the trailing pair to match the preceding pair's code (typically $a).
    # Iterate: while the last two tokens are <initial> '.' and the previous two
    # are <initial> '.', and they're tagged differently, retag.
    new = tags
    while True:
        if n < 4:
            break
        i = n - 4
        a, dot1, b, dot2 = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
        if dot1 != "." or dot2 != ".":
            break
        if not (is_initial(a) and is_initial(b)):
            break
        prev_code = tag_code(new[i])
        last_code = tag_code(new[i + 2])
        if prev_code == last_code:
            break
        # Only intervene if the preceding pair is in $a (the typical name case).
        if prev_code != "a":
            break
        # Retag last initial+period as I-a (continuation of $a).
        new = list(new)
        new[i + 2] = "I-a"
        new[i + 3] = "I-a"
        # Also handle the leading B-a issue: this is a continuation of the
        # name, so it must NOT start a new B-. Already I-a above. Done.
        break  # only one retag — avoid runaway
    return new


def rule_a_uniform_title_paren_tail(header: str, tokens: list[str],
                                    tags: list[str]) -> list[str]:
    """Pattern 5: '... (Chicago, Ill. : 1902)' inside a uniform-title $a — the
    closing year and ')' should stay in $a, not split off as $d.

    Trigger:
      - header is '130|#|0'
      - the row contains a paren group whose entire content is currently tagged
        $a EXCEPT the last few tokens (something_:_year_)) which got retagged.
      - specifically: we find a span ' : 9999 )' where '9999' is a 4-digit token
        labelled $d but inside an unclosed paren section whose start is $a.
    """
    if header != "130|#|0":
        return tags
    n = len(tokens)
    new = tags
    # Find the last '(' before any ')' such that we're inside an $a span.
    # Simpler: look for sequence ': <4digit> )' where 4digit is the LAST token-2.
    if n < 4:
        return tags
    if tokens[-1] != ")":
        return tags
    if not (len(tokens[-2]) == 4 and tokens[-2].isdigit()):
        return tags
    if tokens[-3] != ":":
        return tags
    # Find the matching '(' to the left.
    depth = 1
    open_pos = -1
    for j in range(n - 2, -1, -1):
        if tokens[j] == ")":
            depth += 1
        elif tokens[j] == "(":
            depth -= 1
            if depth == 0:
                open_pos = j
                break
    if open_pos < 0:
        return tags
    # The opening paren must currently be in $a.
    if tag_code(new[open_pos]) != "a":
        return tags
    # If the trailing year+) are already $a, nothing to do.
    if all(tag_code(new[i]) == "a" for i in (n - 2, n - 1)):
        return tags
    # Retag the year and ')' as I-a (continuation of the current $a span).
    new = list(new)
    new[-2] = "I-a"
    new[-1] = "I-a"
    return new


# Registry: name -> function. Order matters because a later rule may operate
# on tags adjusted by an earlier rule. We pick the most specific rules first.
def rule_a_personal_name_continuation(header: str, tokens: list[str],
                                      tags: list[str]) -> list[str]:
    """Pattern A (round 2, biggest single failure cluster): in `100|1|#` rows
    like 'Kutcher, Ashton, 1978-', the CRF sometimes mis-splits the given name
    as `$c`. If we see a `$c` span sandwiched between `$a` content on the left
    and `$d` (date) on the right, with the `$c` content being just word(s) +
    optional commas, retag as `$a`.

    Trigger:
      - header begins with '100'
      - somewhere in the tag sequence: an `$a` span, then a `$c` span (which is
        purely word/comma tokens), then a `$d` span starting with a 4-digit year
        OR a '-' followed by a 4-digit year
      - the `$c` content immediately follows a `,` token that was tagged `I-a`
    """
    if not header.startswith("100"):
        return tags
    spans = find_spans(tags)
    if not spans:
        return tags
    new = tags
    for i, (s, e, code) in enumerate(spans):
        if code != "c":
            continue
        # Need an $a span before and a $d span after (immediately adjacent).
        if i == 0 or i + 1 >= len(spans):
            continue
        prev_s, prev_e, prev_code = spans[i - 1]
        next_s, next_e, next_code = spans[i + 1]
        if prev_code != "a" or next_code != "d":
            continue
        # The $a span just before must end with a comma token.
        if tokens[prev_e - 1] != ",":
            continue
        # The $c span content must be all word/comma tokens.
        inside = tokens[s:e]
        if not inside:
            continue
        ok = True
        has_title_word = False
        has_roman_numeral = False
        for t in inside:
            # Allow Roman numerals through the structural filter (they're
            # alphabetic), but flag them so we skip the rule.
            if is_roman_numeral_generation(t):
                has_roman_numeral = True
            if not (is_capitalized_word(t) or t == "," or
                    (len(t) <= 3 and t.isalpha() and t[0].isupper())):
                ok = False
                break
            if t.lower() in TITLE_WORDS:
                has_title_word = True
        if not ok:
            continue
        # If a title word, honorific, or generational numeral appears, the CRF
        # was correct to call this $c; don't fire.
        if has_title_word or has_roman_numeral:
            continue
        # The $d span must start with a 4-digit year, or '-' then year.
        nstart = tokens[next_s]
        if not (nstart == "-" or (nstart.isdigit() and len(nstart) == 4)):
            continue
        new = retag_span(new, s, e, "a")
        # Also: the joining tag at position s-1 (last $a token, a comma) stays
        # I-a — already is. The first token at s is now I-a (continuation).
        # But retag_span sets it to B-a. That'd break BIO invariant since the
        # token before is I-a. Fix: make it I-a since it's a continuation of
        # the previous $a span.
        new = list(new)
        new[s] = "I-a"
    return new


# Honorifics / titles that look like initials but should stay tagged as $c.
# These appear in patterns like 'Hill, John A., Mrs.' where 'Mrs.' is a
# title-of-respect, not part of the name.
HONORIFICS = {
    "Mr", "Mrs", "Ms", "Mme", "Mlle",
    "Dr", "Drs", "Prof", "Profs",
    "Sr", "Sra", "Srta", "Jr", "Snr", "Jnr",
    "Rev", "Revd", "Fr", "Br", "Sr.",  # Father, Brother, Sister
    "Esq", "Esqr",
    "Hon", "Capt", "Cmdr", "Col", "Gen", "Lt", "Sgt", "Maj", "Pvt", "Pte",
    "St",  # Saint as honorific
}

# Title words that appear unabbreviated between name and dates. These are $c
# (titles/positions), not part of the name. Case-insensitive comparison done
# at use site.
TITLE_WORDS = {
    # English / Romance noble titles
    "baron", "baroness", "count", "countess", "duke", "duchess",
    "earl", "lord", "lady", "viscount", "viscountess", "marquis", "marchioness",
    "sir", "dame", "saint",
    # Religious
    "brother", "sister", "father", "mother", "frère", "frere", "soeur",
    "pope", "bishop", "archbishop", "cardinal", "abbot", "abbess", "rabbi",
    "imam", "swami", "guru", "lama", "maulana", "maulvi", "mawlana",
    "mor",  # Syriac honorific
    "siostra",  # Polish "Sister"
    "święty",   # Polish "Saint"
    # Royal / nobility (non-English)
    "king", "queen", "prince", "princess", "emperor", "empress",
    "prinz", "prinzessin", "graf", "gräfin", "fürst", "fürstin",  # German
    "principe", "principessa",  # Italian/Spanish
    "infante", "infanta",  # Iberian
    "tsar", "czar", "tsarina", "kniaz", "kniazʹ", "kniaginia",  # Slavic
    # Military
    "captain", "general", "colonel", "lieutenant", "sergeant", "major",
    "admiral",
    # More foreign-language titles seen in real data.
    "freifrau", "freiherr", "freiin",  # German lower-noble forms
    "bürgermeister", "burgermeister",
    "maung", "sayadaw", "u",           # Burmese
    "sardār", "sardar", "sirdar",      # Persian/Urdu
    "sthavira", "thera",               # Buddhist
    "vardapet",                        # Armenian
    "maulvi", "mawlvi",
    "gosvāmī", "goswami",
    "mistresse", "mistress",
    "reverend",
    "mahā", "maha",
    # Helpers for multi-word titles like 'the Younger', 'of Norwich'
    "the", "of",
}


# Roman-numeral generational suffixes (II, III, IV, ...). These look like
# initials but are $c (numeration/generation).
import re as _re
_ROMAN_RE = _re.compile(r"^[IVX]{2,5}$")


def is_roman_numeral_generation(token: str) -> bool:
    """Return True for tokens like II, III, IV, V, VI, VII, VIII, IX, X."""
    return bool(_ROMAN_RE.match(token))


def rule_a_trailing_initial_single_pair(header: str, tokens: list[str],
                                        tags: list[str]) -> list[str]:
    """Extension of round-1 rule: a SINGLE pair of <initial> '.' at the end of
    a personal name should also stay in $a (not $b, $c, $t etc.).

    Example: 'Rozenberg, M.' -> $aRozenberg, M.  but CRF said $cM.

    We deliberately exclude common honorifics ('Mrs', 'Dr', 'Sr', etc.) because
    those are genuinely $c (title), not part of the name. The CRF was correct
    on those — we'd break them.

    Trigger:
      - header begins with '100'
      - last two tokens are <initial> '.', and <initial> is NOT a honorific
      - the previous token is a ',' tagged I-a
      - the last two tokens have a code different from 'a'
    """
    if not header.startswith("100"):
        return tags
    n = len(tokens)
    if n < 4:
        return tags
    if tokens[-1] != "." or not is_initial(tokens[-2]):
        return tags
    if tokens[-2] in HONORIFICS:
        return tags
    if tokens[-3] != ",":
        return tags
    if tag_code(tags[-3]) != "a":
        return tags
    last_code = tag_code(tags[-2])
    if last_code == "a":
        return tags
    new = list(tags)
    new[-2] = "I-a"
    new[-1] = "I-a"
    return new


def _is_combining_mark(token: str) -> bool:
    """A single character that is a Unicode combining mark (e.g. ︠ ︡).
    These show up in transliterated names like 'I︠A︡' for Я."""
    if len(token) != 1:
        return False
    cat = __import__("unicodedata").category(token)
    return cat in ("Mn", "Me", "Mc")


def _is_name_continuation_token(token: str, allow_lowercase_particle: bool = True) -> bool:
    """A token that's plausibly part of a personal name's $a continuation.

    Allowed: capitalized words ('Victor'), initials ('M', 'Ma', 'Es'),
    periods, combining marks, and (optionally) lowercase particles
    ('de', 'van', 'el') common in Western/Iberian/Arabic names.
    """
    if token == "." or token == "," or _is_combining_mark(token):
        return True
    if is_capitalized_word(token) or is_initial(token):
        return True
    if allow_lowercase_particle and token.lower() in NAME_PARTICLES:
        return True
    return False


# Lowercase particles that legitimately appear inside personal names.
# Conservative list — we only include forms common across many Western names.
NAME_PARTICLES = {
    "de", "del", "della", "delle", "di", "da", "do", "dos", "das",
    "van", "von", "vom", "der", "den", "ten", "ter",
    "le", "la", "les", "du",
    "el", "al", "ibn", "bin", "bint", "abu", "umm",
    "y", "i",  # Iberian/Catalan conjunction
}


def rule_a_personal_name_trailing_block(header: str, tokens: list[str],
                                        tags: list[str]) -> list[str]:
    """Generalized round-3 rule: in 100/110-style headers, if a stretch of
    tokens between an $a span and a 'terminator' (EOS, $d, or $t span) was
    tagged with non-$a codes despite being purely name-continuation tokens
    (initials, periods, words, name particles, combining marks), retag the
    whole stretch as $a.

    This subsumes several round-1/2 rules but is intentionally additive: those
    rules handle slightly different sub-cases (and their guards may differ).
    The combined effect with the older rules is still net-positive (measured).

    Trigger:
      - header begins with '100' or '110' (personal/corporate where this shape applies)
      - locate the longest leading $a span; let its end be `a_end`
      - find the next 'terminator': end-of-tokens, or the start of a span
        whose code is in {'d', 't'} AND whose first token is a 4-digit year
        (for 'd') or a capitalized word that could plausibly start a title.
      - the run [a_end, terminator) must:
          * be non-empty
          * contain ONLY name-continuation tokens (per _is_name_continuation_token)
          * contain NO title-words, honorifics, or roman numerals
          * have a code != 'a' in at least one position
      - retag the run as $a (continuation of the leading span)
    """
    if not header.startswith("100"):
        return tags
    spans = find_spans(tags)
    if not spans:
        return tags
    # The first $a span. It should start at position 0 (canonical heading start).
    if spans[0][2] != "a" or spans[0][0] != 0:
        return tags
    a_end = spans[0][1]
    n = len(tokens)
    if a_end >= n:
        return tags

    # Find the terminator. Preference order:
    #   1. Earliest $t span boundary (period-then-capitalized-word) — this is
    #      the name-title boundary and dominates any later year.
    #   2. Earliest year-shaped token before the $t boundary.
    # We don't trust $d/$n/$b/$t tags around the year because the CRF often
    # mis-tags the year as $n/$b/$t.
    term = n
    for s, e, code in spans[1:]:
        if code == "t" and s < n and is_capitalized_word(tokens[s]):
            if s > 0 and tokens[s - 1] == ".":
                term = s
                break
    # Now scan for a year before term (or EOS if none).
    for i in range(a_end, term):
        t = tokens[i]
        if t.isdigit() and len(t) == 4:
            term = i
            break
        if t == "-" and i + 1 < n and tokens[i + 1].isdigit() and len(tokens[i + 1]) == 4:
            term = i
            break
    middle = tokens[a_end:term]
    if not middle:
        return tags
    # Validate the middle.
    has_non_a = False
    for i, t in enumerate(middle):
        idx = a_end + i
        if not _is_name_continuation_token(t):
            return tags
        if t.lower() in TITLE_WORDS:
            return tags
        if t in HONORIFICS:
            return tags
        if is_roman_numeral_generation(t):
            return tags
        if tag_code(tags[idx]) != "a":
            has_non_a = True
    if not has_non_a:
        return tags

    # Tightener: require an <initial> '.' pair somewhere in tokens[0:term].
    # We allow combining marks (Unicode Mn/Me/Mc) between the initial and the
    # period for transliterated names like 'I︠A︡.'.
    def _has_init_period(toks):
        for i, t in enumerate(toks):
            if not is_initial(t):
                continue
            # Look forward, skipping combining marks, for a '.'.
            j = i + 1
            while j < len(toks) and _is_combining_mark(toks[j]):
                j += 1
            if j < len(toks) and toks[j] == ".":
                return True
        return False

    if not _has_init_period(tokens[:term]):
        return tags
    # Additional honorific-like patterns inside the middle that should bail
    # rather than retag. 'M. D.' and friends look like initials but are
    # academic-degree suffixes belonging to $c.
    middle_text = " ".join(middle)
    if any(degree in middle_text for degree in (
        "M . D .", "M . A .", "Ph . D .", "B . A .", "B . S .",
        "M . S .", "J . D .", "LL . D .", "D . D .", "Esq .",
    )):
        return tags
    # Bare degree abbreviations (no periods): MD, MPA, MBA, PhD, etc. These
    # signify titles ($c), not name continuation.
    BARE_DEGREES = {
        "MD", "MA", "MS", "MSc", "MBA", "MPA", "MFA", "MPH", "MSW",
        "PhD", "PHD", "EdD", "JD", "DDS", "DVM", "DPhil", "ScD",
        "BA", "BS", "BSc", "BFA", "LLB", "LLD", "RN", "BSN", "MSN",
        "CPA", "PE",
    }
    for t in middle:
        if t in BARE_DEGREES:
            return tags
    # Retag the middle as continuation of the leading $a span.
    new = list(tags)
    for idx in range(a_end, term):
        new[idx] = "I-a"
    return new


def rule_c_paren_two_word_occupation(header: str, tokens: list[str],
                                     tags: list[str]) -> list[str]:
    """In 100|1|# personal names, a trailing parenthesized 2-word phrase like
    '(Worm farmer)' or '(Policy analyst)' is an occupation/role ($c), not part
    of the name. The CRF often absorbs it into $a.

    Trigger:
      - header is '100|1|#'
      - last 4 tokens are '(' <Cap-word> <lower-word> ')'
      - the position before '(' is a capitalized word in $a
      - the paren span is currently $a (specific target)
    """
    if header != "100|1|#":
        return tags
    n = len(tokens)
    if n < 5:
        return tags
    if tokens[-1] != ")" or tokens[-4] != "(":
        return tags
    w1 = tokens[-3]
    w2 = tokens[-2]
    # First word inside parens: capitalized word.
    if not is_capitalized_word(w1):
        return tags
    # Second word: lowercase (occupation continuation like 'farmer', 'analyst').
    if not (w2.isalpha() and w2[0].islower()):
        return tags
    # Token before '(' must be a capitalized word currently in $a.
    if not is_capitalized_word(tokens[-5]):
        return tags
    if tag_code(tags[-5]) != "a":
        return tags
    # Currently the paren span must be in $a (the failure mode we target).
    if not all(tag_code(tags[i]) == "a" for i in range(n - 4, n)):
        return tags
    new = list(tags)
    new[n - 4] = "B-c"
    new[n - 3] = "I-c"
    new[n - 2] = "I-c"
    new[n - 1] = "I-c"
    return new


def rule_c_paren_role_after_initial_period(header: str, tokens: list[str],
                                           tags: list[str]) -> list[str]:
    """Like c_paren_role_after_full_name but for the case where the token
    before '(' is a period (i.e. the name ends with an initial+period and
    is followed by an occupation in parens).

    Example: 'Williams, Julius P. (Tenor)' -> $c(Tenor)

    The inside word is required to be a single common-noun-looking capitalized
    word (not a fuller form). To avoid colliding with the fuller-form pattern
    (e.g. 'Coombs, F. (Frederick)'), we require the inside word to NOT share
    its first letter with the most recent initial before the period.
    """
    if not header.startswith("100|"):
        return tags
    n = len(tokens)
    new = tags
    for k in range(2, n):
        if tokens[k] != "(":
            continue
        # Previous token must be '.', and the one before that an initial.
        if tokens[k - 1] != ".":
            continue
        if not is_initial(tokens[k - 2]):
            continue
        # Skip if the "initial" is actually a honorific (Mr., Mrs., Dr., ...);
        # those appear in name-of-wife patterns like 'Heywood, Mr. (James)'
        # where the inside is $q (genuine name), not $c.
        if tokens[k - 2] in HONORIFICS:
            continue
        paren = looks_like_paren_group(tokens, k)
        if paren is None:
            continue
        ps, pe = paren
        inside = tokens[ps + 1 : pe - 1]
        if len(inside) != 1 or not is_capitalized_word(inside[0]):
            continue
        # Avoid the fuller-form ambiguity: skip if the inside word shares its
        # first letter with the preceding initial.
        if _share_first_letter(tokens[k - 2], inside[0]):
            continue
        # CRF must currently have tagged the span as $q (the failure mode).
        if not all(tag_code(new[i]) == "q" for i in range(ps, pe)):
            continue
        new = retag_span(new, ps, pe, "c")
    return new


def rule_c_promote_honorific_after_name(header: str, tokens: list[str],
                                        tags: list[str]) -> list[str]:
    """In 100|*|# personal names, a `, <honorific|title>` (possibly followed
    by '.' and then by a date or EOS) should be $c. The CRF often keeps the
    honorific in $a.

    Example: 'Kolář, Jaroslav, Mgr.' -> $a..., $cMgr.
             'Madana, Acharya, 1920-' -> $a..., $cAcharya, $d1920-

    Trigger:
      - header begins with '100'
      - somewhere in tokens we see ',' <title-or-honorific> optional '.' then
        optional ',' followed by a year-shape OR EOS
      - the title-or-honorific token is currently tagged $a
    """
    if not header.startswith("100"):
        return tags
    n = len(tokens)
    if n < 3:
        return tags
    new = tags
    mutated = False
    i = 1
    while i < n - 1:
        # Look for ',' at position i-1 and a title/honorific at position i.
        if tokens[i - 1] != ",":
            i += 1
            continue
        t = tokens[i]
        is_title = t.lower() in TITLE_WORDS
        is_honorific = t in HONORIFICS
        if not (is_title or is_honorific):
            i += 1
            continue
        # Determine the span: token i, then optional '.' at i+1.
        end_idx = i + 1
        if end_idx < n and tokens[end_idx] == ".":
            end_idx += 1
        # What follows the span must be: EOS, or ',' then year, or ',' then EOS.
        ok = False
        if end_idx >= n:
            ok = True
        elif tokens[end_idx] == ",":
            # Next non-comma must be a 4-digit year or EOS.
            j = end_idx + 1
            if j >= n:
                ok = True
            elif tokens[j].isdigit() and len(tokens[j]) == 4:
                ok = True
            elif tokens[j] == "-" and j + 1 < n and tokens[j + 1].isdigit() and len(tokens[j + 1]) == 4:
                ok = True
        if not ok:
            i += 1
            continue
        # All tokens in [i, end_idx) must currently be tagged $a; otherwise
        # the rule isn't the right one to fire here.
        if not all(tag_code(new[k]) == "a" for k in range(i, end_idx)):
            i += 1
            continue
        # Also the comma at i-1 must be tagged $a (so we're splitting off from
        # the leading $a span).
        if tag_code(new[i - 1]) != "a":
            i += 1
            continue
        # Promote tokens [i, end_idx) to $c.
        if not mutated:
            new = list(new)
            mutated = True
        new[i] = "B-c"
        for k in range(i + 1, end_idx):
            new[k] = "I-c"
        i = end_idx
    return new


def rule_a_corporate_jurisdiction_paren(header: str, tokens: list[str],
                                        tags: list[str]) -> list[str]:
    """In 110|2|# corporate-body headings, a trailing parenthesized location
    like '(Vienna, Austria)' or '(Lancaster, Pa.)' is part of $a (jurisdiction
    qualifier), not $c. The CRF often splits it off.

    Trigger:
      - header is '110|2|#'
      - the heading ends with a closed paren group whose content is
        capitalized words and commas/periods (looks like 'City, State')
      - the paren span is currently tagged $c
      - the position before '(' is in $a
    """
    if header != "110|2|#":
        return tags
    n = len(tokens)
    if n < 3:
        return tags
    if tokens[-1] != ")":
        return tags
    # Find matching '('.
    depth = 1
    open_pos = -1
    for j in range(n - 2, -1, -1):
        if tokens[j] == ")":
            depth += 1
        elif tokens[j] == "(":
            depth -= 1
            if depth == 0:
                open_pos = j
                break
    if open_pos < 1:
        return tags
    # Token before '(' must already be in $a.
    if tag_code(tags[open_pos - 1]) != "a":
        return tags
    # Validate paren content: capitalized words, commas, periods.
    inside = tokens[open_pos + 1 : n - 1]
    if not inside:
        return tags
    has_word = False
    for t in inside:
        if t == "," or t == ".":
            continue
        if is_capitalized_word(t):
            has_word = True
            continue
        # Reject anything else (digits, lowercase, etc.).
        return tags
    if not has_word:
        return tags
    # Only fire if currently the paren span is $c (specific target failure mode).
    if not all(tag_code(tags[i]) == "c" for i in range(open_pos, n)):
        return tags
    new = list(tags)
    for i in range(open_pos, n):
        new[i] = "I-a"
    return new


def rule_d_incomplete_date_range(header: str, tokens: list[str],
                                 tags: list[str]) -> list[str]:
    """Pattern D: incomplete date range starting with '-' (death-date-only):
    'Thomson, Barry, -1960' -> $d-1960. The CRF often absorbs the '-' and year
    into the preceding $a.

    Trigger:
      - header begins with '100' or '110' or '111'
      - the sequence ends with ',' '-' '9999' (optionally followed by more)
      - both the '-' and '9999' are currently tagged $a
    """
    if not (header.startswith("100") or header.startswith("110") or header.startswith("111")):
        return tags
    n = len(tokens)
    if n < 3:
        return tags
    # Find pattern: ',' '-' '<4-digit year>' at the tail.
    # The year may be followed by more tokens (e.g. a name-title), but we look
    # for the dash-year pair coming immediately after a comma.
    for i in range(1, n - 1):
        if tokens[i] != "-":
            continue
        if i + 1 >= n:
            continue
        year_tok = tokens[i + 1]
        if not (year_tok.isdigit() and len(year_tok) == 4):
            continue
        if tokens[i - 1] != ",":
            continue
        if tag_code(tags[i - 1]) != "a":
            continue
        # Only fire if currently the '-' and year are not in $d.
        if tag_code(tags[i]) == "d" and tag_code(tags[i + 1]) == "d":
            continue
        # Only target the specific case where they're in $a.
        if tag_code(tags[i]) != "a" or tag_code(tags[i + 1]) != "a":
            continue
        new = list(tags)
        new[i] = "B-d"
        new[i + 1] = "I-d"
        return new
    return tags


# Active rule registry. Net-positive rules only. q_fuller_form_after_initial
# was disabled in round 1 (net -2); the round-2 q_fuller_form_in_100_1_0
# replaces it with a narrower trigger.
RULES: list[tuple[str, Callable[[str, list[str], list[str]], list[str]]]] = [
    ("c_paren_role_after_full_name", rule_c_paren_role_after_full_name),
    ("a_trailing_initials_in_personal_name", rule_a_trailing_initials_in_personal_name),
    ("a_uniform_title_paren_tail", rule_a_uniform_title_paren_tail),
    ("a_personal_name_continuation", rule_a_personal_name_continuation),
    ("a_trailing_initial_single_pair", rule_a_trailing_initial_single_pair),
    ("a_personal_name_trailing_block", rule_a_personal_name_trailing_block),
    ("a_corporate_jurisdiction_paren", rule_a_corporate_jurisdiction_paren),
    ("c_paren_two_word_occupation", rule_c_paren_two_word_occupation),
    ("c_paren_role_after_initial_period", rule_c_paren_role_after_initial_period),
    ("c_promote_honorific_after_name", rule_c_promote_honorific_after_name),
    ("d_incomplete_date_range", rule_d_incomplete_date_range),
]


def apply_rules(header: str, tokens: list[str], tags: list[str]
                ) -> tuple[list[str], list[str]]:
    """Apply all rules in order. Returns (final_tags, names_of_rules_that_changed_anything).

    A rule "fired" if its output differs from its input."""
    fired: list[str] = []
    cur = tags
    for name, rule in RULES:
        nxt = rule(header, tokens, cur)
        if nxt != cur:
            fired.append(name)
            cur = nxt
    return cur, fired
