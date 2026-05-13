// Browser-friendly inference for the NAF CRF + header LR.
//
// Public API:
//   const model = await NafCRF.load(url);          // url to model.json(.gz)
//   const result = model.tag("Smith, John, 1962-");
//   // -> { header: "100|1|#", tokens: [...], tags: ["B-a","I-a",...], marc: "1001 $aSmith, John,$d1962-" }
//
// The JS featurizer must match Python features.py and train.py:header_doc_features
// exactly, otherwise we'll silently disagree with the trained model.

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.NafCRF = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------- Tokenizer ----------
  // Mirrors Python: re.compile(r"\w+|[^\w\s]", re.UNICODE).
  // \w in JS with the /u flag corresponds to ASCII [A-Za-z0-9_], NOT Unicode
  // letters. We need Unicode letters (Python \w with re.UNICODE includes them).
  // Use \p{L} (letters), \p{N} (numbers), and _ for word characters; everything
  // not whitespace and not a word char is a punct token.
  const TOKEN_RE = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;

  function tokenize(text) {
    return text.match(TOKEN_RE) || [];
  }

  // ---------- Character-class shape (mirrors features.py:shape) ----------
  // We can't easily replicate Python's unicodedata.category("Lu" vs "Ll" etc.)
  // in JS without a Unicode database. We approximate using regex character
  // classes:
  //   Lu  -> /\p{Lu}/u   (uppercase letter)
  //   L   -> /\p{L}/u    (any letter)
  //   N   -> /\p{N}/u    (any number)
  //   P|S -> /[\p{P}\p{S}]/u   (punct or symbol)
  // The featurizer collapses runs of the same class into one character.
  function shape(token) {
    let out = "";
    let last = "";
    for (const ch of token) {
      let c;
      if (/\p{Lu}/u.test(ch)) c = "A";
      else if (/\p{L}/u.test(ch)) c = "a";
      else if (/\p{N}/u.test(ch)) c = "9";
      else if (/[\p{P}\p{S}]/u.test(ch)) c = "p";
      else c = "?";
      if (c !== last) {
        out += c;
        last = c;
      }
    }
    return out;
  }

  function isPunct(token) {
    if (!token) return false;
    return /^[\p{P}\p{S}]/u.test(token);
  }

  function isDigitToken(token) {
    return /^[0-9]+$/.test(token);
  }

  function hasDigit(token) {
    return /[0-9]/.test(token);
  }

  // ---------- Per-token features (mirrors features.py:token_features) ----------
  function tokenFeatures(tokens, i, header) {
    const t = tokens[i];
    const feats = Object.create(null);
    feats["bias"] = 1.0;
    feats["H=" + header] = 1.0;
    feats["w=" + t.toLowerCase()] = 1.0;
    feats["shape=" + shape(t)] = 1.0;

    const len = t.length;
    if (len === 1) feats["len=1"] = 1.0;
    else if (len <= 3) feats["len=2-3"] = 1.0;
    else if (len <= 6) feats["len=4-6"] = 1.0;
    else feats["len=7+"] = 1.0;

    if (isPunct(t)) {
      feats["punct"] = 1.0;
      feats["punct=" + t] = 1.0;
    }
    if (isDigitToken(t)) {
      feats["digit"] = 1.0;
      if (t.length === 4) feats["year_like"] = 1.0;
    } else if (hasDigit(t)) {
      feats["alnum"] = 1.0;
    }
    if (t && /\p{Lu}/u.test(t[0])) feats["upper_first"] = 1.0;
    if (t && t.length > 1 && t === t.toUpperCase() && /\p{L}/u.test(t)) {
      feats["all_upper"] = 1.0;
    }

    const low = t.toLowerCase();
    if (low.length >= 2) {
      feats["pre2=" + low.slice(0, 2)] = 1.0;
      feats["suf2=" + low.slice(-2)] = 1.0;
    }
    if (low.length >= 3) {
      feats["pre3=" + low.slice(0, 3)] = 1.0;
      feats["suf3=" + low.slice(-3)] = 1.0;
    }

    if (i === 0) feats["BOS"] = 1.0;
    if (i === tokens.length - 1) feats["EOS"] = 1.0;

    if (i > 0) {
      const prev = tokens[i - 1];
      feats["-1w=" + prev.toLowerCase()] = 1.0;
      feats["-1shape=" + shape(prev)] = 1.0;
    } else {
      feats["-1=BOS"] = 1.0;
    }
    if (i + 1 < tokens.length) {
      const nxt = tokens[i + 1];
      feats["+1w=" + nxt.toLowerCase()] = 1.0;
      feats["+1shape=" + shape(nxt)] = 1.0;
    } else {
      feats["+1=EOS"] = 1.0;
    }

    return feats;
  }

  // ---------- Header doc features (mirrors train.py:header_doc_features) ----------
  function headerDocFeatures(tokens, vocab) {
    const feats = Object.create(null);
    feats["bias"] = 1.0;
    const n = tokens.length;
    if (n <= 2) feats["nlen=1-2"] = 1.0;
    else if (n <= 4) feats["nlen=3-4"] = 1.0;
    else if (n <= 8) feats["nlen=5-8"] = 1.0;
    else if (n <= 16) feats["nlen=9-16"] = 1.0;
    else feats["nlen=17+"] = 1.0;

    const first = tokens[0];
    const last = tokens[n - 1];
    feats["first_shape=" + shape(first)] = 1.0;
    feats["last_shape=" + shape(last)] = 1.0;
    if (isPunct(first)) feats["first_punct"] = 1.0;
    if (isPunct(last)) feats["last_punct"] = 1.0;

    // Bag-of-shapes (deduped — only emit once per shape).
    const shapesSeen = new Set();
    for (const t of tokens) {
      const sh = shape(t);
      if (!shapesSeen.has(sh)) {
        shapesSeen.add(sh);
        feats["has_shape=" + sh] = 1.0;
      }
    }

    // Bag-of-words gated by vocab.
    const seenLower = new Set();
    for (const t of tokens) {
      const low = t.toLowerCase();
      if (seenLower.has(low)) continue;
      seenLower.add(low);
      if (vocab.has(low)) feats["has_w=" + low] = 1.0;
    }
    const firstLow = first.toLowerCase();
    const lastLow = last.toLowerCase();
    if (vocab.has(firstLow)) feats["first_w=" + firstLow] = 1.0;
    if (vocab.has(lastLow)) feats["last_w=" + lastLow] = 1.0;

    // Structural counts.
    let nComma = 0, nPeriod = 0, nParenOpen = 0, nColon = 0, nDash = 0;
    let nYear = 0, nDigitToken = 0, nPunct = 0, nUpper = 0, nLower = 0;
    let firstCommaPos = -1, firstParenPos = -1, firstPeriodPos = -1;
    for (let i = 0; i < n; i++) {
      const t = tokens[i];
      if (t === ",") { nComma++; if (firstCommaPos < 0) firstCommaPos = i; }
      if (t === ".") { nPeriod++; if (firstPeriodPos < 0) firstPeriodPos = i; }
      if (t === "(") { nParenOpen++; if (firstParenPos < 0) firstParenPos = i; }
      if (t === ":") nColon++;
      if (t === "-") nDash++;
      if (isDigitToken(t) && t.length === 4) nYear++;
      if (isDigitToken(t)) nDigitToken++;
      if (isPunct(t)) nPunct++;
      if (t && /\p{Lu}/u.test(t[0])) nUpper++;
      if (t && /\p{Ll}/u.test(t[0])) nLower++;
    }

    if (nComma) feats["has_comma"] = 1.0;
    if (nPeriod) feats["has_period"] = 1.0;
    if (nParenOpen) feats["has_paren"] = 1.0;
    if (nColon) feats["has_colon"] = 1.0;
    if (nDash) feats["has_dash"] = 1.0;
    if (nYear) feats["has_year"] = 1.0;

    function bucket(name, c) {
      if (c === 0) return;
      if (c === 1) feats[name + "=1"] = 1.0;
      else if (c === 2) feats[name + "=2"] = 1.0;
      else feats[name + "=3+"] = 1.0;
    }
    bucket("n_comma", nComma);
    bucket("n_period", nPeriod);
    bucket("n_paren", nParenOpen);
    bucket("n_year", nYear);

    feats["punct_frac"] = nPunct / Math.max(1, n);
    feats["upper_frac"] = nUpper / Math.max(1, n);
    feats["lower_frac"] = nLower / Math.max(1, n);
    feats["digit_frac"] = nDigitToken / Math.max(1, n);

    if (firstCommaPos >= 0) feats["first_comma_pos"] = firstCommaPos / Math.max(1, n);
    if (firstParenPos >= 0) feats["first_paren_pos"] = firstParenPos / Math.max(1, n);
    if (firstPeriodPos >= 0) feats["first_period_pos"] = firstPeriodPos / Math.max(1, n);

    return feats;
  }

  // ---------- Header LR inference ----------
  // argmax_c (intercept[c] + sum_f weights[c][f] * feat_f)
  function predictHeader(tokens, model) {
    const vocab = model._headerVocab;
    const feats = headerDocFeatures(tokens, vocab);
    const hc = model.header_classifier;
    let best = null;
    let bestScore = -Infinity;
    for (const cls of hc.labels) {
      const w = hc.weights[cls] || {};
      let s = hc.intercepts[cls] || 0;
      for (const f in feats) {
        const wv = w[f];
        if (wv !== undefined) s += wv * feats[f];
      }
      if (s > bestScore) {
        bestScore = s;
        best = cls;
      }
    }
    return best;
  }

  // ---------- CRF Viterbi ----------
  // For each token i, compute state_score(label) = sum_f state_features[label][f] * f.
  // Then DP: viterbi[i][label] = max over prev (viterbi[i-1][prev] + transition[prev][label]) + state_score(label).
  function viterbi(tokens, header, model) {
    const tags = model.tags;
    const labels = tags.labels;
    const stateFeatures = tags.state_features;
    const transitions = tags.transitions;

    const n = tokens.length;
    if (n === 0) return [];

    const NEG_INF = -1e18;

    // Precompute per-token state scores: scores[i][label] = stateScore.
    const scores = new Array(n);
    for (let i = 0; i < n; i++) {
      const feats = tokenFeatures(tokens, i, header);
      const row = new Array(labels.length);
      for (let li = 0; li < labels.length; li++) {
        const sf = stateFeatures[labels[li]];
        if (!sf) {
          row[li] = 0;
          continue;
        }
        let s = 0;
        for (const f in feats) {
          const w = sf[f];
          if (w !== undefined) s += w * feats[f];
        }
        row[li] = s;
      }
      scores[i] = row;
    }

    // Viterbi DP.
    const dp = new Array(n);
    const back = new Array(n);
    for (let i = 0; i < n; i++) {
      dp[i] = new Float64Array(labels.length);
      back[i] = new Int32Array(labels.length);
    }
    for (let li = 0; li < labels.length; li++) {
      dp[0][li] = scores[0][li];
      back[0][li] = -1;
    }
    for (let i = 1; i < n; i++) {
      for (let li = 0; li < labels.length; li++) {
        const curLabel = labels[li];
        let best = NEG_INF;
        let bestPrev = 0;
        for (let pj = 0; pj < labels.length; pj++) {
          const prevLabel = labels[pj];
          const tr = transitions[prevLabel];
          const trVal = (tr && tr[curLabel] !== undefined) ? tr[curLabel] : 0;
          const v = dp[i - 1][pj] + trVal;
          if (v > best) {
            best = v;
            bestPrev = pj;
          }
        }
        dp[i][li] = best + scores[i][li];
        back[i][li] = bestPrev;
      }
    }

    // Backtrace from argmax of dp[n-1].
    let bestLast = 0;
    let bestVal = dp[n - 1][0];
    for (let li = 1; li < labels.length; li++) {
      if (dp[n - 1][li] > bestVal) {
        bestVal = dp[n - 1][li];
        bestLast = li;
      }
    }
    const out = new Array(n);
    let cur = bestLast;
    for (let i = n - 1; i >= 0; i--) {
      out[i] = labels[cur];
      cur = back[i][cur];
    }
    return out;
  }

  // ---------- Reconstruct MARC string from tagged tokens ----------
  // Given header "TAG|i1|i2", tokens, and tags, build the MARC display string
  // like "1001 $aSmith, John,$d1962-".
  function reassembleMarc(header, tokens, tags) {
    const parts = header.split("|");
    const tag = parts[0];
    let i1 = parts[1] === "#" ? " " : parts[1];
    let i2 = parts[2] === "#" ? " " : parts[2];

    let out = tag + i1 + i2;
    let needSep = true;

    // Group consecutive tokens by their subfield code (the BIO tag's payload).
    let curCode = null;
    let curBuf = "";
    function flush() {
      if (curCode !== null && curBuf.length) {
        out += "$" + curCode + curBuf;
      }
      curCode = null;
      curBuf = "";
    }

    function joinToken(buf, tok) {
      // Token is either a word or a single punct/symbol char. Words get a
      // separating space when joined to a previous word; punctuation is
      // attached without a space. Mirrors typical MARC display.
      if (!buf) return tok;
      const lastCh = buf[buf.length - 1];
      const lastIsWord = /[\p{L}\p{N}_]/u.test(lastCh);
      const tokIsWord = /^[\p{L}\p{N}_]/u.test(tok);
      if (lastIsWord && tokIsWord) return buf + " " + tok;
      // Closing parens / quotes attach to previous non-space.
      // Opening parens get a space before unless previous was already punct.
      if (tok === "(" || tok === "[" || tok === '"') {
        if (lastIsWord) return buf + " " + tok;
        return buf + tok;
      }
      return buf + tok;
    }

    for (let i = 0; i < tokens.length; i++) {
      const tagi = tags[i];
      if (tagi === "O") {
        // Pure separator punctuation between subfields. Drop it; the next
        // subfield starts after this.
        continue;
      }
      const code = tagi.slice(2); // "B-a" -> "a", "I-a" -> "a"
      const isB = tagi.charAt(0) === "B";
      if (isB || code !== curCode) {
        flush();
        curCode = code;
        curBuf = tokens[i];
      } else {
        curBuf = joinToken(curBuf, tokens[i]);
      }
    }
    flush();
    return out;
  }

  // ---------- Post-CRF correction rules ----------
  // Each rule takes (header, tokens, tags) and returns a new tag array (or the
  // same one if nothing changed). Rules are deterministic post-processors that
  // target failure modes the CRF misses. See rules.py for the canonical Python
  // version and measure_rules.py for measured fix/break counts.

  function tagCode(t) { return t === "O" ? "O" : t.slice(2); }

  function retagSpan(tags, start, endExclusive, code) {
    const out = tags.slice();
    for (let i = start; i < endExclusive; i++) {
      out[i] = (i === start ? "B-" : "I-") + code;
    }
    return out;
  }

  function isCapitalizedWord(t) {
    if (!t) return false;
    if (!/^\p{Lu}/u.test(t)) return false;
    return /^[\p{L}]+$/u.test(t);
  }

  function isInitial(t) {
    if (!t) return false;
    if (!/^\p{Lu}/u.test(t)) return false;
    if (!/^[\p{L}]+$/u.test(t)) return false;
    return t.length >= 1 && t.length <= 3;
  }

  function findParenGroup(tokens, start) {
    if (start >= tokens.length || tokens[start] !== "(") return null;
    const end = Math.min(tokens.length, start + 20);
    for (let j = start + 1; j < end; j++) {
      if (tokens[j] === ")") return [start, j + 1];
    }
    return null;
  }

  function shareFirstLetter(a, b) {
    if (!a || !b) return false;
    // Accent-insensitive: strip combining marks via NFD.
    const norm = (c) => {
      const nfd = c.normalize("NFD");
      for (const ch of nfd) {
        if (!/^\p{Mn}$/u.test(ch)) return ch.toLowerCase();
      }
      return "";
    };
    return norm(a[0]) === norm(b[0]);
  }

  function findSpans(tags) {
    const out = [];
    const n = tags.length;
    let i = 0;
    while (i < n) {
      const code = tagCode(tags[i]);
      let j = i + 1;
      while (j < n && tagCode(tags[j]) === code) j++;
      out.push([i, j, code]);
      i = j;
    }
    return out;
  }

  const HONORIFICS = new Set([
    "Mr", "Mrs", "Ms", "Mme", "Mlle",
    "Dr", "Drs", "Prof", "Profs",
    "Sr", "Sra", "Srta", "Jr", "Snr", "Jnr",
    "Rev", "Revd", "Fr", "Br",
    "Esq", "Esqr",
    "Hon", "Capt", "Cmdr", "Col", "Gen", "Lt", "Sgt", "Maj", "Pvt", "Pte",
    "St",
  ]);

  const TITLE_WORDS = new Set([
    "baron", "baroness", "count", "countess", "duke", "duchess",
    "earl", "lord", "lady", "viscount", "viscountess", "marquis", "marchioness",
    "sir", "dame", "saint",
    "brother", "sister", "father", "mother", "frère", "frere", "soeur",
    "pope", "bishop", "archbishop", "cardinal", "abbot", "abbess", "rabbi",
    "imam", "swami", "guru", "lama", "maulana", "maulvi", "mawlana",
    "mor", "siostra", "święty",
    "king", "queen", "prince", "princess", "emperor", "empress",
    "prinz", "prinzessin", "graf", "gräfin", "fürst", "fürstin",
    "principe", "principessa", "infante", "infanta",
    "tsar", "czar", "tsarina", "kniaz", "kniazʹ", "kniaginia",
    "captain", "general", "colonel", "lieutenant", "sergeant", "major",
    "admiral",
    "freifrau", "freiherr", "freiin",
    "bürgermeister", "burgermeister",
    "maung", "sayadaw", "u",
    "sardār", "sardar", "sirdar",
    "sthavira", "thera",
    "vardapet",
    "maulvi", "mawlvi",
    "gosvāmī", "goswami",
    "mistresse", "mistress",
    "reverend",
    "mahā", "maha",
    "the", "of",
  ]);

  // Lowercase particles that legitimately appear inside personal names.
  const NAME_PARTICLES = new Set([
    "de", "del", "della", "delle", "di", "da", "do", "dos", "das",
    "van", "von", "vom", "der", "den", "ten", "ter",
    "le", "la", "les", "du",
    "el", "al", "ibn", "bin", "bint", "abu", "umm",
    "y", "i",
  ]);

  function isCombiningMark(t) {
    if (t.length !== 1) return false;
    // Mn = nonspacing mark, Me = enclosing mark, Mc = spacing combining mark.
    return /^[\p{Mn}\p{Me}\p{Mc}]$/u.test(t);
  }

  function isNameContinuationToken(t) {
    if (t === "." || t === "," || isCombiningMark(t)) return true;
    if (isCapitalizedWord(t) || isInitial(t)) return true;
    if (NAME_PARTICLES.has(t.toLowerCase())) return true;
    return false;
  }

  const ROMAN_RE = /^[IVX]{2,5}$/;

  function isRomanNumeralGeneration(t) {
    return ROMAN_RE.test(t);
  }

  function retagSpanInPlace(out, start, endExclusive, code) {
    // Like retagSpan but in-place on an existing mutable array.
    for (let i = start; i < endExclusive; i++) {
      out[i] = (i === start ? "B-" : "I-") + code;
    }
  }

  // Rule: 'Crosse, Thomas (Goldsmith)' — paren'd single capitalized word after
  // a full given name is $c (role/title), but only when the inside word does
  // NOT share its first letter with the preceding name (fuller-form heuristic).
  function ruleCParenRoleAfterFullName(header, tokens, tags) {
    if (!header.startsWith("100|")) return tags;
    let out = tags;
    for (let k = 1; k < tokens.length; k++) {
      if (tokens[k] !== "(") continue;
      const paren = findParenGroup(tokens, k);
      if (!paren) continue;
      const [ps, pe] = paren;
      const inside = tokens.slice(ps + 1, pe - 1);
      if (inside.length !== 1 || !isCapitalizedWord(inside[0])) continue;
      const prev = tokens[k - 1] || "";
      if (!isCapitalizedWord(prev) || prev.length <= 1) continue;
      if (shareFirstLetter(prev, inside[0])) continue;
      let allC = true;
      for (let i = ps; i < pe; i++) {
        if (tagCode(out[i]) !== "c") { allC = false; break; }
      }
      if (allC) continue;
      let allQ = true;
      for (let i = ps; i < pe; i++) {
        if (tagCode(out[i]) !== "q") { allQ = false; break; }
      }
      if (!allQ) continue;
      out = retagSpan(out, ps, pe, "c");
    }
    return out;
  }

  // Rule: 'Flimankov, V. I.' — trailing pair of <initial> '.' inside a personal
  // name should stay in $a, but the CRF sometimes splits the last pair into
  // $b/$t. If the previous pair is $a, collapse the last pair into $a too.
  function ruleATrailingInitialsInPersonalName(header, tokens, tags) {
    if (!header.startsWith("100")) return tags;
    const n = tokens.length;
    if (n < 4) return tags;
    const i = n - 4;
    const a = tokens[i], dot1 = tokens[i + 1], b = tokens[i + 2], dot2 = tokens[i + 3];
    if (dot1 !== "." || dot2 !== ".") return tags;
    if (!isInitial(a) || !isInitial(b)) return tags;
    const prevCode = tagCode(tags[i]);
    const lastCode = tagCode(tags[i + 2]);
    if (prevCode === lastCode) return tags;
    if (prevCode !== "a") return tags;
    const out = tags.slice();
    out[i + 2] = "I-a";
    out[i + 3] = "I-a";
    return out;
  }

  // Rule: '... (Chicago, Ill. : 1902)' inside a 130 uniform-title $a — the
  // closing ': 9999 )' should stay in $a, not split off as $d.
  function ruleAUniformTitleParenTail(header, tokens, tags) {
    if (header !== "130|#|0") return tags;
    const n = tokens.length;
    if (n < 4) return tags;
    if (tokens[n - 1] !== ")") return tags;
    if (!(tokens[n - 2].length === 4 && /^\d{4}$/.test(tokens[n - 2]))) return tags;
    if (tokens[n - 3] !== ":") return tags;
    // Find matching '('.
    let depth = 1;
    let openPos = -1;
    for (let j = n - 2; j >= 0; j--) {
      if (tokens[j] === ")") depth++;
      else if (tokens[j] === "(") {
        depth--;
        if (depth === 0) { openPos = j; break; }
      }
    }
    if (openPos < 0) return tags;
    if (tagCode(tags[openPos]) !== "a") return tags;
    if (tagCode(tags[n - 2]) === "a" && tagCode(tags[n - 1]) === "a") return tags;
    const out = tags.slice();
    out[n - 2] = "I-a";
    out[n - 1] = "I-a";
    return out;
  }

  // Pattern A (round 2): Kutcher, Ashton, 1978- — CRF mis-splits given name
  // as $c between an $a span and a $d span. We retag it to continue $a.
  function ruleAPersonalNameContinuation(header, tokens, tags) {
    if (!header.startsWith("100")) return tags;
    const spans = findSpans(tags);
    if (!spans.length) return tags;
    let out = tags;
    let mutated = false;
    for (let i = 0; i < spans.length; i++) {
      const [s, e, code] = spans[i];
      if (code !== "c") continue;
      if (i === 0 || i + 1 >= spans.length) continue;
      const [, prevE, prevCode] = spans[i - 1];
      const [nextS, , nextCode] = spans[i + 1];
      if (prevCode !== "a" || nextCode !== "d") continue;
      if (tokens[prevE - 1] !== ",") continue;
      const inside = tokens.slice(s, e);
      if (!inside.length) continue;
      let ok = true;
      let hasTitleWord = false;
      let hasRoman = false;
      for (const t of inside) {
        if (isRomanNumeralGeneration(t)) hasRoman = true;
        const isStructTok = (
          isCapitalizedWord(t) ||
          t === "," ||
          (t.length <= 3 && /^[\p{L}]+$/u.test(t) && /^\p{Lu}/u.test(t))
        );
        if (!isStructTok) { ok = false; break; }
        if (TITLE_WORDS.has(t.toLowerCase())) hasTitleWord = true;
      }
      if (!ok) continue;
      if (hasTitleWord || hasRoman) continue;
      const nstart = tokens[nextS];
      if (!(nstart === "-" || /^\d{4}$/.test(nstart))) continue;
      if (!mutated) { out = out.slice(); mutated = true; }
      retagSpanInPlace(out, s, e, "a");
      out[s] = "I-a";  // continuation of previous $a span
    }
    return out;
  }

  // Pattern C ext: trailing single <initial> '.' should stay in $a, but
  // exclude honorifics ('Mrs', 'Dr', 'Sr', ...).
  function ruleATrailingInitialSinglePair(header, tokens, tags) {
    if (!header.startsWith("100")) return tags;
    const n = tokens.length;
    if (n < 4) return tags;
    if (tokens[n - 1] !== "." || !isInitial(tokens[n - 2])) return tags;
    if (HONORIFICS.has(tokens[n - 2])) return tags;
    if (tokens[n - 3] !== ",") return tags;
    if (tagCode(tags[n - 3]) !== "a") return tags;
    if (tagCode(tags[n - 2]) === "a") return tags;
    const out = tags.slice();
    out[n - 2] = "I-a";
    out[n - 1] = "I-a";
    return out;
  }

  // Pattern D: incomplete date range '-9999' inside a 1XX heading should be $d.
  function ruleDIncompleteDateRange(header, tokens, tags) {
    if (!(header.startsWith("100") || header.startsWith("110") || header.startsWith("111"))) {
      return tags;
    }
    const n = tokens.length;
    if (n < 3) return tags;
    for (let i = 1; i < n - 1; i++) {
      if (tokens[i] !== "-") continue;
      const yearTok = tokens[i + 1];
      if (!/^\d{4}$/.test(yearTok)) continue;
      if (tokens[i - 1] !== ",") continue;
      if (tagCode(tags[i - 1]) !== "a") continue;
      if (tagCode(tags[i]) === "d" && tagCode(tags[i + 1]) === "d") continue;
      if (tagCode(tags[i]) !== "a" || tagCode(tags[i + 1]) !== "a") continue;
      const out = tags.slice();
      out[i] = "B-d";
      out[i + 1] = "I-d";
      return out;
    }
    return tags;
  }

  // Generalized rule: in 100|*|# headings, retag a stretch of tokens between
  // an initial $a span and a 'terminator' (year or $t boundary) as $a
  // continuation, provided the stretch contains an <initial> '.' pair and no
  // title-words/honorifics/Roman numerals/bare degrees.
  const BARE_DEGREES = new Set([
    "MD", "MA", "MS", "MSc", "MBA", "MPA", "MFA", "MPH", "MSW",
    "PhD", "PHD", "EdD", "JD", "DDS", "DVM", "DPhil", "ScD",
    "BA", "BS", "BSc", "BFA", "LLB", "LLD", "RN", "BSN", "MSN",
    "CPA", "PE",
  ]);
  const DEGREE_STRINGS = [
    "M . D .", "M . A .", "Ph . D .", "B . A .", "B . S .",
    "M . S .", "J . D .", "LL . D .", "D . D .", "Esq .",
  ];

  function ruleAPersonalNameTrailingBlock(header, tokens, tags) {
    if (!header.startsWith("100")) return tags;
    const spans = findSpans(tags);
    if (!spans.length) return tags;
    if (spans[0][2] !== "a" || spans[0][0] !== 0) return tags;
    const aEnd = spans[0][1];
    const n = tokens.length;
    if (aEnd >= n) return tags;

    // Find the terminator. Preference: earliest $t boundary, else earliest
    // year-shaped token (or '-' + year).
    let term = n;
    for (let si = 1; si < spans.length; si++) {
      const [s, , code] = spans[si];
      if (code === "t" && s < n && isCapitalizedWord(tokens[s])) {
        if (s > 0 && tokens[s - 1] === ".") { term = s; break; }
      }
    }
    for (let i = aEnd; i < term; i++) {
      const t = tokens[i];
      if (/^\d{4}$/.test(t)) { term = i; break; }
      if (t === "-" && i + 1 < n && /^\d{4}$/.test(tokens[i + 1])) {
        term = i; break;
      }
    }
    if (term <= aEnd) return tags;

    const middle = tokens.slice(aEnd, term);
    if (!middle.length) return tags;

    let hasNonA = false;
    for (let i = 0; i < middle.length; i++) {
      const t = middle[i];
      const idx = aEnd + i;
      if (!isNameContinuationToken(t)) return tags;
      if (TITLE_WORDS.has(t.toLowerCase())) return tags;
      if (HONORIFICS.has(t)) return tags;
      if (isRomanNumeralGeneration(t)) return tags;
      if (BARE_DEGREES.has(t)) return tags;
      if (tagCode(tags[idx]) !== "a") hasNonA = true;
    }
    if (!hasNonA) return tags;

    // Require an <initial> '.' pair somewhere in tokens[0:term], allowing
    // combining marks between the initial and the period.
    let hasInitialPair = false;
    for (let i = 0; i < term; i++) {
      if (!isInitial(tokens[i])) continue;
      let j = i + 1;
      while (j < term && isCombiningMark(tokens[j])) j++;
      if (j < term && tokens[j] === ".") { hasInitialPair = true; break; }
    }
    if (!hasInitialPair) return tags;

    const middleText = middle.join(" ");
    for (const d of DEGREE_STRINGS) {
      if (middleText.indexOf(d) >= 0) return tags;
    }

    const out = tags.slice();
    for (let idx = aEnd; idx < term; idx++) {
      out[idx] = "I-a";
    }
    return out;
  }

  // Round-3 rule: trailing '(City, State)' in 110|2|# stays in $a.
  function ruleACorporateJurisdictionParen(header, tokens, tags) {
    if (header !== "110|2|#") return tags;
    const n = tokens.length;
    if (n < 3) return tags;
    if (tokens[n - 1] !== ")") return tags;
    let depth = 1, openPos = -1;
    for (let j = n - 2; j >= 0; j--) {
      if (tokens[j] === ")") depth++;
      else if (tokens[j] === "(") {
        depth--;
        if (depth === 0) { openPos = j; break; }
      }
    }
    if (openPos < 1) return tags;
    if (tagCode(tags[openPos - 1]) !== "a") return tags;

    const inside = tokens.slice(openPos + 1, n - 1);
    if (!inside.length) return tags;
    let hasWord = false;
    for (const t of inside) {
      if (t === "," || t === ".") continue;
      if (isCapitalizedWord(t)) { hasWord = true; continue; }
      return tags;
    }
    if (!hasWord) return tags;
    for (let i = openPos; i < n; i++) {
      if (tagCode(tags[i]) !== "c") return tags;
    }
    const out = tags.slice();
    for (let i = openPos; i < n; i++) out[i] = "I-a";
    return out;
  }

  // Round-3 rule: trailing '(Cap-word lower-word)' in 100|1|# is $c
  // (two-word occupation/role).
  function ruleCParenTwoWordOccupation(header, tokens, tags) {
    if (header !== "100|1|#") return tags;
    const n = tokens.length;
    if (n < 5) return tags;
    if (tokens[n - 1] !== ")" || tokens[n - 4] !== "(") return tags;
    const w1 = tokens[n - 3], w2 = tokens[n - 2];
    if (!isCapitalizedWord(w1)) return tags;
    if (!(/^[\p{L}]+$/u.test(w2) && /^\p{Ll}/u.test(w2))) return tags;
    if (!isCapitalizedWord(tokens[n - 5])) return tags;
    if (tagCode(tags[n - 5]) !== "a") return tags;
    for (let i = n - 4; i < n; i++) {
      if (tagCode(tags[i]) !== "a") return tags;
    }
    const out = tags.slice();
    out[n - 4] = "B-c";
    out[n - 3] = "I-c";
    out[n - 2] = "I-c";
    out[n - 1] = "I-c";
    return out;
  }

  // Round-4: occupation in parens after initial+period (CRF emitted $q).
  function ruleCParenRoleAfterInitialPeriod(header, tokens, tags) {
    if (!header.startsWith("100|")) return tags;
    const n = tokens.length;
    let out = tags;
    let mutated = false;
    for (let k = 2; k < n; k++) {
      if (tokens[k] !== "(") continue;
      if (tokens[k - 1] !== ".") continue;
      if (!isInitial(tokens[k - 2])) continue;
      if (HONORIFICS.has(tokens[k - 2])) continue;
      const paren = findParenGroup(tokens, k);
      if (!paren) continue;
      const [ps, pe] = paren;
      const inside = tokens.slice(ps + 1, pe - 1);
      if (inside.length !== 1 || !isCapitalizedWord(inside[0])) continue;
      if (shareFirstLetter(tokens[k - 2], inside[0])) continue;
      let allQ = true;
      for (let i = ps; i < pe; i++) {
        if (tagCode(out[i]) !== "q") { allQ = false; break; }
      }
      if (!allQ) continue;
      if (!mutated) { out = out.slice(); mutated = true; }
      retagSpanInPlace(out, ps, pe, "c");
    }
    return out;
  }

  // Round-4: ',<honorific|title>' optionally followed by '.' and then year
  // or EOS — promote to $c.
  function ruleCPromoteHonorificAfterName(header, tokens, tags) {
    if (!header.startsWith("100")) return tags;
    const n = tokens.length;
    if (n < 3) return tags;
    let out = tags;
    let mutated = false;
    let i = 1;
    while (i < n - 1) {
      if (tokens[i - 1] !== ",") { i++; continue; }
      const t = tokens[i];
      const isTitle = TITLE_WORDS.has(t.toLowerCase());
      const isHonor = HONORIFICS.has(t);
      if (!(isTitle || isHonor)) { i++; continue; }
      let endIdx = i + 1;
      if (endIdx < n && tokens[endIdx] === ".") endIdx++;
      let ok = false;
      if (endIdx >= n) {
        ok = true;
      } else if (tokens[endIdx] === ",") {
        const j = endIdx + 1;
        if (j >= n) ok = true;
        else if (/^\d{4}$/.test(tokens[j])) ok = true;
        else if (tokens[j] === "-" && j + 1 < n && /^\d{4}$/.test(tokens[j + 1])) ok = true;
      }
      if (!ok) { i++; continue; }
      let allA = true;
      for (let k = i; k < endIdx; k++) {
        if (tagCode(out[k]) !== "a") { allA = false; break; }
      }
      if (!allA) { i++; continue; }
      if (tagCode(out[i - 1]) !== "a") { i++; continue; }
      if (!mutated) { out = out.slice(); mutated = true; }
      out[i] = "B-c";
      for (let k = i + 1; k < endIdx; k++) out[k] = "I-c";
      i = endIdx;
    }
    return out;
  }

  const RULES = [
    ["c_paren_role_after_full_name", ruleCParenRoleAfterFullName],
    ["a_trailing_initials_in_personal_name", ruleATrailingInitialsInPersonalName],
    ["a_uniform_title_paren_tail", ruleAUniformTitleParenTail],
    ["a_personal_name_continuation", ruleAPersonalNameContinuation],
    ["a_trailing_initial_single_pair", ruleATrailingInitialSinglePair],
    ["a_personal_name_trailing_block", ruleAPersonalNameTrailingBlock],
    ["a_corporate_jurisdiction_paren", ruleACorporateJurisdictionParen],
    ["c_paren_two_word_occupation", ruleCParenTwoWordOccupation],
    ["c_paren_role_after_initial_period", ruleCParenRoleAfterInitialPeriod],
    ["c_promote_honorific_after_name", ruleCPromoteHonorificAfterName],
    ["d_incomplete_date_range", ruleDIncompleteDateRange],
  ];

  function applyRules(header, tokens, tags) {
    let cur = tags;
    const fired = [];
    for (const [name, fn] of RULES) {
      const next = fn(header, tokens, cur);
      if (next !== cur) {
        // Check whether any tag actually differs (function may have returned
        // a new array even if contents are identical — defensive).
        let changed = false;
        if (next.length !== cur.length) changed = true;
        else {
          for (let i = 0; i < next.length; i++) {
            if (next[i] !== cur[i]) { changed = true; break; }
          }
        }
        if (changed) {
          fired.push(name);
          cur = next;
        }
      }
    }
    return { tags: cur, fired };
  }

  // ---------- Public API ----------
  function tag(text, model) {
    const tokens = tokenize(text);
    if (tokens.length === 0) {
      return { header: null, tokens: [], tags: [], marc: "", rulesFired: [] };
    }
    const header = predictHeader(tokens, model);
    const crfTags = viterbi(tokens, header, model);
    const { tags, fired } = applyRules(header, tokens, crfTags);
    const marc = reassembleMarc(header, tokens, tags);
    return { header, tokens, tags, marc, rulesFired: fired };
  }

  async function load(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("failed to fetch model: " + resp.status);
    let json;
    if (url.endsWith(".gz")) {
      // Browser must handle Content-Encoding: gzip server-side, OR we read as
      // ArrayBuffer and decompress with DecompressionStream when available.
      if (typeof DecompressionStream !== "undefined") {
        const ds = new DecompressionStream("gzip");
        const stream = resp.body.pipeThrough(ds);
        const text = await new Response(stream).text();
        json = JSON.parse(text);
      } else {
        // Assume server set Content-Encoding: gzip.
        json = await resp.json();
      }
    } else {
      json = await resp.json();
    }
    return loadFromObject(json);
  }

  function loadFromObject(json) {
    // Build a Set for vocab lookups (faster than searching an array).
    const vocab = new Set();
    if (json.header_classifier && json.header_classifier.vocab) {
      for (const w of json.header_classifier.vocab) vocab.add(w);
    }
    json._headerVocab = vocab;
    json.tag = function (text) { return tag(text, json); };
    json.predictHeader = function (tokens) { return predictHeader(tokens, json); };
    return json;
  }

  return {
    load,
    loadFromObject,
    tokenize,
    shape,
    tokenFeatures,
    headerDocFeatures,
    predictHeader,
    viterbi,
    reassembleMarc,
    applyRules,
    tag,
  };
});
