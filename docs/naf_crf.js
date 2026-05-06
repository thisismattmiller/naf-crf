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

  // ---------- Public API ----------
  function tag(text, model) {
    const tokens = tokenize(text);
    if (tokens.length === 0) {
      return { header: null, tokens: [], tags: [], marc: "" };
    }
    const header = predictHeader(tokens, model);
    const tags = viterbi(tokens, header, model);
    const marc = reassembleMarc(header, tokens, tags);
    return { header, tokens, tags, marc };
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
    tag,
  };
});
