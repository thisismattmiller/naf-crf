#!/usr/bin/env node
// Validate that the JS runtime produces the same tag sequences as the Python
// CRF on a sample of dev rows. We load the exported model directly (no fetch),
// then for each dev row:
//   1. Reconstruct the label string from tokens (we don't have the original
//      whitespace, so we can only compare on tokenization-stable inputs).
//   2. Tokenize the *same way Python did* — i.e. just use the dev row's
//      tokens directly. (Avoids tokenizer divergence as a confound.)
//   3. Pass tokens + true header into Viterbi.
//   4. Compare Viterbi's tags to the dev row's gold tags.
//
// This isolates: does the JS Viterbi compute the same argmax as the trained
// CRF on identical inputs? If yes, JS and Python agree on tagging given the
// same featurization. We separately measure header-prediction agreement.

import { readFileSync, createReadStream } from "fs";
import { createGunzip } from "zlib";
import { createInterface } from "readline";
import path from "path";
import url from "url";

// Bring in the runtime. It uses UMD globals; import via require-style.
const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
// Read & eval the module — simplest path since it's UMD and we don't want a
// build step. The canonical copy lives in docs/ (the deployable site root).
const runtimeSrc = readFileSync(path.join(__dirname, "..", "docs", "naf_crf.js"), "utf-8");
const moduleScope = { module: { exports: {} }, exports: {} };
new Function("module", "exports", runtimeSrc)(moduleScope.module, moduleScope.exports);
const NafCRF = moduleScope.module.exports;

const MODEL_PATH = "/Volumes/ImNotGlum/naf-crf/models/model.json";
const DEV_PATH = "/Volumes/ImNotGlum/naf-crf/splits/dev.jsonl";
const SAMPLE_SIZE = 500;

const modelJson = JSON.parse(readFileSync(MODEL_PATH, "utf-8"));
const model = NafCRF.loadFromObject(modelJson);

// Read all dev rows.
const devText = readFileSync(DEV_PATH, "utf-8");
const allRows = devText.split("\n").filter(Boolean).map(l => JSON.parse(l));
// Random sample (seeded for reproducibility).
function lcg(seed) { let s = seed >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; }
const rand = lcg(42);
const sample = [];
for (let i = 0; i < SAMPLE_SIZE && allRows.length; i++) {
  const idx = Math.floor(rand() * allRows.length);
  sample.push(allRows.splice(idx, 1)[0]);
}

// 1. Tag agreement on sampled rows (using ORACLE header).
let nExact = 0;
let nTagAgree = 0;
let nTokens = 0;
const mismatches = [];

for (const row of sample) {
  const tokens = row.tokens;
  const goldTags = row.tags;
  const goldHeader = row.header;
  const predTags = NafCRF.viterbi(tokens, goldHeader, model);

  let ok = true;
  for (let i = 0; i < tokens.length; i++) {
    nTokens++;
    if (predTags[i] === goldTags[i]) nTagAgree++;
    else ok = false;
  }
  if (ok) nExact++;
  else if (mismatches.length < 5) {
    mismatches.push({ tokens, goldHeader, goldTags, predTags });
  }
}

console.log(`=== JS Viterbi (oracle header) vs gold on ${sample.length} dev rows ===`);
console.log(`exact-sequence match: ${nExact}/${sample.length} (${(100*nExact/sample.length).toFixed(2)}%)`);
console.log(`token-level agreement: ${nTagAgree}/${nTokens} (${(100*nTagAgree/nTokens).toFixed(2)}%)`);
console.log(`(reference: Python oracle exact-match was 93.94%)`);
console.log();

if (mismatches.length) {
  console.log("=== first mismatches ===");
  for (const m of mismatches) {
    console.log("  tokens:", m.tokens.join(" / "));
    console.log("  header:", m.goldHeader);
    console.log("  gold:  ", m.goldTags.join(" "));
    console.log("  pred:  ", m.predTags.join(" "));
    console.log();
  }
}

// 2. Header-prediction agreement.
let nHeaderCorrect = 0;
for (const row of sample) {
  const pred = NafCRF.predictHeader(row.tokens, model);
  if (pred === row.header) nHeaderCorrect++;
}
console.log(`header LR accuracy on sample: ${nHeaderCorrect}/${sample.length} `
  + `(${(100*nHeaderCorrect/sample.length).toFixed(2)}%)`);
console.log(`(reference: Python LR accuracy on full dev was 89.28%)`);

// 3. End-to-end (predicted header into Viterbi).
let nE2E = 0;
for (const row of sample) {
  const result = model.tag(row.tokens.join(" "));
  // We can't fully compare because tokenization may differ when re-tokenizing
  // a joined string. Use the original tokens instead.
  const predHdr = NafCRF.predictHeader(row.tokens, model);
  const predTags = NafCRF.viterbi(row.tokens, predHdr, model);
  let ok = true;
  for (let i = 0; i < row.tags.length; i++) {
    if (predTags[i] !== row.tags[i]) { ok = false; break; }
  }
  if (ok) nE2E++;
}
console.log(`end-to-end (predicted header) exact-match: ${nE2E}/${sample.length} `
  + `(${(100*nE2E/sample.length).toFixed(2)}%)`);
console.log(`(reference: Python end-to-end was 89.95%)`);
