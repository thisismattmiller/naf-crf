#!/usr/bin/env node
// Compare JS Viterbi predictions to Python predictions (NOT to gold).
// Goal: confirm JS == Python on identical inputs.

import { readFileSync } from "fs";
import path from "path";
import url from "url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const runtimeSrc = readFileSync(path.join(__dirname, "..", "docs", "naf_crf.js"), "utf-8");
const moduleScope = { module: { exports: {} }, exports: {} };
new Function("module", "exports", runtimeSrc)(moduleScope.module, moduleScope.exports);
const NafCRF = moduleScope.module.exports;

const MODEL_PATH = "/Volumes/ImNotGlum/naf-crf/models/model.json";
const PRED_PATH = "/Volumes/ImNotGlum/naf-crf/splits/dev_python_pred.jsonl";

const model = NafCRF.loadFromObject(JSON.parse(readFileSync(MODEL_PATH, "utf-8")));
const rows = readFileSync(PRED_PATH, "utf-8").split("\n").filter(Boolean).map(l => JSON.parse(l));

let nMatch = 0;
let nTokenMatch = 0;
let nTokens = 0;
const mismatches = [];

for (const row of rows) {
  const jsPred = NafCRF.viterbi(row.tokens, row.header, model);
  let ok = true;
  for (let i = 0; i < row.tokens.length; i++) {
    nTokens++;
    if (jsPred[i] === row.py_pred[i]) nTokenMatch++;
    else ok = false;
  }
  if (ok) nMatch++;
  else if (mismatches.length < 5) {
    mismatches.push({ tokens: row.tokens, header: row.header, py: row.py_pred, js: jsPred });
  }
}

console.log(`JS vs Python predictions on ${rows.length} dev rows:`);
console.log(`  exact-sequence match: ${nMatch}/${rows.length} (${(100*nMatch/rows.length).toFixed(2)}%)`);
console.log(`  token-level agreement: ${nTokenMatch}/${nTokens} (${(100*nTokenMatch/nTokens).toFixed(4)}%)`);

if (mismatches.length) {
  console.log();
  console.log("=== first mismatches ===");
  for (const m of mismatches) {
    console.log("  tokens:", m.tokens.join(" / "));
    console.log("  header:", m.header);
    console.log("  py:    ", m.py.join(" "));
    console.log("  js:    ", m.js.join(" "));
    const diffs = [];
    for (let i = 0; i < m.tokens.length; i++) {
      if (m.py[i] !== m.js[i]) diffs.push(`${i}:${m.tokens[i]!=null?JSON.stringify(m.tokens[i]):""} py=${m.py[i]} js=${m.js[i]}`);
    }
    console.log("  diff:  ", diffs.join("; "));
    console.log();
  }
}
