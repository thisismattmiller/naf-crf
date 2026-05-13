#!/usr/bin/env node
// Verify JS rules produce identical tags to Python rules, given identical
// CRF input. We don't re-run the CRF in JS here; we just feed py_crf into
// the JS applyRules() and compare its output to py_after.

import { readFileSync } from "fs";
import path from "path";
import url from "url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const runtimeSrc = readFileSync(path.join(__dirname, "..", "docs", "naf_crf.js"), "utf-8");
const moduleScope = { module: { exports: {} }, exports: {} };
new Function("module", "exports", runtimeSrc)(moduleScope.module, moduleScope.exports);
const NafCRF = moduleScope.module.exports;

const PRED_PATH = "/Volumes/ImNotGlum/naf-crf/splits/dev_python_with_rules.jsonl";
const rows = readFileSync(PRED_PATH, "utf-8").split("\n").filter(Boolean).map(l => JSON.parse(l));

let nMatch = 0;
let nMismatch = 0;
let nRowsRulesFiredPy = 0;
let nRowsRulesFiredJs = 0;
let nAgreeOnFire = 0;
const examples = [];

for (const row of rows) {
  const r = NafCRF.applyRules(row.header, row.tokens, row.py_crf);
  const jsTags = r.tags;
  const jsFired = r.fired;

  if (row.fired.length > 0) nRowsRulesFiredPy++;
  if (jsFired.length > 0) nRowsRulesFiredJs++;

  let ok = true;
  for (let i = 0; i < jsTags.length; i++) {
    if (jsTags[i] !== row.py_after[i]) { ok = false; break; }
  }
  if (ok) {
    nMatch++;
    if (row.fired.length === jsFired.length &&
        row.fired.every((f, i) => f === jsFired[i])) {
      nAgreeOnFire++;
    }
  } else {
    nMismatch++;
    if (examples.length < 5) {
      examples.push({
        header: row.header,
        tokens: row.tokens,
        py_crf: row.py_crf,
        py_after: row.py_after,
        js_after: jsTags,
        py_fired: row.fired,
        js_fired: jsFired,
      });
    }
  }
}

console.log(`JS rules vs Python rules on ${rows.length} dev rows:`);
console.log(`  exact-sequence match: ${nMatch}/${rows.length} (${(100*nMatch/rows.length).toFixed(3)}%)`);
console.log(`  mismatches: ${nMismatch}`);
console.log(`  rule-fire rows (py): ${nRowsRulesFiredPy}`);
console.log(`  rule-fire rows (js): ${nRowsRulesFiredJs}`);

if (examples.length) {
  console.log();
  console.log("=== first mismatches ===");
  for (const e of examples) {
    console.log("  tokens:    ", e.tokens.join(" "));
    console.log("  header:    ", e.header);
    console.log("  py_crf:    ", e.py_crf.join(" "));
    console.log("  py_after:  ", e.py_after.join(" "));
    console.log("  js_after:  ", e.js_after.join(" "));
    console.log("  py_fired:  ", JSON.stringify(e.py_fired));
    console.log("  js_fired:  ", JSON.stringify(e.js_fired));
    console.log();
  }
}
