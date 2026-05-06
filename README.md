# naf-crf

Conditional Random Fields applied to NAF (Name Authority File) authorized
labels to recover MARC subfield structure. Trained from ~10M aligned LCNAF
records, runs in the browser.

## Live demo

Deployed via GitHub Pages from `docs/`:
https://thisismattmiller.github.io/naf-crf/

## Layout

| Path | What it is |
|---|---|
| `docs/` | Static site (GH Pages root): `index.html`, `naf_crf.js`, `model.json.gz` |
| `tests/` | Node-based parity tests (`cross_check.mjs`, `validate.mjs`) |
| `align_1xx.py`, `build_splits_v2.py`, `build_vocab.py`, `train.py`, `export_models.py` | Training pipeline |
| `features.py` | Per-token CRF feature extractor (Python — JS port lives in `docs/naf_crf.js`) |
| `eval_e2e.py` | End-to-end evaluation |
| `.github/workflows/deploy.yml` | Auto-deploys `docs/` to GH Pages on push to main |

## Browser inference

```html
<script src="naf_crf.js"></script>
<script>
  const model = await NafCRF.load("model.json.gz");
  const r = model.tag("Smith, John, 1962-");
  // r.header   = "100|1|#"
  // r.tokens   = ["Smith", ",", "John", ",", "1962", "-"]
  // r.tags     = ["B-a", "I-a", "I-a", "I-a", "B-d", "I-d"]
  // r.marc     = "1001 $aSmith, John,$d1962-"
</script>
```

## Numbers (dev split)

- Header LR accuracy: **89.3%**
- CRF exact-sequence match (oracle header): **93.9%**
- End-to-end exact-sequence match: **89.9%**
- Browser model: 5.8 MB raw / 2.1 MB gzipped

## Deploy

Push to `main`. The workflow uploads `docs/` and publishes via the
`actions/deploy-pages` action. First time, enable Pages in repo settings →
Pages → Source = "GitHub Actions".
