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
| `rules.py` | Post-CRF correction rules (Python — JS port lives in `docs/naf_crf.js`) |
| `mine_failures.py`, `measure_rules.py` | Tools for discovering failure patterns and validating rules |
| `eval_e2e.py`, `eval_random_sample.py` | Evaluation |
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

## Numbers

On a 50k uniform random sample (held out from training):
- CRF exact-sequence match: **96.39%**
- CRF + post-processing rules: **97.49%** (+543 fixes, 17 regressions)
- Browser model: 5.8 MB raw / 2.1 MB gzipped

The pipeline:
1. Header LR predicts the field+indicator label (e.g. `100|1|#`)
2. CRF emits BIO subfield tags (e.g. `B-a I-a I-a B-d I-d`)
3. Post-CRF deterministic rules patch known CRF failure patterns (see `rules.py`)

Active rules (each net-positive on a held-out sample):
- `a_personal_name_trailing_block` — generalized trailing initials/words → $a (+204)
- `a_personal_name_continuation` — `Kutcher, Ashton, 1978-` keeps "Ashton" in $a (+186)
- `a_trailing_initials_in_personal_name` — `Tay, Andrew A. O.` keeps trailing initials in $a (+119)
- `a_corporate_jurisdiction_paren` — `Colchester Zoo (Colchester, England)` keeps location in $a (+35)
- `c_paren_role_after_full_name` — `Crosse, Thomas (Goldsmith)` tags "(Goldsmith)" as $c (+32)
- `a_trailing_initial_single_pair` — `Rozenberg, M.` keeps trailing "M." in $a (+27)
- `d_incomplete_date_range` — `Thomson, Barry, -1960` tags "-1960" as $d (+20)
- `c_paren_role_after_initial_period` — `Williams, Julius P. (Tenor)` tags "(Tenor)" as $c (+9)
- `c_promote_honorific_after_name` — `Madana, Acharya, 1920-` promotes "Acharya" to $c (+8)
- `c_paren_two_word_occupation` — `Fisher, Eric (Worm farmer)` tags "(Worm farmer)" as $c (+8)
- `a_uniform_title_paren_tail` — uniform-title closing paren stays in $a (+7)

## Deploy

Push to `main`. The workflow uploads `docs/` and publishes via the
`actions/deploy-pages` action. First time, enable Pages in repo settings →
Pages → Source = "GitHub Actions".
