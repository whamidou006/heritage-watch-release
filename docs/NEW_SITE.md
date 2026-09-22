# Bring your own site

## 1. Prepare imagery and annotations

Supply north-up, three-band uint8 RGB GeoTIFFs, with one common CRS, resolution
and aligned pixel lattice. Footprints may differ. Reproject and co-register
outside this package if necessary; the code deliberately refuses incompatible
grids rather than silently comparing different ground windows. RGB band order
must actually be RGB; no automatic band identification is attempted.

Bring imagery of comparable physical scale. Matching 128px alone is not enough
when a new site's resolution differs. The original 128px window is roughly
32×38m. Nodata inside a footprint is not currently screened: inspect it before
use (the reference Herat scenes had no internal blank pixels). A footprint
intersection is a rectangle test, not a valid-data mask.

Annotations must be **labelled points already identified as changes**, not
masks. This package does not find new change locations automatically.
Coordinates must use the raster CRS. Use UTF-8 comma-delimited CSV, with exact
headers mapped in the config. Decimal-comma coordinates such as `"34,34"` are
accepted when properly CSV-quoted. The original x header is `"X "` with a
trailing space; the loader neither strips nor guesses headers.

Required fields:

| Semantic field | Meaning |
|---|---|
| fid | Unique integer point ID; duplicates are an error |
| category | Class string (values are stripped) |
| layer | Two YYYYMM endpoint tokens, e.g. `change_202205_and_202310` |
| year | Eight-digit fallback interval, e.g. `20222023` |
| x / y | Finite coordinates in the raster CRS |

## 2. Configure pairing and filtering

Copy the heavily commented `configs/template_new_site.yaml` to a new YAML.
Every setting is validated. File paths resolve relative to that config;
`${ENV_VAR}` and `${ENV_VAR:-default}` are supported.

In month mode, target each month's **15th** and select the nearest scene within
±75 days. Ties prefer the earlier scene. If either endpoint has no match,
drop the point instead of misrepresenting its provenance. Layers without two
month tokens use the `year` field (latest scene of the first year, earliest of
the second). `pairing: year` forces the legacy rule for controlled experiments.
Date ordering must remain before < after.

`pair` preserves the source CSV's interval identifier, even if its year-only
label is inconsistent with the resolved acquisition dates (a historical Herat
case is `20102011` resolving to 2010→2012). Always inspect `t1_date` and `t2_date`
for actual scene dates.

The manifest checks **the intersection of the two chosen rasters**, then class
membership and complete chip windows at **both** dates. Partial chips are
rejected, never zero-padded. The ledger is mutually exclusive, so counts sum
to raw count minus usable count. Inspect the per-pair class distribution before
training. Empty manifests cannot be embedded.

## 3. Fit and evaluate a new-site model

```bash
heritage-watch manifest --config configs/my_site.yaml --out cache/site.json
heritage-watch embed --config configs/my_site.yaml --manifest cache/site.json \
  --encoder satlas --out cache/site_satlas.npz
heritage-watch evaluate --config configs/my_site.yaml \
  --features cache/site_satlas.npz --representation satlas_mi_si_diff
heritage-watch train --config configs/my_site.yaml \
  --features cache/site_satlas.npz --representation satlas_mi_si_diff --out out/model.joblib
```

Evaluate **before** fitting the deployment bundle. The model is frozen-feature
logistic regression, not encoder fine-tuning. Taxonomy changes require a new
head. For representations using DINOv2, also embed that encoder and pass both
aligned caches. For alternative heads use `protocol.evaluate(..., clf_factory=...)`;
the factory must create a fresh pipeline with every learned preprocessing step
inside it. Optional PCA is available via `default_classifier(pca_components=...)`
and is off for the published table.

## 4. Measure transfer of an existing model

Prediction requires the same class list **in the same order** and chip size as
the bundle. It validates the required encoder set, representation and both
declared and actual classifier dimensions. Encoders are selected from the bundle,
not independently from the new config.

```bash
heritage-watch predict --model out/model.joblib --config configs/new_site.yaml \
  --out out/predictions.csv
python -u - <<'PY'
import csv
from sklearn.metrics import classification_report, f1_score
from heritage_watch.config import load_config
cfg = load_config("configs/new_site.yaml")
with open("out/predictions.csv") as f:
    rows = list(csv.DictReader(f))
y = [r["truth"] for r in rows]
p = [r["prediction"] for r in rows]
print("Transfer macro-F1:", f1_score(y, p, labels=cfg.classes,
                                    average="macro", zero_division=0))
print(classification_report(y, p, labels=cfg.classes, zero_division=0))
PY
```

Declare absent-class handling and sample counts. Do not tune the model on this
transfer evaluation set. A single held-out-site score is not an eight-replicate
paired CV result; do not feed it into `compare`.

For a **within-new-site** score, extract its features and run `evaluate`.
This retrains each fold's head on the new site's labels: it is a different
question from cross-site transfer.

## 5. Temporal controls

```bash
python -u scripts/controls.py date-only --config configs/my_site.yaml \
  --features cache/site_satlas.npz --out out/date_control.json
python -u scripts/controls.py loio --config configs/my_site.yaml \
  --features cache/site_satlas.npz --representation satlas_mi_si_diff \
  --manifest cache/site.json --train-size 500 --out out/loio.json
```

Reduce train size for small datasets, and declare the change. LOIO holds identical
test rows and training sizes across “seen interval” and “unseen interval” arms,
scores only classes actually present in each test half, and reports shared
endpoint scenes from the actual manifest. It is **not spatially disjoint**:
it diagnoses interval composition, not clean spatial-and-temporal transfer.

Only load trusted joblib bundles. Model outputs should assist expert review,
not replace heritage-conservation ground truth or authorization decisions.
