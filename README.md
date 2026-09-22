# Heritage Watch

**Semantic change classification at annotated points in bi-temporal aerial imagery.**
This is not a change localiser, segmentation system, or a model trained from scratch.
It classifies a before/after pair at a point already identified by an analyst.

The reference study covers Herat Old City, Afghanistan, 2009–2025: 21 GeoTIFF
scenes, 10 usable acquisition pairs, and 879 labels. Herat is on UNESCO's
**Tentative List**, not an inscribed World Heritage Site. Nominal image resolution
is 0.25 m/px; the geographic raster grid gives approximately 0.247 × 0.297 m pixels.

Frozen DINOv2 ViT-B/14 and SatlasPretrain Aerial Swin-v2-B encoders feed a balanced
multinomial logistic-regression head. No encoder fine-tuning is performed.
Labels are **points, not masks**.

## Installation

Python 3.10+ is required. On a fresh machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
heritage-watch --help
```

The reference environment is Python 3.12, NumPy 2.5, sklearn 1.8, torch 2.8,
rasterio 1.5 and timm 1.0.26. Encoder weights may download on first use;
their licenses are separate from this MIT-licensed code. The data, features,
annotations and pretrained weights are **not included** and are not licensed
by this repository. Obtain authorized copies from the dataset custodians.
There is no public dataset URL or paper DOI asserted here.

Without installation, use existing dependencies:

```bash
export PYTHONPATH=src
python -u -m heritage_watch.cli --help
```

All commands below can replace `heritage-watch` with `python -u -m heritage_watch.cli`.
Cache evaluation is CPU-only and does not load encoders. BLAS is capped at four
threads at import and call time; `HERITAGE_THREADS=2` can reduce shared-host load.
Embedding defaults to CPU. GPU use is explicit, for example
`CUDA_VISIBLE_DEVICES=3 heritage-watch embed ... --device cuda`. Never assume
that a GPU is free. No full extraction or seven-model sweep is needed for a smoke test.

## 1. Reproduce the study

Put the authorized dataset at `data/dataset/`, or set `HERITAGE_DATA_ROOT` to
its absolute root. It must contain `Herat_all_changes.csv` and
`herat_site_sat_images/herat_site_TM_z19/*.tif`. Relative paths in YAML resolve
relative to the YAML file, not the working directory.

```bash
heritage-watch manifest --config configs/herat.yaml --out cache/manifest.json
# Expect 879: New Construction 380, Solar Panel 308, Destruction 138,
# Temporary Structure 53. Every removed row has a ledger reason.

# With authorized precomputed 128px, month-paired features in cache/:
heritage-watch evaluate --config configs/herat.yaml \
  --features cache/features_month.npz cache/features_satlas_month.npz \
  --representation satlas_mi_si_diff

# Full seven-row table; substantially more expensive than one evaluation:
heritage-watch reproduce --config configs/herat.yaml --from-cache cache/
# Equivalent: python -u scripts/reproduce_results.py --config configs/herat.yaml --from-cache cache/

heritage-watch compare --config configs/herat.yaml \
  --features cache/features_month.npz cache/features_satlas_month.npz \
  --a satlas_mi_si_diff --b merged
```

For extraction from scratch, omit `--from-cache` on `reproduce`, or run:

```bash
heritage-watch embed --config configs/herat.yaml --manifest cache/manifest.json \
  --encoder dinov2 --out cache/features_month.npz
heritage-watch embed --config configs/herat.yaml --manifest cache/manifest.json \
  --encoder satlas --out cache/features_satlas_month.npz
```

The caches contain `f1`, `f2`, Satlas `fmi`, and `y`, `pair`, `x`, `yy`, `fid`.
New caches also record encoder and chip size. Legacy caches lack chip-size
provenance: loading warns, and the caller must independently verify the extraction
size. All cache rows and metadata must agree exactly; we never assume a row-order
join is safe. Never load an untrusted joblib model.

## 2. Train on new data

Copy `configs/template_new_site.yaml`, adapt its paths, exact CSV headers,
taxonomy, and scene-date regex. Then:

```bash
heritage-watch manifest --config configs/my_site.yaml --out cache/my_manifest.json
heritage-watch embed --config configs/my_site.yaml --manifest cache/my_manifest.json \
  --encoder satlas --out cache/my_satlas.npz
heritage-watch evaluate --config configs/my_site.yaml \
  --features cache/my_satlas.npz --representation satlas_mi_si_diff
heritage-watch train --config configs/my_site.yaml \
  --features cache/my_satlas.npz --representation satlas_mi_si_diff --out out/model.joblib
```

Training fits the head on **all** labels; it is not a new test score. The bundle
includes class order, representation, required encoders, chip size, feature
dimension, software version and a SHA-256 fingerprint of the ordered training
metadata and features.

## 3. Evaluate transfer to another site

Use the **same taxonomy, class order and chip size** with new-site imagery and
labelled evaluation points, and comparable physical resolution:

```bash
heritage-watch predict --model out/model.joblib \
  --config configs/new_site.yaml --out out/predictions.csv
```

Prediction rebuilds the new manifest and uses exactly the bundle's encoders and
representation. Incompatible metadata or feature dimensions cause an error
before scoring. Output contains point IDs, true labels, predictions, dates and
uncalibrated probabilities. Compute transfer macro-F1 as described in
[NEW_SITE.md](docs/NEW_SITE.md). Running `evaluate` instead measures
**within-new-site CV**, not transfer of the already trained model.

## Published results

Eight paired jittered-grid replicates, five spatially grouped folds, n=879.
Macro-F1 averages the four classes equally; majority-class floor **0.151**.

| Representation | CLI name | dim | Macro-F1 |
|---|---|---:|---:|
| DINOv2, post-event only | `dinov2_post` | 768 | 0.6439 |
| DINOv2, both + diff | `dinov2_both_diff` | 2304 | 0.6656 |
| Satlas-SI, post-event only | `satlas_si_post` | 1920 | 0.6698 |
| Satlas-SI, both + diff | `satlas_si_both_diff` | 5760 | 0.6918 |
| Satlas-MI, order-invariant | `satlas_mi` | 1920 | 0.6823 |
| **Satlas-MI + SI diff (selected)** | `satlas_mi_si_diff` | **3840** | **0.7259** |
| Merged DINOv2 + Satlas | `merged` | 8064 | 0.7062 |

The selected representation is `[f_mi, f_si_t2 - f_si_t1]`. MI alone max-pools
over time and is order-invariant; the signed SI difference restores direction.
Satlas's four feature-pyramid levels are global-average-pooled and concatenated.
DINOv2 resizes native 128px chips to 224px with ImageNet normalization;
Satlas uses native chips scaled to [0,1].

**The top two are not separated.** The selected system wins 8/8 paired replicates
by +0.0197, above 2·SE but **0.0003 below** the 0.02 minimum effect. Selection is
on parsimony: 3840 dimensions and one encoder *family* (SI and MI checkpoints),
versus 8064 dimensions and two encoder families.

Important limits:

- The YYYY-versus-YYYYMM provenance bug was fixed. Its strictly paired effect,
  +0.0067 on 811 shared points, is inside the noise floor: **score-neutral,
  justified on provenance, not accuracy**. Different dataset versions contain
  different populations; their headline changes cannot be attributed to this fix.
- Spatial blocking is not time blocking. All acquisition pairs appear on both
  sides of every fold. The report's leave-one-interval-out diagnostic is
  directionally clear — a held-out interval is worse in 9 of 10 cases — but it
  was run on the **superseded 838-sample manifest**, its test rows are drawn
  **randomly within the interval rather than spatially blocked**, and it scores
  only the classes present in each subset. It therefore supports "temporal
  generalisation is materially worse than the headline" and **no specific
  number**. Do not quote a macro-F1 penalty from it.
- Destruction is weak: published fixed-grid F1 **0.548**, **50.0%** correct and
  **41.3%** read as New Construction—opposite temporal orders of the same states.
  Those fixed-grid per-class numbers use a different averaging scheme from the
  jittered headline and need not equal `report()`'s per-class outputs.
- Encoders are frozen and untuned: **0.7259 is a lower bound, not a ceiling**
  on achievable performance in this setting, not a guaranteed deployment score.
- Site-specific performance and probabilities are not deployment assurances.
  New-site quantitative claims require independent ground truth.

See [protocol and decision rule](docs/PROTOCOL.md), [results and solver
fidelity](docs/RESULTS.md), and [new-site instructions](docs/NEW_SITE.md).

## Tests and citation

```bash
PYTHONPATH=src python -u -m pytest tests/ -q
```

Tests create synthetic GeoTIFFs under ignored `out/pytest-work/`, never in a
system temporary directory. They require no real data, downloaded weights or GPU.

Until a paper identifier is supplied, cite the software without inventing one:

> Heritage Watch contributors. *Heritage Watch: Semantic Change Classification
> on Bi-temporal Aerial Imagery*. Software, version 1.0.0, 2026.

Also cite the original DINOv2 and SatlasPretrain work when using those encoders.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE).
