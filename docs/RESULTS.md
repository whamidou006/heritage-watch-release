# Results and reproduction

## Published section-5 table

All rows use n=879 month-resolved labels and eight paired jittered spatial
replicates, 128px chips, frozen encoders, balanced logistic regression, C=1,
and **no PCA**.

| Representation | dim | Macro-F1 | sd |
|---|---:|---:|---:|
| DINOv2, post-event only | 768 | 0.6439 | 0.017 |
| DINOv2, both + diff | 2304 | 0.6656 | 0.011 |
| Satlas-SI, post-event only | 1920 | 0.6698 | 0.010 |
| Satlas-SI, both + diff | 5760 | 0.6918 | 0.006 |
| Satlas-MI, order-invariant | 1920 | 0.6823 | 0.011 |
| **Satlas-MI + SI diff** | **3840** | **0.7259** | **0.010** |
| Merged DINOv2 + Satlas | 8064 | 0.7062 | 0.006 |

The top-two difference is +0.0197, 8/8 wins, 2·SE=0.0094.
It **fails** the 0.02 minimum effect by 0.0003: the two are **not separated**.
Select Satlas on parsimony, not a claimed resolved superiority.

## Regenerate, do not transcribe

```bash
heritage-watch manifest --config configs/herat.yaml --out cache/manifest.json
heritage-watch reproduce --config configs/herat.yaml --from-cache cache/
# Or:
PYTHONPATH=src python -u scripts/reproduce_results.py \
  --config configs/herat.yaml --from-cache cache/
```

The cache directory must contain **features_month.npz** (DINOv2) and
**features_satlas_month.npz** (Satlas SI+MI). The loader verifies exact equality
of IDs, labels, pairs and coordinates before any combination. Legacy caches
lack chip-size metadata; verify 128px provenance separately. Data and caches
are obtained independently and never committed here.

Omitting `--from-cache` runs the manifest and both encoders first. This requires
authorized imagery and pretrained weights, and can be expensive. A full seven-row
sweep may take over an hour on a contended host. For cheap fidelity testing:

```bash
heritage-watch evaluate --config configs/herat.yaml \
  --features cache/features_month.npz cache/features_satlas_month.npz \
  --representation satlas_mi_si_diff
```

### Numerical implementation and release checks

The published table used torch float32 LBFGS (500 iterations, strong-Wolfe
search) and sample standard deviations. The clean reference harness used
sklearn StandardScaler and LogisticRegression. Both optimize the balanced
multinomial objective with C=1, but solver stopping and floating point matter.

On the release-check machine, the harness's old sklearn default `tol=1e-4`
gave **0.7214825**. Tightening to `tol=1e-9`, `max_iter=3000`, with unchanged
rows, grids and features gave **0.7262**, inside the required **0.7259 ±0.002**.
This is a measured implementation difference, not a changed expected target or
a new scientific result. The released default uses the tight tolerance.
Bitwise equality with the original GPU solver is not promised.

The real-data manifest check measured **879**, with class counts **380 / 308 /
138 / 53** and the exact ledger:

| Drop reason | Count |
|---|---:|
| Outside chosen-pair footprint | 350 |
| No scene near the stated month | 110 |
| Other | 25 |
| Reconstruction | 1 |
| Full chip clipped at edge | 5 |

Only the selected representation was re-scored for release verification.
**The complete seven-row sweep and full feature extraction were not re-run.**
The table above is the published source table, not seven newly measured release
results. The reproduction command computes all rows rather than returning stored
scores; small numerical deviations from the printed table can occur.

## Interpretation, provenance, controls

- Month parsing corrected the loss of YYYYMM information. A strictly paired
  comparison on 811 common points measured +0.0067, inside the noise floor:
  **score-neutral; adopted for provenance, not accuracy**. Scores across
  dataset versions cannot isolate this effect because the populations differ.
- Generalisation across time is materially worse, but by an unmeasured amount.
  The report's leave-one-interval-out diagnostic gave −0.1755 overall (9/10
  intervals worse) and −0.1110 on a subset. **Neither figure is quotable as a
  correction to the headline**: it ran on the superseded 838-sample manifest
  (whose interval list still contains `20122013`, absent from the corrected
  data), neither arm is spatially blocked — test rows are a random permutation
  within the interval — and F1 is computed over the classes present in each
  subset rather than the fixed four. Non-unanimity independently rules out
  "resolved" under the three-bar rule.
- The historical LOIO script used a hardcoded “shared endpoint” interval set.
  The release control derives endpoint sharing from actual scene paths instead.
  Month-corrected pairs chain more often, so the subset can change or disappear;
  **do not expect the historical −0.11 subset number from a different endpoint
  definition**. Neither control has been rerun as part of the cheap release check.
- Date-only macro-F1 **0.275** versus floor **0.099** was measured on the raw
  **six-class** month dataset with fixed-grid/three-seed evaluation. It is not a
  four-class score. The release date-only control defaults to the official
  jittered protocol and configured classes. To investigate the raw taxonomy,
  use a six-class config and matching raw caches; do not compare unlike metrics
  or call its jittered result an exact regeneration of the historical number.
- Destruction's published fixed-grid F1 is **0.548**, with 50.0% correct and
  41.3% confused as New Construction. The temporal direction distinguishes the
  same two ground states. Fixed-grid per-class values are not the jittered
  average returned by the release harness.
- Encoders remain **frozen and untuned**: the headline is a lower bound, not a
  ceiling, on achievable performance in this experimental setting.

Earlier appendix ablations were mostly measured on a year-resolved n=838
population; their point estimates are not silently relabelled as n=879 results.
