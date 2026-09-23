# Results and reproduction

## Published section-5 table

All rows use n=882 month-resolved labels and eight paired jittered spatial
replicates, 64px chips, frozen encoders, balanced logistic regression, C=1,
and **no PCA**.

| Representation | dim | Macro-F1 | sd |
|---|---:|---:|---:|
| DINOv2, post-event only | 768 | 0.6921 | 0.012 |
| DINOv2, both + diff | 2304 | 0.7147 | 0.007 |
| Satlas-SI, post-event only | 1920 | 0.7204 | 0.005 |
| Satlas-SI, both + diff | 5760 | 0.7559 | 0.008 |
| Satlas-MI, order-invariant | 1920 | 0.6881 | 0.010 |
| **Satlas-MI + SI diff** | **3840** | **0.7788** | **0.004** |
| Merged DINOv2 + Satlas | 8064 | 0.7873 | 0.009 |

The top-two difference is +0.0085 for the merged pair, 7/8 replicates,
2·SE=0.0066. It clears the uncertainty bar but **fails** unanimity and the 0.02
minimum effect: the two are **not separated**. Select Satlas-MI + SI diff on
parsimony (3840 dims against 8064, one encoder against two), not a claimed
superiority. Every other row is resolved against it.

Chips are 64px, not the 128px used in earlier versions of this table. A paired
sweep over 32/64/128/256 on identical rows, one shared grid and identical folds
put 64 ahead of 128 by 0.0885 macro-F1, 8/8 replicates. Because DINOv2 resizes
every chip to a fixed 224px input, its arms differ only in ground extent, and it
gains just as much -- so the effect is field of view, not input resolution. At
0.247 m/px a 64px window spans about 16 m, roughly one building.

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
lack chip-size metadata; verify 64px provenance separately. Data and caches
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

This was checked against the 128px generation, whose target was 0.7259. On the
release-check machine the harness's old sklearn default `tol=1e-4` gave
**0.7214825**. Tightening to `tol=1e-9`, `max_iter=3000`, with unchanged rows,
grids and features gave **0.7262**, inside the required **±0.002**. This is a
measured implementation difference, not a changed expected target or a new
scientific result. The released default uses the tight tolerance. Bitwise
equality with the original GPU solver is not promised.

The real-data manifest check measured **882**, with class counts **382 / 309 /
138 / 53** and the exact ledger:

| Drop reason | Count |
|---|---:|
| Outside chosen-pair footprint | 350 |
| No scene near the stated month | 110 |
| Other | 25 |
| Reconstruction | 1 |
| Chip window clipped at the raster edge (64px) | 2 |

Only the last row depends on chip size: 64px clips 2 points, 128px clips 5,
which is why numbers quoted at the superseded 128px size carry n=879.
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
- Generalisation across time is materially worse, and now measured. The second
  official score, `evaluate_interval`, gives **0.5745** against the headline
  **0.7788**. Shuffling the interval labels — same capped training pool, same
  cell blocking, no date structure — gives **0.7392**, so 0.040 of the drop is
  the smaller pool and **0.165 is date novelty**. That 0.165 is a lower bound:
  the scenes chain, so a held-out interval's model has usually still seen one
  of its two endpoint images.
- Date-only macro-F1 is **0.4053** against a **0.1511** floor under the headline
  score, and collapses to **0.1739** against a **0.1112** floor under the
  interval score. The shortcut is real and the second score removes most of it.
  Regenerate with `scripts/baselines.py`; do not transcribe.
- Destruction is the weak class: F1 **0.642**, with 62.3% correct and 29.0%
  confused as New Construction — the same two ground states in the opposite
  temporal direction. Fixed-grid per-class values are not the jittered average
  returned by the release harness.
- Encoders remain **frozen and untuned**: the headline is a lower bound, not a
  ceiling, on achievable performance in this experimental setting.

Earlier appendix ablations were mostly measured on a year-resolved n=838
population; their point estimates are not silently relabelled as n=882 results.

## Regenerating these tables

The baseline rows in this file and in `hackathon/README.md` come from one
script. Re-run it after any change to the chips, the encoder or the protocol;
do not hand-edit the tables.

```bash
PYTHONPATH=src python scripts/baselines.py --cache-dir cache --out out/baselines.json
```

The colour-only rows need no encoder cache and are reproduced on CPU in about
two minutes by `hackathon/baseline.py`.
