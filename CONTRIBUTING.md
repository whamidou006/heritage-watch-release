# Contributing

Keep changes small, documented and tested. Open an issue describing the scientific
or engineering problem before changing the published protocol.

```bash
PYTHONPATH=src python -u -m pytest tests/ -q
PYTHONPATH=src python -u -m heritage_watch.cli --help
```

Use project-local scratch outputs under ignored `out/`. Tests create tiny
synthetic GeoTIFFs there and do not need downloaded weights, real data or GPUs.
Do not launch full embedding/comparison sweeps on a shared machine without
resource authorization. Cap BLAS threads and explicitly select a free GPU.

Required invariants:

- Actual scene-pair footprint intersection and full-window rejection at both
  dates; never silently zero-pad or co-register.
- Stable sample IDs and exact cross-cache metadata alignment.
- Scaling, PCA and every other learned transform fitted inside each fold.
- Paired grid jitter and all three decision bars, including minimum effect.
- Model-bundle compatibility checks before scoring.
- No hardcoded local filesystem roots, private data, credentials, weights,
  feature caches or model bundles in commits.

Document whether a number is newly measured or quoted, the dataset population,
solver, split, replicate count and whether a contrast is truly paired.
Do not turn “consistent but SMALL” into a ranking claim. Do not infer accuracy
gains from dataset-version differences.

The code is MIT licensed. Verify separate imagery, annotation and pretrained
model permissions before redistribution. The citation intentionally has no
invented publication identifier.
