# Research branch — how the protocol and the baselines were arrived at

`main` carries the student-facing benchmark: the protocol, the baselines, and the harness.
This branch carries the working material behind it — the full report, the ablations, the
controls that failed, and the raw result JSONs.

It is kept separate on purpose. Nothing here is needed to run the benchmark, and some of it
records claims that **did not survive re-measurement**. That history is the useful part.

## Contents

| Path | What it is |
|---|---|
| `REPORT.md` | The long report: dataset construction, protocol design, results, ten findings |
| `APPENDIX.md` | Supporting ablations (§A1–A7). **Still measured at the superseded 128 px chip size** |
| `scripts/` | One script per experiment; each writes a JSON into `out/` |
| `out/` | Every measurement quoted in the report, as produced |

## Reading the result files

Studies exist in two generations, selected by the `HW_SUFFIX` environment variable:

```
HW_SUFFIX=""        legacy, year-resolved pairs   (kept only so old numbers reproduce)
HW_SUFFIX="_month"  current, month-resolved pairs (everything reported)
```

`scripts/cachesel.py` maps both the input cache and the output filename, so the two
generations can never overwrite each other. A file named `*_month.json` is current; one
without the suffix is legacy and should not be quoted.

Re-run the current generation end to end with:

```bash
cd research/scripts && bash rerun_month.sh
```

## The decision rule

`scripts/decision.py` is the single source of truth. A difference counts as **resolved**
only if it clears all three bars:

1. **unanimous** across paired replicates,
2. **|Δ| > 2·SE**,
3. **|Δ| ≥ 0.02** macro-F1 — the minimum effect size, set from the measured noise floor.

Run `python decision.py` to execute its self-check. The rule is deliberately hard to pass;
several results that were once reported as findings do not pass it.

## Four claims that changed when they were re-measured

Worth reading before trusting anything in a first draft, including ours.

| Claim | First reported | After re-measurement |
|---|---|---|
| Two encoders are complementary | merged stack leads | inseparable, across three separate measurements |
| Pretraining domain beats temporal input | resolved | inside the noise floor — withdrawn |
| 64 px vs 128 px chips | not resolved (+0.019, 7/8) | **+0.0885, 8/8, resolved** — the arms had not been paired |
| Post-event imagery beats bi-temporal | 8/8 both encoders, resolved | all four contrasts **within noise** — does not reproduce |

Two of those four had been reported as *resolved* under the same rule that now rejects them.
The rule catches under-powered claims; it does not catch wrongly-controlled ones. Only a
control with the same blocking as the thing it controls does that — see the leave-one-interval-out
result in `REPORT.md` §4, which halved once both of its arms were spatially blocked.
