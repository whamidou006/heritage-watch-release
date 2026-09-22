# Evaluation protocol

This module is based on the research `herat_eval.py` harness. Its grid function,
replicate seeding, fold assignment, out-of-fold aggregation, and paired decision
rule are preserved. Site-specific constants are parameters, not mutable globals.
The classifier convergence tolerance is tightened for numerical fidelity to the
published solver; see [RESULTS.md](RESULTS.md).

## Why not random CV?

Labels cluster at construction sites. A random split places neighbouring pixels
from the same rooftop in training and testing, inflating macro-F1 by **2.5–6.3
percentage points** in the measured controls. Five-fold StratifiedGroupKFold keeps
every cell of a **6×6** coordinate grid whole on one side of a split.
All preprocessing—including any optional PCA—is fitted **inside the training
fold only**. Cross-validation scores the pooled out-of-fold predictions of each
replicate, not an unweighted average of per-fold F1.

Precisely, the reference grid normalizes **label coordinate extrema**, with
`span = ptp(v) + 1e-12`, and computes
`floor((v-v.min()) / span * blocks + offset) % (blocks+1)`.
It is commonly called an equal-area site grid in the report; strictly it is
equal-width in the supplied coordinates, not an equal-area map projection.
Jitter can create partial edge cells, so there can be more than 36 populated
cell IDs. Replacing this formula with scene bounds or clipped cell IDs changes
the published partitions and is **not** a fidelity-preserving refactor.
Cells control neighbourhood leakage but do not impose a physical buffer at
their boundaries. Nearby points on opposite sides can still overlap.

## Why jitter the grouping?

StratifiedGroupKFold is a greedy assignment. Its random state can affect only
tie-breaking in the historical measurement setting: the 64px dataset produced
**identical partitions for every seed**, with 0/841 predictions changing.
Reseeding therefore gave a misleading apparent standard deviation near zero.

For replicate `r=0..7`, draw two independent origin offsets using
`np.random.default_rng(100+r).random()`. These move every boundary by a random
fraction of a cell. Use splitter `random_state=r`. The grouping itself is now
stochastic, giving genuinely different spatially blocked partitions.
The measured single-system noise floor is approximately **0.016 macro-F1**,
roughly four times the old seed-only estimate.

Folds vary in size because spatial cells have unequal label density. The old
fixed-grid counts ranged from 137 to 245 points per fold. All preprocessing is
re-fitted each time. A new site must have enough populated cells and class
support; the package rejects invalid/insufficient group counts.

## Paired differences and all three bars

For each representation, replicate r uses identical labels, coordinates, groups
and fold seeds. Test `d_r = score_A_r - score_B_r`, **not** the difference of two
independent means. Results produced by `evaluate` include a pairing fingerprint;
`compare` rejects mismatched populations or protocol settings.

A difference is resolved only when:

1. **Unanimous**: all d are strictly positive, or all strictly negative; ties fail.
2. **Above uncertainty**: `abs(mean(d)) > 2 * std(d) / sqrt(R)`.
3. **Material**: `abs(mean(d)) >= 0.02`.

Standard deviations use the original harness's population convention (`ddof=0`).
A unanimous effect above 2·SE but below 0.02 is **“consistent but SMALL”**,
not separated and not a ranking claim. The published top-two +0.0197 is exactly
such a case, despite 8/8 wins.

Why all three bars? Against a null of two equally good representations (60
disjoint random halves of one feature set), 7/8 wins plus 2·SE called **20%**
of non-differences resolved. Unanimity alone reduced this to 6.7%; adding 0.02
reduced it to **1.7%**. Random halves may retain genuine differences, so these
are upper bounds on false positives, not universal significance levels.
They are an empirical calibration on Herat, not a guarantee for every site.

## What this does not establish

The headline blocks **space, not time**; all acquisition pairs can appear in
training and testing. The temporal diagnostic reported alongside it (−0.1755
averaged over ten intervals, 9/10 worse; −0.1110 on a subset) is **exploratory
and not comparable to the headline**: it was measured on the superseded
838-sample manifest, its held-out test rows are drawn by random permutation
*within* the interval rather than spatially blocked, and it is scored over
whichever classes appear in each subset rather than the fixed four. One interval
also dissents, so it is not "resolved" in any case. Read it as evidence that
holding out an interval hurts, repeatedly and substantially — not as a
quantified penalty. Do not mistake 0.7259 for unseen-interval performance.

For competitive evaluation, the original protocol recommends final scoring
under an unpublished jitter seed to reduce partition tuning. This repository
does not contain an evaluation server, hidden seed, or blind holdout.
Twenty-five spatial holdouts measured sd 0.060, about four times noisier than
the replicated-CV mean; at this small sample size a holdout is not automatically
more reliable. Choosing hyperparameters on these same public folds still
introduces selection bias.

Macro-F1 averages the configured classes equally. Tiny classes need scrutiny:
including a single-sample class can change the macro average without improving
any model. Prediction probabilities are not calibrated confidence estimates.
