# The benchmark

A short, fixed way to measure a change-classification model on Herat, so that
two people's numbers mean the same thing. Everything here is already
implemented; you do not have to write an evaluation.

**The task.** You are given an analyst's point and two aerial images of that
point, years apart. Say which of four things happened there:

`New Construction` · `Solar Panel` · `Destruction` · `Temporary Structure`

882 labelled points, 10 acquisition intervals between 2009 and 2025.

---

## Two scores, not one

You report **both**. They answer different questions and they disagree by a
lot, which is the point.

| Score | Question it answers | How it splits |
|---|---|---|
| `evaluate` | Another building, same flight | Test points are whole spatial cells; every acquisition appears on both sides |
| `evaluate_interval` | An acquisition you have never seen | Each acquisition is held out in turn; its test cells are withheld from training too |

A single number hides the gap between them. On the reference system it is
0.779 against 0.574.

Both return macro-F1 over the four classes — each class counts equally
regardless of size, so you cannot win by getting the two big classes right.
Both average over 8 replicates with a different random grid origin each time,
and both report the standard deviation across those replicates.

---

## Baselines to beat

```bash
PYTHONPATH=src python scripts/baselines.py --cache-dir cache --out out/baselines.json
```

| Baseline | reads imagery? | dim | Another building | An unseen acquisition |
|---|---|---:|---:|---:|
| Majority floor | no | – | 0.151 | 0.111 |
| Date only | no | 10 | 0.405 | 0.174 |
| DINOv2 post-event | after-image only | 768 | 0.692 | 0.499 |
| **Satlas-MI + SI diff** (reference) | both images | 3840 | **0.779** | **0.574** |

Read the **date-only** row carefully. Knowing nothing but *when* the pair was
taken scores 0.405 — nearly three times the floor — because different kinds of
change happened in different years, and the classes are not spread evenly
across the ten intervals. A model given imagery can quietly learn that same
shortcut and bank the 0.405 without ever looking at a building.

The second score is what removes it. Held out from its own acquisition,
date-only falls to 0.174, barely above the 0.111 floor. That collapse is the
reason the benchmark has two numbers.

---

## The rule for "better"

A change counts as real only if it clears **all three** bars:

1. **Consistent** — wins on every one of the 8 replicates.
2. **Bigger than the spread** — exceeds 2 × the standard error of the paired
   per-replicate differences.
3. **Bigger than 0.02 macro-F1** — the smallest gap this data can resolve.

Clearing bars 1 and 2 but not 3 means "consistent but too small to matter". It
is a normal, publishable outcome. So is "within noise".

```python
from heritage_watch.protocol import evaluate, evaluate_interval, compare, report

a = evaluate(my_features, y, meta)
b = evaluate(baseline_features, y, meta)
report(a, "mine")
compare(a, b, "mine", "baseline")     # applies all three bars
```

Comparisons must be **paired**: both arms scored on the same rows with the same
grids and the same folds. `compare` assumes this. Scoring two systems on
different subsets and subtracting is the single easiest way to invent a result
that is not there — an earlier version of this work found a 0.0885 effect that
was invisible until the arms were properly paired.

---

## What you may change, and what you may not

**Change anything inside the model.** Swap the encoder, fine-tune it, replace
the logistic head, change the chip size, engineer features, use an ensemble.
`evaluate` takes any scikit-learn-style estimator:

```python
evaluate(X, y, meta, clf_factory=lambda: MyClassifier())
```

**Do not change the evaluation.** Not the folds, the blocking, the replicate
count, the class set, or the metric. If you want a different protocol, report
it *next to* these two numbers, not instead of them.

**Do not tune on the seed.** Development uses a public grid seed so everyone
sees identical partitions. Final scoring uses an unpublished seed, so fitting
to these exact 8 partitions buys you nothing.

---

## Submitting

Report, for both scores: macro-F1, the standard deviation across replicates,
and per-class F1. Include the `compare` verdict against the reference baseline,
and say what you changed. A submission that loses to the baseline but reports
honestly is more useful than one that wins by scoring a different split.

---

## Suggested starting points

Roughly in order of effort:

1. **Fine-tune the encoder.** Everything here uses frozen features and a linear
   head. Nothing has been tuned on this task at all.
2. **Attack Destruction.** It is the weak class: F1 0.642, and 29 % of it is
   read as New Construction — the same two states in the opposite temporal
   order. A model that used the sign of the change rather than its magnitude
   should do better.
3. **Close the acquisition gap.** 0.165 of the drop in the second score is date
   novelty. Colour and seasonal normalisation across dates is untried.
4. **Use the window better.** The sweep that chose 64 px only ever showed one
   window per point. Multiple scales around the same point is untried.

Background, full measurements and the controls behind these numbers are in
[`RESULTS.md`](RESULTS.md) and [`PROTOCOL.md`](PROTOCOL.md). The long research
report and all study scripts live on the `research` branch.
