# Herat Heritage Watch — Semantic Change Classification

**Bi-temporal 0.25 m aerial imagery, Herat Old City conservation and buffer zones, Afghanistan (2009–2025)**

<sub>Herat Old City is on Afghanistan's UNESCO **Tentative** List (ref. 1927, 2004); it is not an
inscribed World Heritage Site. Protection is local — municipal conservation and buffer zones
administered with the Aga Khan Trust for Culture, with construction inside them requiring Old City
Commission permits. Enforcement is weak, which is why an imagery-based record of what was built,
demolished or roofed over is worth having at all.</sub>
Supporting ablations and the hackathon pack are in **`APPENDIX.md`**.

---

## 1. Problem description

The data comes as pairs of aerial photographs. Each pair shows the same spot in Herat — one photo
taken earlier (`t1`), one taken later (`t2`) — at a place where an analyst has already marked that
something changed. The task is to look at the pair and say which of four changes it was:

- **New Construction** — a building went up
- **Solar Panel** — solar panels appeared, on a roof or on the ground
- **Destruction** — a building came down
- **Temporary Structure** — tents or informal shelter appeared

Scores are macro-F1, which averages the four classes equally so the rare ones still count.
Always guessing the most common class scores 0.151, so that is the bar to clear.

These four categories come from the analyst who drew the original annotations, working through the
imagery one date-interval at a time from 2009 to 2025.

Destruction and New Construction are the hard part. Both involve the same two scenes — an empty
plot and a standing building — and the only thing that separates them is which one came first.
Mistake the order and a demolition looks like a new building. This is the central difficulty of
the task, not a detail.

Three data properties determine the design:

| Property | Consequence |
|---|---|
| 882 usable labels | Too few to train a segmentation network → frozen pretrained encoder + light head |
| Labels are points, not masks | Dense change-detection heads unusable → classify a patch around each point |
| 0.25 m RGB, 8-bit | Excludes Sentinel/Landsat foundation models (resolution and band mismatch) |

A fourth property constrains *evaluation*: labels are **spatially clustered by construction site**,
so random cross-validation places neighbouring pixels in both train and test.

## 2. Dataset

### Step 1 — the 21 scenes

All 21 cover the same 2.20 × 2.19 km tile of Herat on an **identical raster grid** — one distinct
bounds/size tuple across all 21 files, 8923 × 7368 px, EPSG:4326. The tile sits on a degree grid, so
pixels are not square on the ground: ≈0.247 m/px in x, ≈0.297 m/px in y.

Coverage of 2009–2025 is uneven. Twelve years are imaged, **five are missing entirely** (2011, 2014,
2015, 2019, 2020), and some years hold several scenes:

| year | scenes | | year | scenes |
|---|---|---|---|---|
| 2009 | 06-12, 06-23 | | 2018 | 01-04, 02-22 |
| 2010 | 02-25 | | 2021 | 03-23, 08-06, 08-25 |
| 2012 | 01-07 | | 2022 | 05-11 |
| 2013 | 09-21, 09-30, 12-04 | | 2023 | 04-22, 10-11, 10-16 |
| 2016 | 09-27 | | 2024 | 08-10 |
| 2017 | 07-19, 11-14 | | 2025 | 01-26 |

### Step 2 — the annotations

An analyst worked through the archive one interval at a time, producing **12 change layers** and
**1,370 points**. Each point marks a place where something changed between the two ends of its
layer, and carries one category. Layer names give a **month as well as a year** —
`change_between_202205_and_202310` means May 2022 to October 2023, not "2022 to 2023". Eleven of the
twelve layers are specified to the month; one gives years alone.

### Step 3 — from scenes to pairs

A layer names two *moments*; the archive holds scenes on 21 particular days. Each end is matched to
the **scene nearest the stated month, within ±75 days** (the month token read as the 15th). If no
scene falls inside that window the layer cannot be imaged honestly and is dropped rather than
approximated. Three consequences matter later:

- **Two layers die here.** 2011→2012 and 2012→2013 both need imagery near October 2012, and the
  nearest scene is 282 days away. That leaves **10 pairs**. Note what does *not* die: 2010→2011
  names November 2011, and although no scene falls in that month, 2012-01-07 is only 53 days later
  and inside the window — so the pair survives with an after-image stamped **2012-01-07**, not 2011.
- **One of the ten pairs is not month-resolved.** The 2024→2025 layer carries no `YYYYMM` tokens and
  falls back to the year rule. That is **93 of 884 labels, 10.5 %**; the other nine pairs, 791
  labels, are month-resolved.
- Only **12 of the 21 scenes** are ever used. The rest lie in un-annotated years, or within days of a
  scene already chosen as an endpoint.

### Step 4 — the 10 pairs

| pair | before → after | gap (days) | n | New Constr. | Solar | Destr. | Temp. | links |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| 2009→2010 | 2009-06-12 → 2010-02-25 | 258 | 43 | 33 | 0 | 3 | 7 | |
| 2010→2011 | 2010-02-25 → 2012-01-07 | 681 | 68 | 48 | 0 | 15 | 5 | ✓ |
| 2013→2016 | 2013-09-21 → 2016-09-27 | **1102** | 142 | 93 | 0 | 46 | 3 | |
| 2016→2017 | 2016-09-27 → 2017-11-14 | 413 | 48 | 32 | 0 | 15 | 1 | ✓ |
| 2017→2018 | 2017-11-14 → 2018-02-22 | **100** | 35 | 26 | 0 | 8 | 1 | ✓ |
| 2018→2021 | 2018-02-22 → 2021-03-23 | **1125** | 105 | 64 | 25 | 16 | 0 | ✓ |
| 2021→2022 | 2021-03-23 → 2022-05-11 | 414 | 54 | 34 | 16 | 2 | 2 | ✓ |
| 2022→2023 | 2022-05-11 → 2023-10-16 | 523 | 197 | 23 | 136 | 5 | 33 | ✓ |
| 2023→2024 | 2023-10-16 → 2024-08-10 | 299 | 99 | 15 | 68 | 15 | 1 | ✓ |
| 2024→2025 | 2024-08-10 → 2025-01-26 | 169 | 93 | 14 | 65 | 14 | 0 | ✓ |
| **total** | | | **884** | **382** | **310** | **139** | **53** | |

*(Counts are the 884 surviving steps 1–3 and the class filter; the edge-clip in step 5 removes a
further 2 at the 64 px chip size, giving the 882 used throughout. ✓ marks a pair whose before-image
is the previous pair's after-image.)*

Two properties of this table drive several later results:

- **The gap is not constant** — 100 to 1,125 days, an 11× spread, median 414. A change observed over
  three years is a different visual problem from one observed over three months, even when the label
  word is the same. The two widest gaps carry 247 of 884 points (28 %).
- **The class mix moves with time.** There is **not one Solar Panel label before 2018**; all 310 come
  from the last five pairs. Early pairs are dominated by New Construction (33/43, 48/68, 93/142),
  late pairs by Solar Panel (136/197, 68/99, 65/93), and 33 of the 53 Temporary Structure labels sit
  in a single pair. Knowing the date therefore already narrows the answer — this is the source of the
  date shortcut measured in §A4.

### Step 5 — filtering to usable labels

| Filtering step | Remaining |
|---|---:|
| Raw annotations | 1,370 |
| − no scene within ±75 days of the stated month (110) | 1,260 |
| − point outside the imagery tile (350) | 910 |
| − classes `Other` (25) / `Reconstruction` (1) | 884 |
| − patch window clipped at the raster edge (2 at 64 px) | **882** |

*(Shown in the order the code applies it. The two middle steps do not commute in their intermediate
counts — filtering by footprint first would attribute 367 to the tile test and 93 to missing imagery
— but the total, 884, is the same either way. The last step is the only one that depends on the
chip size: a 64 px window clips 2 points, a 128 px window clips 5, which is why results quoted at
the superseded 128 px size carry n = 879 rather than 882.)*

The tile test is a lon/lat bounding-box check against the imagery footprint. It removes 350 points
(27.8 % of those surviving pairing) simply because the annotator worked across the whole city while
the imagery covers one 2.2 km square. For this archive the rectangle is exact rather than an
approximation: all 21 scenes share one footprint, and no scene contains blank or nodata pixels
inside it (measured blank fraction 0.0000 on scenes from 2009, 2018 and 2025).

**Class distribution (final 882).** New Construction 382 (43.3 %), Solar Panel 309 (35.0 %),
Destruction 138 (15.6 %), Temporary Structure 53 (6.0 %). Skewed 7.2 : 1, so all metrics are
macro-averaged; the measured effect of this imbalance is in `APPENDIX.md` §A1.

### Step 5a — why the month and not the year

Reading only the **year** — latest scene of the first year, earliest of the second — mis-selects
whenever an interval does not begin in December and end in January, which is most of them. Both
rules yield 9 usable pairs from the 11 month-named layers, but not the same 9: **6 of the 8 pairs
usable under both receive different endpoint imagery**, the month rule recovers 2010→2011 which the
year rule cannot image at all, and it refuses 2012→2013 which the year rule imaged 282 days off
target. Coverage — the fraction of the annotated interval actually spanned by the imagery — for the
four worst-affected pairs:

| layer | annotated | year rule | month rule |
|---|---|---:|---:|
| 2016→2017 | 2016-09 → 2017-11 | 69.2 % | 97.0 % |
| 2017→2018 | 2017-11 → 2018-02 | **54.4 %** | 100.0 % |
| 2021→2022 | 2021-03 → 2022-05 | 60.8 % | 97.2 % |
| 2022→2023 | 2022-05 → 2023-10 | 66.0 % | 100.0 % |
| *each rule's own 9 pairs, mean* | | *81.9 %* | ***98.8 %*** |

Coverage rewards overshoot, so the worst endpoint offset is reported beside it: median **80 → 8
days**, maximum **282 → 53 days**. All 11 layers are archived in `out/window_coverage.json`. The
concern is that a patch keeps its change label while the pixels either side of it no longer span the
change — a "New Construction" example in which nothing is yet built. That is the mechanism being
guarded against; the affected chips have **not** been inspected to confirm it.

One property corroborates the rule independently of any score. Consecutive pairs should meet, the
second scene of one interval being the first of the next. Among the nine month-named pairs,
junctions that chain exactly rise from **2 of 8 to 7 of 8**; across all ten pairs of the final
manifest the chain holds at **8 of 9**. Nothing in the rule optimises for this.

**What it is worth, measured.** The two manifests were scored against each other paired — identical
labels, spatial blocks, folds and classifier, only the pixels differing — over 8 seeds:

| population | n | year rule | month rule | Δ | 2·SE | seeds won |
|---|---:|---:|---:|---:|---:|---|
| all paired points | 811 | 0.6783 | 0.6850 | +0.0067 | 0.0117 | 5/8 |
| imagery actually moved | 516 | 0.6901 | 0.6984 | +0.0083 | 0.0072 | 6/8 |

Neither difference is unanimous across seeds, and neither reaches the 0.02 minimum detectable effect
of §A2, so **the correction is reported as score-neutral** — adopted because the provenance is
right, not because it improves the numbers.

### Step 6 — what one sample is

| | |
|---|---|
| two image patches | 128 × 128 px cut from the *same* pixel coordinates in the before- and after-scene (≈32 m across, ≈38 m tall on the ground) |
| a label | one of the four change classes |
| a location | longitude / latitude, used only to build spatial folds |
| a date pair | which of the 10 acquisition pairs it came from |

There is no mask and no box — the annotation is a point, and the patch is a fixed window centred
on it.

Because the two scenes share a raster grid (step 1), the two patches are **pixel-aligned by
construction**. Co-registration, normally the dominant error source in high-resolution change
detection, is eliminated rather than modelled.

## 3. Models

An **encoder** is a neural network
that turns an image patch into a fixed list of numbers summarising its content — edges, rooftops,
bare ground, vegetation, texture. That list is the patch's **embedding**, and its length is the
**dimension** (`d`). Here the encoders are pretrained by others on large image collections and are
**frozen**: their weights are never updated, so they act as fixed measuring instruments rather than
as something we train. The **head** is the only part fitted to Herat data — a small classifier that
reads the embeddings of the before and after patches and returns one of the four classes.

The head is a multinomial logistic regression (`C=1.0`, balanced class
weights). At n = 882, d = 768–8064, a linear head is the appropriate capacity — gradient boosting
was slower and weaker. Of seven configurations evaluated, three are strongest:

**A. DINOv2 ViT-B/14** — generic self-supervised features, strong few-shot linear probing. Patches
resized to a fixed 224 px input, ImageNet normalisation, represented as `[e₁, e₂, e₂−e₁]`.
*2304-d, 0.715.*

**B. Satlas-MI + SI diff (selected)** — SatlasPretrain `Aerial_SwinB_MI`, a Swin-v2-B pretrained on
0.5–2 m/px RGB **aerial** imagery (closest available resolution match); chips fed at their native
size, 4-level FPN pooled and concatenated. The multi-image encoder max-pools over time and is
therefore **exactly order-invariant** — swapping `t1`/`t2` changes its output by 0.0. Since
Destruction and New Construction differ *only* by temporal order, MI alone cannot separate them;
this variant repairs that by re-injecting an explicit directional term `s₂ − s₁`. *3840-d,
**0.779**.*

**C. Merged: DINOv2 + Satlas-SI** — both encoders at both dates plus both differences. It scores
marginally *above* B (0.787) at more than twice the dimensionality, but the gap is inside the noise
floor and so is not resolvable either way; the second encoder buys nothing measurable. *8064-d,
0.787.*

## 4. Evaluation data and metrics

**Evaluation data.** The same 882 labelled patches; no separate hold-out is withheld, because at
this sample size cross-validation uses the data more efficiently and yields a variance estimate.

**Protocol — spatially-blocked CV.** Labels cluster tightly at construction sites, so a random
split would put neighbouring pixels of the *same* rooftop in both train and test. To prevent that,
the 2.2 km tile is cut into a 6 × 6 grid and each point is tagged with its cell; the cell ID is
then passed as the group to `StratifiedGroupKFold(5)`, which keeps every cell **whole** on one
side of the split. No test patch can share a cell with a training patch. Random CV is reported
**only** to quantify the leakage it introduces.

The grid is equal-area but the labels are not evenly spread, so the folds are uneven:

| | |
|---|---|
| cells | 36 of 36 populated |
| points per cell | min 6, median 20, **max 142** |
| test-fold size | 137 – 245 points (5 folds) |
| cells per test fold | 7 – 8 |

Every fold contains all four classes (Destruction 23–33, Temporary Structure 9–13 per fold), which
is what `Stratified` buys; but the 1.8× spread in fold size is a direct consequence of grouping by
cell, and it is part of why the noise floor is as wide as it is.

**Does the blocking actually work? Measured.** Chips are extracted around points, and the points
are dense, so the windows overlap: at 128 px, **664 of 879 rows (75.5 %) share pixels with at least
one other row**. Cell-grouping is supposed to keep those pairs on the same side of the split. It
does — after blocking, only **3.7 % of test rows (mean 32 of 879)** still share pixels with their
own training fold, so **blocking removes about 95 % of the chip overlap**. Purging the remainder is
worth nothing measurable:

| arm | macro-F1 | Δ vs keep | 2·SE | wins | rows dropped/fold |
|---|---:|---:|---:|:---:|---:|
| keep (the protocol) | 0.6929 | — | — | — | 0 |
| purge test rows leaking into their own fold | 0.6923 | −0.0006 | 0.0021 | 4/8 | 6.4 |
| purge every overlapping row | 0.6895 | −0.0034 | 0.0037 | 6/8 | 17.8 |

Both purges land *within noise*, and both go very slightly **down** rather than up — which is what
you would expect if the residual overlap were contributing nothing and the purge were merely
removing training data. The headline is not inflated by chip overlap.

**What is *not* blocked: time.** All 10 acquisition pairs appear on both sides of every fold. The
splitting controls *where* a point is, never *when* it was observed, so a model may exploit
interval-specific appearance and class mix (§2). §A4 quantifies the resulting shortcut.

**How much does that flatter the headline? Measured.** We ran a leave-one-interval-out control on
the corrected month-resolved manifest: train on the other intervals, test on a held-out one,
against a same-sized control trained on data that *does* include the test interval. Training-set
size and classifier are held fixed, and — unlike the earlier version of this control — **both arms
are spatially blocked**, so the contrast isolates acquisition familiarity rather than re-measuring
spatial leakage.

| Held-out interval | n | seen | unseen | Δ | wins |
|---|---:|---:|---:|---:|:---:|
| 2009→2010 | 41 | 0.3521 | 0.3556 | **+0.0034** | 2/6 |
| 2010→2011 | 68 | 0.4661 | 0.4192 | −0.0469 | 5/6 |
| 2013→2016 | 141 | 0.4703 | 0.3786 | −0.0917 | 6/6 |
| 2016→2017 | 48 | 0.5149 | 0.4190 | −0.0959 | 5/6 |
| 2017→2018 | 35 | 0.4765 | 0.5022 | **+0.0257** | 3/6 |
| 2018→2021 | 104 | 0.4297 | 0.3497 | −0.0800 | 6/6 |
| 2021→2022 | 54 | 0.5192 | 0.4315 | −0.0876 | 4/6 |
| 2022→2023 | 197 | 0.7077 | 0.5522 | −0.1556 | 6/6 |
| 2023→2024 | 98 | 0.6046 | 0.4713 | −0.1332 | 6/6 |
| 2024→2025 | 93 | 0.7000 | 0.6427 | −0.0573 | 5/6 |
| **mean** | | | | **−0.0719** | 8/10 worse |

Mean Δ **−0.0719**, sd 0.0559, 2·SE 0.0353. It clears two standard errors and clears the 0.02
minimum effect size, but **8 of 10 is not unanimous**, so under the rule of §4 it is **not
resolved**. The two dissenting intervals are the two smallest tested (35 and 41 labels), which is
where the estimate is noisiest — but the rule is the rule, and we do not get to drop the
inconvenient rows.

**Two things worth stating plainly.**

*The legacy version of this control was roughly twice as large: −0.1755 at 9/10.* The difference is
that neither of its arms was spatially blocked. So **about half of what looked like a temporal
generalisation penalty was ordinary spatial leakage** being counted twice. This is the clearest
example in the report of why a control needs the same blocking as the thing it is controlling.

*No interval is clean.* Under the month rule the scenes **chain** — the after-image of one pair is
the before-image of the next — so every "unseen" arm has already seen one of its two endpoint
images through a neighbouring pair. A genuinely date-disjoint hold-out cannot be built from this
archive at all. The −0.0719 is therefore a **lower bound** on the cost of a new acquisition, and
the only way to measure the real thing is a second site.

The practical reading: the headline in §5 is blocked in space but only partially in time, and
generalising to a genuinely new acquisition should be expected to cost **at least** 0.07 macro-F1.

**A second estimator of the same quantity, and why it is larger.** The released benchmark
(`evaluate_interval`) measures this differently: it holds out each interval in turn, takes whole
spatial cells as test, withholds those cells from training, and then **pools predictions across all
ten intervals** before computing one macro-F1 over the fixed four classes. Its matched control is
the same function run with the interval labels shuffled — identical capped training pool, identical
cell blocking, no date structure. Measured on the selected configuration at 64 px:

| Arm | macro-F1 |
|---|---:|
| Headline (§5, spatially blocked only) | 0.7792 |
| Interval machinery, interval labels shuffled | 0.7392 |
| Interval machinery, real intervals | 0.5745 |

So the smaller, doubly-blocked training pool costs 0.040 and **date novelty costs 0.165**.

That is more than twice the −0.0719 above, and the two are not in conflict — they are different
estimators. The table above averages a per-interval macro-F1 taken over *whichever classes appear
in that interval*; an interval holding three classes is scored on three. Pooling first and scoring
over the fixed four removes that compression and is the stricter, more honest measurement, which is
why the released benchmark uses it. Neither number is wrong, but **0.165 is the one to quote**, and
the −0.0719 should be read as what the older per-interval estimator reports on the same data.

Both remain lower bounds for the same reason: the scenes chain.

**Metric — macro-F1** over four classes, so the 53-sample Temporary Structure class carries the
same weight as the 380-sample New Construction class. Majority-class floor **0.151**.

**Controls.** All preprocessing is fitted **within fold**; the GPU solver reproduces the reference
sklearn implementation to ≤0.009 macro-F1.

> **Protocol correction (found during review).** `StratifiedGroupKFold` is a *greedy* assignment
> whose `random_state` only breaks ties: on the 64 px set it returned an **identical partition for
> every seed** (0/841 predictions changed), and at 128 px only 27/838 changed. *(Both measured on
> the 838-sample set in use at the time; the finding is a property of the splitter, not of the
> sample count.)* Reseeding the splitter therefore measured almost nothing, and the ±0.004 it
> produced was not a noise estimate.
> We instead **jitter the grid origin** by a random fraction of a block per replicate, which moves
> every block boundary and yields genuinely different — still spatially blocked — partitions.
> **The true single-configuration noise floor is ±0.016, about 4× the original figure.**
> All comparisons below are **paired**: replicate *r* uses the same grid for every configuration,
> so a difference is tested per replicate rather than between two independently noisy means.

**Decision rule, calibrated against a stress test.** Pairing alone is not enough. Because paired
deltas are strongly correlated, a "consistent" win can be arbitrarily small. To size that risk we
ran 60 trials in which the two "systems" were disjoint random halves of one feature set: a rule of
*wins ≥ 7/8 and |Δ| > 2·SE* declared **20 % of those pairs "resolved"**. A win must therefore
clear three bars — unanimous across all replicates, |Δ| > 2·SE, and **|Δ| ≥ 0.02** (a minimum
effect size):

| decision rule | fraction of stress-test pairs called "resolved" |
|---|---:|
| wins ≥ 7/8, \|Δ\| > 2·SE | 20.0 % |
| wins = 8/8, \|Δ\| > 2·SE | 6.7 % |
| **wins = 8/8, \|Δ\| > 2·SE, \|Δ\| ≥ 0.02** | **1.7 % (1 of 60)** |

Every claim in §5 is reported under the calibrated rule. **Read these as acceptance fractions in
one stress test on one dataset, not as a validated false-positive rate.** Random halves of a
feature set are not guaranteed to be equally predictive, so this is not a true null; 1 of 60 is too
thin to support a portable rate; and the trials themselves are not archived in `out/`. What the
table does establish is the *ranking* of the three rules, which is the reason the third was adopted.
Extrapolating it to a per-hackathon count of expected false wins would be exactly the kind of
projection this report refuses elsewhere, so we do not.

**Guard against partition tuning — and its limits.** The grid-jitter seed is fixed and public so
that all teams see identical partitions during development; final scoring runs under an
**unpublished seed**. Re-scoring the baseline under a secret seed moved it 0.5949 → 0.5989, well
inside the floor, which shows the baseline is not partition-sensitive.

Be precise about what this buys. It removes **partition-specific** tuning only. Because every label
is released, model selection, feature selection and any supervised training still see the same
observations used for scoring, so the final figure retains ordinary model-selection optimism; a
changed seed does not undo that. There is also a concrete leak the harness cannot catch: `evaluate()`
refits only the classifier handed to it, so a backbone fine-tuned on all 882 labels *before*
embedding would contaminate every fold. The participant pack states this as a rule, but an unbiased
final claim would need nested evaluation or genuinely unseen labels — ideally a second site.

**Why not a blind hold-out?** We measured the alternative. Twenty-five spatially disjoint 20 %
hold-outs gave macro-F1 **sd 0.060** with a 0.24 spread, and dropped Temporary Structure to a
median of **11** test samples. That single-hold-out sd is ~4× the ±0.016 single-run CV noise floor
(and ~10× the SE of the 8-replicate CV mean, ≈0.006), and ranking teams that way would largely rank
which blocks they drew. With a single site, replicated blocked CV under a secret seed is the better
instrument — though, per the paragraph above, neither option removes model-selection bias.

## 5. Results

All figures below are measured on the **month-resolved dataset** (n = 882, §Step 5a) at the
**64 px** chip size selected in finding 8. Means over 8 jittered grid replicates. "Paired Δ" is the
per-replicate deficit against the best mean; "wins" counts replicates in which the better model is
ahead. Verdicts apply the three-bar rule of §4 — unanimous, |Δ| > 2·SE, **and** |Δ| ≥ 0.02.

| Representation | dim | **Macro-F1** | sd | Paired Δ vs C | wins | verdict |
|---|---:|---:|---:|---:|---:|---|
| Satlas-MI (order-invariant) | 1920 | 0.6881 | 0.0092 | −0.0993 | 8/8 | resolved |
| DINOv2, post-event only | 768 | 0.6921 | 0.0113 | −0.0953 | 8/8 | resolved |
| DINOv2, both + diff | 2304 | 0.7147 | 0.0063 | −0.0726 | 8/8 | resolved |
| Satlas-SI, post-event only | 1920 | 0.7204 | 0.0050 | −0.0669 | 8/8 | resolved |
| Satlas-SI, both + diff | 5760 | 0.7559 | 0.0071 | −0.0314 | 8/8 | resolved |
| **B · Satlas-MI + SI diff (selected)** | 3840 | **0.7788** | 0.0039 | −0.0085 | 7/8 | **not resolved** |
| C · Merged DINOv2 + Satlas | 8064 | **0.7873** | 0.0082 | — | — | — |

**On the choice between the top two.** The merged encoder has the higher mean (0.7873 vs 0.7788)
and wins 7 of 8 paired replicates, but the gap is **+0.0085** — less than half the 0.02 minimum
effect size, not unanimous, and smaller than its own replicate-to-replicate spread (sd 0.0093).
The decision rule of §4 returns *within noise*. The two are therefore **not separated**, and we do
not claim the merged encoder is better. We select Satlas-MI + SI diff on **parsimony**: equal
measured performance at **less than half the dimensionality** (3840 vs 8064) and drawing on a single
pretrained *family* (SatlasPretrain Aerial) rather than two unrelated ones. Note the precise claim:
the selected configuration still loads **two Satlas checkpoints**, `Aerial_SwinB_SI` and
`Aerial_SwinB_MI`, so this is one model family, not one network — we have not measured inference
cost, only dimension and dependency count.

*(At the superseded 128 px size the same two configurations sat the other way round — B ahead of C
by 0.0197, also not resolved. The ordering of the two is not stable across chip size; their
inseparability is. That is the substantive point, and it is why neither is claimed as better.)*

Pairing is what makes the rest of the table readable: the gaps between adjacent configurations
(0.02–0.04) are of the same order as the ±0.016 single-run noise, so unpaired means could not have
separated them.

Leakage from random CV, measured under the original protocol, remains **+2.5 to +6.3 pp**.

**Selected model (B), per class.** Precision / recall / F1 averaged over folds, with the
row-normalised confusion matrix:

| Class | P | R | **F1** | n | share |
|---|---:|---:|---:|---:|---:|
| Solar Panel | 0.942 | 0.890 | **0.915** | 309 | 35.0 % |
| New Construction | 0.813 | 0.887 | **0.849** | 382 | 43.3 % |
| Temporary Structure | 0.721 | 0.585 | **0.646** | 53 | 6.0 % |
| Destruction | 0.662 | 0.623 | **0.642** | 138 | 15.6 % |
| **Macro** | 0.784 | 0.746 | **0.763** | 882 | |

*(Per-class figures are out-of-fold over the fixed 6 × 6 grid, averaged across three splitter seeds;
the confusion matrix is seed 0. The headline 0.7788 uses the 8 jittered grids instead, so the 0.763
here is a **different partition protocol**, not merely a different averaging of the same folds. The
three-seed spread prints as 0.000 and should not be read as a precision claim — three seeds is too
few to estimate it; the eight-grid sd of 0.0039 is the usable one.)*

| true ↓ / predicted → | Destruction | New Constr. | Solar Panel | Temporary |
|---|---:|---:|---:|---:|
| **Destruction** | **62.3 %** | 29.0 % | 6.5 % | 2.2 % |
| **New Construction** | 7.9 % | **88.7 %** | 1.6 % | 1.8 % |
| **Solar Panel** | 1.3 % | 9.1 % | **89.0 %** | 0.6 % |
| **Temporary Structure** | 18.9 % | 18.9 % | 3.8 % | **58.5 %** |

**Destruction is now read correctly more than twice as often as it is misread as New
Construction** — 62.3 % against 29.0 %, a margin of 33.3 points on 138 samples. An earlier version
of this report made the *reversed* ordering (43.7 % correct vs 46.8 % misread) its headline failure.

**What fixed it was chip size, not the imagery correction and not the choice of encoder.** The
three candidate explanations can be separated because all four cells of the 2 × 2 have been
measured on the same corrected manifest:

| | 128 px | 64 px |
|---|---|---|
| **C · merged** | 40.6 % correct / 51.4 % misread — *reversed* | 67.4 % / 27.5 % |
| **B · MI + SI diff** | 50.0 % / 41.3 % | 62.3 % / 29.0 % |

Holding the encoder fixed and changing only the chip size is worth **+26.8 points** of correct
Destruction recall on the merged encoder; holding the chip size fixed at 128 px and changing only
the encoder is worth **+9.4**. Chip size is roughly three times the lever that representation is,
and at 64 px *both* encoders get the ordering right. The imagery correction explains none of it:
its only clean estimate remains the paired experiment in §Step 5a, **+0.0067, inside the noise
floor.**

The confusion remains the largest in the matrix, and the direction of the residual error is
unchanged — **both minority classes fail toward the majority class they physically resemble**,
though Temporary Structure's leak into New Construction has fallen from 30.2 % to 18.9 %.

> **Caveat on Solar Panel (§A4).** Its 0.915 is partly *temporal*, not purely visual: there are
> **zero** solar-panel labels before 2018, so the acquisition pair alone is partially diagnostic.
> On this same four-class population and chip size, an imagery-free control using only the
> one-hot date reaches macro-F1 **0.4260** against a **0.1511** majority floor — a real and
> substantial shortcut.
>
> **But the model does not appear to be using it.** Supplying the date *alongside* the imagery
> moves the score by **+0.0010** (0.7886 → 0.7896), a twentieth of the noise floor. That is
> consistent with the imagery already encoding whatever the date would contribute, and it bounds
> the risk — but it does not prove the Solar Panel F1 is shortcut-free, because a model can exploit
> a correlate of the date without being handed the date.

**Findings.**

1. **Spatial blocking is essential, and it is sufficient.** Leakage inflates every configuration by
   2.5–6.3 pp, so random-CV scores on this dataset are not trustworthy. But the blocking that fixes
   it is doing its job: 75.5 % of chip windows overlap another row's, and cell-grouping removes
   about **95 %** of that overlap, leaving 3.7 % of test rows touching their own training fold.
   Purging even those is worth **−0.0006** (4/8) and purging every overlapping row **−0.0034**
   (6/8) — both within noise, and both slightly negative. The headline is not inflated by chip
   overlap (§4).
2. **Pretraining domain does *not* measurably outweigh temporal input — claim withdrawn.** An
   earlier version of this report stated that Satlas-SI with a single date beats DINOv2 with both
   dates plus their difference "in 8/8 paired replicates". That was an error: the 8/8 figure
   belonged to each configuration's comparison against the *selected* model, not to this
   head-to-head. Measured directly, the pair is **+0.0189, 6/8** on the old manifest and
   **+0.0042, 4/8** on the corrected one — inside the noise floor in both cases. Satlas-SI does
   lead on the mean, but this dataset cannot resolve it. *(What does survive: every Satlas variant
   outranks its DINOv2 counterpart, and the selected model uses no DINOv2 features at all — see
   finding 4.)*
3. **Order-invariance is harmful exactly where it matters.** Restoring the directional term
   `s₂ − s₁` to the order-invariant Satlas-MI encoder is worth **+0.0908 macro-F1, 8/8 paired,
   resolved** (0.6881 → 0.7788) — the largest architectural effect measured here. Since
   Destruction and New Construction differ *only* by temporal order, a symmetric encoder is
   structurally unable to separate them, and the fix is to re-inject direction explicitly.
4. **A second encoder adds nothing that can be measured.** The merged DINOv2 + Satlas stack
   (8064-d, 0.7873) is ahead of Satlas-MI + SI diff alone (3840-d, 0.7788) by **+0.0085, 7/8** —
   below the 0.02 bar and below its own replicate spread, so *within noise*. On the previous
   manifest the merged stack appeared to lead by a wide margin, and "the encoders are
   complementary" was reported as a finding; at the superseded 128 px size the ordering reversed
   again, to 0.0197 *against* the merged model. Across all three measurements the one stable
   result is that the two cannot be separated. **One well-matched aerial encoder is enough.**
5. **Difference imagery alone is the weakest signal**, below the post-event image alone (§A7).
   Four-class macro-F1 for f2−f1 is 0.566 on DINOv2 and 0.637 on Satlas, against 0.645 and 0.669
   for the post-event image: paired, post-only wins by +0.0788 (8/8, resolved) and +0.0316 (7/8,
   suggestive). Much of this taxonomy is readable from appearance rather than from change.
6. **Class imbalance redistributes performance but does not cap it** (§A1). Reweighting the loss is
   worth +0.011 macro-F1 — below the 0.02 floor, so unresolved — and what it actually does is move
   score between classes: Temporary Structure gains +0.047 F1 and +0.082 recall, paid for by New
   Construction. Changing the support ratio is starker still: downsampling New Construction from
   352 to 52 swings its own F1 from 0.785 to 0.395 and Destruction's from 0.497 to 0.749, while
   macro-F1 never leaves 0.683–0.710. The binding constraint is directional confusability, not
   label counts.
7. **Cleaning affects the metric far more than the model** (§A2). The headline contrast — cleaned
   four-class 0.6961 against raw six-class 0.5767, a gap of **+0.1194** — is mostly a metric
   artefact: scoring the *same* raw six-class predictions over only the four target classes gives
   0.6790, so **+0.1023** of the gap is the choice of which classes the metric averages over. The
   genuine modelling effects are small: **−0.0172** for keeping the discarded classes in training
   (scored on the target classes), and **−0.0060** for deduplicating to one label per 32 m. That
   last number is the useful one — it halves the evaluated population (879 → 442) and removes
   every near-neighbour, yet costs less than a third of the noise floor.
8. **Chip size matters more than anything else we varied** (§A3) — more than the encoder, and more
   than the imagery correction. Measured strictly paired — the 871 locations common to all four
   caches, one shared spatial grid, identical folds, 8 replicates, so the *only* thing that differs
   is the pixels:

   | chip | ground | macro-F1 | vs 64 px | 2·SE | wins | verdict |
   |---|---|---:|---:|---:|:---:|---|
   | 32 px | 8 m | 0.7554 ± 0.0058 | +0.0272 | 0.0042 | 8/8 | resolved |
   | **64 px** | **16 m** | **0.7827 ± 0.0037** | — | — | — | best |
   | 128 px *(used elsewhere)* | 32 m | 0.6942 ± 0.0041 | +0.0885 | 0.0003 | 8/8 | resolved |
   | 256 px | 64 m | 0.6447 ± 0.0028 | +0.1380 | 0.0028 | 8/8 | resolved |

   64 px beats 128 px by **+0.0885**, about 4.4× the 0.02 floor, with a paired standard deviation
   of 0.0004 — the gap is near-identical on every partition. This supersedes an earlier unpaired
   sweep that put the same contrast at +0.019, 7/8, *not resolved*; that sweep compared arms with
   different populations (832–842 points) and re-normalised the jittered grid to each population's
   own extent, which is what hid the effect.

   The mechanism is **ground extent, not input resolution**. The DINOv2 path interpolates every
   chip to a fixed 224 px model input regardless of how much ground it covers, so its arms differ
   only in field of view — and it improves just as much (post-event only 0.6439 → 0.6921; both +
   diff 0.6656 → 0.7147 going from 128 px to 64 px). A 16 m window is roughly one building and
   fills the frame; at 32 m the target occupies a quarter of the frame and the neighbours fill the
   rest. The label is a point, not a footprint, so the wider the window the more of it is somebody
   else's building.
9. **A modest date shortcut exists, and the model does not use it** (§A4). Class labels correlate
   with the acquisition pair — Solar Panel is absent before 2018, and 20 of 25 `Other` labels come
   from a single annotation layer. On the same four-class set as the headline (n = 882, 64 px), the
   acquisition pair *alone*, with no imagery at all, reaches macro-F1 **0.4260** against a 0.1511
   majority floor, so the shortcut is real and substantial. But adding the date as an extra input
   alongside the imagery moves the score by **+0.0010** — 0.7886 to 0.7896, a twentieth of the
   noise floor. The shortcut is available; the classifier is not taking it.

10. **Bi-temporal input does not help the damage class** (§A7). Against `microsoft/haste`'s
   single-date design, the post-event image *alone* separates Destruction from New Construction
   just as well as the full bi-temporal stack: 0.555 vs 0.535 F1 on Satlas, 0.421 vs 0.430 on
   DINOv2. All four contrasts (two encoders × F1 and AP) fall **within noise** — the largest is
   0.019 against a 2·SE of 0.015, and the signs disagree — so we cannot separate them. Rubble is
   visible without the date order.
   What the date order *does* buy is measurable but lands elsewhere. On the four-class task
   bi-temporal beats post-only by +0.0221 macro-F1 on Satlas (8/8, resolved) and +0.0230 on
   DINOv2 (7/8, suggestive) — yet on Destruction itself the gain is −0.004 and +0.019, i.e.
   nothing. The macro gain comes from Temporary Structure (+0.075 on Satlas) and New
   Construction (+0.043 on DINOv2).
   The pre-event image alone is *much* worse — bi-temporal beats it by +0.115/+0.112 macro-F1,
   8/8 resolved on both encoders — as is the difference image f2−f1 (+0.054/+0.102, both
   resolved). So the signal is post-event appearance, not change. This is consistent with
   finding 3: Satlas-MI is *symmetric* in the pair and cannot encode direction, whereas
   selecting the post-event date is itself directional.

**Which numbers sit on which dataset.** Everything in §1–§5 and findings 2–10 is now measured on
the **corrected, month-resolved** manifest. The ablations in §6 and the appendix were run at the
superseded 128 px chip size and carry n = 879 rather than 882; they are reported as *relative*
effects, which is how they should be read regardless, and finding 8 is the reason to re-run any of
them before quoting an absolute number from them.

| Population | n | classes | What uses it |
|---|---:|---|---|
| Corrected, month-resolved, 64 px | 882 | 4 | §5 tables, per-class table, findings 3, 4, 8, 9 |
| Corrected, month-resolved, 128 px | 879 | 4 | findings 5, 6, 7, 10; the §4 interval control; `APPENDIX.md` |
| Corrected, unfiltered | 905 | 6 | the six-class arm of the cleaning study (§A2, finding 7) |

**Four claims changed when they were re-measured**, which is the main reason the scope of each
number is stated at all: findings 2 and 4 changed on the imagery correction, finding 8 reversed
outright once the arms were properly paired, and finding 10 stopped reproducing altogether. Two of
those four had previously been reported as *resolved* under the same decision rule that now rejects
them — the rule catches under-powered claims, not wrongly-controlled ones.

**Limitations.** Destruction and New Construction remain the dominant confusion: 29.0 % of
Destruction is read as New Construction, against 62.3 % read correctly — still the largest error in
the matrix, on only 138 samples. The split of F1 between the two (0.642 / 0.849) reflects their
2.8 : 1 support ratio rather than intrinsic difficulty (§A1). **Destruction is the weakest class at
0.642**, with Temporary Structure just above it at 0.646 on only 53 samples, leaking 18.9 % into
New Construction. Part of Solar Panel's 0.915 is a **date shortcut** rather than visual skill
(§A4). No pretrained head predicts "Solar Panel" — pretraining supplies features, not classes.
Encoders are frozen and untuned, so these figures are lower bounds. Finally, the ±0.016 noise floor
means any future claim below ~0.03 must be tested **paired**, not by comparing means.

**Next steps.** Items 2 and 3 of the previous version are done and folded into §4 and finding 9.
What remains, in priority order:

1. **Re-run the `APPENDIX.md` ablations at 64 px.** Finding 8 makes chip size the largest single
   lever measured here, and every appendix ablation predates it. Their *directions* are unlikely to
   change — most are large — but none of their absolute numbers should be quoted.
2. **Measure on a second site.** The interval control in §4 establishes that a new acquisition costs
   at least 0.165 macro-F1, but it cannot do better than a lower bound, because the scenes chain and
   no date-disjoint hold-out exists in this archive. Every remaining question about generalisation
   is blocked on data, not on method.
3. **Ablate the Solar Panel date shortcut directly** — how much of its 0.915 survives when the
   acquisition pair is made uninformative. Adding the date as an input is a null (+0.0010), which
   bounds the risk but does not settle it, since the model can use a correlate of the date without
   being handed the date.

On method, the ceiling is likely directional discrimination rather than data volume: finding 3
shows the signal is present but under-exploited, and §A1 shows more labels would mainly move F1
between the confusable pair — though note §A1 removes New Construction points from training *and*
evaluation, so it does not estimate the value of acquiring more Destruction labels against a fixed
benchmark, and a fixed-test-set learning curve is needed to settle that. §A7 suggests the target is
a **better encoder of the post-event state** (fine-tuning the aerial backbone on rubble-vs-structure)
rather than more temporal channels, since the post-event image alone already outperforms the
bi-temporal stack on this pair. That is a promising hypothesis from frozen-feature experiments, not
a demonstrated necessity: it does not exclude gains from temporal fine-tuning, a different fusion
operator, or better early imagery.
Supporting routes remain SAM-based **AnyChange** proposals to decouple localisation from
classification and **Changen2 / ChangeStar2** synthetic ordered pairs.

## 6. Related work

Two 2026 studies from Microsoft AI for Good, Iconem and Planet Labs address heritage monitoring in
Afghanistan on the same national corpus of **1,943 archaeological sites** (898 looted, 1,045
preserved) using PlanetScope monthly mosaics at **4.7 m/px**. Neither covers Herat, and neither is
a usable baseline here — our imagery is ~19× finer (0.25 m/px aerial) and our label space is
four-way semantic typing rather than binary looting or event timing. They are, however, the closest
published work, and three of their results bear directly on ours.

| | Tadesse et al., *Satellite-Based Detection of Looted Archaeological Sites* (arXiv 2602.19608) | Tadesse et al., *WATCH: Wide-Area Archaeological Site Tracking* (arXiv 2605.08160) |
|---|---|---|
| task | binary: looted vs preserved | month-level localisation of a change event |
| data | PlanetScope 4.7 m/px, 2016–2023 | PlanetScope 4.7 m/px, 2017–2024 |
| method | CNN on RGB patches vs classical ML on handcrafted features and foundation-model embeddings | three scorers — TED (training-free), SSCD (self-supervised), WS (weakly supervised) — over six foundation models |
| headline | ImageNet CNN + spatial masking **F1 0.926**; best foundation-model pipeline 0.710 | TED+SatMAE **55.0 %** exact-month (best at m=0) and 85.0 % within ±3 months; **92.5 %** within ±3 months is reached by TED+CLIP / GeoRSCLIP / Satlas, not by SatMAE |
| protocol | site-level stratified split, 5-fold, PCA fit on training folds only | tolerance-based recall, m = 0…6 months |

**1. Imprecise annotation dates are a known, general problem — not a local defect.** WATCH states
plainly that heritage ground truth is *"sparse and often temporally imprecise"*, and builds
symmetric tolerance margins of up to six months to absorb *"the imprecision inherent in recorded
looting dates"*; its recall rises steeply from m = 0 to m = 3. This is the same failure mode as the
provenance bug in §Step 5a, and the ±75-day tolerance we adopted is a remedy for the same *kind* of
problem at a finer scale. The two operations are not equivalent, though — WATCH tolerates error in a
*predicted event date*, whereas our tolerance governs *which input image is selected* — so we read
this as motivation for taking date uncertainty seriously, not as external validation of ±75 days.

**2. Their encoder ranking is consistent with our selected model.** Our selected configuration uses
**Satlas features only** — adding DINOv2 buys nothing measurable (finding 4), and every Satlas
variant outranks its DINOv2 counterpart on the mean. WATCH ranks **Satlas-Pretrain best or
joint-best at relaxed tolerance under all three of its scorers** (92.5 / 92.5 / 90.0), with DINOv3
leading only at exact-month under one scorer, and attributes the gap to geospatially-aligned
pretraining outperforming generic visual pretraining. That is the same ordering we observe, on
different imagery, resolution and task.

How strongly can we state our side of it? **Matched like-for-like, the family gap is resolved**:
Satlas-SI both+diff over DINOv2 both+diff is **+0.0262, 8/8 replicates, 2·SE = 0.0083**, clearing
all three bars. What was withdrawn in finding 2 was the narrower and stranger claim that Satlas
*single-date* beats DINOv2 *bi-temporal* — an unmatched comparison, which measures **+0.0042, 4/8**
and is plainly unresolved. So: the family ranking agrees with WATCH and is measured on matched
inputs; the cross-representation shortcut claim is not. The looting paper supplies the corrective in
the other direction — there an **ImageNet-pretrained CNN (0.926) beat every foundation-model
pipeline (best 0.710)** — so "foundation model" is not by itself a guarantee of anything.

**3. Precedent for reporting cross-site transfer qualitatively.** WATCH applies its framework to
Syria, Turkey, Pakistan and Egypt but explicitly frames this as *"a qualitative operational
demonstration rather than a quantitative evaluation"*, because no ground-truth event months exist
outside Afghanistan. Any cross-site section for Herat is in the same position and should carry the
same caveat.

**Where this report is stricter.** Neither study reports a measured noise floor, paired replicates,
or a minimum detectable effect, and the looting paper's site-level split is stratified but not
spatially blocked. The jittered-grid paired protocol with a declared ±0.016 floor and a
null-calibrated decision rule (§4) is a step up on that specific axis — though on tasks different
enough that this is a methodological observation, not a performance claim.

---

*Reproducibility:* `build_manifest.py --pairing month` → `embed_chips.py` / `embed_satlas.py` →
`compare_jitter.py` (main table), `perclass_month.py` (per-class + confusion),
`date_shortcut_month.py` (date control), `month_vs_year.py` (paired imagery-correction test),
`loio_control.py` (time-blocking control), `noise_floor.py` (protocol check),
`haste_control.py` (HASTE single-date control, §A7),
`export_student_manifest.py` (ships `hackathon/data/manifest.json`).
Numeric output for every table in `out/*.json`. Ablations,
protocol rationale and the participant pack: **`APPENDIX.md`**.
