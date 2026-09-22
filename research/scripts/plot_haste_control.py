#!/usr/bin/env python3
"""Figure for the HASTE single-date control (work/out/haste_control.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "out" / "haste_control.json"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORDER = ["pre_only", "diff", "haste_post", "bitemporal"]
NICE = {"pre_only": "pre-event only", "diff": "difference",
        "haste_post": "post-event only\n(HASTE design)", "bitemporal": "bi-temporal\n(ours)"}
COL = {"pre_only": "#b0b0b0", "diff": "#8fb8de",
       "haste_post": "#d1495b", "bitemporal": "#2e5eaa"}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

# Panel 1: Task A, the order-confusable pair
ax = axes[0]
w, xs = 0.38, np.arange(len(ORDER))
for i, enc in enumerate(["dinov2", "satlas"]):
    t = R["encoders"][enc]["taskA_destruction_vs_newconstruction"]["reps"]
    v = [t[k]["f1"] for k in ORDER]
    e = [t[k]["f1_sd"] for k in ORDER]
    ax.bar(xs + (i - 0.5) * w, v, w, yerr=e, capsize=3,
           color=[COL[k] for k in ORDER], alpha=1.0 if i == 0 else 0.55,
           edgecolor="k", linewidth=0.6, label="DINOv2" if i == 0 else "Satlas")
ax.set_xticks(xs); ax.set_xticklabels([NICE[k] for k in ORDER], fontsize=8)
ax.set_ylabel("Destruction F1")
ax.set_title("A. Destruction vs New Construction\n(n=478, the order-confusable pair)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", alpha=0.3)

# Panel 2: Task B macro
ax = axes[1]
for i, enc in enumerate(["dinov2", "satlas"]):
    t = R["encoders"][enc]["taskB_4class"]
    v = [t[k]["macro_f1"] for k in ORDER]
    ax.bar(xs + (i - 0.5) * w, v, w, color=[COL[k] for k in ORDER],
           alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([NICE[k] for k in ORDER], fontsize=8)
ax.set_ylabel("macro-F1 (4 classes)")
ax.set_title("B. Full 4-class task\nbi-temporal wins overall", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Panel 3: Task B, Destruction only
ax = axes[2]
for i, enc in enumerate(["dinov2", "satlas"]):
    t = R["encoders"][enc]["taskB_4class"]
    v = [t[k]["per_class"]["Destruction"] for k in ORDER]
    ax.bar(xs + (i - 0.5) * w, v, w, color=[COL[k] for k in ORDER],
           alpha=1.0 if i == 0 else 0.55, edgecolor="k", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([NICE[k] for k in ORDER], fontsize=8)
ax.set_ylabel("Destruction F1 (4-class)")
ax.set_title("C. Damage class only\nbi-temporal brings no gain", fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("HASTE's single-date design is not the bottleneck for damage: "
             "post-event imagery alone separates Destruction best",
             fontsize=11, y=1.02)
fig.tight_layout()
out = FIG / "haste_control.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
