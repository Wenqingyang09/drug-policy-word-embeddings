"""
plot_results.py

Draws Figure 1 and Figure 2 from the files written by results_summary.py
(~/Desktop/results_v1/main_results.csv and per_run_scores.csv).

Figure 1: positions of each concept in both corpora (mean of 10 runs, bars = ±1 SD),
          grouped by where the concepts fall in the two corpora.
Figure 2: difference China minus US for each shared concept, with the range
          across the 10 runs, marked by the H1 decision rule.

Groups follow the within-country sign rule (same sign in at least 9 of 10 runs).

Run:
  conda activate thesis
  python ~/Desktop/plot_results.py
"""

import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIR = os.path.join(os.path.expanduser("~/Desktop"), "results_v1")
US_C, CN_C = "#2F6DB5", "#C0392B"
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E3E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK})

main = {}
with open(os.path.join(DIR, "main_results.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        main[r["Concept"]] = r

runs = defaultdict(dict)
with open(os.path.join(DIR, "per_run_scores.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        runs[(r["country"], r["concept"])][int(r["seed"])] = float(r["score"])


def val(r, k):
    return float(r[k]) if r[k] not in ("", None) else None


def group(r):
    us, cn = r["US sign"], r["CN sign"]
    if cn == "absent":
        return "E"
    if us == cn == "positive":
        return "A"
    if us == cn == "negative":
        return "B"
    if "near zero" in (us, cn):
        return "D"
    return "C"


TITLES = {
    "A": "Medical side in both corpora",
    "B": "Punitive side in both corpora",
    "C": "Opposite sides",
    "D": "Near zero in one corpus",
    "E": "US corpus only",
}
groups = defaultdict(list)
for k, r in main.items():
    groups[group(r)].append(k)
for g in groups:
    if g != "E":
        groups[g].sort(key=lambda k: abs(val(main[k], "Diff mean")))
order = [g for g in "ABCDE" if groups[g]]

# Figure 1
heights = [len(groups[g]) + 0.6 for g in order]
fig, axes = plt.subplots(len(order), 1, figsize=(7.2, 0.42 * sum(heights) + 1.6),
                         sharex=True, gridspec_kw={"height_ratios": heights})
for ax, g in zip(axes, order):
    names = groups[g]
    for i, k in enumerate(names):
        r = main[k]
        u, us_sd = val(r, "US mean"), val(r, "US sd")
        c, cn_sd = val(r, "CN mean"), val(r, "CN sd")
        if c is not None:
            ax.plot([u, c], [i, i], color="#BDBDBD", lw=2, zorder=1)
            ax.errorbar(c, i, xerr=cn_sd, fmt="s", ms=7, color=CN_C, mec="white",
                        mew=1.2, ecolor=CN_C, elinewidth=1, capsize=2, zorder=3)
        else:
            ax.annotate("absent from the Chinese corpus", (u, i), xytext=(10, 0),
                        textcoords="offset points", va="center", color=MUTED,
                        fontsize=8, style="italic")
        ax.errorbar(u, i, xerr=us_sd, fmt="o", ms=7, color=US_C, mec="white",
                    mew=1.2, ecolor=US_C, elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1.4, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(TITLES[g], loc="left", fontsize=9, color=INK, fontweight="bold", pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[-1].set_xlim(-0.48, 0.48)
axes[-1].set_xlabel("Projection score (cosine, mean of 10 runs)\n"
                    "← closer to punitive pole        closer to medical pole →")
handles = [Line2D([], [], marker="o", ls="", color=US_C, ms=7, label="United States"),
           Line2D([], [], marker="s", ls="", color=CN_C, ms=7, label="China")]
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(DIR, f"figure1.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2
shared = [k for k, r in main.items() if r["CN sign"] != "absent"]
shared.sort(key=lambda k: val(main[k], "Diff mean"))
fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(shared) + 1.6))
style = {"CN lower": dict(marker="o", color=INK, mfc=INK),
         "CN higher": dict(marker="D", color=CN_C, mfc=CN_C),
         "undetermined": dict(marker="o", color=MUTED, mfc="white")}
for i, k in enumerate(shared):
    r = main[k]
    st = style[r["Direction"]]
    lo, hi = val(r, "Diff min"), val(r, "Diff max")
    ax.plot([lo, hi], [i, i], color=st["color"], lw=1.6, alpha=0.6)
    ax.plot(val(r, "Diff mean"), i, ls="", ms=7, mec=st["color"], mew=1.4,
            marker=st["marker"], mfc=st["mfc"], zorder=3)
ax.set_yticks(range(len(shared)), shared)
ax.invert_yaxis()
ax.axvline(0, color=INK, lw=1.4)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("Difference in projection score, China minus United States\n"
              "← China closer to punitive pole        China closer to medical pole →")
handles = [Line2D([], [], ls="", ms=7, mew=1.4, mec=style[d]["color"], mfc=style[d]["mfc"],
                  marker=style[d]["marker"], label=lab)
           for d, lab in (("CN lower", "China lower in ≥ 9 of 10 runs"),
                          ("CN higher", "China higher in ≥ 9 of 10 runs"),
                          ("undetermined", "Direction not determined"))]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(DIR, f"figure2.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figure1 and figure2 (png and pdf) in", DIR)