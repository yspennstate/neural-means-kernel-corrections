"""Residual-correlation heatmap for the paper (reads runs/corr_<tag>.json).
Usage: python fig_corr.py <tag> [labels-comma-separated]
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "div"
RUNS = pathlib.Path(__file__).resolve().parent / "runs"
FIGS = pathlib.Path(__file__).resolve().parent / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

d = json.load(open(RUNS / f"corr_{TAG}.json"))
C = np.array(d["corr"]); names = d["members"]
errs = [100 * d["errors"][n] for n in names]
if len(sys.argv) > 2:
    labels = sys.argv[2].split(",")
else:
    labels = names
labels = [f"{l}\n({e:.2f}%)" for l, e in zip(labels, errs)]

M = len(names)
fig, ax = plt.subplots(figsize=(0.85 * M + 1.6, 0.85 * M + 1.1))
im = ax.imshow(C, vmin=0.5, vmax=1.0, cmap="viridis")
for i in range(M):
    for j in range(M):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                color="white" if C[i, j] < 0.85 else "black", fontsize=8)
ax.set_xticks(range(M)); ax.set_yticks(range(M))
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(labels)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="residual correlation")
fig.tight_layout()
fig.savefig(FIGS / "corr.pdf")
print("corr.pdf")
