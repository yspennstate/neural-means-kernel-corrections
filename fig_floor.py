"""Figure: the ensembling floor is class-specific but hard to beat (paper/figs/floor.pdf).

Left: residual correlations on WCO2, seed-to-seed against cross-architecture.
Right: the accuracy-decorrelation trade-off of the Fourier member across
bandwidths, with the region where a member would lower the floor shaded.

    python fig_floor.py [figs_dir]

The right panel is read from runs/fourier_wco2.log, the run log of the bandwidth
sweep. The four left-panel correlations were computed on 2026-07-19 from the
seed and architecture prediction arrays of that sweep (seed pairs averaged over
the three box seeds, cross-architecture pairs from the diverse set); those arrays
are not part of this repository, so the values are carried here as constants.
"""
import sys, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "paper" / "figs")
LOG = HERE / "runs" / "fourier_wco2.log"

LEFT = [("seed vs seed\n(same arch)", 0.97), ("ReLU vs SiLU", 0.84),
        ("SiLU vs wide", 0.40), ("SiLU vs\nFourier", 0.11)]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c", "#495057"]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5))

ax = axes[0]
labels = [l for l, _ in LEFT]
cors = [v for _, v in LEFT]
cols = [C[4]] + [C[0]] * (len(LEFT) - 1)
bars = ax.bar(range(len(cors)), cors, color=cols)
ax.axhline(cors[0], ls="--", color=C[4], lw=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("residual correlation")
ax.set_ylim(0, 1)
ax.set_title("the floor is class-specific")
for b_, v in zip(bars, cors):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

ax = axes[1]
sig, err, corr = [], [], []
e_ref = None
for line in LOG.read_text(encoding="utf-8").splitlines():
    m = re.search(r"reference silu: test ([\d.]+)", line)
    if m:
        e_ref = float(m.group(1)) * 100
    m = re.search(r"sigma ([\d.]+): test ([\d.]+)%\s+corr ([\d.]+)", line)
    if m:
        sig.append(float(m.group(1)))
        err.append(float(m.group(2)))
        corr.append(float(m.group(3)))
sig, err, corr = np.array(sig), np.array(err), np.array(corr)

sc = ax.scatter(corr, err, c=sig, cmap="viridis", s=40, zorder=3)
# Bandwidths 1.0, 1.5 and 2.0 land on one point (corr 0.11, error 144%); label the
# cluster once with a leader instead of stacking three labels on one marker.
cluster = (corr < 0.13) & (err > 140)
for x, y, s in zip(corr[~cluster], err[~cluster], sig[~cluster]):
    ax.annotate(f"{s:g}", (x, y), fontsize=7, xytext=(6, 4), textcoords="offset points")
if cluster.any():
    cx, cy = float(corr[cluster].mean()), float(err[cluster].mean())
    names = ", ".join(f"{s:g}" for s in sorted(sig[cluster]))
    ax.annotate(names, (cx, cy), fontsize=7, xytext=(0.26, 0.86), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=C[4], lw=0.6), va="center")
plt.colorbar(sc, ax=ax, label="Fourier bandwidth")
cc = np.linspace(0.01, 1, 200)
ax.plot(cc, e_ref / cc, color=C[1], lw=1.0)
ax.fill_between(cc, 0, e_ref / cc, color=C[1], alpha=0.08)
ax.axhline(e_ref, ls="--", color=C[4], lw=0.8, label="SiLU reference")
ax.text(0.62, e_ref * 0.38, "members here\nlower the floor", fontsize=7, color=C[1])
ax.set_xlabel("correlation with the SiLU mean")
ax.set_ylabel("test error (%)")
ax.set_ylim(0, min(160, err.max() * 1.1))
ax.set_xlim(0, 1)
ax.set_title("no member is both accurate and decorrelated")
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "floor.pdf")
print("wrote", FIGS / "floor.pdf")
