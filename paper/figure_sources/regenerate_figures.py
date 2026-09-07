"""Regenerate the three figures from the accompanying recorded numerical values.

Run from any directory: python figure_sources/regenerate_figures.py
Requires Python 3, matplotlib, and numpy.
"""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print(OUT / name)


def fourier_sweep():
    data = load("fourier_sweep_data.json")
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, len(rows)))
    for row, color in zip(rows, colors):
        ax.scatter(row["reported_correlation"], row["reported_error_percent"],
                   color=color, s=40, zorder=3,
                   label=f"Bandwidth {row['bandwidth']:g}")
    ax.axhline(data["reference_reported_error_percent"], ls="--", color="0.4",
               lw=1, label="SiLU reference (16.37%)")
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 160)
    ax.set_xlabel("Reported correlation with the SiLU reference")
    ax.set_ylabel("Reported test error (%)")
    ax.set_title("WCO2 Fourier-feature bandwidth sweep")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "floor.pdf")


def scaling():
    data = load("scaling_data.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.7))
    oco = data["oco2"]
    ax1.loglog([r["n"] for r in oco], [r["error_percent"] for r in oco],
               "o-", color="tab:blue", ms=4, lw=1.2,
               label="corrected surrogate (one seed)")
    ax1.axhline(4.12, ls=":", color="0.4", lw=1.1,
                label="full pipeline, ten seeds (4.12%)")
    ax1.set_xlabel("training pairs")
    ax1.set_ylabel("reduced relative error (%)")
    ax1.set_title("OCO-2 O2: error against training size")
    ax1.legend(fontsize=7.5)

    rows = data["climsim"]
    ns = sorted({r["n"] for r in rows})
    for key, label, fmt, color in [
        ("mean_r2", "neural mean", "o-", "tab:blue"),
        ("kernel_r2", "kernel (6000 fitting rows)", "s-", "tab:purple"),
    ]:
        groups = {n: [r[key] for r in rows if r["n"] == n and r[key] is not None]
                  for n in ns}
        groups = {n: vals for n, vals in groups.items() if vals}
        means = [statistics.mean(vals) for vals in groups.values()]
        sds = [statistics.stdev(vals) if len(vals) > 1 else 0.0
               for vals in groups.values()]
        ax2.errorbar(list(groups), means, yerr=sds, fmt=fmt, color=color,
                     ms=5, capsize=3, label=label)
    ax2.axhline(0.6, ls=":", color="0.4", lw=1.1,
                label="published MLP baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("training samples")
    ax2.set_ylabel(r"test $R^2$")
    ax2.set_title("ClimSim: performance against training size")
    ax2.legend(fontsize=7.5, loc="center right")
    save(fig, "scaling_seeded.pdf")


def seed_spread():
    data = load("seedspread_data.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    published = [("DeepONet", 5.20), ("kernel", 5.18), ("FNO", 4.76),
                 ("PCA-Net", 4.67), ("PARA-Net", 4.55)]
    for name, value in published:
        ax.axhline(value, color="0.75", lw=0.9, zorder=1)
        dy = {"kernel": -0.013, "DeepONet": 0.013}.get(name, 0.0)
        ax.text(10.65, value + dy, " " + name, va="center", fontsize=8,
                color="0.35")
    seeds = data["seeds"]
    errors = [100 * v for v in data["corrected"]]
    ax.plot(seeds, errors, "o", ms=5, color="#1f4e79", zorder=3,
            label="corrected pipeline, per seed")
    ax.axhline(statistics.mean(errors), color="#1f4e79", lw=1.2,
               ls="--", zorder=2)
    ax.set_xlim(-0.7, 10.4)
    ax.set_ylim(4.5, 5.26)
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel(r"relative $L^2$ test error (%)")
    ax.set_title("Structural mechanics: ten seeds")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    stages = [("best member", "best_member"), ("equal weights", "ens_equal"),
              ("fitted weights", "ens_fitted"), ("affine stack", "stack"),
              ("+ correction", "corrected")]
    for i, (_, key) in enumerate(stages):
        values = [100 * z for z in data[key]]
        bx.plot([i] * len(values), values, "o", ms=4, color="#1f4e79", alpha=0.55)
        mu = statistics.mean(values)
        bx.plot([i - 0.22, i + 0.22], [mu, mu], color="#c0392b", lw=1.8)
    bx.set_xticks(range(len(stages)))
    bx.set_xticklabels([s[0] for s in stages], fontsize=8, rotation=15)
    bx.set_ylabel(r"relative $L^2$ test error (%)")
    bx.set_title("Error by pipeline stage")
    save(fig, "seedspread.pdf")


if __name__ == "__main__":
    fourier_sweep()
    scaling()
    seed_spread()
