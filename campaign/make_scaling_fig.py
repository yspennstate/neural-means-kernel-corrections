"""Regenerate paper/figs/scaling_seeded.pdf.

The figure this replaces was drawn in July, before either seed campaign, and
its ClimSim panel came from a sweep whose kernel hyperparameters were selected
against test rows. That sweep is retained in the release as superseded; it is
not plotted here. The panel below carries only the leak-free campaign points,
which exist at three training sizes, with error bars over the seeds actually
run at each.

Usage:
    python campaign/make_scaling_fig.py <collected_dir> <runs_dir> <out.pdf>
"""
import collections
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def climsim_points(collected):
    by_n = collections.defaultdict(lambda: {"mean": [], "kernel": []})
    for f in glob.glob(os.path.join(collected, "box*", "a3_climsim_train_*.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("mean_r2") is None:
            continue
        by_n[r["n"]]["mean"].append(r["mean_r2"])
        if r.get("kernel_r2") is not None:
            by_n[r["n"]]["kernel"].append(r["kernel_r2"])
    return dict(sorted(by_n.items()))


def agg(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def main():
    collected, runs_dir, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.7))

    # Left: the OCO-2 O2 data-scaling run. One seed, pre-campaign, and labelled
    # as such; the reference line is the current ten-seed combination.
    with open(os.path.join(runs_dir, "scaling_reduced.json"), encoding="utf-8") as fh:
        sc = json.load(fh)
    rows = sc["results"]
    ns = [r["n"] for r in rows]
    es = [r["best"] for r in rows]
    ax1.loglog(ns, es, "o-", color="tab:blue", ms=4, lw=1.2,
               label="corrected surrogate (one seed)")
    ax1.axhline(4.12, ls=":", color="0.4", lw=1.1,
                label="full pipeline, ten seeds (4.12%)")
    ax1.set_xlabel("training pairs")
    ax1.set_ylabel("reduced relative error (%)")
    ax1.set_title("OCO-2 O2: error against training size", fontsize=10)
    ax1.legend(fontsize=7.5)

    # Right: ClimSim, leak-free campaign points only.
    pts = climsim_points(collected)
    ns = sorted(pts)
    mm = [agg(pts[n]["mean"]) for n in ns]
    kk = [agg(pts[n]["kernel"]) for n in ns if pts[n]["kernel"]]
    ax2.errorbar(ns, [m[0] for m in mm], yerr=[m[1] for m in mm],
                 fmt="o-", color="tab:blue", ms=5, capsize=3,
                 label="neural mean")
    ax2.errorbar(ns, [k[0] for k in kk], yerr=[k[1] for k in kk],
                 fmt="s-", color="tab:purple", ms=5, capsize=3,
                 label="exact kernel on the state")
    ax2.axhline(0.6, ls=":", color="0.4", lw=1.1, label="published MLP baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("training samples")
    ax2.set_ylabel(r"test $R^2$")
    ax2.set_title("ClimSim: leak-free points only", fontsize=10)
    ax2.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out_pdf)
    print("wrote %s" % out_pdf)
    for n, m, k in zip(ns, mm, kk):
        print("  n=%-9d mean %.4f +- %.4f (%d seeds)   kernel %.4f +- %.4f (%d)"
              % (n, m[0], m[1], m[2], k[0], k[1], k[2]))


if __name__ == "__main__":
    main()
