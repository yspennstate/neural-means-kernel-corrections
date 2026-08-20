"""Regenerate paper/figs/oco2_seeded.pdf from the ten-seed campaign.

The figure this replaces was produced before the seed campaign, from a single
run at a longer training budget, and its right-hand legend labelled the
weighted-coefficient network as radiance-trained. Both are fixed here by
drawing everything from the campaign artifacts: the left panel from the
per-band member table and the right panel from per-coordinate test RMSE
averaged over the ten seeds. Members are named by the loss they are trained
on.

Usage:
    python campaign/make_oco2_fig.py <percoord_dir> <results_dir> <out.pdf>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LANE = {
    ("o2", 0): "b02", ("o2", 1): "b01", ("o2", 2): "b01", ("o2", 3): "b02",
    ("o2", 4): "b02", ("o2", 5): "b03", ("o2", 6): "b03", ("o2", 7): "b04",
    ("o2", 8): "b04", ("o2", 9): "b04",
}
BAND = "o2"


def main():
    pc_dir, res_dir, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3]
    boxes = {b: json.load(open(os.path.join(pc_dir, b + ".json"), encoding="utf-8"))
             for b in ("b01", "b02", "b03", "b04")}

    # per-coordinate RMSE, averaged over the ten seeds of the table's lane
    stacks = {}
    for seed in range(10):
        rec = boxes[LANE[(BAND, seed)]]["%s_%d" % (BAND, seed)]
        for name, vals in rec.items():
            stacks.setdefault(name, []).append(np.asarray(vals, float))
    pc = {k: np.mean(v, axis=0) for k, v in stacks.items() if len(v) == 10}

    # ladder values, ten seeds, from the same artifacts as the table
    def band_stat(key):
        vals = []
        for seed in range(10):
            with open(os.path.join(res_dir, "oco_%s_s%d.json" % (BAND, seed)),
                      encoding="utf-8") as f:
                vals.append(json.load(f)["results"][key]["reduced"])
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    ladder = [("Matérn,\nraw input", "kernel_raw"),
              ("Matérn,\nARD", "kernel_ard"),
              ("kernel-flow\nemulator", "kernel_flow"),
              ("neural\nmean", "mean_flat"),
              ("Matérn on\nfeatures", "dkr_flat")]
    means, sds = zip(*[band_stat(k) for _, k in ladder])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ["0.30", "0.30", "tab:purple", "tab:blue", "tab:orange"]
    x = np.arange(len(ladder))
    ax1.bar(x, means, yerr=sds, color=colors, capsize=3)
    for xi, m in zip(x, means):
        ax1.text(xi, m + 1.2, "%.1f" % m, ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n for n, _ in ladder], fontsize=8)
    ax1.set_ylabel("reduced relative error (%)")
    ax1.set_title("the same kernel, three feature sets (O2)", fontsize=10)
    ax1.set_ylim(0, max(means) * 1.15)

    order = np.argsort(-pc["kernel_flow"] * 0 - np.arange(len(pc["kernel_flow"])))
    order = np.arange(len(pc["kernel_flow"]))     # coefficients in PCA order
    for key, label, style in (
            ("kernel_flow", "kernel-flow emulator", dict(color="tab:purple", marker="o")),
            ("mean_flat", "network, flat loss", dict(color="tab:blue", marker="s")),
            ("mean_wnum", "network, weighted-coefficient loss",
             dict(color="tab:orange", marker="^")),
            ("mean_radx", "network, exact radiance loss",
             dict(color="tab:red", marker="v"))):
        if key in pc:
            ax2.plot(order, pc[key][order], lw=1.1, ms=3, **style, label=label)
    ax2.set_yscale("log")
    ax2.set_xlabel("reduced coordinate, ordered by radiance weight (shaded)")
    ax2.set_ylabel("per-coordinate RMSE")
    ax2.set_title("who is accurate where", fontsize=10)
    ax2.axvspan(-0.5, 1.5, color="0.85", zorder=0)
    ax2.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_pdf)
    print("wrote %s" % out_pdf)
    for k in ("kernel_flow", "mean_flat", "mean_wnum", "mean_radx"):
        if k in pc:
            print("  %-12s leading coefficient RMSE %.4f   trailing-30 mean %.4f"
                  % (k, pc[k][0], pc[k][10:].mean()))


if __name__ == "__main__":
    main()
