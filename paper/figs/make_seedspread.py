"""Figure: the ten-seed spread of the pipeline against the spread of published methods.

Left panel places every seed's corrected test error next to the published single numbers
on the same axis, which is the whole argument in one picture: the seed-to-seed scatter of
one pipeline is far smaller than the differences the benchmark is used to adjudicate.
Right panel follows the spread down the pipeline, stage by stage.

Reads the campaign records collected under paper/figs/seedspread_data.json so the figure
regenerates without box access. Writes seedspread.pdf (a name not otherwise claimed under
figs/).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "seedspread_data.json")))

PUB = [("DeepONet", 5.20), ("kernel", 5.18), ("FNO", 4.76),
       ("PCA-Net", 4.67), ("PARA-Net", 4.55)]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.6))

for name, v in PUB:
    ax.axhline(v, color="0.75", lw=0.9, zorder=1)
    dy = {"kernel": -0.013, "DeepONet": 0.013}.get(name, 0.0)
    ax.text(10.65, v + dy, " " + name, va="center", fontsize=8, color="0.35")
seeds = list(range(len(D["corrected"])))
ax.plot(seeds, [100 * v for v in D["corrected"]], "o", ms=5, color="#1f4e79",
        zorder=3, label="this pipeline, per seed")
m = 100 * sum(D["corrected"]) / len(D["corrected"])
ax.axhline(m, color="#1f4e79", lw=1.2, ls="--", zorder=2)
ax.set_xlim(-0.7, 10.4)
ax.set_ylim(4.5, 5.26)
ax.set_xticks(seeds)
ax.set_xlabel("seed")
ax.set_ylabel("relative $L^2$ test error (%)")
ax.set_title("ten replicates against five published methods", fontsize=10)
ax.legend(loc="upper left", fontsize=8, frameon=False)

STAGES = [("best member", "best_member"), ("equal weights", "ens_equal"),
          ("fitted weights", "ens_fitted"), ("affine stack", "stack"),
          ("+ correction", "corrected")]
xs = range(len(STAGES))
for i, (label, key) in enumerate(STAGES):
    v = [100 * z for z in D[key]]
    bx.plot([i] * len(v), v, "o", ms=4, color="#1f4e79", alpha=0.55)
    mu = sum(v) / len(v)
    bx.plot([i - 0.22, i + 0.22], [mu, mu], color="#c0392b", lw=1.8)
bx.set_xticks(list(xs))
bx.set_xticklabels([s[0] for s in STAGES], fontsize=8, rotation=15)
bx.set_ylabel("relative $L^2$ test error (%)")
bx.set_title("the same ten seeds, stage by stage", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
    bx.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "seedspread.pdf"))
print("wrote seedspread.pdf")
