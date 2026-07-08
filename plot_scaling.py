"""Scaling figure: test error vs training-set size, log-log, with a power-law
fit to the neural mean and its extrapolation to larger n. Makes the argument
that more data yields stronger results.

    python plot_scaling.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE / "paper" / "figs"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c"]

red = json.load(open(HERE / "runs" / "scaling_reduced.json"))
R = red["results"]
n = np.array([r["n"] for r in R], float)
mean = np.array([r["mean"] for r in R])
kernel = np.array([r["kernel"] for r in R])
best = np.array([r["best"] for r in R])

# power-law fit to the neural mean: log(err) = a + b log(n)
b, a = np.polyfit(np.log(n), np.log(mean), 1)
ne = np.array([n[0], 2e5])
fit = np.exp(a) * ne ** b

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.loglog(n, mean, "o-", color=C[0], lw=1.3, ms=4, label="neural mean")
ax.loglog(n, kernel, "s-", color=C[1], lw=1.3, ms=4, label="kernel on state")
ax.loglog(n, best, "d-", color=C[2], lw=1.3, ms=4, label="mean + kernel")
ax.loglog(ne, fit, "--", color=C[0], lw=1.0, alpha=0.7,
          label=f"power-law fit, slope {b:.2f}")
# extrapolated markers at 20k and 100k
for N in (2e4, 1e5):
    e = np.exp(a) * N ** b
    ax.plot(N, e, "*", color=C[3], ms=10)
    ax.annotate(f"{e:.1f}%\n@ n={int(N):,}", (N, e), fontsize=7, color=C[3],
                xytext=(4, 4), textcoords="offset points")
ax.axhline(3.8, ls=":", color="#868e96", lw=0.8)
ax.text(n[0], 4.0, "emulator (20k samples): 3.8%", fontsize=7, color="#868e96")
ax.set_xlabel("training samples $n$")
ax.set_ylabel("reduced-radiance test error (%)")
ax.set_title("OCO-2 O2 emulator: test error versus training-set size")
ax.legend(frameon=False, loc="lower left")
ax.grid(True, which="both", alpha=0.15)

fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "scaling.pdf")
fig.savefig(HERE / "scaling.png")
print(f"wrote scaling.pdf | mean slope {b:.3f}; extrapolated: "
      f"{np.exp(a)*2e4**b:.2f}% at 20k, {np.exp(a)*1e5**b:.2f}% at 100k", flush=True)
