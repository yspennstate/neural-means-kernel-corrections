"""Combined data-scaling figure: OCO-2 emulator (error vs n, reaching the
published emulator) and ClimSim (R^2 vs n, the neural mean overtaking the
data-efficient kernel and climbing toward the published baseline). Two panels,
the paper's "more data -> stronger, and the crossover is measurable" story.

    python plot_scaling_combined.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE / "paper" / "figs"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c"]

fig, (axo, axc) = plt.subplots(1, 2, figsize=(8.2, 3.3))

# --- OCO-2 panel: corrected-surrogate error vs n, power law toward full-data result ---
R = json.load(open(HERE / "runs" / "scaling_reduced.json"))["results"]
n = np.array([r["n"] for r in R], float); best = np.array([r["best"] for r in R])
b, a = np.polyfit(np.log(n), np.log(best), 1)
ne = np.array([n[0], 2.2e4]); axo.loglog(ne, np.exp(a) * ne ** b, "--", color=C[0], lw=1.0, alpha=.6,
                                          label=f"slope {b:.2f}")
axo.loglog(n, best, "o-", color=C[0], lw=1.3, ms=4, label="neural mean + kernel")
axo.axhline(3.83, ls=":", color="#868e96", lw=.8); axo.text(2.5e3, 3.95, "full pipeline 3.8%", fontsize=7, color="#868e96")
axo.set_xlabel("training pairs $n$"); axo.set_ylabel("reduced-radiance error (%)")
axo.set_title("OCO-2 O2 emulator"); axo.legend(frameon=False, loc="upper right")
axo.grid(True, which="both", alpha=.15)

# --- ClimSim panel: R^2 vs n, neural mean crossing the kernel, toward baseline ---
tf = HERE / "runs" / "scaling_climsim_train.json"
T = json.load(open(tf))["results"]
nc = np.array([r["n"] for r in T], float)
mr2 = np.array([r.get("mean_r2", np.nan) for r in T])
kr2 = np.array([r.get("kernel_r2", np.nan) for r in T])
axc.semilogx(nc, mr2, "o-", color=C[0], lw=1.3, ms=4, label="neural mean")
km = ~np.isnan(kr2)
axc.semilogx(nc[km], kr2[km], "s-", color=C[1], lw=1.3, ms=4, label="kernel on state")
axc.axhline(0.6, ls=":", color="#868e96", lw=.8); axc.text(nc[0], 0.61, "published MLP ~0.6", fontsize=7, color="#868e96")
axc.axhline(0, ls="-", color="#ced4da", lw=.6)
axc.set_xlabel("training samples $n$"); axc.set_ylabel(r"test $R^2$ (active outputs)")
axc.set_title("ClimSim (10M-sample scale)"); axc.legend(frameon=False, loc="lower right")
axc.set_ylim(-1.0, 0.75); axc.grid(True, which="both", alpha=.15)

fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "scaling_combined.pdf"); fig.savefig(HERE / "scaling_combined.png")
print(f"wrote scaling_combined.pdf | OCO slope {b:.2f} | ClimSim R2 to n={int(nc[-1])}: {mr2[-1]:.3f}", flush=True)
