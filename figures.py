"""Paper figures. Requires: hybrid run (tag), uq_spectra artifacts, krr_scaling.
Produces PDF figures under paper/figs/.
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS

TAG = sys.argv[1] if len(sys.argv) > 1 else "hyb"
FIGS = pathlib.Path(__file__).resolve().parent / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_LINE = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c", "#495057"]

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
pred = np.load(RUNS / f"{TAG}_pred_test.npy").astype(np.float64)
err_s = np.linalg.norm(pred - Yte, axis=1) / np.linalg.norm(Yte, axis=1)

# ---------- fields figure ----------
order = np.argsort(err_s)
picks = [order[len(order) // 2], order[int(len(order) * 0.98)]]   # median + 98th pct
fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1, 1, 1]))
for r, i in enumerate(picks):
    u = loads[te[i]]
    yt = Yte[i].reshape(41, 41); yp = pred[i].reshape(41, 41)
    ax = axes[r, 0]
    ax.plot(np.linspace(0, 1, 41), u, color=C_LINE[0], lw=1.4)
    ax.set_ylabel("load $\\bar t(x_1)$" if r == 1 else "load $\\bar t(x_1)$")
    if r == 1: ax.set_xlabel("$x_1$")
    ax.set_title(f"input (rel.\\ err {err_s[i]*100:.1f}\\%)" if False else f"input (rel. err {err_s[i]*100:.1f}%)")
    vmin, vmax = yt.min(), yt.max()
    for c, (img, ttl) in enumerate([(yt, "true stress"), (yp, "prediction"), (np.abs(yp - yt), "absolute error")]):
        ax = axes[r, c + 1]
        cmap = "viridis" if c < 2 else "magma"
        vm = dict(vmin=vmin, vmax=vmax) if c < 2 else {}
        im = ax.imshow(img.T, origin="lower", cmap=cmap, extent=[0, 1, 0, 1], **vm)
        ax.set_title(ttl if r == 0 else "")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.tight_layout()
fig.savefig(FIGS / "fields.pdf"); plt.close(fig)
print("fields.pdf")

# ---------- UQ figure ----------
uq = np.load(RUNS / f"{TAG}_uq_spectra.npz")
uqj = json.load(open(RUNS / f"{TAG}_uq.json"))
P, err = uq["P"], uq["err"]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax = axes[0]
sub = np.random.default_rng(0).choice(len(P), 4000, replace=False)
ax.scatter(P[sub], err[sub] * 100, s=3, alpha=0.25, color=C_LINE[0], edgecolors="none")
ax.set_xlabel("posterior sd $P_\\lambda(u)$"); ax.set_ylabel("relative error (%)")
ax.set_title(f"test samples (Pearson {uqj['pearson']:.2f}, Spearman {uqj['spearman']:.2f})")
ax = axes[1]
ax.plot(uq["dec_P"], np.array(uq["dec_err"]) * 100, "o-", color=C_LINE[1], lw=1.4, ms=4)
ax.set_xlabel("mean $P_\\lambda$ in decile"); ax.set_ylabel("mean relative error (%)")
ax.set_title(f"decile calibration; 90% conformal coverage {uqj['cover90']:.3f}")
fig.tight_layout(); fig.savefig(FIGS / "uq.pdf"); plt.close(fig)
print("uq.pdf")

# ---------- spectra figure ----------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
ev = uq["evals_load"]; n = len(ev)
ax = axes[0]
pos = np.maximum(ev[::-1], 1e-14) / n
ax.semilogy(np.arange(1, n + 1), pos, color=C_LINE[0], lw=1.2)
ax.set_xlabel("eigenvalue index"); ax.set_ylabel("eigenvalue of $K/n$")
ax.set_title("Gram spectrum, Matérn on loads")
ax.set_xlim(0, 400)
ax = axes[1]
lam_grid, d_eff = uq["lam_grid"], uq["d_eff"]
ax.semilogx(lam_grid, d_eff, color=C_LINE[2], lw=1.4)
ax.axvline(uqj["lam"], color=C_LINE[1], lw=1.0, ls="--")
ax.annotate("CV-chosen $\\lambda$", xy=(uqj["lam"], np.interp(np.log(uqj["lam"]), np.log(lam_grid), d_eff)),
            xytext=(uqj["lam"] * 8, max(d_eff) * 0.55), fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.7))
ax.set_xlabel("nugget $\\lambda$"); ax.set_ylabel("$d_{\\mathrm{eff}}(\\lambda)$")
ax.set_title("effective dimension (Lemma 4.4 identity)")
fig.tight_layout(); fig.savefig(FIGS / "spectra.pdf"); plt.close(fig)
print("spectra.pdf")

# ---------- scaling figure ----------
sc = json.load(open(RUNS / "krr_scaling.json"))["errors"]
Ns = sorted(int(k) for k in sc)
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.loglog(Ns, [sc[str(N)] * 100 for N in Ns], "o-", color=C_LINE[0], lw=1.4, ms=4, label="kernel ridge (loads)")
pub = {20000: 4.55}
ax.loglog([20000], [4.55], "s", color=C_LINE[4], ms=5, label="best published (PARA-Net)")
low = json.load(open(RUNS / "hybLD.json"))["report"] if (RUNS / "hybLD.json").exists() else None
pts_x, pts_y = [], []
if low: pts_x.append(1250); pts_y.append(low["final_test"] * 100)
hi = json.load(open(RUNS / f"{TAG}.json"))["report"]
pts_x.append(19000); pts_y.append(hi["final_test"] * 100)
ax.loglog(pts_x, pts_y, "D", color=C_LINE[1], ms=5, label="full pipeline (ours)")
ax.loglog([1250], [6.49], "^", color=C_LINE[4], ms=5, label="best published, 1250 (FNO-mean GP)")
ax.set_xlabel("training samples"); ax.set_ylabel("relative test error (%)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGS / "scaling.pdf"); plt.close(fig)
print("scaling.pdf")
print("done")
