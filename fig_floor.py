"""Plot the rounded historical WCO2 Fourier sweep, without a theorem overlay.

Usage: python fig_floor.py [figs_dir]

The archived log does not define the correlation normalization or retain the
predictions. Its values cannot establish the RMS/uncentered-second-moment
condition of Proposition 6.1. The former architecture bars also lacked retained
prediction inputs and are omitted. This figure reports only the sweep log.
"""
import sys, pathlib, re, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "paper" / "figs")
LOG = HERE / "runs" / "fourier_wco2.log"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = ["#3b5bdb", "#e8590c", "#2b8a3e", "#862e9c", "#495057"]

fig, ax = plt.subplots(figsize=(6.2, 3.5))
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
if len(sig) != 5 or e_ref is None or not np.all(np.isfinite([sig, err, corr])):
    raise ValueError("Expected five complete archived sweep rows and a reference")
if len(set(sig)) != 5 or np.any(err < 0) or np.any(np.abs(corr) > 1):
    raise ValueError("Invalid archived sweep values")

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
ax.axhline(e_ref, ls="--", color=C[4], lw=0.8, label="Recorded SiLU error")
ax.set_xlabel("reported residual correlation with SiLU")
ax.set_ylabel("reported test error (%)")
ax.set_ylim(0, min(160, err.max() * 1.1))
ax.set_xlim(0, 1)
ax.set_title("Archived WCO2 Fourier-feature sweep")
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "floor.pdf", metadata={"CreationDate": None, "ModDate": None})
receipt = dict(
    source="runs/fourier_wco2.log",
    source_sha256=hashlib.sha256(LOG.read_bytes()).hexdigest(),
    producer_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    figure_sha256=hashlib.sha256((FIGS / "floor.pdf").read_bytes()).hexdigest(),
    reference_reported_error_percent=e_ref,
    rows=[dict(bandwidth=float(s), reported_error_percent=float(e),
               reported_correlation=float(c)) for s, e, c in zip(sig, err, corr)],
    metric_provenance="Rounded historical log; underlying arrays and exact moment convention unavailable",
    theorem_admission_overlay=False, architecture_comparison_bars=False)
(FIGS / "floor_provenance.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print("wrote", FIGS / "floor.pdf")
