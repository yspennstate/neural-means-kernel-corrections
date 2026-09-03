"""Figure S6.4 (paper/figs/uq.pdf) from the seeded test-block calibration record, after review.

The first release drew this figure from the earlier validation-calibrated single run and labelled its
calibration panel as relative error. The panels now come from collected/dgx/uq_fig_seeded_s0.npz, written by
dgx_checks/uq_fig_dump.py on the campaign host: the deployed six-member correction at seed 0, evaluation half
of the test block (19000 cases), band calibrated on the other 1000 exactly as uq_conformal_plam.py does.
Left: P_lambda against relative error. Middle: decile calibration of P_lambda against absolute error, the
quantity Theorem 6.9 bounds. Right: coverage of the P_lambda-scaled 90% band within each decile of P_lambda.

    python campaign/fig_uq_seeded.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "collected", "dgx", "uq_fig_seeded_s0.npz"))
P, er = z["P"], z["err_rel"]
pr, pa, sr, cov = float(z["pearson_rel"]), float(z["pearson_abs"]), float(z["spearman_rel"]), float(z["cover90"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
sub = np.random.default_rng(0).choice(len(P), 4000, replace=False)
ax = axes[0]
ax.scatter(P[sub], 100 * er[sub], s=3, alpha=0.25, color="#1f4e79", edgecolors="none")
ax.set_xlabel(r"posterior sd $P_\lambda(u)$")
ax.set_ylabel("relative error (%)")
ax.set_title(f"relative error\nPearson {pr:.2f}, Spearman {sr:.2f}")
ax = axes[1]
ax.plot(z["dec_P"], z["dec_abs"], "o-", color="#b5651d", lw=1.4, ms=4)
ax.set_xlabel(r"mean $P_\lambda$ in decile")
ax.set_ylabel(r"mean absolute error $\|e(u)\|_2$")
ax.set_title(f"absolute error by decile\nPearson {pa:.2f}")
ax = axes[2]
ax.plot(range(1, 11), 100 * np.array(z["dec_cov"]), "s-", color="#2e7d32", lw=1.4, ms=4)
ax.axhline(90, color="k", lw=0.8, ls="--")
ax.set_ylim(60, 101)
ax.set_xticks(range(1, 11))
ax.set_xlabel(r"decile of $P_\lambda$")
ax.set_ylabel("coverage of the 90% band (%)")
ax.set_title(f"coverage by decile\noverall {100 * cov:.1f}%")
fig.tight_layout()
out = os.path.join(HERE, "..", "paper", "figs", "uq.pdf")
fig.savefig(out)
print("wrote", os.path.normpath(out))
