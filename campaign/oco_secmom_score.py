"""Prospective scoreboard for Proposition~\\ref{prop:secmom}(i) on OCO-2.

The rule is decided on validation and checked on test, which is what makes the
check prospective: for a pair of members it predicts that mixing the weaker
one in strictly helps exactly when the residual correlation falls below the
error ratio, and the test set then says whether the optimal convex mix beat
the better member. Nothing is retrained; the stored member predictions of each
band-seed are read as they are.

Run on a research box, where the seed directories live:
    python oco_secmom_score.py /srv/aiwork/nmkc10seed/seeds <out.json>
"""
import glob
import itertools
import json
import math
import os
import sys

import numpy as np

# Predictions of the published emulator are a fixed baseline, not a member of
# our stack, so they take no part in a statement about our own members.
SKIP = {"kernel_flow"}


def sq_metric(R, den):
    """E ell^2 in the relative metric, for a residual array."""
    return float(np.mean((np.linalg.norm(R, axis=1) / den) ** 2))


def cross(Ra, Rb, den):
    """S_ab = E <rho_a, rho_b> in the relative metric."""
    return float(np.mean((Ra * Rb).sum(axis=1) / den ** 2))


def best_mix(e1s, e2s, s12):
    """Squared risk of the optimal convex mix of two members, and its weight."""
    denom = e1s + e2s - 2 * s12
    if denom <= 0:
        return min(e1s, e2s), (1.0 if e1s <= e2s else 0.0)
    t = (e2s - s12) / denom              # weight on member 1
    t = min(max(t, 0.0), 1.0)
    val = t * t * e1s + (1 - t) ** 2 * e2s + 2 * t * (1 - t) * s12
    return float(val), float(t)


def main():
    root = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    rows = []

    for band in ("o2", "wco2", "sco2"):
        for d in sorted(glob.glob(os.path.join(root, "oco_%s_s*" % band))):
            tag = os.path.basename(d)
            try:
                seed = int(tag.rsplit("_s", 1)[1])
            except ValueError:
                continue
            if seed >= 90:
                continue
            p = os.path.join(d, "member_preds.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            names = sorted({k[4:] for k in z.files if k.startswith("val_")}
                           & {k[3:] for k in z.files if k.startswith("te_")})
            names = [n for n in names if n not in SKIP and not n.startswith("combined")]
            Yval, Yte = z["Yval"].astype(np.float64), z["Yte"].astype(np.float64)
            dv, dt = np.linalg.norm(Yval, axis=1), np.linalg.norm(Yte, axis=1)
            Rv = {n: z["val_" + n].astype(np.float64) - Yval for n in names}
            Rt = {n: z["te_" + n].astype(np.float64) - Yte for n in names}

            for a, b in itertools.combinations(names, 2):
                # decided on validation
                e1v, e2v = sq_metric(Rv[a], dv), sq_metric(Rv[b], dv)
                lo, hi = (a, b) if e1v <= e2v else (b, a)
                elo_v, ehi_v = min(e1v, e2v), max(e1v, e2v)
                if ehi_v <= 0 or elo_v <= 0:
                    continue
                s12v = cross(Rv[lo], Rv[hi], dv)
                rho_v = s12v / np.sqrt(elo_v * ehi_v)
                ratio_v = np.sqrt(elo_v / ehi_v)
                predict_helps = bool(rho_v < ratio_v)
                margin = float(abs(rho_v - ratio_v))

                # checked on test
                elo_t, ehi_t = sq_metric(Rt[lo], dt), sq_metric(Rt[hi], dt)
                s12t = cross(Rt[lo], Rt[hi], dt)
                # the same two quantities on the test block, so the
                # validation-to-test transfer of the statistic can be measured
                # rather than assumed
                rho_t = s12t / math.sqrt(elo_t * ehi_t)
                ratio_t = math.sqrt(min(elo_t, ehi_t) / max(elo_t, ehi_t))
                mix_t, w_t = best_mix(elo_t, ehi_t, s12t)
                # "helps" means the optimum is interior and beats the better
                # member by more than numerical noise
                realized_helps = bool(mix_t < elo_t * (1 - 1e-9))
                gain = float(np.sqrt(elo_t) - np.sqrt(max(mix_t, 0.0)))

                rows.append(dict(band=band, seed=seed, better=lo, weaker=hi,
                                 rho_val=round(float(rho_v), 5),
                                 ratio_val=round(float(ratio_v), 5),
                                 rho_test=round(float(rho_t), 5),
                                 ratio_test=round(float(ratio_t), 5),
                                 margin=round(margin, 5),
                                 predict_helps=predict_helps,
                                 realized_helps=realized_helps,
                                 correct=bool(predict_helps == realized_helps),
                                 gain_pct=round(100 * gain, 5),
                                 mix_weight=round(w_t, 4)))

    print("pairs scored: %d over %d band-seeds"
          % (len(rows), len({(r["band"], r["seed"]) for r in rows})))
    if rows:
        n_ok = sum(r["correct"] for r in rows)
        print("overall: %d/%d = %.1f%%" % (n_ok, len(rows), 100 * n_ok / len(rows)))
        bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 9)]
        for lo, hi in bins:
            sel = [r for r in rows if lo <= r["margin"] < hi]
            if sel:
                k = sum(r["correct"] for r in sel)
                print("  margin [%.2f,%.2f): %d/%d = %.0f%%"
                      % (lo, hi, k, len(sel), 100 * k / len(sel)))
        for band in ("o2", "wco2", "sco2"):
            sel = [r for r in rows if r["band"] == band]
            if sel:
                k = sum(r["correct"] for r in sel)
                print("  %-5s %d/%d = %.1f%%" % (band, k, len(sel), 100 * k / len(sel)))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
