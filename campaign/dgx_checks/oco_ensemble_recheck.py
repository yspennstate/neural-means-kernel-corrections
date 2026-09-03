"""Recompute the OCO-2 ensemble quantities in the terms Proposition 6.1 uses, after review.

E2 is the RMS relative error (the proposition's metric; the tables report the mean relative error E1). S is the
second-moment matrix of the normalized residuals rho = (f(u) - z) / ||z|| on the validation split, uncentered as
the proposition defines it (the first release quoted a centered correlation). Reported: the two-member admission
example (best flat network against the raw-input kernel) per band and seed, and the prospective scoreboard over
the Table-6 members, decided on validation and scored on test against the better-on-TEST member in two readings:
(a) the convex two-member mix optimized on the test moments (does any convex mix beat both members on test) and
(b) the mix at the validation weight (does the mix the rule would ship beat both members on test). The first
release scored the test-optimal mix against the member that was better on validation, a different object when
the order of the two members flips between the splits.

    ~/nmkc_venv/bin/python oco_ensemble_recheck.py [--root ~/nmkc2/oco_ridge/seeds] [--out ~/nmkc2/results/oco_ensemble_recheck.json]
"""
import argparse, json, math, os, statistics as st
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", default=os.path.expanduser("~/nmkc2/oco_ridge/seeds"))
p.add_argument("--out", default=os.path.expanduser("~/nmkc2/results/oco_ensemble_recheck.json"))
a = p.parse_args()
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 9.0)]


def E(R):
    n = np.linalg.norm(R, axis=1)
    return float(n.mean()), float(np.sqrt((n ** 2).mean()))


def S_of(Ra, Rb):
    return float((Ra * Rb).sum(1).mean())


out = dict(bands={}, scoreboard=dict(rows=[]), admission=[])
for band in ("o2", "wco2", "sco2"):
    for seed in range(10):
        f = f"{a.root}/oco_{band}_s{seed}/member_preds.npz"
        if not os.path.exists(f):
            print("missing", f, flush=True)
            continue
        z = np.load(f)
        names = sorted(k[4:] for k in z.files if k.startswith("val_") and not k[4:].startswith("ridge"))
        tp = next(k for k in z.files if not k.startswith("val_") and k not in ("Yval", "Yte")).split("_")[0] + "_"
        Yv, Yt = z["Yval"].astype(float), z["Yte"].astype(float)
        Rv = {m: (z["val_" + m].astype(float) - Yv) / np.linalg.norm(Yv, axis=1)[:, None] for m in names}
        Rt = {m: (z[tp + m].astype(float) - Yt) / np.linalg.norm(Yt, axis=1)[:, None] for m in names}
        ev = {m: E(Rv[m]) for m in names}
        et = {m: E(Rt[m]) for m in names}
        if "mean_flat" in names and "kernel_raw" in names:
            A, B = "mean_flat", "kernel_raw"
            e1, e2 = ev[A][1], ev[B][1]
            rho = S_of(Rv[A], Rv[B]) / (e1 * e2)
            cen = float(np.corrcoef(Rv[A].ravel(), Rv[B].ravel())[0, 1])
            den = e1 ** 2 + e2 ** 2 - 2 * rho * e1 * e2
            wstar = (e2 ** 2 - rho * e1 * e2) / den
            vstar = math.sqrt(e1 ** 2 * e2 ** 2 * (1 - rho ** 2) / den)
            wc = min(1.0, max(0.0, wstar))
            out["admission"].append(dict(
                band=band, seed=seed, e_net_E1=ev[A][0], e_net_E2=e1, e_ker_E1=ev[B][0], e_ker_E2=e2, ratio=e1 / e2,
                rho_uncentered=rho, rho_centered=cen, w_star=wstar, v_star=vstar, helps=bool(rho < e1 / e2),
                test_net_E2=et[A][1], test_mix_E2_clipped=E(wc * Rt[A] + (1 - wc) * Rt[B])[1],
                test_mix_E2_signed=E(wstar * Rt[A] + (1 - wstar) * Rt[B])[1]))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                m1, m2 = names[i], names[j]
                if ev[m1][1] > ev[m2][1]:
                    m1, m2 = m2, m1
                e1v, e2v = ev[m1][1], ev[m2][1]
                rv = S_of(Rv[m1], Rv[m2]) / (e1v * e2v)
                ratio_v = e1v / e2v
                e1t, e2t = et[m1][1], et[m2][1]
                ct = S_of(Rt[m1], Rt[m2])
                rt = ct / (e1t * e2t)
                dent = e1t ** 2 + e2t ** 2 - 2 * ct
                wt = min(1.0, max(0.0, (e2t ** 2 - ct) / dent)) if dent > 1e-30 else 1.0
                mix_t = E(wt * Rt[m1] + (1 - wt) * Rt[m2])[1]
                denv = e1v ** 2 + e2v ** 2 - 2 * rv * e1v * e2v
                wv = min(1.0, max(0.0, (e2v ** 2 - rv * e1v * e2v) / denv)) if denv > 1e-30 else 1.0
                mix_v = E(wv * Rt[m1] + (1 - wv) * Rt[m2])[1]
                best_t = min(e1t, e2t)
                out["scoreboard"]["rows"].append(dict(
                    band=band, seed=seed, pair=[m1, m2], rho_val=rv, ratio_val=ratio_v, margin=abs(rv - ratio_v),
                    predict=bool(rv < ratio_v), order_flips=bool(e1t > e2t),
                    outcome_old=bool(mix_t < e1t * (1 - 1e-6)), outcome_any=bool(mix_t < best_t * (1 - 1e-6)),
                    outcome_ship=bool(mix_v < best_t * (1 - 1e-6)), D_val=rv - ratio_v,
                    D_test=rt - min(e1t, e2t) / max(e1t, e2t), w_val=wv, w_test=wt))
        out["bands"].setdefault(band, {})[seed] = dict(
            members=names, E1_val={m: ev[m][0] for m in names}, E2_val={m: ev[m][1] for m in names},
            E1_test={m: et[m][0] for m in names}, E2_test={m: et[m][1] for m in names})
        print(band, seed, "members", len(names), flush=True)

rows = out["scoreboard"]["rows"]
summ = dict(n=len(rows), members=len(out["bands"]["o2"][0]["members"]), order_flips=sum(r["order_flips"] for r in rows))
for key in ("outcome_old", "outcome_any", "outcome_ship"):
    summ[key] = dict(correct=sum(r["predict"] == r[key] for r in rows),
                     bins={f"{lo}-{hi}": [sum(r["predict"] == r[key] for r in rows if lo <= r["margin"] < hi),
                                          sum(1 for r in rows if lo <= r["margin"] < hi)] for lo, hi in BINS})
D = np.array([r["D_test"] - r["D_val"] for r in rows])
summ["D_transfer_sd"] = float(D.std())
summ["D_transfer_mean"] = float(D.mean())
out["scoreboard"]["summary"] = summ
adm = out["admission"]
keys = ("rho_uncentered", "rho_centered", "ratio", "e_net_E1", "e_net_E2", "e_ker_E1", "e_ker_E2", "v_star", "w_star",
        "test_net_E2", "test_mix_E2_clipped", "test_mix_E2_signed")
out["admission_summary"] = {band: dict(n=len(rs), helps_count=sum(r["helps"] for r in rs),
                                       **{k: st.mean(r[k] for r in rs) for k in keys})
                            for band, rs in ((b, [r for r in adm if r["band"] == b]) for b in ("o2", "wco2", "sco2")) if rs}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1))
print(json.dumps(out["admission_summary"], indent=1))
print("wrote", a.out, flush=True)
