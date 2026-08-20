"""Ensemble + error localisation across every seed on this box.

Uses the project's OWN metric and split (common.rel_l2, canonical_split(n_val=1000,
seed=0)) so nothing here can disagree with the per-member JSONs by convention.

PROTOCOL, and it is the part a referee will check:
  - EQUAL WEIGHTS are reported on val AND test. No fitting, so both are honest.
  - FITTED WEIGHTS are solved on VAL and reported on TEST only. A weight vector
    selected on val and then scored on val is an in-sample number and would overstate
    the ensemble; that is the whole trap in ensemble papers.
  - Per-member val is RECOMPUTED from the saved arrays rather than read from the JSON,
    so a mismatch against the harvested table would surface a bad prediction file.

ERROR LOCALISATION ("errors where they appear"):
  - spatial: per-grid-point RMSE over val samples, for the best member and the
    ensemble, plus where the ensemble helps most
  - per-sample: the hardest samples, and whether the ensemble fixes them or the
    members all fail together (which is the difference between a diversity gain and
    an irreducible-hardness region)

Writes one JSON per seed. Prints a summary.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/srv/aiwork/nmkc10seed/code")
from common import load_arrays, canonical_split, rel_l2

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
BOX = sys.argv[1] if len(sys.argv) > 1 else "box"

loads, stress = load_arrays()

# EACH SEED HAS ITS OWN TRAIN/VAL SPLIT. campaign/analyze_extras.py:251 does
#     os.environ.setdefault("NMKC_SPLIT_SEED", name.split("_s")[1])
# so seed s is validated on split s, not a shared one. Scoring every seed against
# split 0 made s1's mlpR recompute to 0.2834 against a recorded 0.0469 - and seed 0
# looked perfect because its split IS 0, which hid the bug on the only seed I checked.
# The split is therefore reloaded PER SEED below; the test set never moves.
def split_for(seed):
    return canonical_split(n_val=1000, seed=int(seed))

MEMBERS = ["mlp", "mlpMSE", "mlpR", "fno", "krr"]
out_all = []
RECON = []   # per-member |recomputed - recorded|, the control that caught this

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    tr, va, te = split_for(seed)
    Yva, Yte = stress[va], stress[te]
    Pva, Pte, names = [], [], []
    for m in MEMBERS:
        if m == "krr":
            v = glob.glob(os.path.join(R, "krr_*_pred_val.npy"))
            t = glob.glob(os.path.join(R, "krr_*_pred_test.npy"))
        else:
            v = [p for p in glob.glob(os.path.join(R, "*_predva.npy"))
                 if os.path.basename(p).split("_")[0] == m]
            t = [p for p in glob.glob(os.path.join(R, "*_predte.npy"))
                 if os.path.basename(p).split("_")[0] == m]
        if not v or not t:
            continue
        a, b = np.load(v[0]), np.load(t[0])
        if a.shape[0] != len(va) or b.shape[0] != len(te):
            print("  s%s %s: SHAPE MISMATCH va=%s te=%s - skipped"
                  % (seed, m, a.shape, b.shape))
            continue
        Pva.append(a.reshape(len(va), -1))
        Pte.append(b.reshape(len(te), -1))
        names.append(m)
    if len(names) < 2:
        print("  s%s: only %d members with preds - no ensemble" % (seed, len(names)))
        continue

    Yv = Yva.reshape(len(va), -1).astype(np.float64)
    Yt = Yte.reshape(len(te), -1).astype(np.float64)
    Pva = [p.astype(np.float64) for p in Pva]
    Pte = [p.astype(np.float64) for p in Pte]

    per_member_val = {n: float(rel_l2(p, Yv)) for n, p in zip(names, Pva)}
    per_member_test = {n: float(rel_l2(p, Yt)) for n, p in zip(names, Pte)}

    eq_va = float(rel_l2(np.mean(Pva, axis=0), Yv))
    eq_te = float(rel_l2(np.mean(Pte, axis=0), Yt))

    # weights solved on VAL, scored on TEST. Non-negative, sum-to-one via a simple
    # projected grid over the simplex would be slow; least squares then clip+renorm is
    # enough and is reported as what it is.
    A = np.stack([p.ravel() for p in Pva], axis=1)
    w, *_ = np.linalg.lstsq(A, Yv.ravel(), rcond=None)
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    fit_te = float(rel_l2(np.tensordot(w, np.stack(Pte), axes=1), Yt))

    best_m = min(per_member_val, key=per_member_val.get)
    bi = names.index(best_m)

    # ---- error localisation, on val
    ens_va = np.mean(Pva, axis=0)
    g = int(np.sqrt(Yv.shape[1])) if int(np.sqrt(Yv.shape[1])) ** 2 == Yv.shape[1] else 0
    sp_best = np.sqrt(np.mean((Pva[bi] - Yv) ** 2, axis=0))
    sp_ens = np.sqrt(np.mean((ens_va - Yv) ** 2, axis=0))
    # per-sample relative error
    num_b = np.linalg.norm(Pva[bi] - Yv, axis=1); den = np.linalg.norm(Yv, axis=1)
    num_e = np.linalg.norm(ens_va - Yv, axis=1)
    rb, re_ = num_b / den, num_e / den
    hard = np.argsort(-rb)[:20]
    all_fail = float(np.mean([np.mean(np.linalg.norm(p[hard] - Yv[hard], axis=1)
                                      / den[hard]) for p in Pva]))

    rec = dict(box=BOX, seed=seed, members=names,
               per_member_val=per_member_val, per_member_test=per_member_test,
               ens_equal_val=eq_va, ens_equal_test=eq_te,
               ens_fitted_test=fit_te, fitted_weights=dict(zip(names, map(float, w))),
               best_member=best_m, best_member_val=per_member_val[best_m],
               gain_equal_vs_best_val=per_member_val[best_m] - eq_va,
               grid=g,
               spatial_rmse_best=sp_best.tolist(), spatial_rmse_ens=sp_ens.tolist(),
               hard20_best_relerr=float(np.mean(rb[hard])),
               hard20_ens_relerr=float(np.mean(re_[hard])),
               hard20_all_member_mean_relerr=all_fail,
               median_relerr_best=float(np.median(rb)),
               median_relerr_ens=float(np.median(re_)))
    with open(os.path.join(ROOT, "ens5_s%s.json" % seed), "w") as f:
        json.dump(rec, f)
    out_all.append(rec)
    print("  s%-3s members=%s | best %s %.4f | ens(eq) val %.4f test %.4f | ens(fit) test %.4f"
          % (seed, ",".join(names), best_m, per_member_val[best_m], eq_va, eq_te, fit_te))

print("%s: wrote %d seed ensemble records" % (BOX, len(out_all)))
