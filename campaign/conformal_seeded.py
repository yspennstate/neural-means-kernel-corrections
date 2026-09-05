"""Split-conformal coverage of the corrected surrogate, at every seed.

Identical arithmetic to campaign/uq_conformal.py; the only change is that member
prediction files are resolved by their campaign run names (mlpMSE_s3_w1024_..._predte.npy)
rather than the short names the single-seed script assumed.

The public test block is split into 1000 calibration and 19000 evaluation
cases. These rows are separate from within-run fitting and selection, but
the benchmark has been inspected repeatedly across the study. A fresh-population
coverage claim requires a predictor and scale fixed independently of exchangeable
calibration/evaluation observations; this script measures retrospective coverage.
"""
import glob, json, math, os, sys
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split  # noqa: E402

NAMES = ["mlp", "mlpMSE", "mlpR", "krr"]
loads, stress = load_arrays()
out_all = []

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    pred_f = os.path.join(R, "pix4_corr_pred_test.npy")
    if not os.path.exists(pred_f):
        print("s%s no corrected prediction" % seed)
        continue
    os.environ["NMKC_SPLIT_SEED"] = seed
    _, _, te = canonical_split(n_val=1000, seed=int(seed))
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)
    pred = np.load(pred_f).astype(np.float64)

    mem = []
    for m in NAMES:
        if m == "krr":
            g = glob.glob(os.path.join(R, "krr_*_pred_test.npy"))
        else:
            g = [p for p in glob.glob(os.path.join(R, "*_predte.npy"))
                 if os.path.basename(p).split("_")[0] == m]
        mem.append(np.load(g[0]).astype(np.float64))
    mem = np.stack(mem)
    disagree = mem.std(0).mean(1)
    err = np.linalg.norm(pred - Yte, axis=1)

    n = len(Yte)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    cal, ev = perm[:1000], perm[1000:]
    rec = dict(seed=int(seed), n_cal=len(cal), n_eval=len(ev))
    msg = "s%-2s" % seed
    for alpha in (0.1, 0.05):
        k = math.ceil((1 - alpha) * (len(cal) + 1))
        q_s = np.sort(err[cal] / disagree[cal])[k - 1]
        cov_s = float(np.mean(err[ev] <= q_s * disagree[ev]))
        q_r = np.sort(err[cal])[k - 1]
        cov_r = float(np.mean(err[ev] <= q_r))
        rec["a%g" % alpha] = dict(target=1 - alpha, scaled=cov_s, raw=cov_r,
                                  q_scaled=float(q_s), q_raw=float(q_r))
        msg += "  target %.2f: scaled %.4f raw %.4f" % (1 - alpha, cov_s, cov_r)
    print(msg, flush=True)
    out_all.append(rec)
    with open(os.path.join(ROOT, "conf_s%s.json" % seed), "w") as f:
        json.dump(rec, f)
