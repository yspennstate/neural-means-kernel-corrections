"""Split-conformal coverage of the corrected surrogate with the score the theory names.

Section S3 defines the reported band as q P_lambda(u), with q the split-conformal quantile of
the calibration scores ||e(u_i)||_2 / P_lambda(u_i) and P_lambda the power function (posterior
standard deviation with nugget) of the residual correction's own Matern kernel on the training
design. campaign/uq_conformal.py and conformal_seeded.py compute the constant-width band
(raw scores) and the ensemble-disagreement-scaled band; this script computes the P_lambda-scaled
band for the same seeds, the same calibration / evaluation split of the test block and the same
corrected predictions, so the three bands are directly comparable.

P_lambda is rebuilt from the correction record of each seed: the standardized loads of the
training split, the median length scale from the same 2000-point subsample the correction used
(default_rng(seed).choice), the selected scale multiplier and nugget, and
    P_lambda(u)^2 = k(u,u) - k_u^T (K + n lambda I)^{-1} k_u,   k(u,u) = 1.

    python campaign/uq_conformal_plam.py --seed 0 --tags hpix5,hpix
Writes <root>/seeds/sm_s<seed>/runs/<tag>_uq_plam.json.
"""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
from conformal_utils import conformal_quantile
from scipy.linalg import cho_factor, cho_solve, solve_triangular

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--tags", default="hpix5,hpix")
p.add_argument("--ncal", type=int, default=1000)
p.add_argument("--alphas", default="0.1,0.05")
args = p.parse_args()

ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2")))
CODE = pathlib.Path(os.environ.get("NMKC_CODE", ROOT / "code"))
os.environ.setdefault("NMKC_DATA", str(ROOT / "data" / "structmech"))
os.environ["NMKC_SPLIT_SEED"] = str(args.seed)
sys.path.insert(0, str(CODE))
from common import load_arrays, canonical_split  # noqa: E402

THREADS = int(os.environ.get("NMKC_THREADS", "8"))
RUNS = ROOT / "seeds" / f"sm_s{args.seed}" / "runs"

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=args.seed)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd
Xe = (loads[te].astype(np.float64) - mu) / sd
n = len(Xt)


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)


rng = np.random.default_rng(args.seed)
sub = rng.choice(n, min(2000, n), replace=False)                    # the correction's first draw
med = float(np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)])))

# calibration / evaluation split of the test block: identical to conformal_seeded.py
if not 0 < args.ncal < len(te):
    raise ValueError("Calibration and evaluation sets must both be nonempty")
perm = np.random.default_rng(args.seed).permutation(len(te))
cal, ev = perm[:args.ncal], perm[args.ncal:]
alphas = [float(a) for a in args.alphas.split(",")]

cache = {}


def power_function(smult, lam):
    key = (smult, lam)
    if key in cache:
        return cache[key]
    t0 = time.time()
    s = smult * med
    K = np.empty((n, n))
    for k in range(0, n, 1000):
        K[k:k + 1000] = m52(sqdist(Xt[k:k + 1000], Xt), s)
    K.flat[::n + 1] += lam * n
    c, low = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    P2 = np.empty(len(Xe))
    for k in range(0, len(Xe), 2000):
        Kue = m52(sqdist(Xt, Xe[k:k + 2000]), s)                          # n x b
        V = solve_triangular(c, Kue, lower=True, check_finite=False)       # L^{-1} k_u
        P2[k:k + 2000] = 1.0 - (V * V).sum(0)
    del K, c
    P = np.sqrt(np.maximum(P2, 1e-300))
    print(f"s{args.seed} P_lambda(smult={smult}, lam={lam:g}): med {med:.4f}, "
          f"P range {P.min():.3g}-{P.max():.3g} [{time.time() - t0:.0f}s]", flush=True)
    cache[key] = P
    return P


for tag in args.tags.split(","):
    rec_f = RUNS / f"{tag}_corr.json"
    pred_f = RUNS / f"{tag}_corr_pred_test.npy"
    source = tag
    if not rec_f.exists() and tag == "hpix5":
        # where the half-split rule declined the per-pixel weights, the deployed five-member
        # pipeline is the corrected global stack (hstk5), as in audit_reported_macros.py
        rec_f, pred_f, source = RUNS / "hstk5.json", RUNS / "hstk5_pred_test.npy", "hstk5"
    if not rec_f.exists() or not pred_f.exists():
        print(f"s{args.seed} {tag}: record or prediction missing, skipped", flush=True)
        continue
    rec = json.load(open(rec_f))
    pc = rec["report"]["plus_corr"]
    smult, lam = float(pc["smult"]), float(pc["lam"])
    P = power_function(smult, lam)
    pred = np.load(pred_f).astype(np.float64)
    err = np.linalg.norm(pred - Yte, axis=1)
    out = dict(kind="split_conformal_plam", tag=tag, source=source, seed=args.seed, n_cal=int(len(cal)),
               n_eval=int(len(ev)), smult=smult, lam=lam, med=med,
               final_stage=rec["report"].get("final_stage"),
               spearman_P_err=float(np.corrcoef(np.argsort(np.argsort(P)), np.argsort(np.argsort(err)))[0, 1]),
               pearson_P_err=float(np.corrcoef(P, err)[0, 1]))
    for alpha in alphas:
        q_p = conformal_quantile(err[cal] / P[cal], alpha)
        q_r = conformal_quantile(err[cal], alpha)
        out[f"a{alpha:g}"] = dict(
            target=1 - alpha,
            plam=dict(q=float(q_p), coverage=float(np.mean(err[ev] <= q_p * P[ev])),
                      mean_width=float(q_p * P[ev].mean())),
            raw=dict(q=float(q_r), coverage=float(np.mean(err[ev] <= q_r)), mean_width=float(q_r)))
        print(f"s{args.seed} {tag} alpha={alpha}: P_lambda-scaled cover {out[f'a{alpha:g}']['plam']['coverage']:.4f}"
              f"  raw cover {out[f'a{alpha:g}']['raw']['coverage']:.4f}", flush=True)
    json.dump(out, open(RUNS / f"{tag}_uq_plam.json", "w"), indent=1)
    np.save(RUNS / f"{tag}_plam_test.npy", P.astype(np.float32))
print(f"[uq_plam_s{args.seed}] complete", flush=True)
