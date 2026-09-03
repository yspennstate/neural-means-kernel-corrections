"""Fitted-norm diagnostics for the structural-mechanics correction, recomputed after review.

For one seed of the complete-schedule campaign: the ridge fit of the same validated correction kernel
(scale multiplier and nugget from the seed's hpix_corr record, median length scale as the correction drew
it) to (a) the raw stress fields and (b) the six-member stack residuals on the training split. Reports the
fitted squared RKHS norm alpha^T K alpha, the regularization term n lambda ||alpha||^2, the energy of what is
regressed, the norm per unit energy, the raw and energy-normalized ratios between (a) and (b), the effective
dimension, and the test-point power function summary. Everything is a finite-design diagnostic; the text
describes it as such.

    python sm_norm_check.py --seed 0 [--tag hpix]
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="hpix")
p.add_argument("--out", default="")
args = p.parse_args()

ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2")))
CODE = pathlib.Path(os.environ.get("NMKC_CODE", ROOT / "code"))
os.environ.setdefault("NMKC_DATA", str(ROOT / "data" / "structmech"))
os.environ["NMKC_SPLIT_SEED"] = str(args.seed)
sys.path.insert(0, str(CODE))
from common import load_arrays, canonical_split  # noqa: E402

RUNS = ROOT / "seeds" / f"sm_s{args.seed}" / "runs"
rec = json.load(open(RUNS / f"{args.tag}_corr.json"))["report"]["plus_corr"]
smult, lam = float(rec["smult"]), float(rec["lam"])

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=args.seed)
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
E_tr = np.load(RUNS / f"{args.tag}_stack_tr.npy").astype(np.float64)
R = Ytr - E_tr
X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd; Xe = (loads[te].astype(np.float64) - mu) / sd
n = len(Xt)


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)


t0 = time.time()
rng = np.random.default_rng(args.seed)
sub = rng.choice(n, min(2000, n), replace=False)
med = float(np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)])))
s = smult * med
K = np.empty((n, n))
for k in range(0, n, 1000):
    K[k:k + 1000] = m52(sqdist(Xt[k:k + 1000], Xt), s)
Kr = K.copy(); Kr.flat[::n + 1] += lam * n
c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
out = dict(seed=args.seed, tag=args.tag, smult=smult, lam=lam, med=med, n=n, q=int(Ytr.shape[1]))
for name, T in (("raw_targets", Ytr), ("stack_residuals", R)):
    a = cho_solve(c, T, check_finite=False)
    aKa = float(np.einsum("nd,nd->", a, K @ a)); aT = float(np.einsum("nd,nd->", a, T))
    energy = float((T * T).sum())
    out[name] = dict(aKa=aKa, aT=aT, reg_term=float(lam * n * (a * a).sum()), energy=energy, aKa_per_energy=aKa / energy)
    print(f"{name}: aKa {aKa:.4g}  aT {aT:.4g}  energy {energy:.4g}  per energy {aKa / energy:.4g} [{time.time() - t0:.0f}s]", flush=True)
out["ratio_aKa"] = out["raw_targets"]["aKa"] / out["stack_residuals"]["aKa"]
out["ratio_aT"] = out["raw_targets"]["aT"] / out["stack_residuals"]["aT"]
out["energy_ratio"] = out["raw_targets"]["energy"] / out["stack_residuals"]["energy"]
out["ratio_aKa_normalized"] = out["raw_targets"]["aKa_per_energy"] / out["stack_residuals"]["aKa_per_energy"]
# effective dimension and test-point power function of the same kernel and nugget
ev = np.maximum(np.linalg.eigvalsh(K), 0.0)
out["deff"] = float((ev / (ev + n * lam)).sum())
P2 = np.empty(len(Xe))
for k in range(0, len(Xe), 2000):
    Ku = m52(sqdist(Xt, Xe[k:k + 2000]), s)
    V = solve_triangular(c[0], Ku, lower=True, check_finite=False)
    P2[k:k + 2000] = 1.0 - (V * V).sum(0)
P = np.sqrt(np.maximum(P2, 0.0))
out["P_test"] = dict(mean=float(P.mean()), median=float(np.median(P)), q10=float(np.quantile(P, 0.1)), q90=float(np.quantile(P, 0.9)))
out["minutes"] = round((time.time() - t0) / 60, 1)
print(f"ratio aKa {out['ratio_aKa']:.1f}  ratio aT {out['ratio_aT']:.1f}  energy ratio {out['energy_ratio']:.1f}  "
      f"normalized ratio {out['ratio_aKa_normalized']:.3f}  deff {out['deff']:.0f}  P median {out['P_test']['median']:.4f}", flush=True)
path = pathlib.Path(args.out or (ROOT / "results" / f"sm_norm_check_{args.tag}_s{args.seed}.json"))
json.dump(out, open(path, "w"), indent=1); print("wrote", path)
