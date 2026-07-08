"""UQ and spectral analysis for the hybrid pipeline.

Given a finished hybrid run (tag), refit the input-space residual KRR with the
tuned hyperparameters and compute:
 1. GP posterior sd (power function) P_lam(u) on the test set; correlation and
    calibration against realized errors of the corrected predictor.
 2. RKHS norms alpha^T K alpha: direct fit vs residual fit (theory Thm 3b).
 3. Gram spectrum (load space + deep features, n_sub subsample), empirical
    d_eff(lambda) curve, CV-chosen lambda marked.
Saves arrays + a JSON summary for figures.

Usage: python uq_spectra.py --tag hyb [--nsub 6000]
"""
import argparse, json, time, sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS

p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, default="hyb")
p.add_argument("--nsub", type=int, default=6000)
args = p.parse_args()

cfg = json.load(open(RUNS / f"{args.tag}.json"))
best = cfg["report"]["plus_input_krr"]
smult, lam = best["smult"], best["lam"]
print("using tuned input-KRR params:", smult, lam, flush=True)

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if cfg.get("ntrain") and cfg["ntrain"] != len(tr): tr = tr[:cfg["ntrain"]]
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
E_tr = np.load(RUNS / f"{args.tag}_stack_tr.npy").astype(np.float64)
E_va = np.load(RUNS / f"{args.tag}_stack_va.npy").astype(np.float64)
E_te = np.load(RUNS / f"{args.tag}_stack_te.npy").astype(np.float64)

X = loads[tr].astype(np.float64)
mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd
Xv = (loads[va].astype(np.float64) - mu) / sd
Xe = (loads[te].astype(np.float64) - mu) / sd

def sqdist(A, B):
    aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)

def matern52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)

t0 = time.time()
D2 = sqdist(Xt, Xt)
rng = np.random.default_rng(0)
subm = rng.choice(len(Xt), size=min(2000, len(Xt)), replace=False)
med = np.sqrt(np.median(D2[np.ix_(subm, subm)][np.triu_indices(len(subm), 1)]))
s = smult * med
n = len(Xt)
K = matern52(D2, s)
A = K + lam * n * np.eye(n)
c, low = cho_factor(A, lower=True, check_finite=False)
print(f"chol: {time.time()-t0:.0f}s", flush=True)

R_tr = Ytr - E_tr
alpha_res = cho_solve((c, low), R_tr, check_finite=False)
alpha_dir = cho_solve((c, low), Ytr - Ytr.mean(0), check_finite=False)

# RKHS norms of the fitted interpolants (dual quadratic forms)
norm_res = float(np.sum(alpha_res * (K @ alpha_res)))
norm_dir = float(np.sum(alpha_dir * (K @ alpha_dir)))
print(f"RKHS^2 norms: direct {norm_dir:.4e}  residual {norm_res:.4e}  ratio {norm_dir/norm_res:.2f}", flush=True)

# posterior sd (power function with nugget) on test, chunked
L = np.tril(c) if low else np.tril(c.T)
kuu = matern52(np.zeros((1,)), s)[0]  # = 1.0
P = np.empty(len(Xe))
pred_corr = np.empty_like(E_te)
CH = 2000
t1 = time.time()
for k in range(0, len(Xe), CH):
    Kq = matern52(sqdist(Xt, Xe[k:k+CH]), s)          # (n, ch)
    W = solve_triangular(L, Kq, lower=True, check_finite=False)
    P[k:k+CH] = np.sqrt(np.maximum(kuu - (W * W).sum(0), 0.0))
    pred_corr[k:k+CH] = E_te[k:k+CH] + Kq.T @ alpha_res
print(f"posterior sd on test: {time.time()-t1:.0f}s", flush=True)

err_abs = np.linalg.norm(pred_corr - Yte, axis=1)                  # absolute L2 error (theory quantity)
err = err_abs / np.linalg.norm(Yte, axis=1)                       # relative error (benchmark metric)
from scipy.stats import spearmanr
pear_abs = float(np.corrcoef(P, err_abs)[0, 1])
spear_abs = float(spearmanr(P, err_abs).statistic)
pear = float(np.corrcoef(P, err)[0, 1])
spear = float(spearmanr(P, err).statistic)
ynorm = np.linalg.norm(Yte, axis=1)
pear_Pnorm = float(np.corrcoef(P, ynorm)[0, 1])
print(f"corr(P, ABS err): pearson {pear_abs:.3f}  spearman {spear_abs:.3f}", flush=True)
print(f"corr(P, REL err): pearson {pear:.3f}  spearman {spear:.3f}", flush=True)
print(f"corr(P, output norm): pearson {pear_Pnorm:.3f}  (explains rel-vs-abs sign flip)", flush=True)

# calibration by deciles of P (absolute error, the theory-consistent quantity)
qs = np.quantile(P, np.linspace(0, 1, 11))
dec_err = [float(err_abs[(P >= qs[i]) & (P <= qs[i + 1])].mean()) for i in range(10)]
dec_P = [float(P[(P >= qs[i]) & (P <= qs[i + 1])].mean()) for i in range(10)]

# conformal-style scaling: on val, compute ratio err_val / P_val quantile
Kqv = matern52(sqdist(Xt, Xv), s)
Wv = solve_triangular(L, Kqv, lower=True, check_finite=False)
Pv = np.sqrt(np.maximum(kuu - (Wv * Wv).sum(0), 0.0))
pred_va = E_va + Kqv.T @ alpha_res
err_va_abs = np.linalg.norm(pred_va - Yva, axis=1)
# exact split-conformal quantile: the ceil((1-alpha)(m+1))-th smallest score,
# the order statistic that carries the finite-sample coverage guarantee
scores = np.sort(err_va_abs / np.maximum(Pv, 1e-12))
m = len(scores)
scale90 = float(scores[int(np.ceil(0.9 * (m + 1))) - 1])
cover90 = float(np.mean(np.linalg.norm(pred_corr - Yte, axis=1) <= scale90 * P))
print(f"conformal 90% scaling: empirical coverage on test = {cover90:.3f}", flush=True)

# spectrum of Gram (loads) + d_eff curve
nsub = min(args.nsub, n)
sub = rng.choice(n, size=nsub, replace=False)
evals_load = np.linalg.eigvalsh(matern52(D2[np.ix_(sub, sub)], s))
lam_grid = np.logspace(-9, -1, 40)
d_eff = [float(np.sum(evals_load / (evals_load + l * nsub))) for l in lam_grid]

feat_path = RUNS / f"{args.tag}_feat_tr.npy"
evals_feat = None
if feat_path.exists():
    Ftr = np.load(feat_path).astype(np.float64)
    Fs = (Ftr - Ftr.mean(0)) / (Ftr.std(0) + 1e-12)
    Fsub = Fs[sub]
    D2f = sqdist(Fsub, Fsub)
    medf = np.sqrt(np.median(D2f[np.triu_indices(nsub, 1)][::7]))
    evals_feat = np.linalg.eigvalsh(matern52(D2f, medf))

np.savez(RUNS / f"{args.tag}_uq_spectra.npz",
         P=P, err=err, Pv=Pv, dec_P=np.array(dec_P), dec_err=np.array(dec_err),
         evals_load=evals_load, evals_feat=(evals_feat if evals_feat is not None else np.zeros(1)),
         lam_grid=lam_grid, d_eff=np.array(d_eff))
save_run(f"{args.tag}_uq", dict(kind="uq", pearson=pear, spearman=spear,
         pearson_abs=pear_abs, spearman_abs=spear_abs, pearson_P_ynorm=pear_Pnorm,
         cover90=cover90, scale90=scale90, norm_dir=norm_dir, norm_res=norm_res,
         ratio=norm_dir / norm_res, smult=smult, lam=lam, med=float(med), nsub=nsub,
         test_rel_l2_corrected=rel_l2(pred_corr, Yte)))
print("saved analysis artifacts", flush=True)
