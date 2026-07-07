"""Exact kernel ridge regression on the 41-dim load -> 1681-dim stress field.

Full f64 solve on CPU (BLAS): distance matrix via Gram trick, Matern-5/2 /
RBF kernels, grid search over length-scale multiplier and nugget on the
validation split. Saves the best config's predictions and dual coefficients
for later ensembling / residual analysis.

Usage: python train_krr.py [--ntrain 0 (=full pool minus val) | 1250] [--tag krr]
"""
import argparse, json, time, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA

p = argparse.ArgumentParser()
p.add_argument("--ntrain", type=int, default=0, help="0 = full train pool minus val (19000)")
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--kernel", type=str, default="matern52", choices=["matern52", "rbf", "matern32"])
p.add_argument("--tag", type=str, default="krr")
p.add_argument("--save_pred", type=int, default=1)
p.add_argument("--smults", type=str, default="0.5,0.75,1.0,1.5,2.0")
p.add_argument("--lams", type=str, default="1e-8,1e-6,1e-4")
args = p.parse_args()

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if args.ntrain > 0:
    tr = np.load(DATA / "idx_train.npy")[:args.ntrain]
    if args.lowval > 0:
        va = tr[-args.lowval:]; tr = tr[:-args.lowval]
print(f"train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)

Xtr = loads[tr].astype(np.float64); Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Xva = loads[va].astype(np.float64); Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Xte = loads[te].astype(np.float64); Yte = stress[te].reshape(len(te), -1).astype(np.float64)

mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-12
Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
muY = Ytr.mean(0)
Ytr_c = Ytr - muY

def sqdist(A, B):
    aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
    D2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(D2, 0.0, out=D2)
    return D2

def kernel(D2, s):
    r2 = D2 / (s * s)
    if args.kernel == "rbf":
        return np.exp(-0.5 * r2)
    r = np.sqrt(r2)
    if args.kernel == "matern32":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)

t0 = time.time()
D2_tt = sqdist(Xtr, Xtr)
D2_vt = sqdist(Xva, Xtr)
print(f"distance matrices: {time.time()-t0:.1f}s", flush=True)

sub = np.random.default_rng(0).choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)
med = np.sqrt(np.median(D2_tt[np.ix_(sub, sub)][np.triu_indices(len(sub), 1)]))
print(f"median heuristic sigma = {med:.3f}", flush=True)

from scipy.linalg import cho_factor, cho_solve
results = []
best = dict(val=1e9)
n = len(Xtr)
for smult in [float(x) for x in args.smults.split(",")]:
    s = smult * med
    Kv = kernel(D2_vt, s)
    for lam in [float(x) for x in args.lams.split(",")]:
        t1 = time.time()
        Kreg = kernel(D2_tt, s)                     # fresh buffer, factor in place
        Kreg.flat[::n + 1] += lam * n
        try:
            c = cho_factor(Kreg, lower=True, check_finite=False, overwrite_a=True)
            alpha = cho_solve(c, Ytr_c, check_finite=False)
        except np.linalg.LinAlgError:
            print(f"  chol failed s={smult} lam={lam}"); del Kreg; continue
        del c, Kreg
        e_va = rel_l2(Kv @ alpha + muY, Yva)
        print(f"  smult={smult:4.2f} lam={lam:.0e}  val {e_va:.4f}  [{time.time()-t1:.0f}s]", flush=True)
        results.append(dict(smult=smult, lam=lam, val=e_va))
        if e_va < best["val"]:
            best = dict(val=e_va, smult=smult, lam=lam, s=s)
        del alpha
    del Kv

print(f"\nbest: smult={best['smult']} lam={best['lam']} val={best['val']:.4f}", flush=True)
# refit best and evaluate test, chunked
s = best["s"]
Kreg = kernel(D2_tt, s)
Kreg.flat[::n + 1] += best["lam"] * n
c = cho_factor(Kreg, lower=True, check_finite=False, overwrite_a=True)
alpha = cho_solve(c, Ytr_c, check_finite=False)
del c, Kreg, D2_tt
preds = []
CH = 4000
for k in range(0, len(Xte), CH):
    Kq = kernel(sqdist(Xte[k:k+CH], Xtr), s)
    preds.append(Kq @ alpha + muY)
Yhat_te = np.vstack(preds)
e_te = rel_l2(Yhat_te, Yte)
print(f"TEST rel-L2 = {e_te:.4f}", flush=True)

name = f"{args.tag}_{args.kernel}_n{len(tr)}"
if args.save_pred:
    np.save(RUNS / f"{name}_pred_test.npy", Yhat_te.astype(np.float32))
    Kv = kernel(D2_vt, s)
    np.save(RUNS / f"{name}_pred_val.npy", (Kv @ alpha + muY).astype(np.float32))
save_run(name, dict(kind="krr", args=vars(args), grid=results, best=dict(smult=best["smult"], lam=best["lam"]),
                    val=best["val"], test=e_te, minutes=(time.time()-t0)/60))
print("saved", name, flush=True)
