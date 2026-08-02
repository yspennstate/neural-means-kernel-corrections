"""Measure the certificate constant kappa for one structmech seed.

kappa = sup over test points and grid configurations of ||c_gamma(u)||_1,
where c_gamma(u) = (K_gamma + n lambda I)^{-1} k_gamma(X, u) are the
correction weights of Corollary cor:certificate. The grid and length-scale
convention mirror campaign/correct_stack.py exactly (scale {1,2,4} x median
distance on a 2000-point subsample drawn with the pipeline seed, nuggets
{1e-6,1e-5,1e-3}); the supremum over u is taken over a fixed 500-sample test
subset, which makes the reported value a measured lower bound on the
population supremum and is labeled as such.

Environment: NMKC_ROOT, NMKC_SEED, TASK_ID, NMKC_THREADS.
"""
import json, os, pathlib, sys, time
import fcntl
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = pathlib.Path(os.environ["NMKC_ROOT"])
SEED = int(os.environ["NMKC_SEED"])
TASK_ID = os.environ.get("TASK_ID", f"kappa_s{SEED}")
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("NMKC_DATA", str(ROOT / "data" / "structmech"))
os.environ["NMKC_SPLIT_SEED"] = str(SEED)
from common import load_arrays, canonical_split

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=SEED)
X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd
rng = np.random.default_rng(SEED)                       # pipe-seed convention
sub = rng.choice(len(Xt), min(2000, len(Xt)), replace=False)


def sqdist(A, B):
    return np.maximum((A*A).sum(1)[:, None] + (B*B).sum(1)[None, :] - 2*A@B.T, 0)


def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r
    return (1 + a + (5/3)*r2)*np.exp(-a)


med = np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)]))
ute = rng.choice(len(te), 500, replace=False)
Xe = (loads[te[ute]].astype(np.float64) - mu) / sd
n = len(Xt)

t0 = time.time()
rows = []
kappa = 0.0
lock = open(ROOT / ".gram.lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)
try:
    D2tt = sqdist(Xt, Xt)
    D2et = sqdist(Xe, Xt)
    for smult in (1.0, 2.0, 4.0):
        K = m52(D2tt, smult*med)
        Ke = m52(D2et, smult*med)
        for lam in (1e-6, 1e-5, 1e-3):
            Kr = K.copy(); Kr.flat[::n+1] += lam*n
            c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
            Cmat = cho_solve(c, Ke.T, check_finite=False)      # (n, 500)
            l1 = np.abs(Cmat).sum(0)
            rows.append(dict(smult=smult, lam=lam,
                             kappa_max=float(l1.max()), kappa_med=float(np.median(l1))))
            kappa = max(kappa, float(l1.max()))
            del Kr, c, Cmat
            print(f"smult={smult} lam={lam:g} kappa_max={rows[-1]['kappa_max']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()

out = dict(task_id=TASK_ID, kind="kappa", seed=SEED, med=float(med),
           n_train=n, n_test_probe=500, kappa=kappa, per_config=rows,
           note="supremum over a 500-point test probe; a measured lower bound",
           minutes=round((time.time()-t0)/60, 1))
res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, res)
print(f"[{TASK_ID}] kappa={kappa:.3f}", flush=True)
