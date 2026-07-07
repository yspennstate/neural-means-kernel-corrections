"""KRR test error vs number of training samples (fixed tuned kernel).
Powers the error-vs-N figure. Fast: one solve per N."""
import numpy as np, sys, time, json
from scipy.linalg import cho_factor, cho_solve
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS, save_run, DATA

SMULT, LAM = 1.0, 1e-6
loads, stress = load_arrays()
tr_all, va, te = canonical_split(n_val=1000, seed=0)
pool = np.load(DATA / "idx_train.npy")
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

def sqdist(A, B):
    return np.maximum((A*A).sum(1)[:,None] + (B*B).sum(1)[None,:] - 2*A@B.T, 0)
def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r
    return (1 + a + (5/3)*r2)*np.exp(-a)

out = {}
for N in [500, 1000, 2500, 5000, 10000, 19000]:
    t0 = time.time()
    tr = pool[:N] if N < 19000 else tr_all
    X = loads[tr].astype(np.float64)
    mu = X.mean(0); sd = X.std(0) + 1e-12
    Xt = (X - mu) / sd
    Xe = (loads[te].astype(np.float64) - mu) / sd
    Y = stress[tr].reshape(len(tr), -1).astype(np.float64)
    muY = Y.mean(0)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), min(2000, len(Xt)), replace=False)
    D2s = sqdist(Xt[sub], Xt[sub])
    med = np.sqrt(np.median(D2s[np.triu_indices(len(sub), 1)]))
    K = m52(sqdist(Xt, Xt), SMULT * med)
    K.flat[::len(K)+1] += LAM * len(K)
    c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    a1 = cho_solve(c, Y - muY, check_finite=False)
    del K, c
    preds = []
    for k in range(0, len(Xe), 4000):
        preds.append(m52(sqdist(Xe[k:k+4000], Xt), SMULT * med) @ a1 + muY)
    e = rel_l2(np.vstack(preds), Yte)
    out[N] = e
    print(f"N={N:6d}  test {e:.4f}  [{time.time()-t0:.0f}s]", flush=True)

save_run("krr_scaling", dict(kind="krr_scaling", smult=SMULT, lam=LAM, errors=out))
print("saved", flush=True)
