"""Out-of-fold KRR predictions on the training pool (for the refiner net).

4-fold CV with the tuned kernel (smult 1.0 x median, lam 1e-6). The default
uses each fold's fitting-target mean. --target-centering pooled reproduces
the historical field construction, including its held-out-target reuse.
Val/test predictions come from the full 19000-sample fit (train_krr.py).
Saves krr_oof_train.npy in the canonical train order and a metadata sidecar.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np, sys, time
from scipy.linalg import cho_factor, cho_solve
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS


def target_mean(y, fit, mode):
    if mode == "fold-local":
        return y[fit].mean(0)
    if mode == "pooled":
        return y.mean(0)
    raise ValueError("Unknown target-centering mode")


p = argparse.ArgumentParser()
p.add_argument("--target-centering", choices=("fold-local", "pooled"), default="fold-local")
args = p.parse_args()
SMULT, LAM = 1.0, 1e-6
loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
X = loads[tr].astype(np.float64)
Y = stress[tr].reshape(len(tr), -1).astype(np.float64)
mu = X.mean(0); sd = X.std(0) + 1e-12
Xn = (X - mu) / sd

def sqdist(A, B):
    return np.maximum((A*A).sum(1)[:,None] + (B*B).sum(1)[None,:] - 2*A@B.T, 0)
def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r
    return (1 + a + (5/3)*r2)*np.exp(-a)

rng = np.random.default_rng(0)
sub = rng.choice(len(Xn), 2000, replace=False)
D2s = sqdist(Xn[sub], Xn[sub])
med = np.sqrt(np.median(D2s[np.triu_indices(2000, 1)]))
s = SMULT * med
print("sigma =", s, flush=True)

n = len(Xn)
folds = np.array_split(rng.permutation(n), 4)
oof = np.empty_like(Y)
for f, hold in enumerate(folds):
    t0 = time.time()
    fit = np.setdiff1d(np.arange(n), hold)
    K = m52(sqdist(Xn[fit], Xn[fit]), s)
    K.flat[::len(fit)+1] += LAM * len(fit)
    c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    muY = target_mean(Y, fit, args.target_centering)
    a1 = cho_solve(c, Y[fit] - muY, check_finite=False)
    del K, c
    Kq = m52(sqdist(Xn[hold], Xn[fit]), s)
    oof[hold] = Kq @ a1 + muY
    del Kq, a1
    print(f"fold {f}: {time.time()-t0:.0f}s  oof rel-L2 {rel_l2(oof[hold], Y[hold]):.4f}", flush=True)

print("total OOF rel-L2:", rel_l2(oof, Y), flush=True)
np.save(RUNS / "krr_oof_train.npy", oof.astype(np.float32))
metadata = dict(target_centering=args.target_centering, ntrain=n, folds=4,
                smult=SMULT, lam=LAM, scale=float(s),
                split_seed=int(os.environ.get("NMKC_SPLIT_SEED", "0")),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                field_sha256=hashlib.sha256((RUNS / "krr_oof_train.npy").read_bytes()).hexdigest(),
                training_index_sha256=hashlib.sha256(tr.tobytes()).hexdigest())
(RUNS / "krr_oof_train.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("saved", flush=True)
