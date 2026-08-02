"""Residual Matern correction of a saved stacked prediction.

Reads {tag}_stack_tr/va/te.npy (written by stack_perpixel.py), fits the
validated Matern-5/2 correction of the stack residual on the loads -- the
protocol of stack_correct.py: median length scale from a 2000-point subsample,
scale grid {1,2,4} x median, nugget grid {1e-6,1e-5,1e-3}, tuned against
validation on an 8000-sample subsample, winner refit on the full train set.
The final stage (corrected or plain stack) is chosen on validation, as before.
Subsample draws are seeded by NMKC_PIPE_SEED so campaign seeds differ.

Writes {tag}_corr.json and {tag}_corr_pred_test.npy.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS

p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, default="hpix")
args = p.parse_args()

PIPE_SEED = int(os.environ.get("NMKC_PIPE_SEED", "0"))

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

E_tr = np.load(RUNS / f"{args.tag}_stack_tr.npy").astype(np.float64)
E_va = np.load(RUNS / f"{args.tag}_stack_va.npy").astype(np.float64)
E_te = np.load(RUNS / f"{args.tag}_stack_te.npy").astype(np.float64)

X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd
Xv = (loads[va].astype(np.float64) - mu) / sd
Xe = (loads[te].astype(np.float64) - mu) / sd

def sqdist(A, B):
    return np.maximum((A*A).sum(1)[:, None] + (B*B).sum(1)[None, :] - 2*A@B.T, 0)

def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r
    return (1 + a + (5/3)*r2)*np.exp(-a)

n = len(Xt)
rng = np.random.default_rng(PIPE_SEED)
sub = rng.choice(n, min(2000, n), replace=False)
med = np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)]))
R = Ytr - E_tr
best = dict(val=np.inf)
t0 = time.time()
nsub = min(8000, n)
tsub = rng.choice(n, nsub, replace=False)
D2s = sqdist(Xt[tsub], Xt[tsub]); D2vs = sqdist(Xv, Xt[tsub])
for smult in [1.0, 2.0, 4.0]:
    Ks = m52(D2s, smult*med); Kvs = m52(D2vs, smult*med)
    for lam in [1e-6, 1e-5, 1e-3]:
        Kr = Ks.copy(); Kr.flat[::nsub+1] += lam*nsub
        try:
            c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
        except Exception:
            continue
        al = cho_solve(c, R[tsub], check_finite=False)
        e = rel_l2(E_va + Kvs @ al, Yva)
        if e < best["val"]:
            best = dict(val=e, smult=smult, lam=lam)
        del Kr, c, al
print(f"corr tune (sub {nsub}): smult={best['smult']} lam={best['lam']} "
      f"val={best['val']:.4f} [{time.time()-t0:.0f}s]", flush=True)

K = m52(sqdist(Xt, Xt), best["smult"]*med); K.flat[::n+1] += best["lam"]*n
c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
al = cho_solve(c, R, check_finite=False)
corr_te = np.zeros_like(E_te)
for k in range(0, len(Xe), 4000):
    corr_te[k:k+4000] = m52(sqdist(Xe[k:k+4000], Xt), best["smult"]*med) @ al
E2_te = E_te + corr_te
E2_va = E_va + m52(sqdist(Xv, Xt), best["smult"]*med) @ al

stack_val, stack_te = rel_l2(E_va, Yva), rel_l2(E_te, Yte)
corr_val, corr_te_err = rel_l2(E2_va, Yva), rel_l2(E2_te, Yte)
final = "plus_corr" if corr_val < stack_val else "stack"
report = dict(kind="hybrid", runs=[args.tag], pipe_seed=PIPE_SEED,
              report=dict(stack=dict(val=stack_val, test=stack_te),
                          plus_corr=dict(val=corr_val, test=corr_te_err,
                                         smult=best["smult"], lam=best["lam"]),
                          final_stage=final,
                          final_test=corr_te_err if final == "plus_corr" else stack_te),
              ntrain=n)
np.save(RUNS / f"{args.tag}_corr_pred_test.npy",
        (E2_te if final == "plus_corr" else E_te).astype(np.float32))
np.save(RUNS / f"{args.tag}_corr_pred_val.npy",
        (E2_va if final == "plus_corr" else E_va).astype(np.float32))
save_run(f"{args.tag}_corr", report)
print(f"STACK {stack_te:.4f} -> CORR {corr_te_err:.4f} (final {final})", flush=True)
