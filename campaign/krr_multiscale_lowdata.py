"""Multiscale (staged) kernel ridge under the strict low-data protocol.

The stage-wise construction behind the low-data kernel row: train 1000,
validation 250 carved from the first 1250 samples, up to four boosting stages,
each a validated Matern-5/2 fit to the current residual; a stage is kept only
if it improves validation error. Deterministic given the data. Writes
runs/krr_multiscale_lowdata.json and, when TASK_ID is set, the campaign
result file.
"""
import json, os, pathlib, sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load_arrays, canonical_split, rel_l2, save_run, DATA

loads, stress = load_arrays()
_, _, te = canonical_split(n_val=1000, seed=0)
pool = np.load(DATA / "idx_train.npy")[:1250]
tr, va = pool[:1000], pool[1000:]

X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd
Xv = (loads[va].astype(np.float64) - mu) / sd
Xe = (loads[te].astype(np.float64) - mu) / sd
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
muY = Ytr.mean(0)


def sqdist(A, B):
    return np.maximum((A*A).sum(1)[:, None] + (B*B).sum(1)[None, :] - 2*A@B.T, 0)


def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r
    return (1 + a + (5/3)*r2)*np.exp(-a)


D2tt = sqdist(Xt, Xt); D2vt = sqdist(Xv, Xt); D2et = sqdist(Xe, Xt)
med = np.sqrt(np.median(D2tt[np.triu_indices(len(Xt), 1)]))
n = len(Xt)

P_tr = np.tile(muY, (n, 1)); P_va = np.tile(muY, (len(Xv), 1)); P_te = np.tile(muY, (len(Xe), 1))
stages = []
for stage in range(4):
    R = Ytr - P_tr
    best = None
    for smult in [0.25, 0.5, 1.0, 2.0, 3.0]:
        K = m52(D2tt, smult*med); Kv = m52(D2vt, smult*med)
        for lam in [1e-8, 1e-6, 1e-4]:
            Kr = K.copy(); Kr.flat[::n+1] += lam*n
            c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
            a1 = cho_solve(c, R, check_finite=False)
            ev = rel_l2(P_va + Kv@a1, Yva)
            if best is None or ev < best[0]:
                best = (ev, smult, lam, a1)
    ev, smult, lam, a1 = best
    if ev >= rel_l2(P_va, Yva) - 1e-5:
        break
    K = m52(D2tt, smult*med); Kv = m52(D2vt, smult*med); Ke = m52(D2et, smult*med)
    P_tr = P_tr + K@a1; P_va = P_va + Kv@a1; P_te = P_te + Ke@a1
    stages.append(dict(smult=smult, lam=lam,
                       val=round(rel_l2(P_va, Yva), 4), test=round(rel_l2(P_te, Yte), 4)))
    print(f"stage {len(stages)}: smult={smult} lam={lam:.0e}  "
          f"val {stages[-1]['val']}  test {stages[-1]['test']}", flush=True)

out = dict(kind="krr", protocol="strict n=1250 (train 1000, val 250)",
           stages=stages, val=stages[-1]["val"], test=stages[-1]["test"],
           note="multiscale boosted KRR (campaign/krr_multiscale_lowdata.py)")
save_run("krr_multiscale_lowdata", out)
if os.environ.get("TASK_ID"):
    root = pathlib.Path(os.environ["NMKC_ROOT"])
    res = root / "results" / (os.environ["TASK_ID"] + ".json")
    res.parent.mkdir(exist_ok=True)
    tmp = res.with_suffix(".tmp")
    json.dump(dict(out, task_id=os.environ["TASK_ID"]), open(tmp, "w"), indent=1)
    os.replace(tmp, res)
print("final:", out["val"], out["test"], flush=True)
