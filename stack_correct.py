"""Architecture-agnostic stack + residual kernel correction from saved
predictions. Members are given as prediction-run tags (each has _predtr/_predva/
_predte.npy from gen_preds.py); the standalone KRR is added as a member via its
saved arrays. Convex stacking weights are fit on validation; the stack residual
is corrected by a validated Matern kernel on the loads. CPU only.

Usage: python stack_correct.py --members mlp_s0_w1024_d4_n19000_mir,mlpR_s0_w1024_d4
        [--krr 1] [--ntrain 0] [--lowval 0] [--tag hstack]
"""
import argparse, sys, json, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS, DATA

p = argparse.ArgumentParser()
p.add_argument("--members", type=str, required=True)
p.add_argument("--krr", type=int, default=1, help="add standalone KRR as a member")
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowval", type=int, default=0)
p.add_argument("--tag", type=str, default="hstack")
args = p.parse_args()

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
if args.ntrain > 0:
    tr = np.load(DATA / "idx_train.npy")[:args.ntrain]
    if args.lowval > 0:
        va = tr[-args.lowval:]; tr = tr[:-args.lowval]
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

Ptr, Pva, Pte, names = [], [], [], []
for m in args.members.split(","):
    m = m.strip()
    Ptr.append(np.load(RUNS / f"{m}_predtr.npy").astype(np.float64))
    Pva.append(np.load(RUNS / f"{m}_predva.npy").astype(np.float64))
    Pte.append(np.load(RUNS / f"{m}_predte.npy").astype(np.float64))
    names.append(m)
if args.krr and args.ntrain == 0:
    Ptr.append(np.load(RUNS / "krr_oof_train.npy").astype(np.float64))
    Pva.append(np.load(RUNS / "krr_full_matern52_n19000_pred_val.npy").astype(np.float64))
    Pte.append(np.load(RUNS / "krr_full_matern52_n19000_pred_test.npy").astype(np.float64))
    names.append("krr")

for m, pt in zip(names, Pte):
    print(f"  member {m}: test {rel_l2(pt, Yte):.4f}  val {rel_l2(Pva[names.index(m)], Yva):.4f}", flush=True)

# convex stacking weights on val (projected gradient on the simplex via softmax)
Pv = np.stack(Pva); Yv = Yva
w = np.zeros(len(names))
def stack_err(logits, PP, YY):
    ww = np.exp(logits - logits.max()); ww /= ww.sum()
    pred = np.einsum("m,mnd->nd", ww, PP)
    return rel_l2(pred, YY), ww
# simple coordinate search + smoothing is enough for few members
best = (1e9, None)
rng = np.random.default_rng(0)
logit = np.zeros(len(names))
step = 0.5
for it in range(2000):
    cand = logit + rng.normal(0, step, len(names))
    e, ww = stack_err(cand, Pv, Yv)
    e0, _ = stack_err(logit, Pv, Yv)
    if e < e0:
        logit = cand
    step *= 0.999
_, w = stack_err(logit, Pv, Yv)
print("stack weights:", {n: round(float(x), 3) for n, x in zip(names, w)}, flush=True)

E_tr = np.einsum("m,mnd->nd", w, np.stack(Ptr))
E_va = np.einsum("m,mnd->nd", w, np.stack(Pva))
E_te = np.einsum("m,mnd->nd", w, np.stack(Pte))
report = dict(members={n: dict(test=rel_l2(pt, Yte)) for n, pt in zip(names, Pte)},
              stack=dict(val=rel_l2(E_va, Yva), test=rel_l2(E_te, Yte)), weights=dict(zip(names, w.tolist())))
print(f"STACK: val {report['stack']['val']:.4f}  test {report['stack']['test']:.4f}", flush=True)

# residual kernel correction on loads
X = loads[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd; Xv = (loads[va].astype(np.float64) - mu) / sd; Xe = (loads[te].astype(np.float64) - mu) / sd
def sqdist(A, B): return np.maximum((A*A).sum(1)[:,None] + (B*B).sum(1)[None,:] - 2*A@B.T, 0)
def m52(D2, s):
    r2 = D2/(s*s); r = np.sqrt(r2); a = np.sqrt(5)*r; return (1 + a + (5/3)*r2)*np.exp(-a)
n = len(Xt); rng = np.random.default_rng(0)
sub = rng.choice(n, min(2000, n), replace=False)
med = np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)]))
R = Ytr - E_tr
best = dict(val=1e9)
t0 = time.time()
D2v = sqdist(Xv, Xt)
# tune on a subsample (hybrid.py convention: lam scales with the fit size),
# refit the winner on the full train set below
nsub = min(8000, n)
tsub = rng.choice(n, nsub, replace=False)
D2s = sqdist(Xt[tsub], Xt[tsub]); D2vs = sqdist(Xv, Xt[tsub])
for smult in [1.0, 2.0, 4.0]:
    Ks = m52(D2s, smult*med)
    Kvs = m52(D2vs, smult*med)
    for lam in [1e-6, 1e-5, 1e-3]:
        Kr = Ks.copy(); Kr.flat[::nsub+1] += lam*nsub
        try: c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
        except Exception: continue
        al = cho_solve(c, R[tsub], check_finite=False)
        e = rel_l2(E_va + Kvs @ al, Yva)
        if e < best["val"]: best = dict(val=e, smult=smult, lam=lam)
        del Kr, c, al
print(f"corr tune (sub {nsub}): smult={best['smult']} lam={best['lam']} val={best['val']:.4f} [{time.time()-t0:.0f}s]", flush=True)
K = m52(sqdist(Xt, Xt), best["smult"]*med); K.flat[::n+1] += best["lam"]*n
c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
al = cho_solve(c, R, check_finite=False)
corr_te = np.zeros_like(E_te)
for k in range(0, len(Xe), 4000):
    corr_te[k:k+4000] = m52(sqdist(Xe[k:k+4000], Xt), best["smult"]*med) @ al
E2_te = E_te + corr_te
E2_va = E_va + m52(D2v, best["smult"]*med) @ al
report["plus_corr"] = dict(val=rel_l2(E2_va, Yva), test=rel_l2(E2_te, Yte), smult=best["smult"], lam=best["lam"])
print(f"+CORR: val {report['plus_corr']['val']:.4f}  test {report['plus_corr']['test']:.4f}", flush=True)

final = "plus_corr" if report["plus_corr"]["val"] < report["stack"]["val"] else "stack"
report["final_stage"] = final
report["final_test"] = report[final]["test"]
np.save(RUNS / f"{args.tag}_pred_test.npy", (E2_te if final == "plus_corr" else E_te).astype(np.float32))
np.save(RUNS / f"{args.tag}_members_te.npy", np.stack(Pte).astype(np.float32))
np.save(RUNS / f"{args.tag}_stack_tr.npy", E_tr.astype(np.float32))
np.save(RUNS / f"{args.tag}_stack_va.npy", E_va.astype(np.float32))
np.save(RUNS / f"{args.tag}_stack_te.npy", E_te.astype(np.float32))
save_run(args.tag, dict(kind="hybrid", runs=names, report=report, ntrain=len(tr)))
print(f"\nFINAL ({final}): test {report['final_test']:.4f}", flush=True)
print("saved", args.tag, flush=True)
