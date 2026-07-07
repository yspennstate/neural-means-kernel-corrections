"""Residual-correlation analysis across ensemble members.

For each member with saved test predictions, compute the test error and the
pairwise Pearson correlation of residuals (on a fixed subsample of test cases,
all pixels). Low off-diagonal correlation is what stacking can exploit; the
MLP-family plateau of 2026-07-06 showed correlations near one.

Usage: python analyze_corr.py --members mlp_s0_w1024_d4_n19000_mir,mlpR_s0_w1024_d4,krr
Saves runs/corr_<tag>.json.
"""
import argparse, sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS

p = argparse.ArgumentParser()
p.add_argument("--members", type=str, required=True)
p.add_argument("--nsub", type=int, default=5000)
p.add_argument("--tag", type=str, default="ens")
args = p.parse_args()

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
sub = np.random.default_rng(0).choice(len(te), args.nsub, replace=False)

names, errs, res = [], {}, []
for m in args.members.split(","):
    m = m.strip()
    f = RUNS / ("krr_full_matern52_n19000_pred_test.npy" if m == "krr" else f"{m}_predte.npy")
    P = np.load(f).astype(np.float64)
    names.append(m)
    errs[m] = rel_l2(P, Yte)
    res.append((P - Yte)[sub].ravel())
    del P

R = np.stack(res)                      # (m, nsub*1681)
R -= R.mean(1, keepdims=True)
nrm = np.linalg.norm(R, axis=1, keepdims=True)
C = (R / nrm) @ (R / nrm).T

print("member errors:")
for n in names:
    print(f"  {n:42s} {100*errs[n]:.2f}%")
print("\nresidual correlation matrix:")
hdr = "".join(f"{i:>7d}" for i in range(len(names)))
print("      " + hdr)
for i, n in enumerate(names):
    print(f"  [{i}] " + "".join(f"{C[i,j]:7.3f}" for j in range(len(names))))

# pairwise 50/50 average error: what decorrelation buys before any tuning
pair = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        fi = RUNS / ("krr_full_matern52_n19000_pred_test.npy" if names[i] == "krr" else f"{names[i]}_predte.npy")
        fj = RUNS / ("krr_full_matern52_n19000_pred_test.npy" if names[j] == "krr" else f"{names[j]}_predte.npy")
        e = rel_l2(0.5 * (np.load(fi).astype(np.float64) + np.load(fj).astype(np.float64)), Yte)
        pair[f"{names[i]}+{names[j]}"] = e
        print(f"  avg({names[i]}, {names[j]}) = {100*e:.2f}%")

out = dict(members=names, errors={k: float(v) for k, v in errs.items()},
           corr=[[float(C[i, j]) for j in range(len(names))] for i in range(len(names))],
           pair_avg={k: float(v) for k, v in pair.items()}, nsub=args.nsub)
with open(RUNS / f"corr_{args.tag}.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved", f"corr_{args.tag}.json", flush=True)
