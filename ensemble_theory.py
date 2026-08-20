"""Second-moment reading of the stack: with normalized residuals
r_m,n = (f_m(u_n) - y_n)/||y_n||, the squared relative error of a convex
combination is the quadratic form w' S w with
S_mk = mean_n <r_m,n, r_k,n>. The classical ambiguity decomposition says the
optimal w on the simplex trades individual accuracy against decorrelation.
This script measures S on the validation split, solves the simplex QP, and
compares the predicted sqrt(w'Sw) with the realized metric of that w on the
test split -- the check that the correlation matrix, not member accuracy,
is what limits the stack.

Usage: python ensemble_theory.py --members a,b,c --tag ens
"""
import argparse, sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS

p = argparse.ArgumentParser()
p.add_argument("--members", type=str, required=True)
p.add_argument("--tag", type=str, default="ens")
args = p.parse_args()

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

def pv(m, split):
    if m == "krr":
        f = {"va": "krr_full_matern52_n19000_pred_val.npy",
             "te": "krr_full_matern52_n19000_pred_test.npy"}[split]
        return np.load(RUNS / f).astype(np.float64)
    return np.load(RUNS / f"{m}_pred{split}.npy").astype(np.float64)

names = [m.strip() for m in args.members.split(",")]
nv = np.linalg.norm(Yva, axis=1, keepdims=True)
Rva = np.stack([(pv(m, "va") - Yva) / nv for m in names])       # (M, n, d)
M = len(names)
S = np.einsum("mnd,knd->mk", Rva, Rva) / Rva.shape[1]

# Simplex QP. This was a projected-gradient loop with clip-and-renormalise, which is
# NOT a projection onto the simplex: it walks to near-vertex points and stops there. On
# the five-member matrix of the seed campaign it terminated 0.8-3.6% above the true
# minimum and reported all the weight on a single member, which reads as a finding
# ("the kernel gets zero weight") when it is a numerical failure. Solved properly the
# optimum is a spread over every member. Kept as a cautionary note rather than deleted,
# because the earlier reported weight vectors came from the old routine.
from scipy.optimize import minimize
_res = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z,
                bounds=[(0.0, 1.0)] * M,
                constraints={"type": "eq", "fun": lambda z: z.sum() - 1.0},
                method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
w = np.maximum(_res.x, 0.0); w /= w.sum()
pred_err = float(np.sqrt(w @ S @ w))
# the check that catches a bad solve: no pure vertex may beat the returned mixture
_vert = float(np.min(np.diag(S)))
assert w @ S @ w <= _vert + 1e-12, "simplex solve is worse than a single member"

E_te = np.einsum("m,mnd->nd", w, np.stack([pv(m, "te") for m in names]))
real_te = rel_l2(E_te, Yte)
E_va = np.einsum("m,mnd->nd", w, np.stack([pv(m, "va") for m in names]))
real_va = rel_l2(E_va, Yva)

print("members:", names)
print("S (x1e4):\n", np.array2string(S * 1e4, precision=2))
print("QP weights:", np.round(w, 3))
print(f"predicted RMS rel err (val, sqrt w'Sw): {100*pred_err:.2f}%")
print(f"realized mean rel err: val {100*real_va:.2f}%  test {100*real_te:.2f}%")
out = dict(members=names, S=S.tolist(), w=w.tolist(), pred_rms_val=pred_err,
           real_val=real_va, real_test=real_te)
with open(RUNS / f"enstheory_{args.tag}.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved", f"enstheory_{args.tag}.json")
