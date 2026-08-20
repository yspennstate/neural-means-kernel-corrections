"""Is the inherited simplex QP actually solving the problem?

ensemble_theory.py minimises w'Sw over the simplex by projected gradient:

    w = ones/M
    for 20000 steps:  w -= 0.02 * 2 * S @ w;  w = max(w,0);  w /= w.sum()

Clip-and-renormalise is NOT a projection onto the simplex, and on the five-member matrix
this lands on the pure-FNO vertex at six of ten seeds even though the FNO's own RMS
residual is reported as slightly LARGER than mlpMSE's -- in which case the pure-mlpMSE
vertex has a strictly smaller objective and the answer cannot be optimal.

This compares, on the same S:
    - the objective at the inherited routine's answer
    - the objective at each pure-member vertex
    - the objective at a properly solved simplex QP (SLSQP with sum-to-one and w>=0)
If the inherited routine is not at the minimum, every weight vector the paper quotes from
it is wrong, including the four-member ones.
"""
import glob
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split  # noqa: E402

NAMES = ["mlp", "mlpMSE", "mlpR", "fno", "krr"]
loads, stress = load_arrays()

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s[0-9]"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    R = os.path.join(sd, "runs")
    os.environ["NMKC_SPLIT_SEED"] = seed
    _, va, _ = canonical_split(n_val=1000, seed=int(seed))
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)

    def arr(m):
        if m == "krr":
            g = glob.glob(os.path.join(R, "krr_*_pred_val.npy"))
        else:
            g = [p for p in glob.glob(os.path.join(R, "*_predva.npy"))
                 if os.path.basename(p).split("_")[0] == m]
        return np.load(g[0]).astype(np.float64) if g else None

    P = [arr(m) for m in NAMES]
    if any(p is None for p in P):
        continue
    nv = np.linalg.norm(Yva, axis=1, keepdims=True)
    Rva = np.stack([(p - Yva) / nv for p in P])
    M = len(NAMES)
    S = np.einsum("mnd,knd->mk", Rva, Rva) / Rva.shape[1]

    # the inherited routine, verbatim
    w = np.ones(M) / M
    for _ in range(20000):
        w = w - 0.02 * (2 * S @ w)
        w = np.maximum(w, 0)
        w /= w.sum()
    obj_inherited = float(w @ S @ w)

    # every pure vertex
    verts = {NAMES[i]: float(S[i, i]) for i in range(M)}
    best_vert = min(verts, key=verts.get)

    # a properly solved simplex QP
    res = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z,
                   bounds=[(0, 1)] * M,
                   constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
    w_true = np.maximum(res.x, 0)
    w_true /= w_true.sum()
    obj_true = float(w_true @ S @ w_true)

    gap = (obj_inherited - obj_true) / obj_true
    print("s%-2s inherited %.6e  best-vertex(%s) %.6e  solved %.6e  | inherited is %.2f%% above the true min  %s"
          % (seed, obj_inherited, best_vert, verts[best_vert], obj_true, 100 * gap,
             "OK" if gap < 1e-6 else "SUBOPTIMAL"))
    print("      inherited w = %s" % dict(zip(NAMES, np.round(w, 3))))
    print("      solved    w = %s" % dict(zip(NAMES, np.round(w_true, 3))))
