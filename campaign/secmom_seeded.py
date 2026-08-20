"""The second-moment reading of the stack, per seed.

The paper predicts a stack's achievable risk from the members' residual second-moment
matrix S before any weight is fitted, and states an equicorrelated floor. Both were
measured at one seed. This runs the same measurement at every seed the campaign
completed, so the prediction is checked prospectively rather than illustrated once.

Per seed it reports
  - S on the validation split, from normalised residuals r_m = (f_m - y)/||y||
  - the simplex-optimal weights and the predicted RMS relative error sqrt(w'Sw)
  - the realised mean relative error of those weights, on val and on test
  - the dispersion factor realised/predicted (RMS vs mean: Jensen, so below 1)
  - the pairwise residual correlations, their mean, and the equicorrelated floor
    e_bar*sqrt(rho_bar)
  - the pairwise keep/drop test rho < e1/e2 for the kernel against the best network
"""
import glob, json, os, sys
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split, rel_l2  # noqa: E402

loads, stress = load_arrays()
NAMES = ["mlp", "mlpMSE", "mlpR", "krr"]

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    os.environ["NMKC_SPLIT_SEED"] = seed
    tr, va, te = canonical_split(n_val=1000, seed=int(seed))
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)

    def arr(m, split):
        if m == "krr":
            g = glob.glob(os.path.join(R, "krr_*_pred_%s.npy"
                                       % ("val" if split == "va" else "test")))
        else:
            g = [p for p in glob.glob(os.path.join(R, "*_pred%s.npy" % split))
                 if os.path.basename(p).split("_")[0] == m]
        return np.load(g[0]).astype(np.float64) if g else None

    Pva = [arr(m, "va") for m in NAMES]
    Pte = [arr(m, "te") for m in NAMES]
    if any(p is None for p in Pva) or any(p is None for p in Pte):
        print("s%s incomplete" % seed)
        continue

    nv = np.linalg.norm(Yva, axis=1, keepdims=True)
    Rva = np.stack([(p - Yva) / nv for p in Pva])
    M = len(NAMES)
    S = np.einsum("mnd,knd->mk", Rva, Rva) / Rva.shape[1]

    # Solved by SLSQP under an explicit sum-to-one constraint. The projected-gradient
    # loop this replaces used clip-and-renormalise, which is not a simplex projection:
    # it stopped at near-vertex points 0.8-3.6% above the true minimum and reported
    # zero weight on members that in fact earn a few percent. See
    # campaign/check_simplex_solve.py for the test that catches it.
    from scipy.optimize import minimize
    res = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z,
                   bounds=[(0.0, 1.0)] * M,
                   constraints={"type": "eq", "fun": lambda z: z.sum() - 1.0},
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
    w = np.maximum(res.x, 0.0)
    w /= w.sum()
    pred = float(np.sqrt(w @ S @ w))
    real_va = rel_l2(np.einsum("m,mnd->nd", w, np.stack(Pva)), Yva)
    real_te = rel_l2(np.einsum("m,mnd->nd", w, np.stack(Pte)), Yte)

    # correlations of the per-sample normalised residuals
    F = Rva.reshape(M, -1)
    C = np.corrcoef(F)
    off = [C[i, j] for i in range(M) for j in range(i + 1, M)]
    e = np.sqrt(np.diag(S))                     # per-member RMS relative error
    ebar, rbar = float(e.mean()), float(np.mean(off))
    floor = ebar * np.sqrt(max(rbar, 0.0))

    # keep/drop: kernel (index 3) against the most accurate network
    bi = int(np.argmin(e[:3]))
    rho_k = C[bi, 3]
    thresh = e[bi] / e[3]
    print("s%-2s w=%s  pred %.4f  real val %.4f test %.4f  disp %.3f | "
          "rho[min %.3f mean %.3f max %.3f] ebar %.4f floor %.4f | "
          "kernel vs %s: rho %.3f vs e1/e2 %.3f -> %s"
          % (seed, np.round(w, 3), pred, real_va, real_te, real_va / pred,
             min(off), rbar, max(off), ebar, floor, NAMES[bi], rho_k, thresh,
             "KEEP" if rho_k < thresh else "DROP"))
