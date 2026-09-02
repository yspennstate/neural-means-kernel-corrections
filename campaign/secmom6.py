"""Second-moment reading of the stack at every seed, for an arbitrary member list.

Same measurement as campaign/secmom_seeded5.py (S on the validation split from the
normalised residuals, the simplex-optimal weights, the predicted RMS and the realised
mean, the dispersion factor, the pairwise correlations and the equicorrelated floor),
plus three things that script did not record: the full matrices S and C, the
bracket of Corollary floorpinch (delta_e, delta_rho, upper end), and the signed
sum-to-one optimum 1/(1'S^{-1}1) with its weights. Every member of NAMES must have
<name>_predva.npy / <name>_predte.npy in the seed's runs directory (krr uses the
krr_full files). Environment: NMKC_ROOT, NMKC_MEMBERS (comma list, default six),
NMKC_SECMOM_OUT (json path), optional NMKC_SEEDS (comma list).
"""
import glob, json, os, sys
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2"))
OUT_JSON = os.environ.get("NMKC_SECMOM_OUT", "")
NAMES = os.environ.get("NMKC_MEMBERS", "mlp,mlpMSE,mlpR,fno,unet,krr").split(",")
ONLY = os.environ.get("NMKC_SEEDS", "")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split, rel_l2  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

loads, stress = load_arrays()
RECORDS = []
seed_dirs = sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*")))
for sd in seed_dirs:
    seed = os.path.basename(sd).replace("sm_s", "")
    if not seed.isdigit() or int(seed) >= 90:
        continue
    if ONLY and seed not in ONLY.split(","):
        continue
    R = os.path.join(sd, "runs")
    os.environ["NMKC_SPLIT_SEED"] = seed
    tr, va, te = canonical_split(n_val=1000, seed=int(seed))
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)

    def arr(m, split):
        if m == "krr":
            g = glob.glob(os.path.join(R, "krr_full_*_pred_%s.npy" % ("val" if split == "va" else "test")))
        else:
            g = [p for p in glob.glob(os.path.join(R, "*_pred%s.npy" % split))
                 if os.path.basename(p).split("_")[0] == m]
        return np.load(g[0]).astype(np.float64) if g else None

    Pva = [arr(m, "va") for m in NAMES]
    if any(p is None for p in Pva):
        print("s%s incomplete on validation: %s" % (seed, [m for m, p in zip(NAMES, Pva) if p is None]))
        continue
    nv = np.linalg.norm(Yva, axis=1, keepdims=True)
    Rva = np.stack([(p - Yva) / nv for p in Pva])           # (M, n, D) normalised residuals
    M = len(NAMES)
    S = np.einsum("mnd,knd->mk", Rva, Rva) / Rva.shape[1]

    res = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z,
                   bounds=[(0.0, 1.0)] * M,
                   constraints={"type": "eq", "fun": lambda z: z.sum() - 1.0},
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
    w = np.maximum(res.x, 0.0); w /= w.sum()
    # the cheap check the paper describes: no vertex may beat the returned point
    vert = [S[i, i] for i in range(M)]
    assert w @ S @ w <= min(vert) + 1e-12, "simplex solve worse than a vertex"
    pred = float(np.sqrt(w @ S @ w))
    real_va = rel_l2(np.einsum("m,mnd->nd", w, np.stack(Pva)), Yva)
    # test arrays are large; stream the combination member by member
    comb_te = None; eq_te = None
    for wm, m in zip(w, NAMES):
        pte = arr(m, "te")
        comb_te = wm * pte if comb_te is None else comb_te + wm * pte
        eq_te = pte / M if eq_te is None else eq_te + pte / M
        del pte
    real_te = rel_l2(comb_te, Yte)
    eq_test = rel_l2(eq_te, Yte)
    eq_val = rel_l2(np.mean(np.stack(Pva), axis=0), Yva)
    del comb_te, eq_te

    F = Rva.reshape(M, -1)
    C = np.corrcoef(F)
    off = [C[i, j] for i in range(M) for j in range(i + 1, M)]
    e = np.sqrt(np.diag(S))
    ebar, rbar = float(e.mean()), float(np.mean(off))
    floor = ebar * np.sqrt(max(rbar, 0.0))
    # Corollary floorpinch with the one-sided bounds read off the measured S:
    # ebar^2 := min_m S_mm, rho_bar := min_{m!=k} S_mk / ebar^2, delta_e, delta_rho from the maxima
    e2min = float(np.min(np.diag(S)))
    offS = np.array([S[i, j] for i in range(M) for j in range(i + 1, M)])
    rho_lo = float(offS.min() / e2min)
    delta_rho = float(offS.max() / e2min - rho_lo)
    delta_e = float(np.max(np.diag(S)) / e2min - 1.0)
    upper = rho_lo + delta_rho + (1 + delta_e - rho_lo - delta_rho) / M
    minval = float(w @ S @ w) / e2min
    # signed sum-to-one optimum (affine stack with global weights, no positivity)
    Sinv1 = np.linalg.solve(S, np.ones(M))
    w_signed = Sinv1 / Sinv1.sum()
    signed_min = float(1.0 / (np.ones(M) @ Sinv1))
    # pairwise admission of every member against the most accurate residual-MLP member
    net_idx = [NAMES.index(n) for n in ("mlp", "mlpMSE", "mlpR") if n in NAMES]
    bi = min(net_idx, key=lambda i: e[i])
    admit = {}
    for cand in NAMES:
        if cand == NAMES[bi]:
            continue
        ci = NAMES.index(cand)
        e1, e2 = min(e[bi], e[ci]), max(e[bi], e[ci])
        admit[cand] = dict(rho=float(C[bi, ci]), threshold=float(e1 / e2),
                           keep=bool(C[bi, ci] < e1 / e2), ref=NAMES[bi])
    print("s%-2s w=%s pred %.4f real val %.4f test %.4f disp %.3f | rho[%.3f %.3f %.3f] ebar %.4f floor %.4f "
          "| bracket [%.4f, %.4f] min %.4f | signed %.4f eq_test %.4f"
          % (seed, dict(zip(NAMES, np.round(w, 3))), pred, real_va, real_te, real_va / pred,
             min(off), rbar, max(off), ebar, floor, rho_lo, upper, minval, np.sqrt(signed_min), eq_test), flush=True)
    RECORDS.append(dict(
        seed=int(seed), members=NAMES, weights=dict(zip(NAMES, [float(x) for x in w])),
        pred_rms=pred, real_val=float(real_va), real_test=float(real_te),
        disp_factor=float(real_va / pred), rho_min=float(min(off)), rho_mean=rbar, rho_max=float(max(off)),
        ebar=ebar, floor=float(floor), member_err={n: float(v) for n, v in zip(NAMES, e)},
        S=S.tolist(), C=C.tolist(),
        bracket=dict(e2min=e2min, rho_lo=rho_lo, delta_rho=delta_rho, delta_e=delta_e,
                     lower=rho_lo, upper=float(upper), min_over_e2min=minval),
        signed=dict(min_rms=float(np.sqrt(signed_min)), weights=dict(zip(NAMES, [float(x) for x in w_signed]))),
        equal_weight=dict(val=float(eq_val), test=float(eq_test)),
        admission=admit))

if OUT_JSON and RECORDS:
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(RECORDS, fh, indent=1)
    print("wrote %s (%d seeds)" % (OUT_JSON, len(RECORDS)))
