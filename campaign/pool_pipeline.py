"""The pipeline over the sixty-predictor pool: each architecture's ten-seed mean as a member,
the per-pixel affine stack with the half-split honesty protocol, and the residual Matern
correction with its usual grid, every fitted choice made on the calibration half of the test
block (1000 cases the members never saw) and every number read on the evaluation half (19000).
The correction is trained on the stack's residuals over seed 0's 19000 training loads, as the
pipeline trains it, and tuned on the calibration half. Writes results/pool_pipeline.json.
Environment: NMKC_ROOT, NMKC_THREADS.
"""
import glob, json, os, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2"))
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
os.environ["NMKC_SPLIT_SEED"] = "0"
from common import load_arrays, canonical_split  # noqa: E402

t0 = time.time()
loads, stress = load_arrays()
DATA = ROOT + "/data/structmech"
idx_train = np.load(DATA + "/idx_train.npy")
row_of = {int(i): r for r, i in enumerate(idx_train)}          # sample id -> pool row
tr0, va0, te = canonical_split(n_val=1000, seed=0)
Ypool = stress[idx_train].reshape(len(idx_train), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
nte = np.linalg.norm(Yte, axis=1)
rng = np.random.default_rng(20260902)
perm = rng.permutation(len(te)); cal, ev = np.sort(perm[:1000]), np.sort(perm[1000:])
SEEDS = list(range(10))
ARCH = ["mlp", "mlpMSE", "mlpR", "fno", "unet", "krr"]
D = Ypool.shape[1]


def files(a, s):
    d = "%s/seeds/sm_s%d/runs" % (ROOT, s)
    if a == "krr":
        return d + "/krr_oof_train.npy", d + "/krr_full_matern52_n19000_pred_val.npy", d + "/krr_full_matern52_n19000_pred_test.npy"
    g = [p for p in glob.glob(d + "/*_predte.npy") if os.path.basename(p).split("_")[0] == a]
    assert len(g) == 1, (a, s, g)
    stem = g[0][:-len("_predte.npy")]
    return stem + "_predtr.npy", stem + "_predva.npy", stem + "_predte.npy"


# seed-mean members over the whole training pool (in-sample for nine seeds, out-of-sample for one)
# and over the test block (out-of-sample for all)
pool_mean, te_mean = {}, {}
for a in ARCH:
    acc = np.zeros((len(idx_train), D), dtype=np.float64)
    acc_te = np.zeros((len(te), D), dtype=np.float64)
    for s in SEEDS:
        os.environ["NMKC_SPLIT_SEED"] = str(s)
        trs, vas, _ = canonical_split(n_val=1000, seed=s)
        ftr, fva, fte = files(a, s)
        acc[[row_of[int(i)] for i in trs]] += np.load(ftr).astype(np.float64)
        acc[[row_of[int(i)] for i in vas]] += np.load(fva).astype(np.float64)
        acc_te += np.load(fte).astype(np.float64)
    pool_mean[a] = acc / len(SEEDS)
    te_mean[a] = acc_te / len(SEEDS)
    print("assembled", a, "%.0fs" % (time.time() - t0), flush=True)
os.environ["NMKC_SPLIT_SEED"] = "0"
rows_tr0 = np.array([row_of[int(i)] for i in tr0])
Ytr = Ypool[rows_tr0]
P_tr = np.stack([pool_mean[a][rows_tr0] for a in ARCH])               # (M, 19000, D) in-sample means
P_cal = np.stack([te_mean[a][cal] for a in ARCH])                     # (M, 1000, D)
P_ev = np.stack([te_mean[a][ev] for a in ARCH])                       # (M, 19000, D)
Ycal, Yev = Yte[cal], Yte[ev]
M = len(ARCH)


def rel(Pred, Y):
    return float((np.linalg.norm(Pred - Y, axis=1) / np.linalg.norm(Y, axis=1)).mean())


def fit_pixel(P, Y, ridge=1e-3):
    m_, n_, D_ = P.shape
    X = np.concatenate([P, np.ones((1, n_, D_))], 0)
    G = np.einsum("mnd,knd->dmk", X, X) / n_ + ridge * np.eye(m_ + 1)[None]
    b = np.einsum("mnd,nd->dm", X, Y) / n_
    return np.linalg.solve(G, b[..., None])[..., 0]


def apply_pixel(P, W):
    m_, n_, D_ = P.shape
    X = np.concatenate([P, np.ones((1, n_, D_))], 0)
    return np.einsum("dm,mnd->nd", W, X)


def convex_weights(P, Y):
    nv = np.linalg.norm(Y, axis=1, keepdims=True)
    Rn = (P - Y[None]) / nv[None]
    S = np.einsum("mnd,knd->mk", Rn, Rn) / Rn.shape[1]
    res = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z, bounds=[(0, 1)] * M,
                   constraints={"type": "eq", "fun": lambda z: z.sum() - 1}, method="SLSQP", options=dict(maxiter=2000, ftol=1e-15))
    w = np.maximum(res.x, 0); return w / w.sum()


# honesty protocol on the calibration half: per-pixel weights on one half, global convex on the same half,
# compared on the other half; the winner refit on the whole calibration half
hs = np.random.default_rng(1).permutation(len(cal)); A_, B_ = hs[:500], hs[500:]
W_A = fit_pixel(P_cal[:, A_], Ycal[A_])
err_pix_B = rel(apply_pixel(P_cal[:, B_], W_A), Ycal[B_])
w_A = convex_weights(P_cal[:, A_], Ycal[A_])
err_glob_B = rel(np.einsum("m,mnd->nd", w_A, P_cal[:, B_]), Ycal[B_])
use_pix = err_pix_B < err_glob_B
print("half-split: per-pixel %.4f vs global %.4f -> %s" % (err_pix_B, err_glob_B, "per-pixel" if use_pix else "global"), flush=True)
if use_pix:
    W = fit_pixel(P_cal, Ycal)
    E_tr = apply_pixel(P_tr, W); E_cal = apply_pixel(P_cal, W); E_ev = apply_pixel(P_ev, W)
else:
    w = convex_weights(P_cal, Ycal)
    E_tr = np.einsum("m,mnd->nd", w, P_tr); E_cal = np.einsum("m,mnd->nd", w, P_cal); E_ev = np.einsum("m,mnd->nd", w, P_ev)
stack_cal, stack_ev = rel(E_cal, Ycal), rel(E_ev, Yev)
members_ev = {a: rel(te_mean[a][ev], Yev) for a in ARCH}
eq_ev = rel(np.mean(P_ev, axis=0), Yev)
print("members on ev", {a: round(100 * v, 3) for a, v in members_ev.items()}, "equal %.4f stack ev %.4f" % (eq_ev, stack_ev), flush=True)
del P_tr, P_ev

# residual Matern correction: trained on seed 0's 19000 training loads, tuned on the calibration half
X = loads[tr0].astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-12
Xt = (X - mu) / sd; Xc = (loads[te[cal]].astype(np.float64) - mu) / sd; Xe = (loads[te[ev]].astype(np.float64) - mu) / sd


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


def m52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5) * r
    return (1 + a + (5 / 3) * r2) * np.exp(-a)


n = len(Xt)
prng = np.random.default_rng(0)
sub = prng.choice(n, 2000, replace=False)
med = np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(2000, 1)]))
Rtr = Ytr - E_tr
nsub = 8000
tsub = prng.choice(n, nsub, replace=False)
D2s = sqdist(Xt[tsub], Xt[tsub]); D2cs = sqdist(Xc, Xt[tsub])
best = dict(val=np.inf); grid = []
for smult in (1.0, 2.0, 4.0):
    Ks = m52(D2s, smult * med); Kcs = m52(D2cs, smult * med)
    for lam in (1e-6, 1e-5, 1e-3):
        Kr = Ks.copy(); Kr.flat[::nsub + 1] += lam * nsub
        c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
        al = cho_solve(c, Rtr[tsub], check_finite=False)
        e = rel(E_cal + Kcs @ al, Ycal)
        grid.append(dict(smult=smult, lam=lam, cal=e))
        if e < best["val"]:
            best = dict(val=e, smult=smult, lam=lam)
        del Kr, c, al
print("tune on cal: smult %s lam %s cal %.4f (stack cal %.4f) %.0fs" % (best["smult"], best["lam"], best["val"], stack_cal, time.time() - t0), flush=True)
K = m52(sqdist(Xt, Xt), best["smult"] * med); K.flat[::n + 1] += best["lam"] * n
c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
al = cho_solve(c, Rtr, check_finite=False)
del K, c
corr_ev = np.zeros_like(E_ev)
for k in range(0, len(Xe), 4000):
    corr_ev[k:k + 4000] = m52(sqdist(Xe[k:k + 4000], Xt), best["smult"] * med) @ al
corr_cal = E_cal + m52(sqdist(Xc, Xt), best["smult"] * med) @ al
final_uses_corr = rel(corr_cal, Ycal) < stack_cal
E2_ev = E_ev + corr_ev
out = dict(members_ev=members_ev, equal_weight_ev=eq_ev, half_split=dict(perpixel_B=err_pix_B, global_B=err_glob_B, used_perpixel=bool(use_pix)),
           stack_cal=stack_cal, stack_ev=stack_ev, correction=dict(grid=grid, best=best, cal=rel(corr_cal, Ycal), ev=rel(E2_ev, Yev), used=bool(final_uses_corr)),
           final_ev=rel(E2_ev, Yev) if final_uses_corr else stack_ev, n_train=n, minutes=(time.time() - t0) / 60)
os.makedirs(ROOT + "/results", exist_ok=True)
json.dump(out, open(ROOT + "/results/pool_pipeline.json", "w"), indent=1)
print("POOL stack ev %.4f -> corrected ev %.4f (final %.4f) in %.1f min" % (stack_ev, out["correction"]["ev"], out["final_ev"], out["minutes"]), flush=True)
