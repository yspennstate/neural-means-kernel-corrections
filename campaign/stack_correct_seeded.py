"""Per-pixel stack and residual Matern correction, for one seed, on the four members
that completed at every seed of the campaign.

This is the stage the 36-hour timeout cut off. The members (krr, mlp, mlpMSE, mlpR)
finished on all ten seeds; the stack and the correction ran only at the original seed,
which is why the pipeline number had one replicate and its members had ten.

Two departures from campaign/correct_stack.py, both deliberate:

  * MEMORY. The original builds the 19000x19000 Matern by calling m52 on a full
    squared-distance matrix, which allocates r2, r, a and two more temporaries at
    2.9 GB each -- 15-20 GB peak. That is what OOM-killed seed 4 on the 31 GB box.
    Here the Gram is filled in row blocks into one preallocated array and factored in
    place, so the peak is the 2.9 GB matrix plus a block. Same arithmetic, same
    float64, same result.

  * FOUR MEMBERS, not six. FNO and UNet never finalised in the campaign, so including
    them would mean reporting a stack the seeds do not share. Reported as what it is.

Protocol is otherwise unchanged from the published scripts: per-pixel weights are
accepted only if they beat global convex weights on a held-out half of the validation
split; the correction's length scale, scale multiplier and nugget are tuned against
validation on a subsample and refit on the full train set; the final stage (corrected
or plain stack) is chosen on validation and read out once on test.
"""
import json, os, sys, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")

SEED = sys.argv[1]
os.environ["NMKC_SPLIT_SEED"] = SEED
os.environ["NMKC_PIPE_SEED"] = SEED
RUNS_DIR = "%s/seeds/sm_s%s/runs" % (ROOT, SEED)
os.environ["NMKC_RUNS"] = RUNS_DIR

from common import load_arrays, canonical_split, rel_l2  # noqa: E402

RIDGE = 1e-3
TAG = "pix4"
loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=int(SEED))
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)


def member_files(m):
    if m == "krr":
        import glob
        v = glob.glob(RUNS_DIR + "/krr_*_pred_val.npy")[0]
        t = glob.glob(RUNS_DIR + "/krr_*_pred_test.npy")[0]
        return v, RUNS_DIR + "/krr_oof_train.npy", t
    import glob
    v = [p for p in glob.glob(RUNS_DIR + "/*_predva.npy")
         if os.path.basename(p).split("_")[0] == m][0]
    return v, v.replace("_predva", "_predtr"), v.replace("_predva", "_predte")


NAMES = ["mlp", "mlpMSE", "mlpR", "krr"]
Pva, tr_files, te_files = [], [], []
for m in NAMES:
    v, t, e = member_files(m)
    Pva.append(np.load(v).astype(np.float64))
    tr_files.append(t)
    te_files.append(e)
Pva = np.stack(Pva)
M, nv, D = Pva.shape


def fit_pixel_weights(P, Y, ridge):
    Mm, n, Dd = P.shape
    X = np.concatenate([P, np.ones((1, n, Dd))], 0)
    G = np.einsum("mnd,knd->dmk", X, X) / n
    b = np.einsum("mnd,nd->dm", X, Y) / n
    G += ridge * np.eye(Mm + 1)[None]
    return np.linalg.solve(G, b[..., None])[..., 0]


def apply_pixel_weights(P, W):
    Mm, n, Dd = P.shape
    X = np.concatenate([P, np.ones((1, n, Dd))], 0)
    return np.einsum("dm,mnd->nd", W, X)


def apply_files(files, W, block=2000):
    maps = [np.load(f, mmap_mode="r") for f in files]
    n = maps[0].shape[0]
    out = np.empty((n, D), dtype=np.float64)
    for k in range(0, n, block):
        P = np.stack([np.asarray(mp[k:k + block], dtype=np.float64).reshape(-1, D)
                      for mp in maps])
        out[k:k + block] = apply_pixel_weights(P, W)
    return out


rng = np.random.default_rng(1 + 1000 * int(SEED))
perm = rng.permutation(nv)
A, B = perm[:nv // 2], perm[nv // 2:]
W_A = fit_pixel_weights(Pva[:, A], Yva[A], RIDGE)
err_pix_B = rel_l2(apply_pixel_weights(Pva[:, B], W_A), Yva[B])


def stack_err(logits, PP, YY):
    ww = np.exp(logits - logits.max())
    ww /= ww.sum()
    return rel_l2(np.einsum("m,mnd->nd", ww, PP), YY), ww


logit = np.zeros(M)
step = 0.3
cur = stack_err(logit, Pva[:, A], Yva[A])[0]
for it in range(600):
    cand = logit + rng.normal(0, step, M)
    e = stack_err(cand, Pva[:, A], Yva[A])[0]
    if e < cur:
        logit, cur = cand, e
    step *= 0.997
_, wg = stack_err(logit, Pva[:, A], Yva[A])
err_glob_B = rel_l2(np.einsum("m,mnd->nd", wg, Pva[:, B]), Yva[B])
use_pix = bool(err_pix_B < err_glob_B)
print("s%s half-val holdout: per-pixel %.4f vs global %.4f -> %s"
      % (SEED, err_pix_B, err_glob_B, "per-pixel" if use_pix else "global"), flush=True)

if use_pix:
    W = fit_pixel_weights(Pva, Yva, RIDGE)
    E_va = apply_pixel_weights(Pva, W)
    E_te = apply_files(te_files, W)
    E_tr = apply_files(tr_files, W)
else:
    W = None
    E_va = np.einsum("m,mnd->nd", wg, Pva)
    maps_te = [np.load(f, mmap_mode="r") for f in te_files]
    maps_tr = [np.load(f, mmap_mode="r") for f in tr_files]

    def apply_glob(maps, block=2000):
        n = maps[0].shape[0]
        out = np.empty((n, D), dtype=np.float64)
        for k in range(0, n, block):
            P = np.stack([np.asarray(mp[k:k + block], dtype=np.float64).reshape(-1, D)
                          for mp in maps])
            out[k:k + block] = np.einsum("m,mnd->nd", wg, P)
        return out
    E_te = apply_glob(maps_te)
    E_tr = apply_glob(maps_tr)

stack_val, stack_te = rel_l2(E_va, Yva), rel_l2(E_te, Yte)
print("s%s stack: val %.4f test %.4f" % (SEED, stack_val, stack_te), flush=True)

# ---------------- residual Matern correction, memory-lean ----------------
X = loads[tr].astype(np.float64)
mu, sd = X.mean(0), X.std(0) + 1e-12
Xt = (X - mu) / sd
Xv = (loads[va].astype(np.float64) - mu) / sd
Xe = (loads[te].astype(np.float64) - mu) / sd
n = len(Xt)
R = Ytr - E_tr

rng2 = np.random.default_rng(int(SEED))
sub = rng2.choice(n, min(2000, n), replace=False)


def sqdist(Aa, Bb):
    return np.maximum((Aa * Aa).sum(1)[:, None] + (Bb * Bb).sum(1)[None, :] - 2 * Aa @ Bb.T, 0)


def m52(D2, s):
    r2 = D2 / (s * s)
    r = np.sqrt(r2)
    a = np.sqrt(5) * r
    return (1 + a + (5.0 / 3.0) * r2) * np.exp(-a)


med = np.sqrt(np.median(sqdist(Xt[sub], Xt[sub])[np.triu_indices(len(sub), 1)]))
nsub = min(8000, n)
tsub = rng2.choice(n, nsub, replace=False)
D2s = sqdist(Xt[tsub], Xt[tsub])
D2vs = sqdist(Xv, Xt[tsub])
best = dict(val=np.inf)
t0 = time.time()
for smult in [1.0, 2.0, 4.0]:
    Ks = m52(D2s, smult * med)
    Kvs = m52(D2vs, smult * med)
    for lam in [1e-6, 1e-5, 1e-3]:
        Kr = Ks.copy()
        Kr.flat[::nsub + 1] += lam * nsub
        try:
            c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
        except Exception:
            continue
        al = cho_solve(c, R[tsub], check_finite=False)
        e = rel_l2(E_va + Kvs @ al, Yva)
        if e < best["val"]:
            best = dict(val=e, smult=smult, lam=lam)
        del Kr, c, al
    del Ks, Kvs
del D2s, D2vs
print("s%s corr tune: smult=%s lam=%s val=%.4f [%.0fs]"
      % (SEED, best["smult"], best["lam"], best["val"], time.time() - t0), flush=True)

# full Gram, filled in row blocks into one preallocated array (the OOM fix)
s = best["smult"] * med
K = np.empty((n, n), dtype=np.float64)
BLK = 1000
for k in range(0, n, BLK):
    K[k:k + BLK] = m52(sqdist(Xt[k:k + BLK], Xt), s)
K.flat[::n + 1] += best["lam"] * n
t1 = time.time()
c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
al = cho_solve(c, R, check_finite=False)
print("s%s cholesky n=%d in %.0fs" % (SEED, n, time.time() - t1), flush=True)
del K, c

corr_te = np.empty_like(E_te)
for k in range(0, len(Xe), 2000):
    corr_te[k:k + 2000] = m52(sqdist(Xe[k:k + 2000], Xt), s) @ al
E2_te = E_te + corr_te
E2_va = E_va + m52(sqdist(Xv, Xt), s) @ al

corr_val, corr_te_err = rel_l2(E2_va, Yva), rel_l2(E2_te, Yte)
final = "plus_corr" if corr_val < stack_val else "stack"
rec = dict(kind="hybrid4", seed=int(SEED), members=NAMES, used_perpixel=use_pix,
           holdout_pix=err_pix_B, holdout_glob=err_glob_B,
           stack=dict(val=stack_val, test=stack_te),
           plus_corr=dict(val=corr_val, test=corr_te_err,
                          smult=best["smult"], lam=best["lam"], med=med),
           final_stage=final,
           final_test=corr_te_err if final == "plus_corr" else stack_te,
           ntrain=n)
with open("%s/%s_s%s.json" % (ROOT, TAG, SEED), "w") as f:
    json.dump(rec, f, indent=1)
np.save(RUNS_DIR + "/%s_corr_pred_test.npy" % TAG,
        (E2_te if final == "plus_corr" else E_te).astype(np.float32))
print("s%s STACK %.4f -> CORR %.4f (final %s)"
      % (SEED, stack_te, corr_te_err, final), flush=True)
