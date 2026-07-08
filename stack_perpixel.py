"""Per-pixel affine stacking: at each grid point d, weights w_d (and intercept)
over member predictions, ridge-fit. Honesty protocol: fit on half the
validation split, compare with the global-weight stack on the other half;
only if per-pixel wins is it refit on the full validation split and applied
to test. Train predictions get the same refit weights (for the correction
stage downstream).

Usage: python stack_perpixel.py --members a,b,c --krr 1 --tag hpix
"""
import argparse, sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, save_run, RUNS

p = argparse.ArgumentParser()
p.add_argument("--members", type=str, required=True)
p.add_argument("--krr", type=int, default=1)
p.add_argument("--ridge", type=float, default=1e-3)
p.add_argument("--tag", type=str, default="hpix")
args = p.parse_args()

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
Ytr = stress[tr].reshape(len(tr), -1).astype(np.float64)
Yva = stress[va].reshape(len(va), -1).astype(np.float64)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)

names, Ptr, Pva, Pte = [], [], [], []
for m in args.members.split(","):
    m = m.strip(); names.append(m)
    Ptr.append(np.load(RUNS / f"{m}_predtr.npy").astype(np.float64))
    Pva.append(np.load(RUNS / f"{m}_predva.npy").astype(np.float64))
    Pte.append(np.load(RUNS / f"{m}_predte.npy").astype(np.float64))
if args.krr:
    Ptr.append(np.load(RUNS / "krr_oof_train.npy").astype(np.float64))
    Pva.append(np.load(RUNS / "krr_full_matern52_n19000_pred_val.npy").astype(np.float64))
    Pte.append(np.load(RUNS / "krr_full_matern52_n19000_pred_test.npy").astype(np.float64))
    names.append("krr")
Pva = np.stack(Pva); Pte = np.stack(Pte); Ptr = np.stack(Ptr)   # (M,n,D)
M, nv, D = Pva.shape

def fit_pixel_weights(P, Y, ridge):
    """P: (M,n,D), Y: (n,D) -> W: (D, M+1) affine weights per pixel."""
    Mm, n, Dd = P.shape
    X = np.concatenate([P, np.ones((1, n, Dd))], 0)             # (M+1,n,D)
    G = np.einsum("mnd,knd->dmk", X, X) / n                     # (D,M+1,M+1)
    b = np.einsum("mnd,nd->dm", X, Y) / n                       # (D,M+1)
    G += ridge * np.eye(Mm + 1)[None]
    return np.linalg.solve(G, b[..., None])[..., 0]             # (D,M+1)

def apply_pixel_weights(P, W):
    Mm, n, Dd = P.shape
    X = np.concatenate([P, np.ones((1, n, Dd))], 0)
    return np.einsum("dm,mnd->nd", W, X)

rng = np.random.default_rng(1)
perm = rng.permutation(nv)
A, B = perm[:nv // 2], perm[nv // 2:]

W_A = fit_pixel_weights(Pva[:, A], Yva[A], args.ridge)
err_pix_B = rel_l2(apply_pixel_weights(Pva[:, B], W_A), Yva[B])

# global convex reference on the same half (uniform init random search)
def stack_err(logits, PP, YY):
    ww = np.exp(logits - logits.max()); ww /= ww.sum()
    return rel_l2(np.einsum("m,mnd->nd", ww, PP), YY), ww
logit = np.zeros(M); step = 0.3
for it in range(600):
    cand = logit + rng.normal(0, step, M)
    if stack_err(cand, Pva[:, A], Yva[A])[0] < stack_err(logit, Pva[:, A], Yva[A])[0]:
        logit = cand
    step *= 0.997
_, wg = stack_err(logit, Pva[:, A], Yva[A])
err_glob_B = rel_l2(np.einsum("m,mnd->nd", wg, Pva[:, B]), Yva[B])

print(f"half-val holdout: per-pixel {err_pix_B:.4f}  vs global {err_glob_B:.4f}", flush=True)
use_pix = err_pix_B < err_glob_B
out = dict(kind="perpixel", members=names, holdout_pix=err_pix_B,
           holdout_glob=err_glob_B, used=bool(use_pix), ridge=args.ridge)
if use_pix:
    W = fit_pixel_weights(Pva, Yva, args.ridge)
    E_te = apply_pixel_weights(Pte, W)
    E_tr = apply_pixel_weights(Ptr, W)
    E_va = apply_pixel_weights(Pva, W)
    out["test"] = rel_l2(E_te, Yte)
    print(f"per-pixel stack test: {out['test']:.4f}", flush=True)
    np.save(RUNS / f"{args.tag}_stack_te.npy", E_te.astype(np.float32))
    np.save(RUNS / f"{args.tag}_stack_tr.npy", E_tr.astype(np.float32))
    np.save(RUNS / f"{args.tag}_stack_va.npy", E_va.astype(np.float32))
save_run(args.tag, out)
print("saved", args.tag, flush=True)
