"""Helmholtz and Navier-Stokes operator cells of the NMKC paper at one seed, CPU only.

The construction of hidata_benchmark.py (output PCA, exact Matern on standardized inputs, a residual
MLP mean, and the residual kernel correction) with the seed threaded through the sample draw and the
network, the kernel fit subset raised to --fit rows, and two additions: the operator-learning kernel of
Batlle, Darcy, Hosseini and Owhadi (PCA on the INPUT as well, a Matern on the input coefficients) and
a per-coordinate selection on validation.  Writes results/<tag>.json.
    python hidata_seeded.py --name Helmholtz --grid 101 --seed 1
Environment: NMKC_HIDATA (directory with <Name>_inputs.npy / _outputs.npy), P2_OUT, NMKC_THREADS.
"""
import argparse, json, os, pathlib, time
import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve
torch.set_num_threads(int(_T))

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--grid", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ntrain", type=int, default=20000)
p.add_argument("--ntest", type=int, default=2000)
p.add_argument("--npca", type=int, default=40)
p.add_argument("--npca_in", type=int, default=200)
p.add_argument("--fit", type=int, default=8000)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--tag", default="")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
DATA = pathlib.Path(os.environ.get("NMKC_HIDATA", os.path.expanduser("~/nmkc/data")))
OUT = pathlib.Path(os.environ.get("P2_OUT", "results"))
if args.smoke:
    args.ntrain, args.ntest, args.fit, args.epochs = 3000, 500, 1500, 2
t_all = time.time()

Xm = np.load(DATA / f"{args.name}_inputs.npy", mmap_mode="r")   # (g, g, N)
Ym = np.load(DATA / f"{args.name}_outputs.npy", mmap_mode="r")
N = Xm.shape[-1]; d = args.grid * args.grid
rng = np.random.default_rng(args.seed); perm = rng.permutation(N)
tr = np.sort(perm[:args.ntrain]); te = np.sort(perm[args.ntrain:args.ntrain + args.ntest])


def grab(M, idx):
    return np.moveaxis(np.asarray(M[..., idx], np.float32), -1, 0).reshape(len(idx), -1)


Xtr, Xte = grab(Xm, tr), grab(Xm, te)
Ytr, Yte = grab(Ym, tr), grab(Ym, te)
print(f"{args.name} seed {args.seed}: {N} samples, {d}->{d}, using {len(tr)}/{len(te)}, output PCA {args.npca}", flush=True)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
nval = 1000
fit_pool = np.arange(len(Xtr_n) - nval)
Xva_n = Xtr_n[-nval:]; Yva = Ytr[-nval:]
ym = Ytr[:-nval].mean(0)
nsvd = min(6000, len(Ytr) - nval)
_, s, Vh = np.linalg.svd(Ytr[:nsvd] - ym, full_matrices=False)
P = Vh[:args.npca].T.astype(np.float32)
Ctr = (Ytr - ym) @ P
recon = lambda C: C @ P.T + ym
rel = lambda Pr, T: float(np.mean(np.linalg.norm(Pr - T, axis=1) / np.linalg.norm(T, axis=1)))
print(f"  output PCA keeps {100*(s[:args.npca]**2).sum()/(s**2).sum():.2f}% of variance", flush=True)
# the floor every head inherits from the output reduction: the truth projected on the kept modes, scored like a prediction
pca_floor_te = rel(recon((Yte - ym) @ P), Yte)
print(f"  output PCA floor on test (truth projected on {args.npca} modes): {100*pca_floor_te:.3f}%", flush=True)
_, s_in, Vh_in = np.linalg.svd(Xtr_n[:nsvd], full_matrices=False)
Pin = Vh_in[:args.npca_in].T.astype(np.float32)
Xtr_p, Xva_p, Xte_p = Xtr_n @ Pin, Xva_n @ Pin, Xte_n @ Pin
print(f"  input PCA {args.npca_in} keeps {100*(s_in[:args.npca_in]**2).sum()/(s_in**2).sum():.2f}% of variance", flush=True)


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    r = np.sqrt(D2) / ls; return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def kernel_stage(Xall_tr, Xall_va, Xall_te, C_targets, val_err_fn, sub_seed, label):
    sub = np.random.default_rng(sub_seed).choice(fit_pool, min(args.fit, len(fit_pool)), replace=False)
    Xs = Xall_tr[sub].astype(np.float64); Cs = C_targets[sub].astype(np.float64)
    Xva = Xall_va.astype(np.float64)
    dsub = sqd(Xs[:1500], Xs[:1500]); med = float(np.sqrt(np.median(dsub[np.triu_indices(min(1500, len(Xs)), 1)])) + 1e-9)
    D2 = sqd(Xs, Xs); best = (np.inf, None)
    for sc in (0.5, 1, 2, 4, 8, 16):          # the first runs chose 4, the top of the old grid, on both problems
        Kva = m52(sqd(Xva, Xs), sc * med)
        for nug in (1e-8, 1e-6, 1e-4):
            K = m52(D2, sc * med); K.flat[::len(Xs) + 1] += nug * len(Xs)
            try:
                a = cho_solve(cho_factor(K, lower=True, check_finite=False, overwrite_a=True), Cs, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            e = val_err_fn(Kva @ a)
            if e < best[0]:
                best = (e, (a, sc))
    a, sc = best[1]
    pred_te = m52(sqd(Xall_te.astype(np.float64), Xs), sc * med) @ a
    pred_va = m52(sqd(Xva, Xs), sc * med) @ a
    print(f"  {label}: scale {sc} val {100*best[0]:.3f}%", flush=True)
    return pred_te, pred_va, dict(scale=sc, med=med, val=best[0], fit=len(sub))


class MLP(nn.Module):
    def __init__(s, di, do, w=512, dep=4):
        super().__init__(); s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(dep - 1)]); s.out = nn.Linear(w, do)

    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid:
            h = h + F.silu(l(h))
        return s.out(h)


def mean_pred():
    torch.manual_seed(args.seed)
    xt = torch.from_numpy(Xtr_n[:-nval]); ct = torch.from_numpy(Ctr[:-nval].astype(np.float32))
    xv = torch.from_numpy(Xva_n); cv = torch.from_numpy(Ctr[-nval:].astype(np.float32))
    m = MLP(d, args.npca); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 512; best, best_state = np.inf, None
    for ep in range(args.epochs):
        pm = torch.randperm(len(xt))
        for k in range(0, len(xt), bs):
            i = pm[k:k + bs]
            loss = F.mse_loss(m(xt[i]), ct[i]); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        with torch.no_grad():
            vl = F.mse_loss(m(xv), cv).item()
        if vl < best:
            best, best_state = vl, {k2: v.clone() for k2, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()

    def infer(xa):
        xa = torch.from_numpy(xa)
        with torch.no_grad():
            return np.concatenate([m(xa[k:k + 4096]).numpy() for k in range(0, len(xa), 4096)])
    return infer(Xte_n), infer(Xtr_n), infer(Xva_n)


res, hyper = {}, {}
t0 = time.time()
Pk_te, Pk_va, h = kernel_stage(Xtr_n, Xva_n, Xte_n, Ctr, lambda Cv: rel(recon(Cv), Yva), args.seed + 1, "kernel (raw input)")
res["kernel"] = dict(test=rel(recon(Pk_te), Yte), val=h["val"]); hyper["kernel"] = h
print(f"  kernel {100*res['kernel']['test']:.3f}% [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time()
Pb_te, Pb_va, h = kernel_stage(Xtr_p, Xva_p, Xte_p, Ctr, lambda Cv: rel(recon(Cv), Yva), args.seed + 2, "kernel (input PCA)")
res["kernel_pca_in"] = dict(test=rel(recon(Pb_te), Yte), val=h["val"]); hyper["kernel_pca_in"] = h
print(f"  kernel on input PCA {100*res['kernel_pca_in']['test']:.3f}% [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time()
Cp_te, Cp_tr, Cp_va = mean_pred()
res["mean"] = dict(test=rel(recon(Cp_te), Yte), val=rel(recon(Cp_va), Yva))
print(f"  mean {100*res['mean']['test']:.3f}% (val {100*res['mean']['val']:.3f}%) [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time()
R_tr = Ctr - Cp_tr
Rc_te, Rc_va, h = kernel_stage(Xtr_n, Xva_n, Xte_n, R_tr, lambda Rv: rel(recon(Cp_va + Rv), Yva), args.seed + 3, "residual kernel")
res["mean_plus_kernel"] = dict(test=rel(recon(Cp_te + Rc_te), Yte), val=h["val"]); hyper["mean_plus_kernel"] = h
Rp_te, Rp_va, h2 = kernel_stage(Xtr_p, Xva_p, Xte_p, R_tr, lambda Rv: rel(recon(Cp_va + Rv), Yva), args.seed + 4, "residual kernel (input PCA)")
res["mean_plus_kernel_pca_in"] = dict(test=rel(recon(Cp_te + Rp_te), Yte), val=h2["val"]); hyper["mean_plus_kernel_pca_in"] = h2
print(f"  mean+kernel {100*res['mean_plus_kernel']['test']:.3f}%  (input PCA {100*res['mean_plus_kernel_pca_in']['test']:.3f}%) [{time.time()-t0:.0f}s]", flush=True)
cands = {"kernel": (Pk_va, Pk_te), "kernel_pca_in": (Pb_va, Pb_te), "mean": (Cp_va, Cp_te), "mean_plus_kernel": (Cp_va + Rc_va, Cp_te + Rc_te)}
Cva_true = Ctr[-nval:]
sel_te = np.empty_like(Cp_te); picks = {k: 0 for k in cands}
for j in range(args.npca):
    errs = {k: float(((v[0][:, j] - Cva_true[:, j]) ** 2).mean()) for k, v in cands.items()}
    kbest = min(errs, key=errs.get); picks[kbest] += 1; sel_te[:, j] = cands[kbest][1][:, j]
res["select"] = dict(test=rel(recon(sel_te), Yte), picks=picks)
stage_val = {k: v["val"] for k, v in res.items() if "val" in v}
res["val_selected"] = min(stage_val, key=stage_val.get)
for k, v in res.items():
    if isinstance(v, dict) and "test" in v:
        print(f"  {k:26s} {100*v['test']:.3f}%", flush=True)
tag = args.tag or f"hidata_{args.name}_s{args.seed}"
out = dict(tag=tag, kind="hidata", name=args.name, seed=args.seed, ntrain=len(tr), ntest=len(te), npca=args.npca, npca_in=args.npca_in,
           fit=args.fit, epochs=args.epochs, smoke=bool(args.smoke), results=res, hyper=hyper,
           pca_floor_te=float(pca_floor_te), pca_var_out=float((s[:args.npca] ** 2).sum() / (s ** 2).sum()), pca_var_in=float((s_in[:args.npca_in] ** 2).sum() / (s_in ** 2).sum()),
           minutes=round((time.time() - t_all) / 60, 1))
OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / (tag + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT / (tag + ".json"))
print(f"DONE {tag} in {out['minutes']} min", flush=True)
