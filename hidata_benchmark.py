"""High-data operator benchmarks (Helmholtz, Navier-Stokes) with output PCA, to
match Batlle et al.'s protocol so the numbers are comparable to the suite table.

Fields are large (101x101 or 64x64 over 40000 samples), so we memory-map, load
a 20000-train / 2000-test subset as float32, reduce the OUTPUT to a few PCA
components (as Batlle does), fit the kernel (capped) and a neural mean from the
flattened input to the PCA coefficients on the GPU, reconstruct, and report the
mean relative L2 on the full field.

    python hidata_benchmark.py --name NavierStokes --grid 64
    python hidata_benchmark.py --name Helmholtz --grid 101
"""
import argparse, time, json, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"   # <Name>_inputs.npy / <Name>_outputs.npy from the Caltech record
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--grid", type=int, required=True)
p.add_argument("--ntrain", type=int, default=20000)
p.add_argument("--ntest", type=int, default=2000)
p.add_argument("--npca", type=int, default=40)      # output PCA components (Batlle uses ~ this)
p.add_argument("--fit", type=int, default=6000)     # kernel-fit cap
p.add_argument("--epochs", type=int, default=150)
args = p.parse_args()

Xm = np.load(DATA / f"{args.name}_inputs.npy", mmap_mode="r")   # (g, g, N)
Ym = np.load(DATA / f"{args.name}_outputs.npy", mmap_mode="r")
N = Xm.shape[-1]; d = args.grid * args.grid
rng = np.random.default_rng(0); perm = rng.permutation(N)
tr = np.sort(perm[:args.ntrain]); te = np.sort(perm[args.ntrain:args.ntrain + args.ntest])

def grab(M, idx):  # (g,g,N)[...,idx] -> (len(idx), g*g) float32
    return np.moveaxis(np.asarray(M[..., idx], np.float32), -1, 0).reshape(len(idx), -1)

Xtr, Xte = grab(Xm, tr), grab(Xm, te)
Ytr, Yte = grab(Ym, tr), grab(Ym, te)
print(f"{args.name}: {N} samples, {d}->{d}, using {len(tr)}/{len(te)}, output PCA {args.npca}", flush=True)

# standardize inputs
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
# output PCA from a training subsample, on the GPU. A full 20000-by-d LAPACK
# SVD allocates several GB and has died silently on this machine; Otto's
# reduce_dim.jl likewise takes the SVD of the first 5000 columns only.
ym = Ytr.mean(0)
nsvd = min(6000, len(Ytr))
Yg = torch.tensor(Ytr[:nsvd] - ym, device=DEV)
U_t, s_t, Vh = torch.linalg.svd(Yg, full_matrices=False)
s = s_t.cpu().numpy().astype(np.float64)
P = Vh[:args.npca].T.cpu().numpy().astype(np.float32)  # (d, npca)
del Yg, U_t, s_t, Vh
if DEV == "cuda": torch.cuda.empty_cache()
Ctr = (Ytr - ym) @ P                                   # train PCA coeffs
recon = lambda C: C @ P.T + ym
rel = lambda Pr, T: float(np.mean(np.linalg.norm(Pr - T, axis=1) / np.linalg.norm(T, axis=1)))
print(f"  output PCA keeps {100*(s[:args.npca]**2).sum()/(s**2).sum():.2f}% of variance", flush=True)


def sqd(A, B): return np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-2*A@B.T, 0.0)
def m52(D2, ls):
    r = np.sqrt(D2)/ls; return (1+np.sqrt(5)*r+5*D2/(3*ls**2))*np.exp(-np.sqrt(5)*r)

def kernel_pred():
    # validation carve-out from the END of the train block: hyperparameters are
    # selected there, never on test points
    nval = 1000
    fit_pool = np.arange(len(Xtr_n) - nval)
    Xva = Xtr_n[-nval:].astype(np.float64); Yva = Ytr[-nval:]
    sub = np.random.default_rng(1).choice(fit_pool, min(args.fit, len(fit_pool)), replace=False)
    Xs = Xtr_n[sub].astype(np.float64); Cs = Ctr[sub].astype(np.float64)
    dsub = sqd(Xs[:1500], Xs[:1500]); med = float(np.sqrt(np.median(dsub[np.triu_indices(min(1500,len(Xs)),1)]))+1e-9)
    D2 = sqd(Xs, Xs); best = (np.inf, None)
    Kva = {}
    for sc in (0.5,1,2,4):
        Kva[sc] = m52(sqd(Xva, Xs), sc*med)
        for nug in (1e-8,1e-6,1e-4):
            K = m52(D2, sc*med); K.flat[::len(Xs)+1] += nug*len(Xs)
            try: a = cho_solve(cho_factor(K, lower=True, check_finite=False), Cs, check_finite=False)
            except np.linalg.LinAlgError: continue
            e = rel(recon(Kva[sc]@a), Yva)
            if e < best[0]: best = (e, (a, sc, med, sub))
    a, sc, med, sub = best[1]
    Xs = Xtr_n[sub].astype(np.float64)
    return (recon(m52(sqd(Xte_n.astype(np.float64), Xs), sc*med) @ a),
            recon(m52(sqd(Xva, Xs), sc*med) @ a))


class MLP(nn.Module):
    def __init__(s, di, do, w=512, dep=4):
        super().__init__(); s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(dep-1)]); s.out = nn.Linear(w, do)
    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)


def mean_pred():
    # train on the first len-1000 rows only; the val tail stays untouched
    nval = 1000
    torch.manual_seed(0)
    xt = torch.from_numpy(Xtr_n[:-nval]); ct = torch.from_numpy(Ctr[:-nval].astype(np.float32))
    m = MLP(d, args.npca).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 512
    for ep in range(args.epochs):
        pm = torch.randperm(len(xt))
        for k in range(0, len(xt), bs):
            i = pm[k:k+bs]
            loss = F.mse_loss(m(xt[i].to(DEV)), ct[i].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    def infer(xa):
        xa = torch.from_numpy(xa)
        with torch.no_grad():
            return np.concatenate([m(xa[k:k+4096].to(DEV)).cpu().numpy()
                                   for k in range(0, len(xa), 4096)])
    return infer(Xte_n), infer(Xtr_n[:-nval]), infer(Xtr_n[-nval:])   # test, fit-pool, val


def krr_resid(Rfit, Rva):
    """kernel on the mean's residual coefficients; fit and selection both away
    from the val tail (Rfit indexes the fit pool), selection on val residuals."""
    nval = 1000
    fit_pool = np.arange(len(Xtr_n) - nval)
    sub = np.random.default_rng(2).choice(fit_pool, min(args.fit, len(fit_pool)), replace=False)
    Xs = Xtr_n[sub].astype(np.float64); Rs = Rfit[sub].astype(np.float64)
    Xva = Xtr_n[-nval:].astype(np.float64)
    dsub = sqd(Xs[:1500], Xs[:1500]); med = float(np.sqrt(np.median(dsub[np.triu_indices(min(1500,len(Xs)),1)]))+1e-9)
    D2 = sqd(Xs, Xs); best = (np.inf, None)
    for sc in (0.5,1,2,4):
        Kva = m52(sqd(Xva, Xs), sc*med)
        for nug in (1e-8,1e-6,1e-4):
            K = m52(D2, sc*med); K.flat[::len(Xs)+1] += nug*len(Xs)
            try: a = cho_solve(cho_factor(K, lower=True, check_finite=False), Rs, check_finite=False)
            except np.linalg.LinAlgError: continue
            e = np.mean((Kva@a - Rva)**2)
            if e < best[0]: best = (e, (a, sc, med, sub))
    a, sc, med, sub = best[1]
    return m52(sqd(Xte_n.astype(np.float64), Xtr_n[sub].astype(np.float64)), sc*med) @ a, \
           m52(sqd(Xtr_n[-nval:].astype(np.float64), Xtr_n[sub].astype(np.float64)), sc*med) @ a


res = {}
nval = 1000
Yva_f = Ytr[-nval:]                                      # val fields for stage choice
t0 = time.time(); Pk, Pk_va = kernel_pred()
res["kernel"] = round(100*rel(Pk, Yte), 3); kv = rel(Pk_va, Yva_f)
print(f"  kernel        {res['kernel']:.2f}%  (val {100*kv:.2f}%)  [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time(); Cp_te, Cp_fit, Cp_va = mean_pred()
res["mean"] = round(100*rel(recon(Cp_te), Yte), 3); mv = rel(recon(Cp_va), Yva_f)
print(f"  mean          {res['mean']:.2f}%  (val {100*mv:.2f}%)  [{time.time()-t0:.0f}s]", flush=True)
t0 = time.time()
corr_te, corr_va = krr_resid(Ctr[:-nval] - Cp_fit, Ctr[-nval:] - Cp_va)
res["mean_plus_kernel"] = round(100*rel(recon(Cp_te + corr_te), Yte), 3)
cv = rel(recon(Cp_va + corr_va), Yva_f)
print(f"  mean+kernel   {res['mean_plus_kernel']:.2f}%  (val {100*cv:.2f}%)  [{time.time()-t0:.0f}s]", flush=True)
stage = {"kernel": kv, "mean": mv, "mean_plus_kernel": cv}
res["val_selected"] = min(stage, key=stage.get)
print(f"  val selects:  {res['val_selected']}", flush=True)

pathlib.Path(HERE / "hidata_out").mkdir(exist_ok=True)
json.dump(res, open(HERE / "hidata_out" / f"{args.name}.json", "w"), indent=1)
print(f"  saved {args.name}.json:", res, flush=True)
