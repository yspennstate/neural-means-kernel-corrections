"""Data-scaling study: test error vs training-set size, for the JPL emulation
tasks and (optionally) the reduced OCO-2. Trains the neural mean on the GPU,
fits the kernel on a capped subsample (to keep the Cholesky affordable and
avoid silent big-memory deaths), holds out a fixed test set, and records the
error at each n. Produces the numbers behind the scaling figure.

    python scaling_study.py --task full     # 58 -> 3048 full-resolution
    python scaling_study.py --task reduced   # 20 -> 40 reduced O2
"""
import argparse, time, json, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

HERE = pathlib.Path(__file__).resolve().parent
torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--task", default="full")
p.add_argument("--fitcap", type=int, default=6000)   # kernel-fit subsample cap
p.add_argument("--epochs", type=int, default=800)
args = p.parse_args()

if args.task == "full":
    X = np.load(HERE / "data" / "jpl_full_X.npy").astype(np.float64)
    Y = np.load(HERE / "data" / "jpl_full_Y.npy").astype(np.float64)
    sizes = [100, 200, 300, 500, 700]
    ntest = 150
else:
    import h5py
    with h5py.File(HERE / "data" / "dimred_variables_4_mono.jld", "r") as h:
        X = h["xr_o2"][:].astype(np.float64); Y = h["z_o2"][:].astype(np.float64)
        Xte0 = h["xr_o2_test"][:].astype(np.float64); Yte0 = h["z_o2_test"][:].astype(np.float64)
    sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
    ntest = None

rng = np.random.default_rng(0); perm = rng.permutation(len(X))
if ntest is None:
    Xte, Yte = Xte0, Yte0
    pool = perm
else:
    te = perm[:ntest]; Xte, Yte = X[te], Y[te]; pool = perm[ntest:]

mu, sd = X[pool].mean(0), X[pool].std(0) + 1e-9
Xn = (X - mu) / sd; Xte_n = (Xte - mu) / sd
rel = lambda P, T: float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    r = np.sqrt(D2) / ls
    return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def med_ls(Z):
    k = min(1500, len(Z)); idx = np.random.default_rng(0).choice(len(Z), k, replace=False)
    d = sqd(Z[idx], Z[idx]); return float(np.sqrt(np.median(d[np.triu_indices(k, 1)])) + 1e-12)


def kernel_err(Xtr, Ytr, Xva, Yva):
    sub = np.random.default_rng(1).choice(len(Xtr), min(args.fitcap, len(Xtr)), replace=False)
    med = med_ls(Xtr[sub]); D2 = sqd(Xtr[sub], Xtr[sub]); best = np.inf
    for s in (0.5, 1, 2, 4):
        for nug in (1e-8, 1e-6, 1e-4):
            K = m52(D2, s * med); K.flat[::len(sub) + 1] += nug * len(sub)
            try: a = cho_solve(cho_factor(K, lower=True, check_finite=False), Ytr[sub], check_finite=False)
            except np.linalg.LinAlgError: continue
            e = rel(m52(sqd(Xva, Xtr[sub]), s * med) @ a, Yva)
            if e < best: best = e; besta = (a, s, med, sub)
    a, s, med, sub = besta
    return lambda Xq: m52(sqd(Xq, Xtr[sub]), s * med) @ a, best


class MLP(nn.Module):
    def __init__(s, di, do, w=512, d=4):
        super().__init__(); s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)]); s.out = nn.Linear(w, do)
    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)


def train_mean(Xtr, Ytr, Xva, Yva, di, do):
    torch.manual_seed(0)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-9
    xt, xv = f32(Xtr), f32(Xva); yt = (f32(Ytr) - f32(ymu)) / f32(ysd)
    ym, ys = f32(ymu), f32(ysd)
    m = MLP(di, do).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best, sd_ = np.inf, None
    bs = min(256, max(32, len(xt) // 4))
    for ep in range(args.epochs):
        pm = torch.randperm(len(xt), device=DEV)
        for k in range(0, len(xt), bs):
            i = pm[k:k + bs]
            pr = m(xt[i]) * ys + ym; tg = yt[i] * ys + ym
            loss = (torch.linalg.vector_norm(pr - tg, dim=1) / torch.linalg.vector_norm(tg, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            m.eval()
            with torch.no_grad(): e = rel((m(xv) * ys + ym).cpu().numpy(), Yva)
            m.train()
            if e < best: best, sd_ = e, {k2: v.clone() for k2, v in m.state_dict().items()}
    m.load_state_dict(sd_); m.eval()
    def f(Z):
        with torch.no_grad():
            return (m(f32(Z)) * ys + ym).cpu().numpy()
    return f, best


di, do = X.shape[1], Y.shape[1]
results = []
for n in sizes:
    if n > len(pool) - 200:
        break
    t0 = time.time()
    idx = pool[:n]; nv = max(20, min(n // 6, 300))
    tr, va = idx[nv:], idx[:nv]
    Xtr, Ytr, Xva, Yva = Xn[tr], Y[tr], Xn[va], Y[va]
    kf, kve = kernel_err(Xtr, Ytr, Xva, Yva)
    ek = rel(kf(Xte_n), Yte)
    mf, mve = train_mean(Xtr, Ytr, Xva, Yva, di, do)
    em = rel(mf(Xte_n), Yte)
    # mean + kernel correction of the residual
    kfc, _ = kernel_err(Xtr, Ytr - mf(Xtr), Xva, Yva - mf(Xva))
    emk = rel(mf(Xte_n) + kfc(Xte_n), Yte)
    best = min(ek, emk)
    results.append(dict(n=int(len(tr)), kernel=100 * ek, mean=100 * em, mean_kernel=100 * emk, best=100 * best))
    print(f"  n={len(tr):5d}: kernel {100*ek:6.2f}%  mean {100*em:6.2f}%  mean+kernel {100*emk:6.2f}%  best {100*best:6.2f}%  [{time.time()-t0:.0f}s]", flush=True)
    # incremental save so a timeout never loses completed points
    json.dump(dict(task=args.task, din=di, dout=do, ntest=len(Xte), results=results),
              open(HERE / f"scaling_{args.task}.json", "w"), indent=1)

print(f"saved scaling_{args.task}.json ({len(results)} points)", flush=True)
