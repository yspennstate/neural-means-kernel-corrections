"""Data-scaling on ClimSim (LEAP subsampled low-res): does the neural mean keep
getting stronger with data at the 10^3 -> 10^6 scale OCO-2 cannot reach?

State (124 features: T, q profiles + surface) -> tendencies (128: heating,
moistening, fluxes). Pre-normalized (LEAP). We hold a fixed test set and sweep
the training size, training the mean on the GPU and fitting the kernel on a
capped subsample. Reports R^2 (the ClimSim metric) and relative L2.

    python climsim_scaling.py
"""
import time, json, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

HERE = pathlib.Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="val", choices=["val", "train"])
A = ap.parse_args()
# memory-map so we never pull the whole multi-million-row array into RAM (silent-death guard)
Xin = np.load(HERE / "climsim" / f"{A.data}_input.npy", mmap_mode="r")
Yin = np.load(HERE / "climsim" / f"{A.data}_target.npy", mmap_mode="r")
N, di = Xin.shape; do = Yin.shape[1]
print(f"ClimSim: {N} rows, {di} -> {do}", flush=True)

rng = np.random.default_rng(0)
perm = rng.permutation(N)
te = np.sort(perm[:20000]); pool = perm[20000:]
Xte = np.asarray(Xin[te], np.float64); Yte = np.asarray(Yin[te], np.float64)
# standardize inputs on a fixed reference chunk (causal: independent of test)
ref = np.sort(pool[:50000])
mu, sd = np.asarray(Xin[ref], np.float64).mean(0), np.asarray(Xin[ref], np.float64).std(0) + 1e-9
mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)     # bulk load stays float32
Xte_n = (Xte - mu) / sd
yvar = Yte.var(0) + 1e-12                       # per-output variance for R^2
active = yvar > 1e-6 * yvar.max()               # ClimSim has near-constant outputs; mask them
print(f"  R2 over {int(active.sum())}/{do} active outputs (rest are near-constant)", flush=True)

sizes = ([1000, 3000, 10000, 30000, 100000, 300000, 700000, 1000000, 1200000]
         if A.data == "val" else
         [1000, 10000, 100000, 300000, 1000000, 2000000, 3500000])


def r2(P, T):
    num = ((P - T) ** 2).mean(0)
    return float(np.mean(1 - num[active] / yvar[active]))     # masked mean per-output R^2


def rel(P, T):
    n = np.linalg.norm(T, axis=1); m = n > 1e-9
    return float(np.mean(np.linalg.norm(P[m] - T[m], axis=1) / n[m]))


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    r = np.sqrt(D2) / ls
    return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def kernel_pred(Xtr, Ytr, cap=6000):
    sub = np.random.default_rng(1).choice(len(Xtr), min(cap, len(Xtr)), replace=False)
    Xs, Ys = Xtr[sub].astype(np.float64), Ytr[sub].astype(np.float64)   # cast only the subsample
    dsub = sqd(Xs[:1500], Xs[:1500])
    med = float(np.sqrt(np.median(dsub[np.triu_indices(min(1500, len(Xs)), 1)])) + 1e-9)
    D2 = sqd(Xs, Xs); best = (np.inf, None)
    for s in (1, 2, 4):
        for nug in (1e-6, 1e-4, 1e-2):
            K = m52(D2, s * med); K.flat[::len(Xs) + 1] += nug * len(Xs)
            try: a = cho_solve(cho_factor(K, lower=True, check_finite=False), Ys, check_finite=False)
            except np.linalg.LinAlgError: continue
            e = ((m52(sqd(Xte_n[:2000], Xs), s * med) @ a - Yte[:2000]) ** 2).mean()
            if e < best[0]: best = (e, (a, s, med, sub))
    a, s, med, sub = best[1]
    return m52(sqd(Xte_n, Xs), s * med) @ a


class MLP(nn.Module):
    def __init__(s, di, do, w=512, d=4):
        super().__init__(); s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)]); s.out = nn.Linear(w, do)
    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)


def train_mean(Xtr, Ytr, epochs=60):
    torch.manual_seed(0)
    # keep the big arrays in CPU RAM, move only batches to the GPU (millions of
    # rows would not fit as one GPU tensor -- crash-scar guard)
    xt = torch.from_numpy(np.ascontiguousarray(Xtr, np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(Ytr, np.float32))
    xte = torch.tensor(Xte_n, dtype=torch.float32, device=DEV)
    m = MLP(di, do).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = 4096; N = len(xt)
    for ep in range(epochs):
        pm = torch.randperm(N)
        for k in range(0, N, bs):
            i = pm[k:k + bs]
            xb = xt[i].to(DEV, non_blocking=True); yb = yt[i].to(DEV, non_blocking=True)
            loss = F.mse_loss(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        out = []
        for k in range(0, len(xte), 8192):
            out.append(m(xte[k:k + 8192]).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


results = []
for n in sizes:
    if n > len(pool) - 1000:
        break
    t0 = time.time()
    idx = np.sort(pool[:n])
    Xtr = (np.asarray(Xin[idx], np.float32) - mu32) / sd32
    Ytr = np.asarray(Yin[idx], np.float32)
    Pm = train_mean(Xtr, Ytr)
    row = dict(n=int(n), mean_r2=round(r2(Pm, Yte), 4), mean_rel=round(100 * rel(Pm, Yte), 3))
    if n <= 100000:                                  # kernel only where the fit is affordable
        Pk = kernel_pred(Xtr, Ytr)
        row["kernel_r2"] = round(r2(Pk, Yte), 4)
    print(f"  n={n:7d}: mean R2 {row['mean_r2']:.3f} rel {row['mean_rel']:.2f}%"
          + (f"  kernel R2 {row.get('kernel_r2')}" if 'kernel_r2' in row else "")
          + f"  [{time.time()-t0:.0f}s]", flush=True)
    results.append(row)
    json.dump({"dataset": "ClimSim", "din": di, "dout": do, "ntest": len(te), "results": results},
              open(HERE / f"scaling_climsim_{A.data}.json", "w"), indent=1)

print(f"saved scaling_climsim_{A.data}.json ({len(results)} points)", flush=True)
