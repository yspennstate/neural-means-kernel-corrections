"""Bigger kernels trained by Kernel Flows (the JPL cross-validation method).

Replaces the isotropic Matern-5/2 length scale with a learned metric and fits
it by the Kernel Flows rho loss (half-sample cross-validation): on a random
batch, rho = 1 - ||f_c||^2_K / ||f_b||^2_K, where f_b interpolates the whole
batch and f_c half of it. Minimizing rho by SGD selects the metric that leaves
the target smoothest in the kernel's native space -- exactly the quantity the
optimal-recovery bound multiplies. Three metrics:

  iso        one length scale (the current kernel)
  ard        per-dimension log length scales (d params)
  lowrank    Mahalanobis M = L^T L + diag(delta), L is r-by-d (r*d + d params)

Everything runs on the GPU. Reports test error of an exact ridge solve with
each learned metric, on the reduced OCO-2 O2 task (raw state -> reduced radiance).

    python learnable_kernel_gpu.py
"""
import time, pathlib, json
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
DT = torch.float64

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--space", default="state", choices=["state", "features"])
A = ap.parse_args()

import h5py
with h5py.File(HERE / "data" / "dimred_variables_4_mono.jld", "r") as h:
    X = h["xr_o2"][:].astype(np.float64); Y = h["z_o2"][:].astype(np.float64)
    Xte = h["xr_o2_test"][:].astype(np.float64); Yte = h["z_o2_test"][:].astype(np.float64)

if A.space == "features":
    # train an O2 residual MLP and replace inputs with its penultimate features,
    # the representation where the exact kernel reaches 3.82% in the paper
    import torch.nn as nn, torch.nn.functional as F
    class Net(nn.Module):
        def __init__(s, di, do, w=256):
            super().__init__(); s.inp = nn.Linear(di, w)
            s.h1 = nn.Linear(w, w); s.h2 = nn.Linear(w, w); s.out = nn.Linear(w, do)
        def feat(s, x):
            h = F.silu(s.inp(x)); h = h + F.silu(s.h1(h)); return h + F.silu(s.h2(h))
        def forward(s, x): return s.out(s.feat(x))
    di = X.shape[1]; f32 = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEV)
    ym, ys = Y.mean(0), Y.std(0) + 1e-9
    net = Net(di, Y.shape[1]).to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    xt = f32((X - X.mean(0)) / (X.std(0) + 1e-9)); yt = f32((Y - ym) / ys)
    xte = f32((Xte - X.mean(0)) / (X.std(0) + 1e-9))
    torch.manual_seed(0)
    for ep in range(600):
        pm = torch.randperm(len(xt), device=DEV)
        for k in range(0, len(xt), 256):
            i = pm[k:k + 256]
            loss = F.mse_loss(net(xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        X = net.feat(xt).cpu().numpy().astype(np.float64)
        Xte = net.feat(xte).cpu().numpy().astype(np.float64)
    print(f"trained O2 MLP; using {X.shape[1]}-dim features", flush=True)
n, d = X.shape; do = Y.shape[1]
rng = np.random.default_rng(0); perm = rng.permutation(n)
ntr = min(6000, n - 1)                      # cap the exact solve
tr, va = perm[:ntr], perm[ntr:ntr + 2000] if n - ntr > 2000 else perm[ntr:]
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
t = lambda a: torch.tensor((np.asarray(a) - mu) / sd, dtype=DT, device=DEV)
Xtr, Xva, Xt = t(X[tr]), t(X[va]), t(Xte)
Ytr = torch.tensor(Y[tr], dtype=DT, device=DEV)
Yva_np, Yte_np = Y[va], Yte
print(f"O2 state->reduced: d={d}, do={do}, train {len(tr)}, test {len(Xte)}", flush=True)


def rel_np(P, T):
    return float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))


def metric_sqdist(A, B, P):
    """squared Mahalanobis distances ||P(a-b)||^2 with metric map P (k-by-d)."""
    Ap, Bp = A @ P.T, B @ P.T
    return torch.clamp((Ap * Ap).sum(1)[:, None] + (Bp * Bp).sum(1)[None, :] - 2 * Ap @ Bp.T, min=0.0)


def m52(D2):
    r = torch.sqrt(D2 + 1e-30)
    return (1 + np.sqrt(5) * r + 5.0 / 3.0 * D2) * torch.exp(-np.sqrt(5) * r)


def rho_loss(P, Xb, Yb, nug):
    """Kernel Flows half-sample CV loss on a batch."""
    b = len(Xb); h = b // 2
    Kb = m52(metric_sqdist(Xb, Xb, P)) + nug * torch.eye(b, dtype=DT, device=DEV)
    Kc = Kb[:h, :h]
    yb, yc = Yb, Yb[:h]
    # ||f||^2_K = y^T K^{-1} y  (summed over outputs)
    nb = (yb * torch.linalg.solve(Kb, yb)).sum()
    nc = (yc * torch.linalg.solve(Kc, yc)).sum()
    return torch.clamp(1 - nc / (nb + 1e-30), min=1e-6)


def fit_metric(kind, steps=400, bs=256, r=6):
    torch.manual_seed(0)
    logdelta = torch.zeros((), dtype=DT, device=DEV, requires_grad=True)   # log nugget
    med = float(np.sqrt(np.median(((X[tr][:1500, None] - X[tr][None, :1500]) ** 2).sum(-1)
                                   [np.triu_indices(1500, 1)])) + 1e-9)
    s0 = 1.0 / med
    if kind == "iso":
        raw = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [raw, logdelta]
        Pfun = lambda: torch.exp(raw[0]) * torch.eye(d, dtype=DT, device=DEV)
    elif kind == "ard":
        raw = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [raw, logdelta]
        Pfun = lambda: torch.diag(torch.exp(raw))
    else:  # lowrank Mahalanobis
        L = (s0 * torch.randn(r, d, dtype=DT, device=DEV) * 0.3).requires_grad_(True)
        diag = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [L, diag, logdelta]
        Pfun = lambda: torch.cat([L, torch.diag(torch.exp(diag))], 0)
    opt = torch.optim.Adam(params, lr=5e-2)
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    for st in range(steps):
        idx = torch.randint(0, len(Xtr), (bs,), generator=gen, device=DEV)
        loss = rho_loss(Pfun(), Xtr[idx], Ytr[idx], torch.exp(logdelta) + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
    return Pfun().detach(), float(torch.exp(logdelta).detach()) + 1e-8


def test_error(P0, nug0):
    """Exact ridge solve with metric P0. Kernel flows fixes the metric SHAPE;
    the global scale and nugget are set on validation (pure rho-minimization is
    degenerate in the overall scale, so this is the standard division of labor)."""
    best = (np.inf, None)
    for scale in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        P = scale * P0
        Ktr = m52(metric_sqdist(Xtr, Xtr, P))
        Kva = m52(metric_sqdist(Xva, Xtr, P))
        for nug in (1e-8, 1e-6, 1e-4, 1e-2):
            A = Ktr + nug * len(Xtr) * torch.eye(len(Xtr), dtype=DT, device=DEV)
            try: al = torch.linalg.solve(A, Ytr)
            except Exception: continue
            e = rel_np((Kva @ al).cpu().numpy(), Yva_np)
            if e < best[0]: best = (e, (scale, nug))
    scale, nug = best[1]
    P = scale * P0
    A = m52(metric_sqdist(Xtr, Xtr, P)) + nug * len(Xtr) * torch.eye(len(Xtr), dtype=DT, device=DEV)
    al = torch.linalg.solve(A, Ytr)
    pt = (m52(metric_sqdist(Xt, Xtr, P)) @ al).cpu().numpy()
    return rel_np(pt, Yte_np), best[1]


res = {}
for kind in ("iso", "ard", "lowrank"):
    t0 = time.time()
    P, nug = fit_metric(kind)
    e, (scale, nsel) = test_error(P, nug)
    res[kind] = round(100 * e, 3)
    print(f"  {kind:8s}  test {100*e:6.2f}%   (KF shape + val scale {scale}, {time.time()-t0:.0f}s)", flush=True)

json.dump(res, open(HERE / "learnable_kernel_result.json", "w"), indent=1)
print("saved learnable_kernel_result.json:", res, flush=True)
