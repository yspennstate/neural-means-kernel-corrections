"""Kernel-flows learned metrics on the OCO-2 O2 task at larger sizes, on the
GPU, extending learnable_kernel_gpu.py of the NMKC repository (the paper's
statement: an ARD metric and a low-rank Mahalanobis metric fit by the
kernel-flows rho loss give no gain over the isotropic Matern-5/2 kernel, on the
raw state and on the network's features).

Held fixed from the paper: the rho loss (half-sample cross-validation on a
batch), Adam lr 5e-2, the validation selection of the global scale and nugget
after the metric shape is learned, the exact ridge test error, float64.
Knobs: --ntr (exact-solve rows, paper 6000), --steps (paper 400), --bs (256),
--rank (low-rank metric rank, paper 6), --seed, --space state|features.

    python lk_scale.py --space state --ntr 6000 --steps 400 --bs 256 --rank 6 --seed 0
"""
import os, argparse, hashlib, json, sys, threading, time
from pathlib import Path
import numpy as np
import torch

W = Path(os.environ.get("NMKC_W", "C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments"))
ON_CPU = os.environ.get("NMKC_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available()
DEV = "cpu" if ON_CPU else "cuda"; DT = torch.float64
ap = argparse.ArgumentParser()
ap.add_argument("--space", default="state", choices=["state", "features"])
ap.add_argument("--ntr", type=int, default=6000)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--rank", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kinds", default="iso,ard,lowrank")
ap.add_argument("--outdir", default=str(W / "runs" / "lk"))
A = ap.parse_args()
name = f"lk_{A.space}_n{A.ntr}_st{A.steps}_bs{A.bs}_r{A.rank}_s{A.seed}"
OUT = Path(A.outdir); OUT.mkdir(parents=True, exist_ok=True)
if (OUT / f"{name}.json").exists():
    print("done already:", name); sys.exit(0)
torch.set_num_threads(int(os.environ.get("NMKC_THREADS", "8")) if ON_CPU else 2)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


import h5py
DF = W / "data" / "jpl_oco2" / "dimred_variables_4_mono.jld"
with h5py.File(DF, "r") as h:
    X = h["xr_o2"][:].astype(np.float64); Y = h["z_o2"][:].astype(np.float64)
    Xte = h["xr_o2_test"][:].astype(np.float64); Yte = h["z_o2_test"][:].astype(np.float64)

util = []
def sampler(stop):
    try:
        if ON_CPU:
            return
        import pynvml
        pynvml.nvmlInit(); hh = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            try: util.append(pynvml.nvmlDeviceGetUtilizationRates(hh).gpu)
            except Exception: pass
            stop.wait(5)
    except Exception:
        util.append(-1)
stop = threading.Event(); threading.Thread(target=sampler, args=(stop,), daemon=True).start()
t_all = time.time()

if A.space == "features":
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
    torch.manual_seed(A.seed)
    for ep in range(600):
        pm = torch.randperm(len(xt), device=DEV)
        for k in range(0, len(xt), 256):
            i = pm[k:k + 256]
            loss = F.mse_loss(net(xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        X = net.feat(xt).cpu().numpy().astype(np.float64)
        Xte = net.feat(xte).cpu().numpy().astype(np.float64)
    print(f"trained O2 MLP; using {X.shape[1]}-dim features [{time.time()-t_all:.0f}s]", flush=True)
n, d = X.shape; do = Y.shape[1]
rng = np.random.default_rng(A.seed); perm = rng.permutation(n)
ntr = min(A.ntr, n - 2000)
tr, va = perm[:ntr], perm[ntr:ntr + 2000]
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
t = lambda a: torch.tensor((np.asarray(a) - mu) / sd, dtype=DT, device=DEV)
Xtr, Xva, Xt = t(X[tr]), t(X[va]), t(Xte)
Ytr = torch.tensor(Y[tr], dtype=DT, device=DEV)
Yva_np, Yte_np = Y[va], Yte
print(f"O2 {A.space}->reduced: d={d}, do={do}, train {len(tr)}, val {len(va)}, test {len(Xte)}", flush=True)


def rel_np(P, T):
    return float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))


def metric_sqdist(Aa, B, P):
    Ap, Bp = Aa @ P.T, B @ P.T
    return torch.clamp((Ap * Ap).sum(1)[:, None] + (Bp * Bp).sum(1)[None, :] - 2 * Ap @ Bp.T, min=0.0)


def m52(D2):
    r = torch.sqrt(D2 + 1e-30)
    return (1 + np.sqrt(5) * r + 5.0 / 3.0 * D2) * torch.exp(-np.sqrt(5) * r)


def rho_loss(P, Xb, Yb, nug):
    b = len(Xb); h = b // 2
    Kb = m52(metric_sqdist(Xb, Xb, P)) + nug * torch.eye(b, dtype=DT, device=DEV)
    Kc = Kb[:h, :h]
    yb, yc = Yb, Yb[:h]
    nb = (yb * torch.linalg.solve(Kb, yb)).sum()
    nc = (yc * torch.linalg.solve(Kc, yc)).sum()
    return torch.clamp(1 - nc / (nb + 1e-30), min=1e-6)


def fit_metric(kind, steps, bs, r):
    torch.manual_seed(A.seed)
    logdelta = torch.zeros((), dtype=DT, device=DEV, requires_grad=True)
    med = float(np.sqrt(np.median(((X[tr][:1500, None] - X[tr][None, :1500]) ** 2).sum(-1)[np.triu_indices(1500, 1)])) + 1e-9)
    s0 = 1.0 / med
    if kind == "iso":
        raw = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [raw, logdelta]
        Pfun = lambda: torch.exp(raw[0]) * torch.eye(d, dtype=DT, device=DEV)
    elif kind == "ard":
        raw = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [raw, logdelta]
        Pfun = lambda: torch.diag(torch.exp(raw))
    else:
        L = (s0 * torch.randn(r, d, dtype=DT, device=DEV) * 0.3).requires_grad_(True)
        diag = torch.full((d,), np.log(s0), dtype=DT, device=DEV, requires_grad=True)
        params = [L, diag, logdelta]
        Pfun = lambda: torch.cat([L, torch.diag(torch.exp(diag))], 0)
    opt = torch.optim.Adam(params, lr=5e-2)
    gen = torch.Generator(device=DEV); gen.manual_seed(A.seed)
    hist = []
    for st in range(steps):
        idx = torch.randint(0, len(Xtr), (bs,), generator=gen, device=DEV)
        loss = rho_loss(Pfun(), Xtr[idx], Ytr[idx], torch.exp(logdelta) + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        if (st + 1) % 100 == 0:
            hist.append(float(loss))
    return Pfun().detach(), float(torch.exp(logdelta).detach()) + 1e-8, hist


sys.path.insert(0, str(W / "spectrum"))
from gpu_blocked_routes import blocked_cholesky_inplace, pace  # noqa: E402


def kernel_into(out, Aa, B, P, block=1500):
    """Matern-5/2 kernel of the metric map P, written into `out` in row blocks: the only
    matrix-sized array on the card is `out` itself (m52(metric_sqdist(...)) on 12000 rows
    would allocate four 1.15 GB temporaries, above the fleet's 3.2 GB per-process cap)."""
    Ap, Bp = Aa @ P.T, B @ P.T
    nb = (Bp * Bp).sum(1)[None, :]
    tp = time.perf_counter()
    for s in range(0, len(Ap), block):
        e = min(s + block, len(Ap))
        D2 = torch.clamp((Ap[s:e] * Ap[s:e]).sum(1)[:, None] + nb - 2 * Ap[s:e] @ Bp.T, min=0.0)
        out[s:e] = m52(D2)
        del D2
        tp = pace(tp)
    return out


def solve_ridge_inplace(K, nug, Ytr_):
    """Ridge solve through an in-place chunked float64 Cholesky (no monolithic cuSOLVER
    potrf: one long kernel took the display driver down on 6 Sep). K is destroyed."""
    K.diagonal().add_(nug * len(K))
    L = blocked_cholesky_inplace(K, 1000, 6e10)
    return torch.cholesky_solve(Ytr_, L)


def test_error(P0, nug0):
    best = (np.inf, None)
    ntr_, nva_ = len(Xtr), len(Xva)
    Kbuf = torch.empty((ntr_, ntr_), dtype=DT, device=DEV)        # the one matrix-sized buffer
    Kva = torch.empty((nva_, ntr_), dtype=DT, device=DEV)
    for scale in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        P = scale * P0
        kernel_into(Kva, Xva, Xtr, P)
        for nug in (1e-8, 1e-6, 1e-4, 1e-2):
            kernel_into(Kbuf, Xtr, Xtr, P)                         # rebuilt per nugget instead of cloned
            try: al = solve_ridge_inplace(Kbuf, nug, Ytr)
            except Exception: continue
            e = rel_np((Kva @ al).cpu().numpy(), Yva_np)
            if e < best[0]: best = (e, (scale, nug))
    scale, nug = best[1]
    P = scale * P0
    kernel_into(Kbuf, Xtr, Xtr, P)
    al = solve_ridge_inplace(Kbuf, nug, Ytr)
    del Kbuf, Kva
    Kt = torch.empty((len(Xt), ntr_), dtype=DT, device=DEV)
    kernel_into(Kt, Xt, Xtr, P)
    pt = (Kt @ al).cpu().numpy()
    del Kt
    torch.cuda.empty_cache()
    return rel_np(pt, Yte_np), best[1], best[0]


res = {}
for kind in A.kinds.split(","):
    t0 = time.time()
    P, nug, hist = fit_metric(kind, A.steps, A.bs, A.rank)
    e, (scale, nsel), eva = test_error(P, nug)
    res[kind] = dict(test_pct=100 * e, val_pct=100 * eva, scale=scale, nugget=nsel, rho_history=hist,
                     n_metric_params=int(sum(p.numel() for p in [P])), seconds=time.time() - t0)
    print(f"  {kind:8s}  test {100*e:6.2f}%  val {100*eva:6.2f}%  (scale {scale}, nug {nsel}, {time.time()-t0:.0f}s)", flush=True)
stop.set()
u = [x for x in util if x >= 0]
rec = dict(kind="lk_scale", name=name, args=vars(A), results=res, minutes=(time.time() - t_all) / 60,
           gpu_util_mean=float(np.mean(u)) if u else None, data_sha256=sha(DF), torch=torch.__version__,
           device=torch.cuda.get_device_name(0), finished=time.strftime("%Y-%m-%d %H:%M:%S"), script_sha256=sha(Path(__file__)))
(OUT / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
print("saved", name, flush=True)
