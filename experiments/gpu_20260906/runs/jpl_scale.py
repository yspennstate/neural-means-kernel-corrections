"""OCO-2 emulation (JPL) neural means at larger widths and depths, on the GPU,
with the paper's protocol held fixed (jpl_pipeline.py of the NMKC repository):

  split      load_band(band, n_val=2000, seed=0): 2000 validation states drawn from
             the 20000 by default_rng(0) (the paper's fixed carve), 18000 train,
             the 2000-state public test block
  network    residual MLP d_in -> width, SiLU, 3 residual layers, width -> 40
  training   AdamW lr 1e-3, weight decay 1e-5, cosine to 1e-6, batch 512,
             250 epochs, torch.manual_seed(seed); the flat model uses the plain
             relative-L2 loss, the weighted model the radiance-weighted one (s_z)
  selection  validation every 25 epochs, best-validation state kept
  heads      exact Matern-5/2 ridge on the penultimate features (--head), tuned
             on a 6000-row subsample (scale x median, nugget grid) on validation
             and refit on all 18000 rows (float64; on the GPU when --head gpu)
  metrics    reduced (relative L2 on the 40 coefficients) and radiance
             (reconstructed monochromatic spectrum), kernel-flow emulator row

Additions: --width, --depth, --seed, --amp bf16 (labelled), NVML utilization
sampling, a pause on compute distress, saved test predictions.
"""
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

W = Path(os.environ.get("NMKC_W", "C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments"))
sys.path.insert(0, str(W / "code"))
import jpl_data  # noqa: E402
jpl_data.DATA = W / "data" / "jpl_oco2"
from jpl_data import load_band, reconstruction, radiance_error, kernel_flow_predictions  # noqa: E402
sys.path.insert(0, str(W / "spectrum"))
from gpu_blocked_routes import blocked_cholesky_inplace, pace  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2", choices=("o2", "wco2", "sco2"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--width", type=int, default=384)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--amp", choices=("none", "bf16"), default="none")
p.add_argument("--head", choices=("none", "gpu", "cpu"), default="none")
p.add_argument("--outdir", default=str(W / "runs" / "jpl"))
p.add_argument("--duty", type=float, default=0.80)
args = p.parse_args()
DUTY = min(max(args.duty, 0.05), 1.0)
if ON_CPU:
    DUTY = 1.0          # the duty cycle is a display-GPU courtesy; meaningless on the DGX's CPUs
name = f"jpl_{args.band}_s{args.seed}_w{args.width}_d{args.depth}{'_bf16' if args.amp == 'bf16' else ''}{'_head' if args.head != 'none' else ''}"
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)
STATE = W / "runs" / "control"
if (OUT / f"{name}.json").exists():
    print("done already:", name); sys.exit(0)
ON_CPU = os.environ.get("NMKC_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available()
dev = torch.device("cpu" if ON_CPU else "cuda")
if ON_CPU:
    torch.set_num_threads(int(os.environ.get("NMKC_THREADS", "8")))
else:
    torch.backends.cuda.matmul.allow_tf32 = False
DEVICE_NAME = "cpu:" + str(torch.get_num_threads()) + "thr" if ON_CPU else torch.cuda.get_device_name(0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


data_sha = {n: sha(W / "data" / "jpl_oco2" / n) for n in ("dimred_variables_4_mono.jld", "dimred_data_4_mono.jld", f"kf_results_{args.band}_4_mono.jld")}
sp = load_band(args.band)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
recon = reconstruction(args.band)
w_z = np.abs(recon["s_z"]); w_z = w_z / w_z.mean()
rel = lambda P, T: float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))

util = []
def sampler(stop):
    try:
        if ON_CPU:
            return
        import pynvml
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            try: util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception: pass
            stop.wait(5)
    except Exception:
        util.append(-1)
stop = threading.Event(); threading.Thread(target=sampler, args=(stop,), daemon=True).start()


def distress():
    if (STATE / "PAUSE").exists():
        return True
    try:
        d = json.loads(Path(os.environ.get("NMKC_COMPUTE_STATE", "C:/Users/owner/.claude/compute/compute_state.json")).read_text())
        return d.get("status") == "distress" or d.get("mode") == "throttle"
    except Exception:
        return False


class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth=4):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, return_features=False):
        h = F.silu(self.inp(x))
        for layer in self.hidden:
            h = h + F.silu(layer(h))
        return (self.out(h), h) if return_features else self.out(h)


def train(weight=None):
    torch.manual_seed(args.seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
    xt, yt, xv = f32(Xtr), f32(Ytr), f32(Xval)
    Wt = f32(weight) if weight is not None else None
    model = ResidualMLP(Xtr.shape[1], Ytr.shape[1], args.width, args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    n = len(xt); best, best_state = np.inf, None
    amp = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if args.amp == "bf16" else (lambda: torch.autocast(dev.type, enabled=False))
    # GPU duty cycle (see train_mlp_scale.py): the owner's CrashGuard kills python GPU clients at a
    # utilization sample >= 95; every 8 steps the loop syncs and idles (1-DUTY)/DUTY of the busy time
    n_steps, t_w = 0, time.perf_counter()
    for ep in range(args.epochs):
        while distress():
            time.sleep(30)
        t_w = time.perf_counter()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            with amp():
                pred = model(xt[i])
            pred, target = pred.float(), yt[i]
            if Wt is not None:
                pred, target = pred * Wt, target * Wt
            loss = (torch.linalg.vector_norm(pred - target, dim=1) / torch.linalg.vector_norm(target, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            n_steps += 1
            if DUTY < 1.0 and n_steps % 8 == 0:
                if not ON_CPU: torch.cuda.synchronize()
                time.sleep((time.perf_counter() - t_w) * (1.0 - DUTY) / DUTY)
                t_w = time.perf_counter()
        sched.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(xv).cpu().numpy()
            model.train()
            e = rel(pv * (weight if weight is not None else 1), Yval * (weight if weight is not None else 1))
            if e < best:
                best = e; best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        preds = [model(f32(Z)).cpu().numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
        feats = [model(f32(Z), return_features=True)[1].cpu().numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
    return preds, feats, sum(q.numel() for q in model.parameters())


def matern_head(feats, targets_scale=None):
    """Exact Matern-5/2 ridge on features: tuned on a 6000-row subsample on validation,
    refit on all training rows, float64. GPU path: kernels built in row blocks into one
    preallocated matrix and factored in place by the chunked Cholesky (no matrix-sized
    temporaries; the 18000-row full fit holds one 2.6 GB matrix, under the 3.2 GB cap)."""
    sc = targets_scale if targets_scale is not None else 1.0
    Ttr, Tval = Ytr * sc, Yval * sc
    mu, sd = feats[0].mean(0), feats[0].std(0) + 1e-9
    Ftr, Fval, Fte = ((f - mu) / sd for f in feats)
    use_gpu = args.head == "gpu"

    def T(a):
        return torch.tensor(a, dtype=torch.float64, device=dev) if use_gpu else np.asarray(a, np.float64)

    def m52_np(D2, ls):
        a = np.sqrt(5.0) * np.sqrt(D2) / ls
        return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)

    def sqd_np(A, B):
        return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)

    def kernel_into(out, A, B, ls, block=1500):
        """out[i, j] = k(A_i, B_j), built in row blocks (GPU) or directly (CPU)."""
        if not use_gpu:
            out[:] = m52_np(sqd_np(A, B), ls); return out
        nb = (B * B).sum(1)[None, :]
        tp = time.perf_counter()
        for s in range(0, len(A), block):
            e = min(s + block, len(A))
            D2 = torch.clamp((A[s:e] * A[s:e]).sum(1)[:, None] + nb - 2 * A[s:e] @ B.T, min=0.0)
            a = torch.sqrt(D2) * (np.sqrt(5.0) / ls)
            out[s:e] = (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * torch.exp(-a)
            del D2, a
            tp = pace(tp)
        return out

    def solve_inplace(K, nug, rhs):
        n = len(K)
        if use_gpu:
            K.diagonal().add_(nug * n)
            L = blocked_cholesky_inplace(K, 1000, 6e10)
            return torch.cholesky_solve(rhs, L)
        from scipy.linalg import cho_factor, cho_solve
        K.flat[::n + 1] += nug * n
        c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
        return cho_solve(c, rhs, check_finite=False)

    def new(shape):
        return torch.empty(shape, dtype=torch.float64, device=dev) if use_gpu else np.empty(shape, np.float64)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(Ftr), 6000, replace=False)
    Fs, Fv, Ff, Ft = T(Ftr[sub]), T(Fval), T(Ftr), T(Fte)
    Ts, Tf, Tv = T(Ttr[sub]), T(Ttr), np.asarray(Tval)
    # median distance on the subsample from a blockwise squared-distance pass (no 6000x6000 temporary kept)
    if use_gpu:
        nb = (Fs * Fs).sum(1)[None, :]
        meds = []
        for s in range(0, 6000, 1500):
            e = min(s + 1500, 6000)
            D2 = torch.clamp((Fs[s:e] * Fs[s:e]).sum(1)[:, None] + nb - 2 * Fs[s:e] @ Fs.T, min=0.0)
            ii, jj = torch.triu_indices(e - s, 6000, offset=s + 1, device=dev)
            meds.append(D2[ii, jj].cpu().numpy()); del D2
        med = float(np.sqrt(np.median(np.concatenate(meds)))); del meds
    else:
        D2s = sqd_np(Fs, Fs); med = float(np.sqrt(np.median(D2s[np.triu_indices(6000, 1)]))); del D2s
    Ks = new((6000, 6000)); Kvs = new((len(Fval), 6000))
    best = (np.inf, None)
    for scale in (0.5, 1.0, 2.0, 4.0):
        kernel_into(Kvs, Fv, Fs, scale * med)
        for nug in (1e-8, 1e-6, 1e-4):
            kernel_into(Ks, Fs, Fs, scale * med)          # rebuilt per nugget instead of cloned
            try:
                al = solve_inplace(Ks, nug, Ts)
            except Exception:
                continue
            pv = (Kvs @ al)
            e = rel(pv.cpu().numpy() if use_gpu else pv, Tv)
            if e < best[0]:
                best = (e, (scale, nug))
    scale, nug = best[1]
    del Ks, Kvs
    if use_gpu: torch.cuda.empty_cache()
    K = new((len(Ff), len(Ff)))
    kernel_into(K, Ff, Ff, scale * med)
    alpha = solve_inplace(K, nug, Tf)
    del K
    if use_gpu: torch.cuda.empty_cache()
    out = []
    Kq = new((4000, len(Ff)))
    for F_ in (Ff, Fv, Ft):
        pred = []
        for k in range(0, len(F_), 4000):
            m = min(4000, len(F_) - k)
            kernel_into(Kq[:m], F_[k:k + m], Ff, scale * med)
            pk = Kq[:m] @ alpha
            pred.append(pk.cpu().numpy() if use_gpu else pk)
        out.append(np.concatenate(pred) / sc)
    del Kq
    if use_gpu: torch.cuda.empty_cache()
    return out, dict(scale=scale, nug=nug, med=med, val=best[0])


t0 = time.time()
results = {}
kf = kernel_flow_predictions(args.band)
results["kernel_flow"] = dict(reduced=rel(kf, Yte), radiance=radiance_error(kf, Yte, recon))
P_flat, F_flat, n_params = train()
results["mean_flat"] = dict(reduced=rel(P_flat[2], Yte), radiance=radiance_error(P_flat[2], Yte, recon))
P_wt, F_wt, _ = train(weight=w_z)
results["mean_weighted"] = dict(reduced=rel(P_wt[2], Yte), radiance=radiance_error(P_wt[2], Yte, recon))
t_train = time.time() - t0
members_val = [P_flat[1], P_wt[1]]; members_te = [P_flat[2], P_wt[2]]
head_info = {}
if args.head != "none":
    t1 = time.time()
    D_flat, hf = matern_head(F_flat)
    results["dkr_flat"] = dict(reduced=rel(D_flat[2], Yte), radiance=radiance_error(D_flat[2], Yte, recon))
    D_wt, hw = matern_head(F_wt, targets_scale=w_z)
    results["dkr_weighted"] = dict(reduced=rel(D_wt[2], Yte), radiance=radiance_error(D_wt[2], Yte, recon))
    members_val += [D_flat[1], D_wt[1]]; members_te += [D_flat[2], D_wt[2]]
    head_info = dict(flat=hf, weighted=hw, minutes=(time.time() - t1) / 60, device=args.head)
C_te = np.empty_like(Yte)
for j in range(Yte.shape[1]):
    errs = [np.sqrt(((m[:, j] - Yval[:, j]) ** 2).mean()) for m in members_val]
    C_te[:, j] = members_te[int(np.argmin(errs))][:, j]
results["combined"] = dict(reduced=rel(C_te, Yte), radiance=radiance_error(C_te, Yte, recon))
stop.set()
u = [x for x in util if x >= 0]
for k, r in results.items():
    print(f"{k:14s} reduced {100*r['reduced']:7.3f}%   radiance {100*r['radiance']:.4f}%", flush=True)
rec = dict(kind="jpl_scale", name=name, args=vars(args), params=n_params,
           results={k: {m: 100 * v for m, v in r.items()} for k, r in results.items()},
           head=head_info, minutes_wall=(time.time() - t0) / 60, minutes_train=t_train / 60,
           gpu_util_mean=float(np.mean(u)) if u else None, gpu_util_samples=len(u),
           data_sha256=data_sha, torch=torch.__version__, device=DEVICE_NAME,
           finished=time.strftime("%Y-%m-%d %H:%M:%S"), script_sha256=sha(Path(__file__)))
np.savez_compressed(OUT / f"{name}_preds.npz", flat_te=P_flat[2].astype(np.float32), weighted_te=P_wt[2].astype(np.float32),
                    flat_va=P_flat[1].astype(np.float32), weighted_va=P_wt[1].astype(np.float32), combined_te=C_te.astype(np.float32))
(OUT / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
print("saved", name, f"[{(time.time()-t0)/60:.1f} min]", flush=True)
