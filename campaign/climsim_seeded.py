"""ClimSim scaling point, leakage-free and seeded: one (n, seed) per run.

Fixes versus climsim_scaling.py, responding to review: kernel hyperparameters
(scale and nugget) are selected on a 2000-sample validation block drawn from
the training pool immediately after the training slice -- the test set is
touched exactly once, for the final numbers. The grid is the same one the rest
of the paper uses: scale {0.5,1,2,4} x median distance, nugget
{1e-8,1e-6,1e-4,1e-2} (the widest nugget kept because these targets are
noisy). The permutation defining test/pool, the network seed and the kernel
subsample are all functions of --seed.

Data: NMKC_CLIMSIM directory with {train,val}_{input,target}.npy (LEAP
subsampled low-res arrays). Writes <root>/results/<TASK_ID>.json.

    python campaign/climsim_seeded.py --data train --n 100000 --seed 0 --kernel 1
"""
import argparse, json, os, pathlib, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

p = argparse.ArgumentParser()
p.add_argument("--data", default="train", choices=["val", "train"])
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--kernel", type=int, default=0)
p.add_argument("--epochs", type=int, default=60)
args = p.parse_args()

THREADS = int(os.environ.get("NMKC_THREADS", "5"))
torch.set_num_threads(THREADS)
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", "."))
TASK_ID = os.environ.get("TASK_ID", f"climsim_{args.data}_n{args.n}_s{args.seed}")
CS = pathlib.Path(os.environ.get("NMKC_CLIMSIM", ROOT / "climsim"))
OUT = ROOT / "seeds" / f"climsim_{args.data}_n{args.n}_s{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)

Xin = np.load(CS / f"{args.data}_input.npy", mmap_mode="r")
Yin = np.load(CS / f"{args.data}_target.npy", mmap_mode="r")
N, di = Xin.shape; do = Yin.shape[1]
print(f"ClimSim {args.data}: {N} rows, {di} -> {do}; n={args.n} seed={args.seed}", flush=True)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(N)
te = np.sort(perm[:20000]); pool = perm[20000:]
if args.n + 2000 > len(pool):
    raise SystemExit(f"n={args.n} leaves no 2000-sample validation block")
Xte = np.asarray(Xin[te], np.float64); Yte = np.asarray(Yin[te], np.float64)
ref = np.sort(pool[:50000])
mu = np.asarray(Xin[ref], np.float64).mean(0)
sd = np.asarray(Xin[ref], np.float64).std(0) + 1e-9
mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
Xte_n = (Xte - mu) / sd
yvar = Yte.var(0) + 1e-12
active = yvar > 1e-6 * yvar.max()
print(f"  R2 over {int(active.sum())}/{do} active outputs", flush=True)

idx = np.sort(pool[:args.n])
vidx = np.sort(pool[args.n:args.n + 2000])          # kernel-selection block, never test
Xtr = (np.asarray(Xin[idx], np.float32) - mu32) / sd32
Ytr = np.asarray(Yin[idx], np.float32)
Xkv = (np.asarray(Xin[vidx], np.float64) - mu) / sd
Ykv = np.asarray(Yin[vidx], np.float64)


def r2(P, T, var):
    num = ((P - T) ** 2).mean(0)
    return float(np.mean(1 - num[active] / var[active]))


def rel(P, T):
    nn_ = np.linalg.norm(T, axis=1); m = nn_ > 1e-9
    return float(np.mean(np.linalg.norm(P[m] - T[m], axis=1) / nn_[m]))


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    r = np.sqrt(D2) / ls
    return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def kernel_point(cap=6000):
    """Matern-5/2 on the raw state: tuned on the pool validation block, then
    the winner's fit (same subsample) scores the full test set once."""
    sub = np.random.default_rng(args.seed + 1).choice(len(Xtr), min(cap, len(Xtr)), replace=False)
    Xs = Xtr[sub].astype(np.float64); Ys = Ytr[sub].astype(np.float64)
    m1500 = min(1500, len(Xs))
    med = float(np.sqrt(np.median(sqd(Xs[:m1500], Xs[:m1500])[np.triu_indices(m1500, 1)])) + 1e-9)
    D2 = sqd(Xs, Xs); D2v = sqd(Xkv, Xs)
    best = (np.inf, None, None)
    for s in (0.5, 1.0, 2.0, 4.0):
        Kv = m52(D2v, s * med)
        for nug in (1e-8, 1e-6, 1e-4, 1e-2):
            K = m52(D2, s * med); K.flat[::len(Xs) + 1] += nug * len(Xs)
            try:
                a = cho_solve(cho_factor(K, lower=True, check_finite=False, overwrite_a=True),
                              Ys, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            e = ((Kv @ a - Ykv) ** 2).mean()
            if e < best[0]:
                best = (e, (s, nug), a)
    (s, nug), a = best[1], best[2]
    print(f"  kernel: scale {s} nugget {nug:g} (selected on pool block)", flush=True)
    Pk = np.empty((len(Xte_n), do))
    for k in range(0, len(Xte_n), 4000):
        Pk[k:k + 4000] = m52(sqd(Xte_n[k:k + 4000], Xs), s * med) @ a
    return Pk, dict(scale=s, nugget=nug, med=med, cap=int(len(Xs)))


class MLP(nn.Module):
    def __init__(s, di, do, w=512, d=4):
        super().__init__(); s.inp = nn.Linear(di, w)
        s.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)]); s.out = nn.Linear(w, do)

    def forward(s, x):
        h = F.silu(s.inp(x))
        for l in s.hid: h = h + F.silu(l(h))
        return s.out(h)


def train_mean():
    torch.manual_seed(args.seed)
    xt = torch.from_numpy(np.ascontiguousarray(Xtr, np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(Ytr, np.float32))
    xte = torch.tensor(Xte_n, dtype=torch.float32)
    m = MLP(di, do); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 4096; Nt = len(xt)
    for ep in range(args.epochs):
        t0 = time.time()
        pm = torch.randperm(Nt)
        for k in range(0, Nt, bs):
            i = pm[k:k + bs]
            loss = F.mse_loss(m(xt[i]), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1}/{args.epochs} [{time.time()-t0:.0f}s/ep]", flush=True)
    m.eval()
    with torch.no_grad():
        out = []
        for k in range(0, len(xte), 8192):
            out.append(m(xte[k:k + 8192]).numpy())
    return np.concatenate(out).astype(np.float64)


t0 = time.time()
row = dict(task_id=TASK_ID, kind="climsim_point", data=args.data, n=int(args.n),
           seed=args.seed, epochs=args.epochs, ntest=int(len(te)),
           active_outputs=int(active.sum()))
Pm = train_mean()
row["mean_r2"] = round(r2(Pm, Yte, yvar), 4)
row["mean_rel"] = round(100 * rel(Pm, Yte), 3)
np.savez(OUT / "per_output.npz", mean_sqerr=((Pm - Yte) ** 2).mean(0), yvar=yvar, active=active)
if args.kernel:
    Pk, kh = kernel_point()
    row["kernel_r2"] = round(r2(Pk, Yte, yvar), 4)
    row["kernel_rel"] = round(100 * rel(Pk, Yte), 3)
    row["kernel_hyper"] = kh
row["minutes"] = round((time.time() - t0) / 60, 1)
print(f"  n={args.n}: mean R2 {row['mean_r2']}  rel {row['mean_rel']}%"
      + (f"  kernel R2 {row.get('kernel_r2')}" if args.kernel else "")
      + f"  [{row['minutes']} min]", flush=True)

res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(row, open(tmp, "w"), indent=1)
os.replace(tmp, res)
print(f"[{TASK_ID}] complete", flush=True)
