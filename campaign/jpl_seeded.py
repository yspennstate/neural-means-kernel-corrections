"""OCO-2 pipeline at one campaign seed, with matched kernel tuning throughout.

Differences from jpl_pipeline.py:
  - every stochastic choice is seeded by --seed: the train/validation split,
    network initialization and batch order, and the kernel tuning subsample;
  - a third network is trained in the exact radiance-relative metric (the loss
    is computed through the stored PCA reconstruction, numerator and
    denominator, not the diagonally weighted coefficient surrogate);
  - the raw-input kernel rows get the identical protocol as the feature heads:
    the same scale/nugget grid tuned on validation with a 6000-point
    subsample, winner refit on all 18000 points (isotropic and ARD variants);
  - per-sample test errors in both metrics are saved for bootstrap intervals.

Environment: NMKC_ROOT, TASK_ID, NMKC_THREADS. Data under <root>/data/jpl_oco2
via NMKC_JPL_DATA (defaults to the repo layout otherwise).

    python campaign/jpl_seeded.py --band o2 --seed 3
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jpl_data
from jpl_data import load_band, reconstruction, radiance_error, kernel_flow_predictions, to_radiance

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--width", type=int, default=384)
args = p.parse_args()

THREADS = int(os.environ.get("NMKC_THREADS", "5"))
torch.set_num_threads(THREADS)
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", "."))
TASK_ID = os.environ.get("TASK_ID", f"oco_{args.band}_s{args.seed}")
OUT = ROOT / "seeds" / f"oco_{args.band}_s{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
if os.environ.get("NMKC_JPL_DATA"):
    jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])

sp = load_band(args.band, seed=args.seed)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
recon = reconstruction(args.band)
w_z = np.abs(recon["s_z"]); w_z = w_z / w_z.mean()          # legacy numerator weights
s_z = recon["s_z"].astype(np.float64)                        # exact diagonal
P_mat = recon["P"].astype(np.float64)

rel = lambda Pp, T: float(np.mean(np.linalg.norm(Pp - T, axis=1) / np.linalg.norm(T, axis=1)))
rad = lambda Pp, T: radiance_error(Pp, T, recon)
# per-sample radiance denominators (constants of the data, not the model)
r_norm_tr = np.linalg.norm(to_radiance(Ytr, recon), axis=1)
r_norm_va = np.linalg.norm(to_radiance(Yval, recon), axis=1)


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


def train(mode):
    """mode: flat | wnum (legacy numerator weights) | radx (exact radiance)."""
    torch.manual_seed(args.seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(Xtr), f32(Ytr), f32(Xval)
    Wt = f32(w_z) if mode == "wnum" else None
    if mode == "radx":
        sz_t, P_t = f32(s_z), f32(P_mat)
        rn_t = f32(r_norm_tr)
    model = ResidualMLP(Xtr.shape[1], Ytr.shape[1], args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    n = len(xt)
    best, best_state = np.inf, None
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            pred, target = model(xt[i]), yt[i]
            if mode == "flat":
                loss = (torch.linalg.vector_norm(pred - target, dim=1)
                        / torch.linalg.vector_norm(target, dim=1)).mean()
            elif mode == "wnum":
                loss = (torch.linalg.vector_norm((pred - target) * Wt, dim=1)
                        / torch.linalg.vector_norm(target * Wt, dim=1)).mean()
            else:  # radx: exact radiance-relative error through the reconstruction
                diff = ((pred - target) * sz_t) @ P_t
                loss = (torch.linalg.vector_norm(diff, dim=1) / rn_t[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(xv).numpy().astype(np.float64)
            model.train()
            if mode == "flat":
                e = rel(pv, Yval)
            elif mode == "wnum":
                e = rel(pv * w_z, Yval * w_z)
            else:
                e = rad(pv, Yval)
            if e < best:
                best = e
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = [model(f32(Z)).numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
        feats = [model(f32(Z), return_features=True)[1].numpy().astype(np.float64)
                 for Z in (Xtr, Xval, Xte)]
    return preds, feats


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


def matern_head(Ztr, Zva, Zte, err_fn, label):
    """Exact Matern-5/2 kernel ridge, the single protocol used everywhere:
    inputs standardized by train moments, scale grid {0.5,1,2,4} x median
    pairwise distance (6000-point estimate), nugget grid {1e-8,1e-6,1e-4},
    tuned on validation by err_fn over a 6000-sample training subsample,
    winner refit on the full training set."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Ftr, Fval, Fte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(Ftr), min(6000, len(Ftr)), replace=False)
    med = np.sqrt(np.median(sqd(Ftr[sub], Ftr[sub])[np.triu_indices(len(sub), 1)]))
    best = (np.inf, None)
    D2s, D2vs = sqd(Ftr[sub], Ftr[sub]), sqd(Fval, Ftr[sub])
    for scale in (0.5, 1.0, 2.0, 4.0):
        Ks, Kvs = m52(D2s, scale * med), m52(D2vs, scale * med)
        for nug in (1e-8, 1e-6, 1e-4):
            Kr = Ks.copy(); Kr.flat[::len(sub) + 1] += nug * len(sub)
            try:
                c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
            except np.linalg.LinAlgError:
                continue
            e = err_fn(Kvs @ cho_solve(c, Ytr[sub], check_finite=False))
            if e < best[0]:
                best = (e, (scale, nug))
    scale, nug = best[1]
    n = len(Ftr)
    K = m52(sqd(Ftr, Ftr), scale * med); K.flat[::n + 1] += nug * n
    c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    alpha = cho_solve(c, Ytr, check_finite=False)
    out = []
    for F_ in (Ftr, Fval, Fte):
        pred = np.empty((len(F_), Ytr.shape[1]))
        for k in range(0, len(F_), 4000):
            pred[k:k + 4000] = m52(sqd(F_[k:k + 4000], Ftr), scale * med) @ alpha
        out.append(pred)
    print(f"  {label}: scale {scale} nugget {nug:g} (med {med:.3f})", flush=True)
    return out, dict(scale=scale, nugget=nug, med=float(med))


t_all = time.time()
results, val_preds, te_preds, hyper = {}, {}, {}, {}

kf = kernel_flow_predictions(args.band)
results["kernel_flow"] = dict(reduced=rel(kf, Yte), radiance=rad(kf, Yte))
te_preds["kernel_flow"] = kf

# matched raw-input kernels: identical grid, tuning and full refit as the heads
raw, h = matern_head(Xtr, Xval, Xte, lambda Pv: rel(Pv, Yval), "kernel_raw")
results["kernel_raw"] = dict(reduced=rel(raw[2], Yte), radiance=rad(raw[2], Yte))
hyper["kernel_raw"] = h
val_preds["kernel_raw"], te_preds["kernel_raw"] = raw[1], raw[2]
A_ls, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
relevance = np.linalg.norm(A_ls, axis=1)
scale_ard = relevance / relevance.mean()
ard, h = matern_head(Xtr * scale_ard, Xval * scale_ard, Xte * scale_ard,
                     lambda Pv: rel(Pv, Yval), "kernel_ard")
results["kernel_ard"] = dict(reduced=rel(ard[2], Yte), radiance=rad(ard[2], Yte))
hyper["kernel_ard"] = h
val_preds["kernel_ard"], te_preds["kernel_ard"] = ard[1], ard[2]

members_val, members_te = [], []
for mode, mname in (("flat", "mean_flat"), ("wnum", "mean_wnum"), ("radx", "mean_radx")):
    t0 = time.time()
    Pm, Fm = train(mode)
    results[mname] = dict(reduced=rel(Pm[2], Yte), radiance=rad(Pm[2], Yte),
                          minutes=round((time.time() - t0) / 60, 1))
    members_val.append(Pm[1]); members_te.append(Pm[2])
    val_preds[mname], te_preds[mname] = Pm[1], Pm[2]
    err_fn = ((lambda Pv: rel(Pv, Yval)) if mode == "flat" else
              (lambda Pv: rel(Pv * w_z, Yval * w_z)) if mode == "wnum" else
              (lambda Pv: rad(Pv, Yval)))
    D, h = matern_head(Fm[0], Fm[1], Fm[2], err_fn, f"dkr_{mode}")
    dname = f"dkr_{mode}"
    results[dname] = dict(reduced=rel(D[2], Yte), radiance=rad(D[2], Yte))
    hyper[dname] = h
    members_val.append(D[1]); members_te.append(D[2])
    val_preds[dname], te_preds[dname] = D[1], D[2]
    print(f"{mname}/{dname} done [{(time.time()-t0)/60:.1f} min]", flush=True)

# per-coordinate selection on validation (optimal for any diagonal metric)
C_te = np.empty_like(Yte)
winners = []
for j in range(Yte.shape[1]):
    errs = [np.sqrt(((m[:, j] - Yval[:, j]) ** 2).mean()) for m in members_val]
    w = int(np.argmin(errs))
    winners.append(w)
    C_te[:, j] = members_te[w][:, j]
results["combined"] = dict(reduced=rel(C_te, Yte), radiance=rad(C_te, Yte))
te_preds["combined"] = C_te

# weighted variant: per-sample weights 1/||z||^2 make the selection exactly
# optimal for the squared relative error (the metric the proposition covers)
u_val = 1.0 / (np.linalg.norm(Yval, axis=1) ** 2 + 1e-30)
Cw_te = np.empty_like(Yte)
winners_w = []
for j in range(Yte.shape[1]):
    errs = [float((u_val * (m[:, j] - Yval[:, j]) ** 2).mean()) for m in members_val]
    w = int(np.argmin(errs))
    winners_w.append(w)
    Cw_te[:, j] = members_te[w][:, j]
results["combined_w"] = dict(reduced=rel(Cw_te, Yte), radiance=rad(Cw_te, Yte))
te_preds["combined_w"] = Cw_te

# member predictions on val and test, so any combination rule can be
# recomputed later without retraining
np.savez_compressed(OUT / "member_preds.npz",
                    Yval=Yval.astype(np.float32), Yte=Yte.astype(np.float32),
                    **{f"val_{k}": v.astype(np.float32) for k, v in val_preds.items()},
                    **{f"te_{k}": v.astype(np.float32) for k, v in te_preds.items()})

# per-sample errors in both metrics, for bootstrap intervals
den_red = np.linalg.norm(Yte, axis=1)
R_te = to_radiance(Yte, recon)
den_rad = np.linalg.norm(R_te, axis=1)
per = {}
for name, Pt in te_preds.items():
    e_red = np.linalg.norm(Pt - Yte, axis=1) / den_red
    e_rad = np.linalg.norm(to_radiance(Pt, recon) - R_te, axis=1) / den_rad
    per[name] = (e_red, e_rad)
np.savez(OUT / "per_sample_errors.npz",
         **{f"{k}_red": v[0] for k, v in per.items()},
         **{f"{k}_rad": v[1] for k, v in per.items()})

B, n = 2000, len(Yte)
rng = np.random.default_rng(200 + args.seed)
bidx = rng.integers(0, n, size=(B, n))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
boot = {}
for name, (er, ea) in per.items():
    boot[name] = dict(reduced=ci(er[bidx].mean(1)), radiance=ci(ea[bidx].mean(1)))
# paired contrasts behind the claims in the text
boot["dkr_flat_minus_mean_flat"] = dict(
    reduced=ci(per["dkr_flat"][0][bidx].mean(1) - per["mean_flat"][0][bidx].mean(1)))
boot["combined_minus_kernel_flow"] = dict(
    reduced=ci(per["combined"][0][bidx].mean(1) - per["kernel_flow"][0][bidx].mean(1)),
    radiance=ci(per["combined"][1][bidx].mean(1) - per["kernel_flow"][1][bidx].mean(1)))

for name, r in results.items():
    print(f"{name:12s} reduced {100*r['reduced']:7.2f}%   radiance {100*r['radiance']:.4f}%",
          flush=True)

out = dict(task_id=TASK_ID, kind="oco_seed", band=args.band, seed=args.seed,
           epochs=args.epochs, width=args.width,
           results={k: {m: round(100 * v, 4) for m, v in r.items() if m in ("reduced", "radiance")}
                    for k, r in results.items()},
           hyper=hyper, winners=winners, winners_w=winners_w,
           boot_ci_pct={k: {m: [round(100 * x, 4) for x in v2] for m, v2 in v.items()}
                        for k, v in boot.items()},
           minutes=round((time.time() - t_all) / 60, 1))
res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, res)
print(f"[{TASK_ID}] complete in {out['minutes']} min", flush=True)
