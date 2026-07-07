"""Neural means and deep kernel heads for the OCO-2 emulator, against the
kernel-flow emulator's own predictions.

Four models per band, sharing one architecture and differing only in the
training metric and the head:

  flat mean       residual MLP trained on the plain relative-L2 loss;
  weighted mean   the same network trained in the radiance metric (loss
                  weighted by s_z, the diagonal form of the reconstruction);
  dkr flat        exact Matern kernel ridge on the flat network's penultimate
                  features;
  dkr weighted    the same head on the weighted network's features, fit to
                  the weighted targets.

The flat models win the reduced metric and the weighted models the radiance
metric; a per-coordinate selection on validation is optimal for both at once
because the radiance metric is diagonal in the reduced coordinates.

    python jpl_pipeline.py --band o2
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

from jpl_data import load_band, reconstruction, radiance_error, kernel_flow_predictions

torch.set_num_threads(8)

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--width", type=int, default=384)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

sp = load_band(args.band)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
recon = reconstruction(args.band)
w_z = np.abs(recon["s_z"])
w_z = w_z / w_z.mean()

rel = lambda P, T: float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))


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
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(Xtr), f32(Ytr), f32(Xval)
    Wt = f32(weight) if weight is not None else None
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
            if Wt is not None:
                pred, target = pred * Wt, target * Wt
            loss = (torch.linalg.vector_norm(pred - target, dim=1)
                    / torch.linalg.vector_norm(target, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(xv).numpy()
            model.train()
            e = rel(pv * (weight if weight is not None else 1), Yval * (weight if weight is not None else 1))
            if e < best:
                best = e
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = [model(f32(Z)).numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
        feats = [model(f32(Z), return_features=True)[1].numpy().astype(np.float64)
                 for Z in (Xtr, Xval, Xte)]
    model.train()
    return preds, feats


def matern_head(feats, targets_scale=None):
    """Exact Matern-5/2 kernel ridge from features to targets, hyperparameters
    tuned on validation with a 6000-point subsample, refit on all of the
    training set. `targets_scale` trains the head in a diagonal output metric."""
    sc = targets_scale if targets_scale is not None else 1.0
    Ttr, Tval = Ytr * sc, Yval * sc
    mu, sd = feats[0].mean(0), feats[0].std(0) + 1e-9
    Ftr, Fval, Fte = ((f - mu) / sd for f in feats)

    def sqd(A, B):
        return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)

    def m52(D2, ls):
        a = np.sqrt(5.0) * np.sqrt(D2) / ls
        return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(Ftr), 6000, replace=False)
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
            e = rel(Kvs @ cho_solve(c, Ttr[sub], check_finite=False), Tval)
            if e < best[0]:
                best = (e, (scale, nug))
    scale, nug = best[1]
    n = len(Ftr)
    K = m52(sqd(Ftr, Ftr), scale * med); K.flat[::n + 1] += nug * n
    c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    alpha = cho_solve(c, Ttr, check_finite=False)
    out = []
    for F_ in (Ftr, Fval, Fte):
        pred = np.empty((len(F_), Ytr.shape[1]))
        for k in range(0, len(F_), 4000):
            pred[k:k + 4000] = m52(sqd(F_[k:k + 4000], Ftr), scale * med) @ alpha
        out.append(pred / sc)
    return out


results = {}
kf = kernel_flow_predictions(args.band)
results["kernel_flow"] = dict(reduced=rel(kf, Yte), radiance=radiance_error(kf, Yte, recon))

t0 = time.time()
P_flat, F_flat = train()
results["mean_flat"] = dict(reduced=rel(P_flat[2], Yte), radiance=radiance_error(P_flat[2], Yte, recon))
P_wt, F_wt = train(weight=w_z)
results["mean_weighted"] = dict(reduced=rel(P_wt[2], Yte), radiance=radiance_error(P_wt[2], Yte, recon))

D_flat = matern_head(F_flat)
results["dkr_flat"] = dict(reduced=rel(D_flat[2], Yte), radiance=radiance_error(D_flat[2], Yte, recon))
D_wt = matern_head(F_wt, targets_scale=w_z)
results["dkr_weighted"] = dict(reduced=rel(D_wt[2], Yte), radiance=radiance_error(D_wt[2], Yte, recon))

# per-coordinate selection on validation: with a diagonal radiance metric this
# is simultaneously optimal for the reduced and the radiance error
members_val = [P_flat[1], P_wt[1], D_flat[1], D_wt[1]]
members_te = [P_flat[2], P_wt[2], D_flat[2], D_wt[2]]
C_te = np.empty_like(Yte)
for j in range(Yte.shape[1]):
    errs = [np.sqrt(((m[:, j] - Yval[:, j]) ** 2).mean()) for m in members_val]
    C_te[:, j] = members_te[int(np.argmin(errs))][:, j]
results["combined"] = dict(reduced=rel(C_te, Yte), radiance=radiance_error(C_te, Yte, recon))

for name, r in results.items():
    print(f"{name:14s} reduced {100*r['reduced']:7.2f}%   radiance {100*r['radiance']:.4f}%", flush=True)
print(f"[{(time.time()-t0)/60:.1f} min]")
json.dump({k: {m: round(100 * v, 4) for m, v in r.items()} for k, r in results.items()},
          open(f"runs/jpl_{args.band}.json", "w"), indent=1)
