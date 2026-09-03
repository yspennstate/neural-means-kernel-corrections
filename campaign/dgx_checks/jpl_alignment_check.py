"""Recompute the OCO-2 alignment diagnostic with the quantities the text names, after review.

The released jpl_diagnostics.py solved alpha = (K + n lambda I)^{-1} Y and reported alpha^T Y, which equals
alpha^T K alpha + n lambda ||alpha||^2, not the fitted interpolant's squared RKHS norm alpha^T K alpha. This
script reports both, the regularization term, the ratio between the raw-input and the feature kernel for each,
the same ratio after normalizing by the target energy ||Y||_F^2, the sensitivity of the ratio to the nugget and
the length-scale multiplier, the effective dimensions, and - the quantity the review asked for directly - the
distribution of the test-point power function P_lambda(u) under the two kernels at matched nugget and median
length scale. Everything is at the paper's setting (4000-row subsample, seed 0) and again on the full 18000-row
training block. Both designs are standardized by training moments before any kernel is formed, as the campaign's
matern_head() does, and the scale grid includes the multiplier 4 that the campaign selected for every feature head. Network and split exactly as in jpl_diagnostics.py.

    NMKC_JPL_DATA=... python jpl_alignment_check.py --band o2 [--out results/jpl_alignment_o2.json]
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve, solve_triangular

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jpl_data  # noqa: E402
from jpl_data import load_band  # noqa: E402

if os.environ.get("NMKC_JPL_DATA"):
    jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--out", default="")
args = p.parse_args()
torch.set_num_threads(int(os.environ.get("NMKC_THREADS", "4")))


def rel(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1)))


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def matern52(D2, ls):
    r = np.sqrt(D2) / ls
    return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def median_ls(Z, n=2000, seed=0):
    idx = np.random.default_rng(seed).choice(len(Z), min(n, len(Z)), replace=False)
    D2 = sqdist(Z[idx], Z[idx])
    return float(np.sqrt(np.median(D2[np.triu_indices(len(idx), 1)])))


class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out, width=384, depth=4):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, feats=False):
        h = F.silu(self.inp(x))
        for layer in self.hidden:
            h = h + F.silu(layer(h))
        return (self.out(h), h) if feats else self.out(h)


def train(sp, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(sp["Xtr"]), f32(sp["Ytr"]), f32(sp["Xval"])
    model = ResidualMLP(sp["Xtr"].shape[1], sp["Ytr"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    n = len(xt); best, best_sd = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            out = model(xt[i])
            loss = (torch.linalg.vector_norm(out - yt[i], dim=1) / torch.linalg.vector_norm(yt[i], dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                e = rel(model(xv).numpy(), sp["Yval"])
            model.train()
            if e < best:
                best, best_sd = e, {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    return model


def diagnostics(Z, Y, Zte, lam_mult, scale_mult, seed):
    """Fitted-norm and power-function diagnostics for one design Z (rows) and its test inputs Zte."""
    n = len(Z); ls = scale_mult * median_ls(Z, seed=seed)
    K = matern52(sqdist(Z, Z), ls)
    ev = np.maximum(np.linalg.eigvalsh(K), 0.0)
    deff = float((ev / (ev + n * lam_mult)).sum())
    Kr = K.copy(); Kr.flat[::n + 1] += lam_mult * n
    c = cho_factor(Kr, lower=True, check_finite=False)
    a = cho_solve(c, Y, check_finite=False)
    aKa = float(np.einsum("nd,nd->", a, K @ a)); aY = float(np.einsum("nd,nd->", a, Y))
    reg = float(lam_mult * n * (a * a).sum())
    # test-point power function with the same nugget: P^2 = k(u,u) - k_u^T (K + n lam I)^{-1} k_u
    Ku = matern52(sqdist(Z, Zte), ls)
    V = solve_triangular(c[0], Ku, lower=True, check_finite=False)
    P2 = np.maximum(1.0 - (V * V).sum(0), 0.0); P = np.sqrt(P2)
    return dict(n=n, ls=ls, deff=deff, aKa=aKa, aY=aY, reg_term=reg, aKa_over_Y2=aKa / float((Y * Y).sum()),
                P_mean=float(P.mean()), P_median=float(np.median(P)), P_q10=float(np.quantile(P, 0.1)),
                P_q90=float(np.quantile(P, 0.9)), P_max=float(P.max()), lead_eig_mass=float(ev[-1] / ev.sum()))


t0 = time.time()
sp = load_band(args.band)
feat = train(sp, args.epochs, args.seed)
f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
with torch.no_grad():
    Ftr = feat(f32(sp["Xtr"]), feats=True)[1].numpy().astype(np.float64)
    Fte = feat(f32(sp["Xte"]), feats=True)[1].numpy().astype(np.float64)
    net_te = feat(f32(sp["Xte"])).numpy()
print(f"network test error {100 * rel(net_te, sp['Yte']):.3f}% [{(time.time() - t0) / 60:.1f} min]", flush=True)

Xtr, Xte, Ytr = sp["Xtr"].astype(np.float64), sp["Xte"].astype(np.float64), sp["Ytr"].astype(np.float64)
# both designs standardized by their training moments, exactly as matern_head() in jpl_seeded.py does before
# any kernel is formed (the first release of this check passed the raw states unstandardized; corrected after review)
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-9
Xtr, Xte = (Xtr - mx) / sx, (Xte - mx) / sx
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
Ftr_s, Fte_s = (Ftr - mu) / sd, (Fte - mu) / sd
out = dict(band=args.band, seed=args.seed, epochs=args.epochs, settings=[])
sub = np.random.default_rng(0).choice(len(Xtr), 4000, replace=False)
for label, rows in (("n4000", sub), ("full", np.arange(len(Xtr)))):
    for lam in (1e-8, 1e-6, 1e-4):
        for sc in (0.5, 1.0, 2.0, 4.0):   # 4.0 is the multiplier the campaign selected for every feature head
            if label == "full" and not (lam == 1e-8 and sc in (1.0, 4.0)):
                continue
            raw = diagnostics(Xtr[rows], Ytr[rows], Xte, lam, sc, args.seed)
            fea = diagnostics(Ftr_s[rows], Ytr[rows], Fte_s, lam, sc, args.seed)
            rec = dict(rows=label, nugget=lam, scale=sc, raw=raw, feature=fea,
                       ratio_aKa=raw["aKa"] / fea["aKa"], ratio_aY=raw["aY"] / fea["aY"],
                       ratio_aKa_normalized=raw["aKa_over_Y2"] / fea["aKa_over_Y2"],
                       ratio_P_mean=raw["P_mean"] / fea["P_mean"], ratio_P_median=raw["P_median"] / fea["P_median"])
            out["settings"].append(rec)
            print(f"{label} nugget {lam:g} scale {sc}: deff raw {raw['deff']:.0f} / feat {fea['deff']:.0f}; "
                  f"aKa raw {raw['aKa']:.4g} feat {fea['aKa']:.4g} (ratio {rec['ratio_aKa']:.1f}; aY ratio {rec['ratio_aY']:.1f}); "
                  f"reg term raw {raw['reg_term']:.3g} feat {fea['reg_term']:.3g}; "
                  f"P median raw {raw['P_median']:.4f} feat {fea['P_median']:.4f} (ratio {rec['ratio_P_median']:.2f}) [{(time.time() - t0) / 60:.1f} min]", flush=True)
out["minutes"] = round((time.time() - t0) / 60, 1)
path = pathlib.Path(args.out or f"results/jpl_alignment_{args.band}_s{args.seed}.json")
path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(path, "w"), indent=1)
print("wrote", path, flush=True)
