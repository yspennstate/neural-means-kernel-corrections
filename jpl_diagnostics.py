"""Diagnostic experiments behind the theory claims for the OCO-2 study.

Self-contained (only jpl_data.py is shared). Reproduces, for one band:

  ladder     the representation ladder of Table 2 and Figure 1: the same
             Matern head on the raw input, on a sensitivity-scaled input, and
             on the trained network's features;
  alignment  the effective dimension and the target's RKHS interpolation norm
             through the raw-input and the feature kernel, the two factors of
             the optimal-recovery bound (Proposition on feature pullback);
  diversity  residual correlations between random seeds of one architecture
             and between different architectures, the quantity that sets the
             ensembling floor (Corollary).

    python jpl_diagnostics.py --band o2 --experiment ladder
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

from jpl_data import load_band

torch.set_num_threads(4)


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
    def __init__(self, d_in, d_out, width=384, depth=4, act="silu"):
        super().__init__()
        self.act = F.silu if act == "silu" else F.relu
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, feats=False):
        h = self.act(self.inp(x))
        for layer in self.hidden:
            h = h + self.act(layer(h))
        return (self.out(h), h) if feats else self.out(h)


class FourierMLP(nn.Module):
    def __init__(self, d_in, d_out, sigma, n_feat=256):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("B", torch.randn(d_in, n_feat, generator=g) * sigma)
        self.net = nn.Sequential(nn.Linear(2 * n_feat, 512), nn.SiLU(),
                                 nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, d_out))

    def forward(self, x):
        p = x @ self.B
        return self.net(torch.cat([torch.sin(p), torch.cos(p)], -1))


def train(make, sp, epochs=250, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(sp["Xtr"]), f32(sp["Ytr"]), f32(sp["Xval"])
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    n = len(xt); best, best_sd = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            out = model(xt[i])
            out = out[0] if isinstance(out, tuple) else out
            loss = (torch.linalg.vector_norm(out - yt[i], dim=1)
                    / torch.linalg.vector_norm(yt[i], dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(xv)
                pv = (pv[0] if isinstance(pv, tuple) else pv).numpy()
            model.train()
            e = rel(pv, sp["Yval"])
            if e < best: best, best_sd = e, {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    return model


def kernel_head(Ftr, Ytr, Fval, Yval, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Ftr, Fval, Fte = ((F_ - mu) / sd for F_ in (Ftr, Fval, Fte))
    best = (np.inf, None)
    med = median_ls(Ftr)
    for s in (0.5, 1, 2, 4):
        for lam in (1e-8, 1e-6, 1e-4):
            K = matern52(sqdist(Ftr, Ftr), s * med); K.flat[::len(Ftr) + 1] += lam * len(Ftr)
            try:
                c = cho_factor(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            a = cho_solve(c, Ytr, check_finite=False)
            e = rel(matern52(sqdist(Fval, Ftr), s * med) @ a, Yval)
            if e < best[0]:
                best = (e, (s, lam))
    s, lam = best[1]
    K = matern52(sqdist(Ftr, Ftr), s * med); K.flat[::len(Ftr) + 1] += lam * len(Ftr)
    a = cho_solve(cho_factor(K, lower=True, check_finite=False), Ytr, check_finite=False)
    return matern52(sqdist(Fte, Ftr), s * med) @ a


def experiment_ladder(sp):
    feat = train(lambda: ResidualMLP(sp["Xtr"].shape[1], sp["Ytr"].shape[1]), sp)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    with torch.no_grad():
        net_te = feat(f32(sp["Xte"]))[0].numpy()
        F = [feat(f32(sp[k]), feats=True)[1].numpy() for k in ("Xtr", "Xval", "Xte")]
    # raw isotropic
    med = median_ls(sp["Xtr"])
    K = matern52(sqdist(sp["Xtr"][:6000], sp["Xtr"][:6000]), med); K.flat[::6001] += 1e-6 * 6000
    a = cho_solve(cho_factor(K, lower=True, check_finite=False), sp["Ytr"][:6000], check_finite=False)
    raw = matern52(sqdist(sp["Xte"], sp["Xtr"][:6000]), med) @ a
    print(f"raw-input kernel     {100*rel(raw, sp['Yte']):.2f}%")
    print(f"neural mean          {100*rel(net_te, sp['Yte']):.2f}%")
    print(f"kernel on features   {100*rel(kernel_head(F[0], sp['Ytr'], F[1], sp['Yval'], F[2]), sp['Yte']):.2f}%")


def experiment_alignment(sp):
    feat = train(lambda: ResidualMLP(sp["Xtr"].shape[1], sp["Ytr"].shape[1]), sp)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    sub = np.random.default_rng(0).choice(len(sp["Xtr"]), 4000, replace=False)
    with torch.no_grad():
        Ff = feat(f32(sp["Xtr"][sub]), feats=True)[1].numpy()
    mu, sd = Ff.mean(0), Ff.std(0) + 1e-9; Ff = (Ff - mu) / sd
    Y = sp["Ytr"][sub]

    def rkhs_and_deff(Z):
        D2 = sqdist(np.asarray(Z, np.float64), np.asarray(Z, np.float64))
        ls = median_ls(np.asarray(Z))
        K = matern52(D2, ls)
        ev = np.maximum(np.linalg.eigvalsh(K), 0)
        deff = float((ev / (ev + len(Z) * 1e-6)).sum())
        Kr = K.copy(); Kr.flat[::len(Z) + 1] += 1e-8 * len(Z)
        a = cho_solve(cho_factor(Kr, lower=True, check_finite=False), Y, check_finite=False)
        return deff, float(np.einsum("nd,nd->", a, Y))

    d_raw, n_raw = rkhs_and_deff(sp["Xtr"][sub])
    d_feat, n_feat = rkhs_and_deff(Ff)
    print(f"effective dimension   raw {d_raw:.0f}   features {d_feat:.0f}   (design factor)")
    print(f"target RKHS norm      raw {n_raw:.3g}   features {n_feat:.3g}   (ratio {n_raw/n_feat:.1f}x)")


def experiment_diversity(sp):
    di, do = sp["Xtr"].shape[1], sp["Ytr"].shape[1]
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    preds = {}
    for name, make in [("silu_s0", lambda: ResidualMLP(di, do, act="silu")),
                       ("silu_s1", lambda: ResidualMLP(di, do, act="silu")),
                       ("relu", lambda: ResidualMLP(di, do, act="relu")),
                       ("fourier", lambda: FourierMLP(di, do, sigma=1.0))]:
        seed = 1 if name == "silu_s1" else 0
        m = train(make, sp, seed=seed)
        with torch.no_grad():
            out = m(f32(sp["Xte"]))
            preds[name] = (out[0] if isinstance(out, tuple) else out).numpy()
        print(f"  {name:9s} test {100*rel(preds[name], sp['Yte']):.2f}%")
    R = {k: (v - sp["Yte"]).ravel() for k, v in preds.items()}
    R = {k: r - r.mean() for k, r in R.items()}
    corr = lambda a, b: float(R[a] @ R[b] / (np.linalg.norm(R[a]) * np.linalg.norm(R[b])))
    print(f"  seed-to-seed correlation (silu s0 vs s1): {corr('silu_s0','silu_s1'):.3f}")
    print(f"  cross-architecture (silu vs relu):        {corr('silu_s0','relu'):.3f}")
    print(f"  cross-architecture (silu vs fourier):     {corr('silu_s0','fourier'):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="o2")
    p.add_argument("--experiment", choices=["ladder", "alignment", "diversity"], default="ladder")
    args = p.parse_args()
    sp = load_band(args.band)
    t0 = time.time()
    {"ladder": experiment_ladder, "alignment": experiment_alignment,
     "diversity": experiment_diversity}[args.experiment](sp)
    print(f"[{(time.time()-t0)/60:.1f} min]")
