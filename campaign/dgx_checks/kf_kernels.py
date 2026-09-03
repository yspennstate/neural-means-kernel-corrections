"""Learned kernels for the exact solve: kernel flows, empirical Bayes and programmed kernels.

Four ways of choosing the kernel of an exact Matern/Gaussian kernel ridge regression are run on
the same split and scored once on test, so that the selection criterion is the only thing that
changes:

  val_iso   isotropic Matern-5/2, scale x nugget grid on validation                 (the baseline)
  kf_ard    per-input length scales learned by the kernel-flow loss rho of Owhadi and Yoo
            (rho = 1 - y_c' K_cc^-1 y_c / y_b' K_bb^-1 y_b on random batches, half sub-batches;
            trace form for vector outputs), then the global scale and nugget re-selected on
            validation; the flow learns the SHAPE of the metric, validation its size
  kf_mahal  the same with a full low-rank Mahalanobis metric M = L'L + diag(exp(2 s))
  eb_ard    per-input length scales, amplitude and nugget by empirical Bayes: the profiled
            negative log marginal likelihood y'(K+lam I)^-1 y + log det(K+lam I) on a training
            subsample (Chen, Owhadi and Stuart's L^EB), no validation step at all
  add_kf    a programmed kernel in the sense of kernel mode decomposition: a weighted sum of
            first-order modes k_i(x_i), pairwise modes k_ij(x_i, x_j) and one full-order mode,
            mode weights and length scales learned by the kernel-flow loss; the learned weights
            and the mode energies of the interpolant are reported, which is the structure
            discovery the hypergraph paper performs with the same additive kernels

After the kernel is chosen the solve is exact on every training row (one Cholesky), and the
test error is read once per method in physical units.  Problems: emit (one component),
oco2 (one band), structmech.  Environment: NMKC_THREADS, P2_OUT and the data variables of p2_data.
    python kf_kernels.py --problem emit --comp Y1 --seed 101
    python kf_kernels.py --problem oco2 --band o2 --seed 3 --methods val_iso,kf_ard,eb_ard
"""
import argparse, json, os, pathlib, time
import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
torch.set_num_threads(int(_T))
torch.set_default_dtype(torch.float64)
import p2_data

p = argparse.ArgumentParser()
p.add_argument("--problem", required=True, choices=["emit", "oco2", "structmech"])
p.add_argument("--comp", default="Y1")
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=101)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--methods", default="val_iso,kf_ard,kf_mahal,eb_ard,add_kf,rfm")
p.add_argument("--rfm_iters", type=int, default=3)
p.add_argument("--rfm_points", type=int, default=2000, help="training points the average gradient outer product is taken over")
p.add_argument("--rfm_power", type=float, default=0.5, help="metric M = AGOP^power (1 = the original RFM, 0.5 = the square-root variant)")
p.add_argument("--rfm_damp", type=float, default=1.0, help="M_t = (1-damp) M_{t-1} + damp AGOP^power (1 = undamped)")
p.add_argument("--kf_steps", type=int, default=400)
p.add_argument("--kf_batch", type=int, default=600)
p.add_argument("--kf_loss", default="l2", choices=["l2", "rho"], help="l2: Yoo-Owhadi squared half-batch prediction error; rho: the RKHS-norm ratio")
p.add_argument("--objective", default="phys", choices=["phys", "whiten", "raw"], help="targets inside the KF and EB objectives: phys = the physical targets with per-sample relative normalization (the validation metric), whiten = standardized coefficients, raw = the coefficients")
p.add_argument("--eb_sub", type=int, default=3000)
p.add_argument("--tag", default="")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
OUT = pathlib.Path(os.environ.get("P2_OUT", "results"))
if args.smoke:
    args.kf_steps, args.eb_sub = 30, 800

t0 = time.time()
if args.problem == "emit":
    D = p2_data.emit(args.seed, args.comp, args.ntrain)
elif args.problem == "oco2":
    D = p2_data.oco2(args.band, args.seed, args.ntrain)
else:
    D = p2_data.structmech(args.seed, args.ntrain)
Xtr, Xva, Xte, Ztr, Zva, Zte = (D[k] for k in ("Xtr", "Xva", "Xte", "Ztr", "Zva", "Zte"))
if args.smoke:
    Xtr, Ztr = Xtr[:3000], Ztr[:3000]
n, d = Xtr.shape
err = D["phys_err"]
print(f"{D['tag']}: n={n} d={d} outputs={Ztr.shape[1]}", flush=True)
sub_tune = np.random.default_rng(args.seed + 7).permutation(n)[:min(6000, n)]
# targets seen by the kernel-flow and empirical-Bayes objectives; the exact solves regress Ztr itself.
# phys: the centered physical targets, so the half-batch error is the physical error the paper reports
if args.objective == "phys":
    Zobj = D["Yobj"][:n]; ynorm2 = D["ynorm2"][:n]
elif args.objective == "whiten":
    Zobj = (Ztr - Ztr.mean(0)) / (Ztr.std(0) + 1e-12); ynorm2 = (Zobj ** 2).sum(1)
else:
    Zobj = Ztr; ynorm2 = (Ztr ** 2).sum(1)
D2s_iso = None


# ---------------- kernels (numpy, for the exact solve) ----------------
def sqd_np(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def matern_np(D2, nu):
    r = np.sqrt(D2)
    if nu == 0.5:
        return np.exp(-r)                 # Laplace (the kernel of the recursive feature machine paper)
    if nu == 1.5:
        a = np.sqrt(3.0) * r; return (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5.0) * r; return (1 + a + a * a / 3.0) * np.exp(-a)
    return np.exp(-0.5 * D2)          # nu = inf, Gaussian


def solve(K, Y, nug):
    Kr = K.copy(); Kr.flat[::len(K) + 1] += nug * len(K)
    c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
    return cho_solve(c, Y, check_finite=False)


class Metric:
    """x -> A x with A = diag(1/ell) (ARD) or a low-rank Mahalanobis; distances computed in the image."""

    def __init__(self, ell=None, L=None):
        self.ell = ell; self.L = L

    def map(self, X):
        parts = []
        if self.ell is not None:
            parts.append(X / self.ell[None, :])
        if self.L is not None:
            parts.append(X @ self.L.T)
        Xm = np.concatenate(parts, axis=1)
        return Xm


def fit_predict(metric, nu, scale, nug, rows=None):
    """Exact solve on all training rows in the metric, kernel k(x,x') = matern_nu(|A(x-x')|/scale)."""
    Ftr, Fva, Fte = metric.map(Xtr), metric.map(Xva), metric.map(Xte)
    K = matern_np(sqd_np(Ftr, Ftr) / scale ** 2, nu)
    alpha = solve(K, Ztr, nug); del K
    out = []
    for F_ in (Fva, Fte):
        pred = np.empty((len(F_), Ztr.shape[1]))
        for k in range(0, len(F_), 4000):
            pred[k:k + 4000] = matern_np(sqd_np(F_[k:k + 4000], Ftr) / scale ** 2, nu) @ alpha
        out.append(pred)
    return out


def tune_scale_nug(metric, nus=(1.5, 2.5), scales=(0.5, 1.0, 2.0, 4.0), nugs=(1e-8, 1e-6, 1e-4, 1e-2)):
    """Global scale (x median image distance) and nugget on validation over the tuning subsample."""
    Fs, Fv = metric.map(Xtr[sub_tune]), metric.map(Xva)
    D2s, D2vs = sqd_np(Fs, Fs), sqd_np(Fv, Fs)
    med = float(np.sqrt(np.median(D2s[np.triu_indices(len(Fs), 1)])))
    best = (np.inf, None)
    for nu in nus:
        for sc in scales:
            Ks, Kvs = matern_np(D2s / (sc * med) ** 2, nu), matern_np(D2vs / (sc * med) ** 2, nu)
            for nug in nugs:
                try:
                    e = err(Kvs @ solve(Ks, Ztr[sub_tune], nug), "va")
                except np.linalg.LinAlgError:
                    continue
                if e < best[0]:
                    best = (e, dict(nu=nu, scale=sc * med, nugget=nug, val_sub=e))
    return best[1]


# ---------------- kernel flows (torch) ----------------
def t(a):
    return torch.as_tensor(a, dtype=torch.float64)


def matern_t(D2, nu):
    r = torch.sqrt(D2 + 1e-12)
    if nu == 1.5:
        a = np.sqrt(3.0) * r; return (1 + a) * torch.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5.0) * r; return (1 + a + a * a / 3.0) * torch.exp(-a)
    return torch.exp(-0.5 * D2)


def sqd_t(A, B):
    return torch.clamp((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, min=0.0)


def kf_rho(K, Y, half, nug=1e-6):
    """rho = 1 - tr(Y_c' K_cc^-1 Y_c) / tr(Y_b' K_bb^-1 Y_b) on one batch (rows already permuted)."""
    nb = K.shape[0]
    Kb = K + nug * nb * torch.eye(nb)
    Kc = Kb[:half, :half]
    Lb = torch.linalg.cholesky(Kb); Lc = torch.linalg.cholesky(Kc)
    qb = (Y * torch.cholesky_solve(Y, Lb)).sum()
    qc = (Y[:half] * torch.cholesky_solve(Y[:half], Lc)).sum()
    return 1.0 - qc / qb


def kf_l2(K, Y, half, yn2, nug=1e-4):
    """Yoo-Owhadi l2 variant: the half batch predicts the other half by kernel ridge; the loss is the mean
    per-sample squared relative error in the objective's metric (the physical metric when --objective phys)."""
    Kc = K[:half, :half] + nug * half * torch.eye(half)
    Lc = torch.linalg.cholesky(Kc)
    pred = K[half:, :half] @ torch.cholesky_solve(Y[:half], Lc)
    return (((Y[half:] - pred) ** 2).sum(1) / yn2[half:]).mean()


def kernel_flow(param_kernel, params, steps, batch, lr=0.05, seed=0):
    """Owhadi-Yoo Algorithm 1: sample N_f rows, N_c = N_f/2 of them, one Adam step on the flow loss
    (rho, the RKHS-norm ratio, or its l2 variant)."""
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    Xt, Yt, Nt = t(Xtr), t(Zobj), t(ynorm2)
    hist = []
    for it in range(steps):
        idx = torch.as_tensor(rng.choice(n, min(batch, n), replace=False))
        Xb, Yb, Nb = Xt[idx], Yt[idx], Nt[idx]
        K = param_kernel(Xb)
        loss = kf_rho(K, Yb, len(idx) // 2) if args.kf_loss == "rho" else kf_l2(K, Yb, len(idx) // 2, Nb)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss))
    return hist


# ---------------- empirical Bayes (torch, profiled amplitude) ----------------
def eb_fit(log_ell0, sub, steps=200, lr=0.05, nu=2.5):
    """Minimize y'(K+lam I)^-1 y + log det(K+lam I) with the amplitude profiled out, over log
    per-input length scales and log nugget, on a training subsample; Chen-Owhadi-Stuart (1.2)."""
    Xs, Ys = t(Xtr[sub]), t(Zobj[sub])
    m, q = Ys.shape
    log_ell = torch.tensor(log_ell0, requires_grad=True)
    log_nug = torch.tensor(np.log(1e-3), requires_grad=True)
    opt = torch.optim.Adam([log_ell, log_nug], lr=lr)
    for it in range(steps):
        F = Xs / torch.exp(log_ell)[None, :]
        K = matern_t(sqd_t(F, F), nu) + torch.exp(log_nug) * torch.eye(m)
        L = torch.linalg.cholesky(K)
        quad = (Ys * torch.cholesky_solve(Ys, L)).sum()
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        # amplitude sigma^2 profiled: sigma^2 = quad/(m q); NLL up to constants = m q log(quad) + q logdet
        nll = m * q * torch.log(quad) + q * logdet
        opt.zero_grad(); nll.backward(); opt.step()
    return log_ell.detach().numpy(), float(torch.exp(log_nug)), float(nll)


results, hyper, preds_te = {}, {}, {}
methods = args.methods.split(",")


def record(name, pv, pt, hp):
    results[name] = dict(val=err(pv, "va"), test=err(pt, "te"))
    if "rad_err" in D:
        results[name]["test_radiance"] = D["rad_err"](pt, "te")
    hyper[name] = hp; preds_te[name] = pt.astype(np.float32)
    print(f"== {name}: val {100*results[name]['val']:.4f}%  test {100*results[name]['test']:.4f}%  [{(time.time()-t0)/60:.1f} min]", flush=True)


if "val_iso" in methods:
    metric = Metric(ell=np.ones(d))
    hp = tune_scale_nug(metric)
    pv, pt = fit_predict(metric, hp["nu"], hp["scale"], hp["nugget"])
    record("val_iso", pv, pt, hp)

if "kf_ard" in methods:
    med0 = np.sqrt(np.median(sqd_np(Xtr[sub_tune[:2000]], Xtr[sub_tune[:2000]])[np.triu_indices(2000, 1)]))
    log_ell = torch.tensor(np.full(d, np.log(med0)), requires_grad=True)

    def k_ard(Xb):
        F = Xb / torch.exp(log_ell)[None, :]
        return matern_t(sqd_t(F, F), 2.5)
    hist = kernel_flow(k_ard, [log_ell], args.kf_steps, args.kf_batch, seed=args.seed)
    ell = np.exp(log_ell.detach().numpy())
    metric = Metric(ell=ell / np.exp(np.log(ell).mean()))          # shape only; validation sets the size
    hp = tune_scale_nug(metric)
    hp.update(ell_rel={nm: round(float(e), 4) for nm, e in zip(D["names"], metric.ell)}, rho_first=hist[0], rho_last=float(np.mean(hist[-20:])))
    pv, pt = fit_predict(metric, hp["nu"], hp["scale"], hp["nugget"])
    record("kf_ard", pv, pt, hp)

if "kf_mahal" in methods:
    r = min(3, d)
    log_ell = torch.tensor(np.full(d, np.log(med0 if "med0" in dir() else 1.0)), requires_grad=True)
    Lm = torch.zeros(r, d, requires_grad=True)

    def k_mahal(Xb):
        F = torch.cat([Xb / torch.exp(log_ell)[None, :], Xb @ Lm.T], dim=1)
        return matern_t(sqd_t(F, F), 2.5)
    hist = kernel_flow(k_mahal, [log_ell, Lm], args.kf_steps, args.kf_batch, seed=args.seed + 1)
    ell = np.exp(log_ell.detach().numpy()); g = np.exp(np.log(ell).mean())
    metric = Metric(ell=ell / g, L=Lm.detach().numpy() * g)
    hp = tune_scale_nug(metric)
    hp.update(ell_rel=[round(float(e), 4) for e in metric.ell], L_norm=float(np.linalg.norm(metric.L)), rho_last=float(np.mean(hist[-20:])))
    pv, pt = fit_predict(metric, hp["nu"], hp["scale"], hp["nugget"])
    record("kf_mahal", pv, pt, hp)

if "eb_ard" in methods:
    sub = np.random.default_rng(args.seed + 11).permutation(n)[:min(args.eb_sub, n)]
    med0 = np.sqrt(np.median(sqd_np(Xtr[sub[:2000]], Xtr[sub[:2000]])[np.triu_indices(min(2000, len(sub)), 1)]))
    log_ell, nug_eb, nll = eb_fit(np.full(d, np.log(med0)), sub, steps=(30 if args.smoke else 200))
    ell = np.exp(log_ell)
    metric = Metric(ell=ell)
    # as learned: scale 1 in the learned metric, the EB nugget (relative form: nug*n added to the diagonal
    # in solve(), so divide by the subsample size to keep the same absolute jitter)
    pv, pt = fit_predict(metric, 2.5, 1.0, nug_eb / n)    # solve() adds nug*n: the same absolute jitter as EB fitted
    record("eb_ard", pv, pt, dict(ell={nm: round(float(e), 4) for nm, e in zip(D["names"], ell)}, nugget_abs=nug_eb, nll=nll))
    # and with validation re-selecting the size and nugget on the EB shape
    metric2 = Metric(ell=ell / np.exp(np.log(ell).mean()))
    hp = tune_scale_nug(metric2)
    pv, pt = fit_predict(metric2, hp["nu"], hp["scale"], hp["nugget"])
    record("eb_ard_val", pv, pt, hp)

if "add_kf" in methods:
    # modes: first order (d), pairwise (d choose 2, only when d <= 8), full order (1)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)] if d <= 8 else []
    med0 = np.sqrt(np.median(sqd_np(Xtr[sub_tune[:2000]], Xtr[sub_tune[:2000]])[np.triu_indices(2000, 1)]))
    n_modes = d + len(pairs) + 1
    log_w = torch.zeros(n_modes, requires_grad=True)
    log_ell = torch.tensor(np.full(d, np.log(med0 / np.sqrt(d))), requires_grad=True)   # per-coordinate scales
    log_ell_full = torch.tensor(np.log(med0), requires_grad=True)

    def modes_t(Xa, Xb=None):
        # a generator, one mode at a time: on structmech (41 inputs, 19,000 rows) materializing all 42 n x n modes
        # at once is 120 GB and the lane dies with rc=137; the sums below consume the modes lazily
        Xb = Xa if Xb is None else Xb
        ell = torch.exp(log_ell)
        for i in range(d):
            yield matern_t(sqd_t(Xa[:, i:i + 1] / ell[i], Xb[:, i:i + 1] / ell[i]), 2.5)
        for (i, j) in pairs:
            yield matern_t(sqd_t(Xa[:, [i, j]] / ell[[i, j]], Xb[:, [i, j]] / ell[[i, j]]), 2.5)
        yield matern_t(sqd_t(Xa / torch.exp(log_ell_full), Xb / torch.exp(log_ell_full)), 2.5)

    def k_add(Xb):
        w = torch.softmax(log_w, 0)
        return sum(wi * m for wi, m in zip(w, modes_t(Xb)))
    hist = kernel_flow(k_add, [log_w, log_ell, log_ell_full], args.kf_steps, args.kf_batch, seed=args.seed + 2)
    w = torch.softmax(log_w, 0).detach()
    # exact solve with the programmed kernel: nugget on validation, kernel evaluated blockwise
    with torch.no_grad():
        Xt = t(Xtr)
        K = sum(wi * m for wi, m in zip(w, modes_t(Xt))).numpy()
        best = (np.inf, None)
        for nug in (1e-8, 1e-6, 1e-4, 1e-2):
            try:
                a_ = solve(K[np.ix_(sub_tune, sub_tune)], Ztr[sub_tune], nug)
            except np.linalg.LinAlgError:
                continue
            Kvs = sum(wi * m for wi, m in zip(w, modes_t(t(Xva), Xt[sub_tune]))).numpy()
            e = err(Kvs @ a_, "va")
            if e < best[0]:
                best = (e, nug)
        nug = best[1]
        alpha = solve(K, Ztr, nug); del K
        Kv = sum(wi * m for wi, m in zip(w, modes_t(t(Xva), Xt))).numpy(); pv = Kv @ alpha
        pt = np.concatenate([sum(wi * m for wi, m in zip(w, modes_t(t(Xte[k:k + 3000]), Xt))).numpy() @ alpha for k in range(0, len(Xte), 3000)])
        # mode energies of the interpolant on a subsample: E_i = alpha' (w_i K_i) alpha / alpha' K alpha (kernel mode decomposition)
        S_ = sub_tune[:1500]; Xs_ = Xt[S_]; a_s = solve(sum(wi * m for wi, m in zip(w, modes_t(Xs_))).numpy(), Ztr[S_], nug)
        Es = [float(np.sum(a_s * ((wi * m).numpy() @ a_s))) for wi, m in zip(w, modes_t(Xs_))]
        tot = sum(Es)
    names = D["names"] + [f"{D['names'][i]}*{D['names'][j]}" for (i, j) in pairs] + ["full"]
    hp = dict(weights={nm: round(float(x), 4) for nm, x in zip(names, w)}, energy={nm: round(e / tot, 4) for nm, e in zip(names, Es)},
              ell={nm: round(float(np.exp(x)), 4) for nm, x in zip(D["names"], log_ell.detach())}, ell_full=float(torch.exp(log_ell_full)),
              nugget=nug, rho_last=float(np.mean(hist[-20:])))
    record("add_kf", pv, pt, hp)

if "rfm" in methods:
    # Recursive feature machine (Radhakrishnan, Beaglehole, Pandit, Belkin 2022): kernel ridge regression in the
    # Mahalanobis metric M, with M re-estimated from the fitted predictor as the average gradient outer product
    # (1/m) sum_i sum_o grad f_o(x_i) grad f_o(x_i)^T over m training points, iterated. Here f_o are the standardized
    # outputs the exact solve regresses; the metric's size and the nugget are re-selected on validation each round.
    def matern_dk_over_r(D2, nu):
        """k'(r)/r for the Matern kernels used in fit_predict (r = distance in the scaled image); finite at r = 0."""
        r = np.sqrt(D2)
        if nu == 0.5:                     # not differentiable at r = 0: the self term is dropped, as in the RFM code
            out = np.zeros_like(r); nz = r > 1e-12; out[nz] = -np.exp(-r[nz]) / r[nz]; return out
        if nu == 1.5:
            a = np.sqrt(3.0) * r; return -3.0 * np.exp(-a)
        if nu == 2.5:
            a = np.sqrt(5.0) * r; return -(5.0 / 3.0) * (1 + a) * np.exp(-a)
        return -np.exp(-0.5 * D2)

    def agop(Lmat, hp_, pts):
        """Average gradient outer product of the exact-solve predictor in the metric x -> Lmat x, over training rows pts."""
        met = Metric(ell=None, L=Lmat)
        Ftr = met.map(Xtr); s2 = hp_["scale"] ** 2
        K = matern_np(sqd_np(Ftr, Ftr) / s2, hp_["nu"]); alpha = solve(K, Ztr, hp_["nugget"]); del K
        MtM = Lmat.T @ Lmat
        Gsum = np.zeros((d, d))
        for k in range(0, len(pts), 200):
            P_ = pts[k:k + 200]
            W = matern_dk_over_r(sqd_np(Ftr[P_], Ftr) / s2, hp_["nu"]) / s2            # (c, n)  k'(r)/(r s^2)
            # g[c, :, o] = sum_j W[c, j] alpha[j, o] (x_c - x_j)  =  x_c (W alpha)[c, o] - (W^T-weighted sum of x_j)
            Wa = W @ alpha                                                               # (c, q)
            XW = np.einsum("cj,jd->cd", W, Xtr)                                          # (c, d): sum_j W_cj x_j (shared over o)
            # grad f_o(x_c) = MtM (x_c Wa[c,o] - sum_j W_cj alpha_jo x_j); the second term needs the o-dependence:
            T2 = np.einsum("cj,jo,jd->cod", W, alpha, Xtr, optimize=True)                # (c, q, d)
            g = Xtr[P_][:, None, :] * Wa[:, :, None] - T2                                # (c, q, d)
            g = g @ MtM                                                                  # apply M = L^T L (symmetric)
            Gsum += np.einsum("cod,coe->de", g, g)
        return Gsum / len(pts)

    pts = np.random.default_rng(args.seed + 13).permutation(n)[:min(args.rfm_points, n)]
    Mcur = np.eye(d) / d; Lcur = np.eye(d) / np.sqrt(d)
    NUS_RFM = (0.5, 1.5, 2.5)                                                            # Laplace included: the RFM paper's kernel
    metric = Metric(ell=None, L=Lcur); hp = tune_scale_nug(metric, nus=NUS_RFM)
    hist_rfm = [("iso", hp["val_sub"])]; best = (hp["val_sub"], Lcur, hp, 0)
    for it in range(args.rfm_iters):
        A = agop(Lcur, hp, pts)
        w_, V_ = np.linalg.eigh(A); w_ = np.maximum(w_, 1e-10 * w_.max())
        Ap = (V_ * w_ ** args.rfm_power) @ V_.T                                          # AGOP^power (symmetric)
        Mnew = (1 - args.rfm_damp) * Mcur + args.rfm_damp * Ap / np.trace(Ap) * d        # trace-normalized; size re-tuned
        w_, V_ = np.linalg.eigh(Mnew); w_ = np.maximum(w_, 1e-8 * w_.max())
        Mcur = Mnew; Lcur = (V_ * np.sqrt(w_)) @ V_.T                                    # symmetric square root
        metric = Metric(ell=None, L=Lcur); hp = tune_scale_nug(metric, nus=NUS_RFM)
        hist_rfm.append((f"iter{it + 1}", hp["val_sub"]))
        if hp["val_sub"] < best[0]:
            best = (hp["val_sub"], Lcur, hp, it + 1)
        print(f"   rfm iter {it + 1}: val_sub {100 * hp['val_sub']:.4f}%  top eig share {w_.max() / w_.sum():.3f}  [{(time.time()-t0)/60:.1f} min]", flush=True)
    _, Lbest, hp, it_best = best                                                          # the iterate validation prefers
    metric = Metric(ell=None, L=Lbest)
    pv, pt = fit_predict(metric, hp["nu"], hp["scale"], hp["nugget"])
    w_, V_ = np.linalg.eigh(Lbest.T @ Lbest)
    hp.update(iters=args.rfm_iters, chosen_iter=it_best, power=args.rfm_power, damp=args.rfm_damp, points=int(len(pts)),
              val_sub_path=[(a, round(100 * b, 4)) for a, b in hist_rfm],
              M_eigs=[round(float(x), 5) for x in w_[::-1]], M_diag={nm: round(float(x), 4) for nm, x in zip(D["names"], np.diag(Lbest.T @ Lbest))})
    record("rfm", pv, pt, hp)

tag = args.tag or (D["tag"] + "_kfk" + (f"_n{n}" if args.ntrain else ""))
out = dict(tag=tag, kind="kf_kernels", problem=args.problem, comp=args.comp, band=args.band, seed=args.seed, ntrain=n,
           smoke=bool(args.smoke), results={k: {m: 100 * v for m, v in r.items()} for k, r in results.items()},
           hyper=hyper, kf_steps=args.kf_steps, kf_batch=args.kf_batch, kf_loss=args.kf_loss, objective=args.objective, minutes=round((time.time() - t0) / 60, 1))
OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / (tag + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT / (tag + ".json"))
(OUT / "preds").mkdir(exist_ok=True)
np.savez_compressed(OUT / "preds" / (tag + ".npz"), **{k: v for k, v in preds_te.items()})
print(f"DONE {tag} in {out['minutes']} min", flush=True)
