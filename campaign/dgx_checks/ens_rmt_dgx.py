"""Sixty-predictor per-pixel stacks with regularization chosen by GCV, leave-one-out and the
Marchenko-Pastur edge: the ensemble-weighting study on the structural-mechanics pool.

Paper 1 found that a per-pixel affine stack over the SIX ten-seed means (4.556) beats the same stack
over all SIXTY predictors (4.682): sixty-one weights per pixel from a thousand calibration cases
overfit. This script asks whether a random-matrix or cross-validated regularization of the
per-pixel regression recovers, or beats, the six-mean stack from the sixty:

  six_means_ridge1e-3   the paper's construction (baseline)
  sixty_ridge1e-3       the naive sixty-member stack (the overfitting case)
  sixty_ridge_gcv       one ridge for all pixels, chosen by generalized cross-validation on the calibration half
  sixty_ridge_loo       the same with the exact leave-one-out error
  sixty_ridge_pixgcv    a ridge per pixel by GCV
  sixty_pcr_mp          per-pixel principal-component regression on the member-prediction covariance,
                        keeping the components above the Marchenko-Pastur edge of the calibration covariance
  sixty_shrink_global   per-pixel weights shrunk toward the global convex weights, shrinkage chosen by LOO
  sixty_ridge_lw        per-pixel weights from the Ledoit-Wolf shrunk Gram
  sixty_rie             per-pixel weights from the rotationally invariant (nonlinear shrinkage) covariance estimator
  _small_calibration    the same estimators with 120 and 300 calibration rows (q = 0.5, 0.2), five draws each

All weights are fitted on the calibration half of the test block (1000 cases, split seed 20260902 as in
seedarch.py) and every number is read on the evaluation half (19000). Runs on the DGX over
~/nmkc2/seeds/sm_s*/runs/*_predte.npy. Writes results/sm_ens_rmt.json.
"""
import glob, json, os, time
import numpy as np

ROOT = os.path.expanduser("~/nmkc2")
DATA = ROOT + "/data/structmech"
t0 = time.time()
stress = np.load(DATA + "/stress.npy"); ite = np.load(DATA + "/idx_test.npy")
Yte = stress[ite].reshape(len(ite), -1).astype(np.float64)
nte = np.linalg.norm(Yte, axis=1)
rng = np.random.default_rng(20260902)
perm = rng.permutation(len(ite)); cal, ev = np.sort(perm[:1000]), np.sort(perm[1000:])
ARCH = ["mlp", "mlpMSE", "mlpR", "fno", "unet", "krr"]
names, P = [], []
for s in range(10):
    d = f"{ROOT}/seeds/sm_s{s}/runs"
    for a in ARCH:
        if a == "krr":
            f = d + "/krr_full_matern52_n19000_pred_test.npy"
        else:
            cands = [p for p in glob.glob(d + "/*_predte.npy") if os.path.basename(p).split("_")[0] == a and "mpprune" not in os.path.basename(p)]
            assert len(cands) == 1, (a, s, cands)          # the pruned copies (*_mpprune_predte.npy) are not pool members
            f = cands[0]
        P.append(np.load(f).astype(np.float32)); names.append(f"{a}_s{s}")
P = np.stack(P)                       # (60, 20000, 1681)
M, n_all, D = P.shape
print(f"loaded {M} predictors {P.shape} in {time.time()-t0:.0f}s", flush=True)
means = np.stack([P[[i for i, nm in enumerate(names) if nm.startswith(a + "_")]].mean(0) for a in ARCH])   # (6, n, D)


def rel(pred, rows):
    return float((np.linalg.norm(pred - Yte[rows], axis=1) / nte[rows]).mean())


def _aug(Pm):
    """(m, n, D) -> (D, n, m+1) with the intercept column, float64: the per-pixel design matrices (batched matmul)."""
    m_, n_, D_ = Pm.shape
    return np.ascontiguousarray(np.concatenate([Pm, np.ones((1, n_, D_), Pm.dtype)], 0).astype(np.float64).transpose(2, 1, 0))


def fit_pixel(Pm, Y, lam):
    """Per-pixel affine weights, one ridge lam (scalar or per-pixel array), on rows of Pm (m, n, D)."""
    m_, n_, D_ = Pm.shape
    Xt = _aug(Pm)
    G = np.matmul(Xt.transpose(0, 2, 1), Xt) / n_                                  # (D, m+1, m+1)
    b = np.matmul(Xt.transpose(0, 2, 1), Y.T[:, :, None])[..., 0] / n_             # (D, m+1)
    lam = np.broadcast_to(np.asarray(lam, np.float64), (D_,))
    return np.linalg.solve(G + lam[:, None, None] * np.eye(m_ + 1)[None], b[..., None])[..., 0]   # (D, m+1)


def apply_pixel(Pm, W, chunk=2000):
    """Per-pixel affine prediction, in row chunks so the (D, n, m+1) design never exceeds a few GB."""
    out = np.empty((Pm.shape[1], Pm.shape[2]))
    for i in range(0, Pm.shape[1], chunk):
        out[i:i + chunk] = np.matmul(_aug(Pm[:, i:i + chunk]), W[:, :, None])[..., 0].T
    return out


def stage(msg):
    print(f"-- {msg} [{(time.time()-t0)/60:.1f} min]", flush=True)


def hat_diag(Xt, lam):
    D_, n_, m1 = Xt.shape
    G = np.matmul(Xt.transpose(0, 2, 1), Xt) / n_ + lam * np.eye(m1)[None]
    T_ = np.matmul(Xt, np.linalg.inv(G))
    return (T_ * Xt).sum(-1).T / n_                                                # (n, D)


def gcv_curve(Pm, Y, lams):
    """GCV and LOO of the per-pixel ridge, summed over pixels, at each lam (one lam for all pixels); also the per-pixel GCV."""
    Xt = _aug(Pm); n_ = Xt.shape[1]
    out_g, out_l, out_pix = [], [], []
    for lam in lams:
        R = Y - apply_pixel(Pm, fit_pixel(Pm, Y, lam))
        H = hat_diag(Xt, lam); tr = H.sum(0)
        gp = (R ** 2).sum(0) / n_ / (1 - tr / n_) ** 2
        out_pix.append(gp); out_g.append(float(gp.sum())); out_l.append(float(((R / (1 - H)) ** 2).sum() / n_))
    return np.array(out_g), np.array(out_l), np.array(out_pix)


def ledoit_wolf_weights(Pm, Y):
    m_, n_, D_ = Pm.shape
    Xt = _aug(Pm)
    G = np.matmul(Xt.transpose(0, 2, 1), Xt) / n_
    b = np.matmul(Xt.transpose(0, 2, 1), Y.T[:, :, None])[..., 0] / n_
    mu = np.trace(G, axis1=1, axis2=2) / (m_ + 1)
    Fro = lambda A: (A ** 2).sum((1, 2))
    d2 = Fro(G - mu[:, None, None] * np.eye(m_ + 1)[None])
    x2 = (Xt ** 2).sum(-1).T
    b2 = ((x2 ** 2).sum(0) / n_ - Fro(G)) / n_
    a = np.clip(np.minimum(b2, d2) / np.maximum(d2, 1e-300), 0.0, 1.0)
    Gl = (1 - a)[:, None, None] * G + (a * mu)[:, None, None] * np.eye(m_ + 1)[None]
    return np.linalg.solve(Gl + 1e-10 * np.eye(m_ + 1)[None], b[..., None])[..., 0], a


def rie_weights(Pc, Yc, eta_scale=1.0):
    """Rotationally invariant (nonlinear shrinkage) estimator of the member covariance -> per-pixel weights."""
    m_, n_, D_ = Pc.shape
    q = m_ / n_
    W = np.zeros((D_, m_ + 1)); change = []
    mu_p = Pc.mean(1); mu_y = Yc.mean(0)
    for d in range(D_):
        Xc = Pc[:, :, d] - mu_p[:, d][:, None]
        C = Xc @ Xc.T / n_; c = Xc @ (Yc[:, d] - mu_y[d]) / n_
        lam, V = np.linalg.eigh(C)
        eta = eta_scale * lam.mean() / np.sqrt(n_)
        z = lam - 1j * eta
        g = (1.0 / (z[:, None] - lam[None, :])).mean(1)
        xi = np.maximum(lam / np.abs(1 - q + q * z * g) ** 2, 1e-8 * lam.max())
        change.append(float(np.median(np.abs(xi / np.maximum(lam, 1e-300) - 1))))
        w = V @ ((V.T @ c) / xi)
        W[d, :m_] = w; W[d, m_] = mu_y[d] - w @ mu_p[:, d]
    return W, dict(q=q, median_rel_change=float(np.median(change)))


def pcr_mp_pred(Pc, Yc, Pe):
    m_, n_, D_ = Pc.shape
    mu = Pc.mean(1, keepdims=True); Xc = Pc - mu; Xe = Pe - mu
    kept, pred = [], np.empty((Pe.shape[1], D_))
    for dpx in range(D_):
        C = Xc[:, :, dpx] @ Xc[:, :, dpx].T / n_
        w, V = np.linalg.eigh(C)
        k = np.where(w > 2.858 * np.median(w))[0]
        if len(k) == 0:
            k = np.array([m_ - 1])
        kept.append(len(k))
        Zc, Ze = Xc[:, :, dpx].T @ V[:, k], Xe[:, :, dpx].T @ V[:, k]
        yc = Yc[:, dpx] - Yc[:, dpx].mean()
        beta = np.linalg.lstsq(Zc, yc, rcond=None)[0]
        pred[:, dpx] = Ze @ beta + Yc[:, dpx].mean()
    return pred, float(np.median(kept))


results = {}
lams = np.logspace(-6, 4, 41)     # the LOO minimum sat at the old upper end (1.0) on the sixty
Ycal, Yev = Yte[cal], Yte[ev]
P6c, P6e = means[:, cal], means[:, ev]
P60c, P60e = P[:, cal], P[:, ev]

W = fit_pixel(P6c, Ycal, 1e-3); results["six_means_ridge1e-3"] = rel(apply_pixel(P6e, W), ev)
W = fit_pixel(P60c, Ycal, 1e-3); results["sixty_ridge1e-3"] = rel(apply_pixel(P60e, W), ev)
results["sixty_equal"] = rel(P60e.mean(0), ev); results["six_means_equal"] = rel(P6e.mean(0), ev)
print({k: round(100 * v, 4) for k, v in results.items()}, f"[{(time.time()-t0)/60:.1f} min]", flush=True)

g, l, gpix = gcv_curve(P60c, Ycal, lams)
lam_g, lam_l = lams[int(np.argmin(g))], lams[int(np.argmin(l))]
W = fit_pixel(P60c, Ycal, lam_g); results["sixty_ridge_gcv"] = rel(apply_pixel(P60e, W), ev)
W = fit_pixel(P60c, Ycal, lam_l); results["sixty_ridge_loo"] = rel(apply_pixel(P60e, W), ev)
g6, l6, _ = gcv_curve(P6c, Ycal, lams)
W = fit_pixel(P6c, Ycal, lams[int(np.argmin(l6))]); results["six_means_ridge_loo"] = rel(apply_pixel(P6e, W), ev)
print("gcv lam", lam_g, "loo lam", lam_l, {k: round(100 * v, 4) for k, v in results.items() if "gcv" in k or "loo" in k}, f"[{(time.time()-t0)/60:.1f} min]", flush=True)

stage('per-pixel gcv')
# per-pixel GCV ridge
m_, n_, D_ = P60c.shape
lam_pix = lams[np.argmin(gpix, 0)]
W = fit_pixel(P60c, Ycal, lam_pix); results["sixty_ridge_pixgcv"] = rel(apply_pixel(P60e, W), ev)
results["_lam_pix_median"] = float(np.median(lam_pix))

stage('pcr at the mp edge')
# per-pixel PCR at the MP edge: covariance of the 60 member predictions across calibration cases, per pixel
Xc = P60c.astype(np.float64) - P60c.astype(np.float64).mean(1, keepdims=True)   # (60, n, D)
Xe = P60e.astype(np.float64) - P60c.astype(np.float64).mean(1, keepdims=True)
kept, pred_ev = [], np.empty((len(ev), D_))
gamma = m_ / n_
for dpx in range(D_):
    C = Xc[:, :, dpx] @ Xc[:, :, dpx].T / n_                     # (60, 60)
    w, V = np.linalg.eigh(C)
    # noise level from the bulk median (Gavish-Donoho square-matrix rule adapted: median-based sigma^2 estimate)
    med = np.median(w); edge = 2.858 * med
    k = np.where(w > edge)[0]
    kept.append(len(k))
    Zc = Xc[:, :, dpx].T @ V[:, k]; Ze = Xe[:, :, dpx].T @ V[:, k]           # scores (n, k)
    yc = Ycal[:, dpx] - Ycal[:, dpx].mean()
    beta = np.linalg.lstsq(Zc, yc, rcond=None)[0]
    pred_ev[:, dpx] = Ze @ beta + Ycal[:, dpx].mean()
results["sixty_pcr_mp"] = rel(pred_ev, ev); results["_pcr_components_median"] = float(np.median(kept))

stage('shrink toward global convex')
# shrinkage toward the global convex weights (fitted on cal), amount by LOO over a grid
from scipy.optimize import minimize
Rn = (P60c.astype(np.float64) - Ycal[None]) / nte[cal][None, :, None]
S = np.einsum("mnd,knd->mk", Rn, Rn) / n_
res = minimize(lambda z: z @ S @ z, np.ones(m_) / m_, jac=lambda z: 2 * S @ z, bounds=[(0, 1)] * m_,
               constraints={"type": "eq", "fun": lambda z: z.sum() - 1}, method="SLSQP", options=dict(maxiter=3000, ftol=1e-15))
wg = np.maximum(res.x, 0); wg /= wg.sum()
Wg = np.concatenate([np.tile(wg, (D_, 1)), np.zeros((D_, 1))], 1)          # (D, 61), no intercept
Wp = fit_pixel(P60c, Ycal, lam_l)
best_s, best_e = 0.0, np.inf
for s_ in np.linspace(0, 1, 21):
    Wm = (1 - s_) * Wp + s_ * Wg
    # LOO is not exact for the mixture; use a 5-fold split of cal instead
    folds = np.array_split(np.random.default_rng(1).permutation(n_), 5); e = 0.0
    for f_ in folds:
        tr_ = np.setdiff1d(np.arange(n_), f_)
        Wf = (1 - s_) * fit_pixel(P60c[:, tr_], Ycal[tr_], lam_l) + s_ * Wg
        e += rel(apply_pixel(P60c[:, f_], Wf), cal[f_]) / 5
    if e < best_e:
        best_e, best_s = e, s_
Wm = (1 - best_s) * Wp + best_s * Wg
results["sixty_shrink_global"] = rel(apply_pixel(P60e, Wm), ev); results["_shrink_s"] = best_s
results["sixty_global_convex"] = rel(np.einsum("m,mnd->nd", wg, P60e), ev)
stage('ledoit-wolf and rie')
Wlw, a_lw = ledoit_wolf_weights(P60c, Ycal); results["sixty_ridge_lw"] = rel(apply_pixel(P60e, Wlw), ev); results["_lw_intensity_median"] = float(np.median(a_lw))
Wr, rie_rec = rie_weights(P60c, Ycal); results["sixty_rie"] = rel(apply_pixel(P60e, Wr), ev); results["_rie"] = rie_rec
print({k: (round(100 * v, 4) if not k.startswith("_") else v) for k, v in results.items()}, f"[{(time.time()-t0)/60:.1f} min]", flush=True)

# the regime where the random-matrix corrections are O(1): the same sixty members on 120 and 300 calibration rows
# (q = 0.5 and 0.2), each averaged over five draws of the calibration rows out of the thousand; read on the same 19,000
stage('small-calibration regime')
small = {}
for ncal in (120, 300):
    acc = {}
    for rep in range(5):
        rows = np.sort(np.random.default_rng(100 + rep).permutation(len(cal))[:ncal])
        Pc_, Yc_ = P60c[:, rows], Ycal[rows]
        gg, ll, gp = gcv_curve(Pc_, Yc_, lams)
        out = {}
        out["ridge_loo"] = rel(apply_pixel(P60e, fit_pixel(Pc_, Yc_, lams[int(np.argmin(ll))])), ev)
        out["ridge_pixgcv"] = rel(apply_pixel(P60e, fit_pixel(Pc_, Yc_, lams[np.argmin(gp, 0)])), ev)
        out["ridge_lw"] = rel(apply_pixel(P60e, ledoit_wolf_weights(Pc_, Yc_)[0]), ev)
        Wr_, rr = rie_weights(Pc_, Yc_); out["rie"] = rel(apply_pixel(P60e, Wr_), ev); out["_rie_change"] = rr["median_rel_change"]
        out["pcr_mp"], out["_pcr_kept"] = pcr_mp_pred(Pc_, Yc_, P60e)
        out["ols"] = rel(apply_pixel(P60e, fit_pixel(Pc_, Yc_, 1e-12)), ev)
        out["six_means_ridge1e-3"] = rel(apply_pixel(P6e, fit_pixel(P6c[:, rows], Yc_, 1e-3)), ev)
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
    small[f"ncal{ncal}"] = {k: (round(100 * float(np.mean(v)), 4) if not k.startswith("_") else round(float(np.mean(v)), 4)) for k, v in acc.items()}
    print("small-calibration regime", ncal, small[f"ncal{ncal}"], f"[{(time.time()-t0)/60:.1f} min]", flush=True)
results["_small_calibration"] = small
os.makedirs(ROOT + "/results", exist_ok=True)
json.dump(dict(results=results, lams=[float(x) for x in lams], gcv=[float(x) for x in g], loo=[float(x) for x in l],
               n_cal=len(cal), n_ev=len(ev), minutes=(time.time() - t0) / 60), open(ROOT + "/results/sm_ens_rmt.json", "w"), indent=1)
print("wrote results/sm_ens_rmt.json", flush=True)
