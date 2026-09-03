"""Random-matrix selection of the ridge and spectrum of an exact kernel ridge regression.

One Gram matrix (Matern-5/2 on the standardized inputs at the validation-chosen scale, on a
training subsample of --nfit rows, default 6000, or all rows), one eigendecomposition, and every
selector read from it:

  val        the ridge that minimizes the validation error (the paper's protocol)
  gcv        generalized cross-validation ||(I-H)y||^2 / (1 - tr H / n)^2 (Golub, Heath, Wahba)
  kare       the kernel alignment risk estimator of Jacot et al. 2020,
             (1/n) y'(K/n + lam)^-2 y / ((1/n) tr (K/n + lam)^-1)^2, minimized over lam
  loo        exact leave-one-out error of ridge regression (Allen's PRESS), minimized over lam
  mp_trunc   principal-component regression in the kernel eigenbasis with the cut at the
             Marchenko-Pastur bulk edge estimated from the median eigenvalue (Gavish-Donoho's
             2.858 x median rule applied to the Gram spectrum), the ridge then chosen by GCV
  mp_shrink  the same spectrum with eigenvalues below the edge shrunk to their bulk mean
             (a rotationally invariant cleaning of the Gram matrix), ridge by GCV

Each selector's ridge is used for the exact solve on the same rows and scored on test once.
Also reported: the Gram spectrum summary (edge, fraction of eigenvalues in the bulk, effective
dimension at each selected ridge) so the theory paper can relate the selectors to the spectrum.
Problems: emit (one component), oco2 (one band), structmech.
    python rmt_krr.py --problem emit --comp Y1 --seed 101
"""
import argparse, json, os, pathlib, time
import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
import numpy as np
import p2_data

p = argparse.ArgumentParser()
p.add_argument("--problem", required=True, choices=["emit", "oco2", "structmech"])
p.add_argument("--comp", default="Y1")
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=101)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--nfit", type=int, default=0, help="rows in the Gram (0 = all training rows)")
p.add_argument("--tag", default="")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
OUT = pathlib.Path(os.environ.get("P2_OUT", "results"))
t0 = time.time()
D = p2_data.emit(args.seed, args.comp, args.ntrain) if args.problem == "emit" else \
    p2_data.oco2(args.band, args.seed, args.ntrain) if args.problem == "oco2" else p2_data.structmech(args.seed, args.ntrain)
Xtr, Xva, Xte, Ztr, Zva, Zte = (D[k] for k in ("Xtr", "Xva", "Xte", "Ztr", "Zva", "Zte"))
if args.smoke:
    Xtr, Ztr = Xtr[:2500], Ztr[:2500]
if args.nfit and args.nfit < len(Xtr):
    Xtr, Ztr = Xtr[:args.nfit], Ztr[:args.nfit]
n, d = Xtr.shape
err = D["phys_err"]
print(f"{D['tag']}: n={n} d={d} outputs={Ztr.shape[1]}", flush=True)


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


# scale on validation at a fixed small ridge (the scale is not the object of this study; the ridge is)
sub = np.random.default_rng(args.seed).permutation(n)[:min(4000, n)]
D2s = sqd(Xtr[sub], Xtr[sub]); med = np.sqrt(np.median(D2s[np.triu_indices(len(sub), 1)]))
best = (np.inf, None)
for sc in (0.5, 1.0, 2.0, 4.0):
    Ks = m52(D2s, sc * med); Kv = m52(sqd(Xva, Xtr[sub]), sc * med)
    w, V = np.linalg.eigh(Ks)
    for lam in (1e-8, 1e-6, 1e-4):
        alpha = V @ ((V.T @ Ztr[sub]) / (w + lam * len(sub))[:, None])
        e = err(Kv @ alpha, "va")
        if e < best[0]:
            best = (e, sc)
scale = best[1] * med
print(f"scale {best[1]} x median ({scale:.3f}); full Gram on {n} rows", flush=True)

K = m52(sqd(Xtr, Xtr), scale)
w, V = np.linalg.eigh(K)          # ascending
w = np.maximum(w, 0.0)
Kva, Kte = m52(sqd(Xva, Xtr), scale), m52(sqd(Xte, Xtr), scale)
Yc = V.T @ Ztr                    # coefficients of the targets in the eigenbasis, (n, q)
q = Ztr.shape[1]
ynorm2 = (Ztr ** 2).sum()
lams = np.logspace(-10, 0, 41)    # relative ridge: K + lam * n * I


def predict(lam, keep=None, shrink=None):
    ww = w.copy()
    if shrink is not None:
        ww = shrink
    if keep is not None:
        Vk, wk, Yk = V[:, keep], ww[keep], Yc[keep]
        alpha = Vk @ (Yk / (wk + lam * n)[:, None])
    else:
        alpha = V @ (Yc / (ww + lam * n)[:, None])
    return Kva @ alpha, Kte @ alpha


def gcv(lam, keep=None):
    ww = w if keep is None else w[keep]; Yk = Yc if keep is None else Yc[keep]
    h = ww / (ww + lam * n)                          # hat-matrix eigenvalues
    resid2 = ((1 - h)[:, None] ** 2 * Yk ** 2).sum() + (0.0 if keep is None else (Yc ** 2).sum() - (Yk ** 2).sum())
    return resid2 / n / (1 - h.sum() / n) ** 2


def kare(lam):
    # KARE(lam) = (1/n) y'(K/n + lam I)^-2 y / ((1/n) tr (K/n + lam I)^-1)^2, summed over outputs
    a = w / n + lam
    num = ((Yc ** 2) / (a[:, None] ** 2)).sum() / n
    den = ((1.0 / a).sum() / n) ** 2
    return num / den


def loo(lam):
    # Allen's PRESS for ridge: sum_i (r_i/(1-H_ii))^2 with r = (I-H)y; H = V diag(h) V'
    h = w / (w + lam * n)
    H_diag = (V ** 2 * h[None, :]).sum(1)
    R = Ztr - V @ (h[:, None] * Yc)
    return ((R / (1 - H_diag)[:, None]) ** 2).sum() / n


results, hyper = {}, {}


def rec(name, lam, keep=None, shrink=None, extra=None):
    pv, pt = predict(lam, keep, shrink)
    results[name] = dict(val=err(pv, "va"), test=err(pt, "te"))
    if "rad_err" in D:
        results[name]["test_radiance"] = D["rad_err"](pt, "te")
    ww = w if shrink is None else shrink
    deff = float((ww / (ww + lam * n)).sum()) if keep is None else float((ww[keep] / (ww[keep] + lam * n)).sum())
    hyper[name] = dict(lam=float(lam), deff=deff, **(extra or {}))
    print(f"== {name}: lam {lam:.2e} deff {deff:.1f} val {100*results[name]['val']:.4f}% test {100*results[name]['test']:.4f}% [{(time.time()-t0)/60:.1f} min]", flush=True)


# validation-selected ridge
ev = [err(predict(l)[0], "va") for l in lams]
rec("val", lams[int(np.argmin(ev))])
rec("gcv", lams[int(np.argmin([gcv(l) for l in lams]))])
rec("kare", lams[int(np.argmin([kare(l) for l in lams]))])
rec("loo", lams[int(np.argmin([loo(l) for l in lams]))])

# Marchenko-Pastur bulk edge from the median eigenvalue (Gavish-Donoho's square-matrix rule applied to the
# Gram spectrum: noise-level eigenvalues sit below 2.858 x median); above it, signal directions
medw = float(np.median(w))
edge = 2.858 * medw
keep = np.where(w > edge)[0]
frac_bulk = 1 - len(keep) / n
gk = [gcv(l, keep) for l in lams]
rec("mp_trunc", lams[int(np.argmin(gk))], keep=keep, extra=dict(edge=edge, kept=int(len(keep)), frac_bulk=float(frac_bulk)))
shr = w.copy(); shr[w <= edge] = float(w[w <= edge].mean()) if (w <= edge).any() else shr[w <= edge]
gs = []
for l in lams:
    h = shr / (shr + l * n); gs.append(((1 - h)[:, None] ** 2 * Yc ** 2).sum() / n / (1 - h.sum() / n) ** 2)
rec("mp_shrink", lams[int(np.argmin(gs))], shrink=shr, extra=dict(edge=edge, bulk_mean=float(shr[w <= edge].mean()) if (w <= edge).any() else None))

tag = args.tag or (D["tag"] + "_rmt" + (f"_n{n}" if args.nfit or args.ntrain else ""))
out = dict(tag=tag, kind="rmt_krr", problem=args.problem, comp=args.comp, band=args.band, seed=args.seed, n=n, scale=float(scale),
           smoke=bool(args.smoke), results={k: {m: 100 * v for m, v in r.items()} for k, r in results.items()}, hyper=hyper,
           spectrum=dict(top=[float(x) for x in w[::-1][:20]], median=medw, edge=edge, frac_bulk=float(frac_bulk),
                         quantiles={str(qq): float(np.quantile(w, qq)) for qq in (0.5, 0.9, 0.99, 0.999)}),
           curves=dict(lams=[float(l) for l in lams], val=[100 * float(e) for e in ev], gcv=[float(gcv(l)) for l in lams],
                       kare=[float(kare(l)) for l in lams]),
           minutes=round((time.time() - t0) / 60, 1))
OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / (tag + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT / (tag + ".json"))
print(f"DONE {tag} in {out['minutes']} min", flush=True)
