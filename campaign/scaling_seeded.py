"""Data-scaling sweep for the raw-input kernels on one OCO-2 band and seed.

Fits the isotropic and ARD Matern rows of jpl_seeded.py at nested training
sizes and reports the two empirical rate exponents. Proposition prop:aniso
predicts the signature: with the input map close to full approximate rank the
two rows must share their exponent and differ by a near-constant factor,
while under a genuine rank gap (the synthetic check in verify_synthetic.py)
the adapted metric must improve the rate itself. Everything is causal per
size: the ARD relevance vector is refit from the first n training rows only,
and the tuning protocol is the exact matern_head grid of jpl_seeded.py with
the subsample capped at n.

Environment: NMKC_ROOT, TASK_ID, NMKC_THREADS, NMKC_JPL_DATA.

    python campaign/scaling_seeded.py --band o2 --seed 3
"""
import argparse, json, os, pathlib, sys, time
import fcntl
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jpl_data
from jpl_data import load_band

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sizes", default="1125,2250,4500,9000,18000")
args = p.parse_args()

ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", "."))
TASK_ID = os.environ.get("TASK_ID", f"scaling_{args.band}_s{args.seed}")
if os.environ.get("NMKC_JPL_DATA"):
    jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])

sp = load_band(args.band, seed=args.seed)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
SIZES = [int(s) for s in args.sizes.split(",") if int(s) <= len(Xtr)]

rel = lambda Pp, T: float(np.mean(np.linalg.norm(Pp - T, axis=1) / np.linalg.norm(T, axis=1)))


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


def matern_head_at(Ztr, Ytr_, Zva, Zte, n_probe, w=None):
    """The matern_head protocol of jpl_seeded.py with the tuning subsample
    capped at the current training size. The diagonal metric w is applied
    after standardization; pre-scaling the inputs cancels against the
    per-dimension std and silently reproduces the isotropic row."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Ftr, Fval, Fte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    if w is not None:
        Ftr, Fval, Fte = Ftr * w, Fval * w, Fte * w
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
            e = rel(Kvs @ cho_solve(c, Ytr_[sub], check_finite=False), Yval)
            if e < best[0]:
                best = (e, (scale, nug))
    scale, nug = best[1]
    n = len(Ftr)
    lk = open(ROOT / ".gram.lock", "w")
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        K = m52(sqd(Ftr, Ftr), scale * med); K.flat[::n + 1] += nug * n
        c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
        alpha = cho_solve(c, Ytr_, check_finite=False)
        del K
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()
    pred = np.empty((len(Fte), Ytr_.shape[1]))
    for k in range(0, len(Fte), 4000):
        pred[k:k + 4000] = m52(sqd(Fte[k:k + 4000], Ftr), scale * med) @ alpha
    return rel(pred, Yte), dict(scale=scale, nugget=nug, med=float(med))


t0 = time.time()
rows = []
for n in SIZES:
    Xn, Yn = Xtr[:n], Ytr[:n]
    e_iso, h_iso = matern_head_at(Xn, Yn, Xval, Xte, n)
    # causal ARD: sensitivity map refit from the first n rows only, in
    # standardized coordinates, applied inside the head after standardization
    Xs_n = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    A_ls, *_ = np.linalg.lstsq(Xs_n, Yn, rcond=None)
    w_ard = np.linalg.norm(A_ls, axis=1)
    w_ard = w_ard / w_ard.mean()
    e_ard, h_ard = matern_head_at(Xn, Yn, Xval, Xte, n, w=w_ard)
    rows.append(dict(n=n, err_iso=e_iso, err_ard=e_ard,
                     gap=e_iso / max(e_ard, 1e-30),
                     hyper_iso=h_iso, hyper_ard=h_ard))
    print(f"n={n}: iso {100*e_iso:.2f}%  ard {100*e_ard:.2f}%  "
          f"gap {rows[-1]['gap']:.2f}  [{(time.time()-t0)/60:.1f} min]", flush=True)


def slope(key):
    ln = np.log([r["n"] for r in rows])
    le = np.log([max(r[key], 1e-30) for r in rows])
    return float(np.polyfit(ln, le, 1)[0])

# anisotropy profile of the full-data sensitivity map, mirroring the
# diagnostic ladder exactly: raw-coordinate least-squares map, effective
# rank = singular values holding 99% of the energy
A_full, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
sig = np.linalg.svd(A_full, compute_uv=False)
eff_rank99 = int(np.searchsorted(np.cumsum(sig ** 2) / (sig ** 2).sum(), 0.99) + 1)
sig = sig / sig[0]
pr = float((sig.sum() ** 2) / (sig ** 2).sum())

s_iso, s_ard = slope("err_iso"), slope("err_ard")
# a ratio is only meaningful when both rows actually decay on this range
ratio = (s_iso / s_ard) if (s_iso < -0.05 and s_ard < -0.05) else None
out = dict(task_id=TASK_ID, kind="oco_scaling", band=args.band, seed=args.seed,
           sizes=SIZES, rows=rows,
           slope_iso=s_iso, slope_ard=s_ard, slope_ratio=ratio,
           gap_min=float(min(r["gap"] for r in rows)),
           gap_max=float(max(r["gap"] for r in rows)),
           d=int(Xtr.shape[1]), sv_profile=[round(float(s), 5) for s in sig],
           eff_rank99=eff_rank99, participation_ratio=pr,
           minutes=round((time.time() - t0) / 60, 1))
res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, res)
print(f"[{TASK_ID}] slopes iso {s_iso:.3f} ard {s_ard:.3f} "
      f"ratio {'%.2f' % ratio if ratio is not None else 'n/a'}  "
      f"gap {out['gap_min']:.2f}..{out['gap_max']:.2f}", flush=True)
