"""Ensemble-floor analysis of the scaled MLP pool, with the paper's definitions.

For each seed s and each pool of member configurations, using the validation
split of seed s (canonical_split with NMKC_SPLIT_SEED = s) and the fixed test
block, with normalized residuals r_m(u) = (f_m(u) - y(u)) / ||y(u)||:

  S_mk        = mean_u <r_m(u), r_k(u)>            (validation; the paper forms S there)
  simplex opt = min_{w in simplex} w' S w           (SLSQP, as ensemble_theory.py)
  weak floor  = sqrt(min_{m != k} S_mk)             (theory.tex, after cor:floor)
  finite-M    = e_bar sqrt(rho_bar + (1 - rho_bar)/M), e_bar^2 = min_m S_mm,
                rho_bar = min_{m != k} S_mk / e_bar^2   (cor:floor at uniform weights)
  equicorr.   = e_bar_mean sqrt(rho_mean) with the mean member RMS and the mean
                centered correlation (the paper's descriptive ensFloor)
  realized    = mean relative L2 of the fitted mixture on test, and of equal weights

Block floor (thm:blockfloor) on the test block: 1000 calibration / 19000
evaluation cases (default_rng(0) permutation of the test block, as
campaign/seedarch.py), K seeds of M configurations, entrywise minima of the
evaluation second-moment matrix, bound e sqrt(rho_b + (rho_w - rho_b)/M + (1 - rho_w)/(MK)).

    python ensemble_floor.py --runs runs/structmech --out analysis/out
"""
import argparse, glob, itertools, json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

W = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments")
os.environ.setdefault("NMKC_DATA", str(W / "data" / "structmech"))
sys.path.insert(0, str(W / "code"))
from common import load_arrays, canonical_split, rel_l2  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default=str(W / "runs" / "structmech"))
p.add_argument("--out", default=str(W / "analysis" / "out"))
p.add_argument("--seeds", default="0,1,2,3,4")
a = p.parse_args()
RUNS = Path(a.runs); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
loads, stress = load_arrays()


def simplex_opt(S):
    M = len(S)
    r = minimize(lambda z: z @ S @ z, np.ones(M) / M, jac=lambda z: 2 * S @ z, bounds=[(0.0, 1.0)] * M,
                 constraints={"type": "eq", "fun": lambda z: z.sum() - 1.0}, method="SLSQP",
                 options=dict(maxiter=500, ftol=1e-14))
    w = np.maximum(r.x, 0.0); w /= w.sum()
    assert w @ S @ w <= np.min(np.diag(S)) + 1e-12
    return w


def second_moment(preds, Y):
    nv = np.linalg.norm(Y, axis=1, keepdims=True)
    R = np.stack([(P - Y) / nv for P in preds])
    return np.einsum("mnd,knd->mk", R, R) / R.shape[1]


def floors(S):
    M = len(S); d = np.diag(S)
    off = S[~np.eye(M, dtype=bool)]
    e2 = float(d.min()); rho = float(off.min() / e2)
    w = simplex_opt(S)
    # descriptive equicorrelation reference: mean RMS and mean centered correlation
    corr = S / np.sqrt(np.outer(d, d))
    rho_mean = float(corr[~np.eye(M, dtype=bool)].mean()); e_mean = float(np.sqrt(d).mean())
    return dict(M=M, member_rms=np.sqrt(d).tolist(), simplex_weights=w.tolist(),
                simplex_rms=float(np.sqrt(w @ S @ w)), weak_floor=float(np.sqrt(off.min())),
                finite_M_floor=float(np.sqrt(e2 * (rho + (1 - rho) / M))), rho_min=rho,
                equicorr_floor=float(e_mean * np.sqrt(max(rho_mean, 0.0))), rho_mean=rho_mean,
                uniform_rms=float(np.sqrt(np.ones(M) @ S @ np.ones(M)) / M),
                corr_min=float(corr[~np.eye(M, dtype=bool)].min()), corr_max=float(corr[~np.eye(M, dtype=bool)].max()))


def member_files(seed):
    out = {}
    for f in glob.glob(str(RUNS / f"*_s{seed}_w*_d*.json")):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        n = rec["name"]
        if (RUNS / f"{n}_predva.npy").exists() and (RUNS / f"{n}_predte.npy").exists():
            cfg = n.replace(f"_s{seed}_", "_")
            out[cfg] = n
    return out


POOLS = {
    "paper_pair_w1024": ["mlp_w1024_d4", "mlpMSE_w1024_d4"],
    "widths_metric": ["mlp_w1024_d4", "mlp_w2048_d4", "mlp_w4096_d4"],
    "widths_mse": ["mlpMSE_w1024_d4", "mlpMSE_w2048_d4", "mlpMSE_w4096_d4"],
    "depths_metric": ["mlp_w1024_d4", "mlp_w1024_d5", "mlp_w1024_d6"],
    "depths_mse": ["mlpMSE_w1024_d4", "mlpMSE_w1024_d5", "mlpMSE_w1024_d6"],
    "core_all": ["mlp_w1024_d4", "mlp_w2048_d4", "mlp_w4096_d4", "mlp_w1024_d5", "mlp_w1024_d6",
                 "mlpMSE_w1024_d4", "mlpMSE_w2048_d4", "mlpMSE_w4096_d4", "mlpMSE_w1024_d5", "mlpMSE_w1024_d6"],
    "wide_pair_w4096": ["mlp_w4096_d4", "mlpMSE_w4096_d4"],
}
seeds = [int(s) for s in a.seeds.split(",")]
report = dict(per_seed={}, pools={}, block_floor={})
Yte_full = stress[np.load(Path(os.environ["NMKC_DATA"]) / "idx_test.npy")].reshape(20000, -1).astype(np.float64)
te_perm = np.random.default_rng(0).permutation(20000)
cal, ev = te_perm[:1000], te_perm[1000:]
all_test_preds = {}   # (cfg, seed) -> test predictions on the evaluation block (float32 kept)

for s in seeds:
    os.environ["NMKC_SPLIT_SEED"] = str(s)
    tr, va, te = canonical_split(n_val=1000, seed=s)
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)
    files = member_files(s)
    singles = {}
    cache = {}
    def pred(cfg, split):
        key = (cfg, split)
        if key not in cache:
            cache[key] = np.load(RUNS / f"{files[cfg]}_pred{split}.npy").astype(np.float64)
        return cache[key]
    for cfg in files:
        singles[cfg] = dict(val=rel_l2(pred(cfg, "va"), Yva), test=rel_l2(pred(cfg, "te"), Yte))
        all_test_preds[(cfg, s)] = np.load(RUNS / f"{files[cfg]}_predte.npy")[ev]
    report["per_seed"][s] = dict(members=singles, pools={})
    for pool, cfgs in POOLS.items():
        if not all(c in files for c in cfgs):
            continue
        S_va = second_moment([pred(c, "va") for c in cfgs], Yva)
        fl = floors(S_va)
        w = np.array(fl["simplex_weights"])
        mix_te = sum(wi * pred(c, "te") for wi, c in zip(w, cfgs))
        eq_te = sum(pred(c, "te") for c in cfgs) / len(cfgs)
        fl.update(realized_test_mean=rel_l2(mix_te, Yte), realized_test_rms=float(np.sqrt(np.mean(
            (np.linalg.norm(mix_te - Yte, axis=1) / np.linalg.norm(Yte, axis=1)) ** 2))),
            equal_weights_test_mean=rel_l2(eq_te, Yte), best_member_test=min(singles[c]["test"] for c in cfgs),
            S_val=S_va.tolist(), members=cfgs)
        report["per_seed"][s]["pools"][pool] = fl
        del mix_te, eq_te
    del cache

# pool summaries over seeds
for pool in POOLS:
    rows = [report["per_seed"][s]["pools"][pool] for s in seeds if pool in report["per_seed"][s]["pools"]]
    if not rows:
        continue
    def ms(key):
        v = np.array([r[key] for r in rows]); return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=len(v))
    report["pools"][pool] = {k: ms(k) for k in ("simplex_rms", "weak_floor", "finite_M_floor", "equicorr_floor", "rho_min", "rho_mean",
                                                "realized_test_mean", "realized_test_rms", "equal_weights_test_mean", "best_member_test", "uniform_rms")}
    report["pools"][pool]["members"] = rows[0]["members"]

# block floor over seeds x configurations on the evaluation block
cfgs_all = sorted({c for (c, s) in all_test_preds})
Yev = Yte_full[ev]
nv = np.linalg.norm(Yev, axis=1, keepdims=True)
for pool, cfgs in POOLS.items():
    use = [(c, s) for c in cfgs for s in seeds if (c, s) in all_test_preds]
    if len(use) < 2 or len({c for c, _ in use}) < len(cfgs):
        continue
    R = np.stack([(all_test_preds[k].astype(np.float64) - Yev) / nv for k in use])
    S = np.einsum("mnd,knd->mk", R, R) / R.shape[1]
    del R
    M = len(cfgs); K = len(seeds)
    e2 = float(np.diag(S).min())
    within = [S[i, j] for i, j in itertools.combinations(range(len(use)), 2) if use[i][0] == use[j][0]]
    between = [S[i, j] for i, j in itertools.combinations(range(len(use)), 2) if use[i][0] != use[j][0]]
    rw = float(min(within) / e2) if within else float("nan"); rb = float(min(between) / e2)
    w = simplex_opt(S)
    mix = sum(wi * all_test_preds[k].astype(np.float64) for wi, k in zip(w, use))
    report["block_floor"][pool] = dict(members=len(use), M=M, K=K, e_min=float(np.sqrt(e2)), rho_w_min=rw, rho_b_min=rb,
                                       bound=float(np.sqrt(e2 * (rb + (rw - rb) / M + (1 - rw) / (M * K)))),
                                       hindsight_convex_rms=float(np.sqrt(w @ S @ w)),
                                       hindsight_convex_mean=rel_l2(mix, Yev),
                                       equal_weights_mean=rel_l2(sum(all_test_preds[k].astype(np.float64) for k in use) / len(use), Yev),
                                       ten_seed_style_means={c: rel_l2(sum(all_test_preds[(c, s)].astype(np.float64) for s in seeds if (c, s) in all_test_preds)
                                                                        / sum(1 for s in seeds if (c, s) in all_test_preds), Yev) for c in cfgs},
                                       single_means={c: float(np.mean([rel_l2(all_test_preds[(c, s)].astype(np.float64), Yev) for s in seeds if (c, s) in all_test_preds])) for c in cfgs})
    del mix

(OUT / "ensemble_floor.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
print(json.dumps(report["pools"], indent=1, default=float))
print(json.dumps(report["block_floor"], indent=1, default=float))
