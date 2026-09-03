"""Dump the arrays behind the seeded UQ figure (test-block calibration path, one seed), after review.

Reads the seed's deployed six-member correction (hpix) prediction on the test block and its P_lambda, recomputes
the split-conformal calibration exactly as uq_conformal_plam.py does (1000 calibration rows drawn from the test
block with the seed, evaluation on the other 19000), and writes the evaluation-half arrays plus decile summaries
of P_lambda against ABSOLUTE and relative error and per-decile coverage.

    ~/nmkc_venv/bin/python uq_fig_dump.py [seed]
"""
import json, math, os, pathlib, sys
import numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROOT = pathlib.Path(os.path.expanduser("~/nmkc2"))
os.environ.setdefault("NMKC_DATA", str(ROOT / "data" / "structmech"))
os.environ["NMKC_SPLIT_SEED"] = str(seed)
sys.path.insert(0, str(ROOT / "code"))
from common import load_arrays, canonical_split  # noqa: E402

loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=seed)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
RUNS = ROOT / "seeds" / f"sm_s{seed}" / "runs"
P = np.load(RUNS / "hpix_plam_test.npy").astype(np.float64)
pred = np.load(RUNS / "hpix_corr_pred_test.npy").astype(np.float64)
err = np.linalg.norm(pred - Yte, axis=1)
rel = err / np.linalg.norm(Yte, axis=1)
perm = np.random.default_rng(seed).permutation(len(te))
cal, ev = perm[:1000], perm[1000:]
k = math.ceil(0.9 * (len(cal) + 1))
q = float(np.sort(err[cal] / P[cal])[k - 1])
cover = float(np.mean(err[ev] <= q * P[ev]))
order = np.argsort(P[ev])
bins = np.array_split(order, 10)
dec = dict(dec_P=[float(P[ev][b].mean()) for b in bins], dec_abs=[float(err[ev][b].mean()) for b in bins],
           dec_rel=[float(rel[ev][b].mean()) for b in bins], dec_cov=[float(np.mean(err[ev][b] <= q * P[ev][b])) for b in bins])
rank = lambda x: np.argsort(np.argsort(x))
outp = ROOT / "results" / f"uq_fig_seeded_s{seed}.npz"
np.savez_compressed(outp, P=P[ev].astype(np.float32), err_abs=err[ev].astype(np.float32), err_rel=rel[ev].astype(np.float32),
                    q=q, cover90=cover, seed=seed, n_cal=len(cal), n_eval=len(ev),
                    pearson_abs=float(np.corrcoef(P[ev], err[ev])[0, 1]), pearson_rel=float(np.corrcoef(P[ev], rel[ev])[0, 1]),
                    spearman_abs=float(np.corrcoef(rank(P[ev]), rank(err[ev]))[0, 1]),
                    spearman_rel=float(np.corrcoef(rank(P[ev]), rank(rel[ev]))[0, 1]), **dec)
print(json.dumps(dict(seed=seed, q=q, cover90=cover, **dec)))
print("wrote", str(outp), flush=True)
