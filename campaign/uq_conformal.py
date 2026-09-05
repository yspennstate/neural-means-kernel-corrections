"""Retrospective split-conformal measurements on the public test block.

The published pipeline calibrated conformal scales on the model-selection
validation split, which the exchangeability argument does not cover. Here the
20000-sample test set is split at random (seeded) into 1000 calibration and
19000 evaluation samples. Within a run these rows are separate from training,
checkpointing, stacking, and kernel tuning. Repeated study-level access to
the public block is not covered by that separation. A fresh-population
guarantee additionally requires a fixed predictor and scale independent of
exchangeable calibration and evaluation observations.

Scores: per-sample L2 error of the corrected surrogate, either raw or scaled
by the ensemble disagreement (mean across-member std). Quantile: the exact
ceil((1-alpha)(m+1))-th smallest calibration score.

Writes {tag}_uq.json and {tag}_uq.npz under NMKC_RUNS.
"""
import argparse, json, math, os, sys
import numpy as np
from conformal_utils import conformal_quantile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from common import load_arrays, canonical_split, RUNS, save_run

p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, default="hpix")
p.add_argument("--members", type=str, default="")
p.add_argument("--ncal", type=int, default=1000)
p.add_argument("--alphas", type=str, default="0.1,0.05")
args = p.parse_args()

SEED = int(os.environ.get("NMKC_SPLIT_SEED", "0"))

loads, stress = load_arrays()
_, _, te = canonical_split(n_val=1000, seed=SEED)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
pred = np.load(RUNS / f"{args.tag}_corr_pred_test.npy").astype(np.float64)

if args.members:
    names = [m.strip() for m in args.members.split(",")]
else:
    names = json.load(open(RUNS / f"{args.tag}.json"))["members"]
mem = []
for m in names:
    f = RUNS / (f"{m}_predte.npy" if m != "krr" else "krr_full_matern52_n19000_pred_test.npy")
    mem.append(np.load(f).astype(np.float64))
mem = np.stack(mem)                                   # (M, n, D)
disagree = mem.std(0).mean(1)                         # (n,)
err = np.linalg.norm(pred - Yte, axis=1)              # (n,) absolute L2 error

n = len(Yte)
if not 0 < args.ncal < n:
    raise ValueError("Calibration and evaluation sets must both be nonempty")
if not np.isfinite(disagree).all() or np.any(disagree <= 0):
    raise ValueError("Disagreement scales must be finite and strictly positive")
rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
cal, ev = perm[:args.ncal], perm[args.ncal:]

out = dict(kind="split_conformal", tag=args.tag, seed=SEED,
           n_cal=len(cal), n_eval=len(ev), members=names)
for alpha in [float(a) for a in args.alphas.split(",")]:
    key = f"a{alpha:g}"
    # scaled scores: err / disagreement
    q_s = conformal_quantile(err[cal] / disagree[cal], alpha)
    cov_s = float(np.mean(err[ev] <= q_s * disagree[ev]))
    # raw scores: constant-width bands
    q_r = conformal_quantile(err[cal], alpha)
    cov_r = float(np.mean(err[ev] <= q_r))
    out[key] = dict(target=1 - alpha,
                    scaled=dict(q=float(q_s), coverage=cov_s,
                                mean_width=float(q_s * disagree[ev].mean())),
                    raw=dict(q=float(q_r), coverage=cov_r, mean_width=float(q_r)))
    print(f"alpha={alpha}: scaled cover {cov_s:.4f} (target {1-alpha})  "
          f"raw cover {cov_r:.4f}", flush=True)

np.savez(RUNS / f"{args.tag}_uq.npz", err=err, disagree=disagree, cal=cal, ev=ev)
save_run(f"{args.tag}_uq", out)
print("saved", flush=True)
