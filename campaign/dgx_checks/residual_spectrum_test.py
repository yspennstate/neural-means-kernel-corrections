"""Experiment 9 of the five-paper plan: is the shared residual of the structural-mechanics pool noise-like or
structured? A floor detector from the spectrum of the residual matrix.

For each seed of the complete-schedule campaign: the test residual matrices R_m = P_m - Y (20000 x 1681) of
every member, of their mean, and of the corrected six-member pipeline; the singular-value spectrum of each
(after centering the columns) against the Marchenko-Pastur law for the same aspect ratio and the same total
variance; the fraction of the residual energy carried by singular values above the MP edge (the 'structured'
part), the number of such components, and the effective rank. A noise-like residual has all but a handful of
its energy inside the bulk; the shared residual of a saturated pool is expected to be far outside it. Also the
alignment between members' leading residual directions (principal angles), the quantity behind the paper's
correlation floor.

    python residual_spectrum_test.py --seed 0   (ROOT ~/nmkc2; writes results/residual_spectrum_s<seed>.json)
"""
import argparse, glob, json, os, pathlib, sys, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="hpix")
args = p.parse_args()
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2")))
CODE = pathlib.Path(os.environ.get("NMKC_CODE", ROOT / "code"))
os.environ.setdefault("NMKC_DATA", str(ROOT / "data" / "structmech")); os.environ["NMKC_SPLIT_SEED"] = str(args.seed)
sys.path.insert(0, str(CODE))
from common import load_arrays, canonical_split  # noqa: E402

RUNS = ROOT / "seeds" / f"sm_s{args.seed}" / "runs"
loads, stress = load_arrays(); tr, va, te = canonical_split(n_val=1000, seed=args.seed)
Y = stress[te].reshape(len(te), -1).astype(np.float64)
n, q = Y.shape; gamma = q / n
t0 = time.time()


def mp_edge(sigma2, gamma):
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def spectrum_report(R, label):
    Rc = R - R.mean(0)
    s = np.linalg.svd(Rc, compute_uv=False); ev = s ** 2 / n            # eigenvalues of the sample covariance
    total = ev.sum(); sigma2 = total / q                                 # the MP law with the same total variance
    edge = mp_edge(sigma2, gamma)
    above = ev > edge
    # a second, more conservative estimate: sigma2 from the median eigenvalue against the MP median
    from numpy import quantile
    lo = sigma2 * (1 - np.sqrt(gamma)) ** 2
    rep = dict(label=label, n=int(n), q=int(q), gamma=gamma, total_var=float(total), sigma2_meanfit=float(sigma2), mp_edge=float(edge),
               n_above_edge=int(above.sum()), energy_above_edge=float(ev[above].sum() / total), top1_energy=float(ev[0] / total),
               top5_energy=float(ev[:5].sum() / total), eff_rank=float(total ** 2 / (ev ** 2).sum()),
               rel_rms=float(np.sqrt((R ** 2).sum(1)).mean() / np.sqrt((Y ** 2).sum(1)).mean()))
    print(f"  {label:14s} energy above MP edge {rep['energy_above_edge']:.3f} in {rep['n_above_edge']} components; top-1 {rep['top1_energy']:.3f}, top-5 {rep['top5_energy']:.3f}, eff rank {rep['eff_rank']:.1f} [{time.time()-t0:.0f}s]", flush=True)
    return rep, Rc


out = dict(seed=args.seed, members={}, minutes=None)
members = {}
for f in sorted(glob.glob(str(RUNS / "*_predte.npy"))):
    name = os.path.basename(f).replace("_predte.npy", "")
    P = np.load(f).astype(np.float64).reshape(n, -1)
    if P.shape[1] != q: continue
    members[name] = P
krr_f = RUNS / "krr_full_matern52_n19000_pred_test.npy"
if krr_f.exists(): members["krr"] = np.load(krr_f).astype(np.float64).reshape(n, -1)
print(f"seed {args.seed}: {len(members)} members, n={n} q={q} gamma={gamma:.3f}", flush=True)
lead = {}
for name, P in members.items():
    rep, Rc = spectrum_report(P - Y, name); out["members"][name] = rep
    U, s, Vt = np.linalg.svd(Rc, full_matrices=False); lead[name] = Vt[:10]
mean_pred = np.mean(list(members.values()), 0)
out["mean_of_members"], _ = spectrum_report(mean_pred - Y, "mean")
corr_f = RUNS / f"{args.tag}_corr_pred_test.npy"
if corr_f.exists():
    out["corrected_pipeline"], _ = spectrum_report(np.load(corr_f).astype(np.float64).reshape(n, -1) - Y, "corrected")
# the target itself, as the scale: the same statistics of Y (not a residual)
out["target"], _ = spectrum_report(Y, "target Y")
# principal angles between members' leading residual subspaces (10 directions): cos of the first angle
names = list(lead)
ang = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        sv = np.linalg.svd(lead[names[i]] @ lead[names[j]].T, compute_uv=False)
        ang[f"{names[i]}|{names[j]}"] = dict(cos_first=float(sv[0]), mean_cos=float(sv.mean()))
out["leading_subspace_alignment"] = ang
out["minutes"] = round((time.time() - t0) / 60, 1)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / f"residual_spectrum_s{args.seed}.json", "w"), indent=1)
print("wrote", ROOT / "results" / f"residual_spectrum_s{args.seed}.json", flush=True)
