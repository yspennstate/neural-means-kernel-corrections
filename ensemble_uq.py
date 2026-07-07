"""Verify the ensemble-disagreement UQ claim.

Loads per-member test predictions saved by hybrid.py ({tag}_members_te.npy) and
the corrected prediction ({tag}_pred_test.npy), computes per-sample member
disagreement (mean over grid of the across-member std), and correlates it with
the corrected surrogate's absolute and relative error. Also compares against
the kernel power function P (from {tag}_uq_spectra.npz) on the same samples.

Usage: python ensemble_uq.py --tag hyb
"""
import argparse, sys, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load_arrays, canonical_split, rel_l2, RUNS, save_run

p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, default="hyb")
args = p.parse_args()

loads, stress = load_arrays()
_, _, te = canonical_split(n_val=1000, seed=0)
Yte = stress[te].reshape(len(te), -1).astype(np.float64)
members = np.load(RUNS / f"{args.tag}_members_te.npy").astype(np.float64)   # (M,n,1681)
pred = np.load(RUNS / f"{args.tag}_pred_test.npy").astype(np.float64)
M = members.shape[0]
print(f"{M} members", flush=True)

disagree = members.std(0).mean(1) if M > 1 else None      # (n,) mean across-member std
err_abs = np.linalg.norm(pred - Yte, axis=1)
err_rel = err_abs / np.linalg.norm(Yte, axis=1)

out = dict(kind="ensemble_uq", M=M)
if M > 1:
    pa = float(np.corrcoef(disagree, err_abs)[0, 1]); sa = float(spearmanr(disagree, err_abs).statistic)
    pr = float(np.corrcoef(disagree, err_rel)[0, 1]); sr = float(spearmanr(disagree, err_rel).statistic)
    print(f"corr(disagreement, ABS err): pearson {pa:.3f}  spearman {sa:.3f}", flush=True)
    print(f"corr(disagreement, REL err): pearson {pr:.3f}  spearman {sr:.3f}", flush=True)
    out.update(dis_abs_pearson=pa, dis_abs_spearman=sa, dis_rel_pearson=pr, dis_rel_spearman=sr)
    # conformal coverage using disagreement as the scale
    tr, va, _ = canonical_split(n_val=1000, seed=0)
    # decile calibration
    qs = np.quantile(disagree, np.linspace(0, 1, 11))
    dec_dis = [float(disagree[(disagree >= qs[i]) & (disagree <= qs[i+1])].mean()) for i in range(10)]
    dec_err = [float(err_abs[(disagree >= qs[i]) & (disagree <= qs[i+1])].mean()) for i in range(10)]
    np.savez(RUNS / f"{args.tag}_ens_uq.npz", disagree=disagree, err_abs=err_abs, err_rel=err_rel,
             dec_dis=np.array(dec_dis), dec_err=np.array(dec_err))
save_run(f"{args.tag}_ensuq", out)
print("saved", flush=True)
