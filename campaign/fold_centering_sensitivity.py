"""Paired pooled/fold-local KRR centering, preserving historical predictions.

One factorization per fold supplies both estimators. The rank-one difference
is checked against a separate solve, and the pooled estimator must reproduce
the archived OOF field before a corrected field can be used downstream.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import canonical_split, load_arrays, RUNS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sqdist(a, b):
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)


def m52(d, scale):
    r2 = d/(scale*scale)
    a = np.sqrt(5*r2)
    return (1+a+(5/3)*r2)*np.exp(-a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", required=True, type=Path)
    p.add_argument("--chunk", type=int, default=1000)
    args = p.parse_args()
    if args.chunk < 1:
        p.error("--chunk must be positive")
    target = RUNS/"krr_oof_train.npy"
    receipt = RUNS/"fold_centering.json"
    if target.exists() or receipt.exists():
        raise SystemExit("Refusing to overwrite an existing sensitivity result")
    RUNS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    loads, stress = load_arrays()
    tr, _, _ = canonical_split(n_val=1000, seed=0)
    x = loads[tr].astype(np.float64)
    y = stress[tr].reshape(len(tr), -1).astype(np.float64)
    x = (x-x.mean(0))/(x.std(0)+1e-12)
    hist = np.load(args.historical, mmap_mode="r")
    if hist.shape != y.shape or not np.isfinite(hist).all():
        raise ValueError("Historical OOF shape or values invalid")
    rng = np.random.default_rng(0)  # original KRR folds, independent of outer split
    sub = rng.choice(len(x), 2000, replace=False)
    med = np.sqrt(np.median(sqdist(x[sub],x[sub])[np.triu_indices(2000,1)]))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("Invalid median kernel scale")
    folds = np.array_split(rng.permutation(len(x)),4)
    out = np.empty_like(y, dtype=np.float32)
    mu_pool = y.mean(0)
    records = []
    for fold, hold in enumerate(folds):
        t0 = time.time()
        fit = np.setdiff1d(np.arange(len(x)),hold)
        k = m52(sqdist(x[fit],x[fit]),med)
        k.flat[::len(fit)+1] += 1e-6*len(fit)
        c = cho_factor(k, lower=True, check_finite=False, overwrite_a=True)
        mu_local = y[fit].mean(0)
        al_pool = cho_solve(c,y[fit]-mu_pool,check_finite=False)
        c_one = cho_solve(c,np.ones(len(fit)),check_finite=False)
        # Independent RHS solve of eight prespecified coordinates checks the identity.
        coords = np.linspace(0,y.shape[1]-1,8,dtype=int)
        al_local_control = cho_solve(c,y[fit][:,coords]-mu_local[coords],check_finite=False)
        del k,c
        hist_sq = hist_den = delta_sq = target_den = 0.0
        relmax = identity_max = 0.0
        for off in range(0,len(hold),args.chunk):
            rows = hold[off:off+args.chunk]
            kq = m52(sqdist(x[rows],x[fit]),med)
            old = kq@al_pool+mu_pool
            rowfactor = 1-kq@c_one
            change = rowfactor[:,None]*(mu_local-mu_pool)[None,:]
            new = old+change
            control = kq@al_local_control+mu_local[coords]
            identity_max = max(identity_max,float(np.max(np.abs(control-new[:,coords]))))
            diff = old-np.asarray(hist[rows],dtype=np.float64)
            hist_sq += float(np.square(diff).sum())
            hist_den += float(np.square(hist[rows].astype(np.float64)).sum())
            delta_sq += float(np.square(change).sum())
            target_den += float(np.square(y[rows]).sum())
            relmax = max(relmax,float(np.max(np.linalg.norm(diff,axis=1)/np.linalg.norm(y[rows],axis=1))))
            out[rows] = new.astype(np.float32)
        rec = dict(fold=fold,nfit=len(fit),nhold=len(hold),
                   historical_relative_frobenius=(hist_sq/max(hist_den,1e-300))**0.5,
                   historical_max_sample_relative_error=relmax,
                   centering_relative_frobenius=(delta_sq/max(target_den,1e-300))**0.5,
                   independent_solve_max_absolute_difference=identity_max,
                   seconds=time.time()-t0)
        print(json.dumps(rec),flush=True)
        records.append(rec)
        if rec["historical_relative_frobenius"] > 1e-5 or relmax > 1e-4:
            raise ValueError("Pooled-centering reproduction failed; do not retrain")
        if identity_max > 1e-6*max(1,float(np.max(np.abs(y)))):
            raise ValueError("Rank-one centering identity failed independent solve")
        del al_pool,c_one,al_local_control
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite corrected field")
    with open(str(target)+".tmp","wb") as f:
        np.save(f,out)
    os.replace(str(target)+".tmp",target)
    rec = dict(kind="paired_krr_target_centering",seed=int(os.environ.get("NMKC_SPLIT_SEED","0")),
               ntrain=len(tr),folds=records,scale=float(med),lam=1e-6,
               historical_sha256=sha(args.historical),output_sha256=sha(target),
               train_index_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
               driver_sha256=sha(__file__),seconds=time.time()-start,
               interpretation="Retrospective implementation sensitivity; not a new untouched evaluation")
    receipt.write_text(json.dumps(rec,indent=2)+"\n",encoding="utf-8")
    print("FOLD_CENTERING_COMPLETE",json.dumps(rec),flush=True)


if __name__ == "__main__":
    main()
