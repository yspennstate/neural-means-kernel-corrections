"""Compare plain and trapezoidal grid norms on retained mechanics predictions.

This does not refit a predictor or change any published comparator's score.
The weighted calculation is checked by an independent nested quadrature.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite a metric check")
    root = args.historical_root
    data = root / "data/structmech"
    target = np.load(data/"stress.npy", mmap_mode="r")
    indices = np.load(data/"idx_test.npy")
    if not np.array_equal(indices, np.arange(20000,40000)):
        raise ValueError("Unexpected test ordering")
    edge = np.ones(41); edge[[0,-1]] = .5
    weights = np.outer(edge, edge).reshape(-1)
    rows = []; identities = {"stress.npy": sha(data/"stress.npy"), "idx_test.npy": sha(data/"idx_test.npy")}
    for seed in range(10):
        path = root / "seeds" / f"sm_s{seed}" / "runs/hpix_corr_pred_test.npy"
        pred = np.load(path, mmap_mode="r")
        if pred.shape != (20000,1681):
            raise ValueError("Unexpected prediction shape")
        plain = []; weighted = []; control = 0.0
        for start in range(0,len(indices),500):
            y = np.asarray(target[indices[start:start+500]],dtype=np.float64).reshape(-1,1681)
            estimate = np.asarray(pred[start:start+500],dtype=np.float64)
            delta = estimate-y
            plain.extend(np.sqrt(np.sum(delta**2,axis=1)/np.sum(y**2,axis=1)))
            value = np.sqrt(np.sum(delta**2*weights,axis=1)/np.sum(y**2*weights,axis=1))
            weighted.extend(value)
            # Nested trapezoidal integration, independent of the outer-product
            # implementation. The common grid spacing cancels in the ratio.
            dgrid = delta.reshape(-1,41,41)**2; ygrid = y.reshape(-1,41,41)**2
            def integrate(z):
                along = np.sum((z[:,:,:-1]+z[:,:,1:])/2,axis=2)
                return np.sum((along[:,:-1]+along[:,1:])/2,axis=1)
            direct = np.sqrt(integrate(dgrid)/integrate(ygrid))
            control = max(control,float(np.max(np.abs(direct-value))))
        if control > 1e-12 or not np.isfinite(plain).all() or not np.isfinite(weighted).all():
            raise ValueError("Metric identity failed")
        mean_plain=math.fsum(map(float,plain))/len(plain)
        mean_weighted=math.fsum(map(float,weighted))/len(weighted)
        rows.append(dict(seed=seed,plain=mean_plain,trapezoidal=mean_weighted,
                         trapezoidal_minus_plain=mean_weighted-mean_plain,
                         independent_quadrature_max_error=control))
        identities[str(path.relative_to(root))]=sha(path)
    out=dict(units="fractions",seeds=list(range(10)),rows=rows,
             input_sha256=identities,driver_sha256=sha(Path(__file__)),
             scope="Historical six-member corrected pipeline only; quoted comparators are not rescored")
    args.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out))


if __name__=="__main__":
    main()
