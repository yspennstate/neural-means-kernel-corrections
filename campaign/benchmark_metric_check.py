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
    parser.add_argument("--members", action="store_true", help="Also recompute all six retained member test scores")
    parser.add_argument("--pool", action="store_true", help="Also reconstruct the sixty-member evaluation matrix in bounded float64 blocks")
    args = parser.parse_args()
    if args.pool and not args.members:
        parser.error("--pool requires --members")
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
    members = []; member_paths = {}
    for seed in range(10):
        path = root / "seeds" / f"sm_s{seed}" / "runs/hpix_corr_pred_test.npy"
        identities[str(path.relative_to(root))]=sha(path)
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
        if args.members:
            run = path.parent
            config_path = run / "hpix.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            names = config["members"]
            if len(names) != 6 or len(set(names)) != 6 or "krr" not in names:
                raise ValueError("Unexpected historical six-member pool")
            identities[str(config_path.relative_to(root))] = sha(config_path)
            for name in names:
                member_path = run / ("krr_full_matern52_n19000_pred_test.npy" if name == "krr" else name + "_predte.npy")
                identities[str(member_path.relative_to(root))] = sha(member_path)
                member = np.load(member_path, mmap_mode="r")
                if member.shape != (20000, 1681):
                    raise ValueError("Unexpected member prediction shape")
                values = []; quadrature = []
                for start in range(0, len(indices), 500):
                    y = np.asarray(target[indices[start:start+500]], dtype=np.float64).reshape(-1, 1681)
                    residual = np.asarray(member[start:start+500], dtype=np.float64) - y
                    values.extend(np.sqrt(np.einsum("ij,ij->i", residual, residual) / np.einsum("ij,ij->i", y, y)))
                    quadrature.extend(np.sqrt(np.sum(residual**2 * weights, axis=1) / np.sum(y**2 * weights, axis=1)))
                if not np.isfinite(values).all() or not np.isfinite(quadrature).all():
                    raise ValueError("Nonfinite member metric")
                members.append(dict(seed=seed, member=name, plain=math.fsum(map(float, values))/len(values),
                                    rms=math.sqrt(math.fsum(float(v)**2 for v in values)/len(values)),
                                    trapezoidal=math.fsum(map(float, quadrature))/len(quadrature)))
                architecture = "krr" if name == "krr" else name.split("_", 1)[0]
                key = architecture + f"_s{seed}"
                if key in member_paths:
                    raise ValueError("Duplicate member in pool")
                member_paths[key] = member_path
            print("MEMBERS_CHECKED", seed, flush=True)
    pool = None
    if args.pool:
        names = [a + f"_s{s}" for a in ("mlp", "mlpMSE", "mlpR", "fno", "unet", "krr") for s in range(10)]
        if set(member_paths) != set(names):
            raise ValueError("Pool does not contain exactly the expected sixty members")
        arrays = [np.load(member_paths[name], mmap_mode="r") for name in names]
        perm = np.random.default_rng(20260902).permutation(len(indices))
        evaluation = np.sort(perm[1000:])
        S = np.zeros((60, 60), dtype=np.float64)
        means = np.zeros(60, dtype=np.float64)
        # The original producer flattens the whole float32 residual tensor.
        # This path subtracts and normalizes in float64, streams case blocks,
        # and never holds all predictions or residuals in memory at once.
        for start in range(0, len(evaluation), 200):
            pos = evaluation[start:start+200]
            y = np.asarray(target[indices[pos]], dtype=np.float64).reshape(-1, 1681)
            norms = np.sqrt(np.einsum("ij,ij->i", y, y))
            block = np.stack([(np.asarray(a[pos], dtype=np.float64)-y)/norms[:,None] for a in arrays])
            flat = block.reshape(60, -1)
            S += flat @ flat.T
            means += np.sum(np.sqrt(np.einsum("mij,mij->mi", block, block)), axis=1)
        S /= len(evaluation); means /= len(evaluation)
        pool = dict(names=names, n_fit=1000, n_eval=len(evaluation), S_ev=S.tolist(),
                    member_evaluation_means=dict(zip(names, map(float, means))),
                    evaluation_index_sha256=hashlib.sha256(evaluation.tobytes()).hexdigest(),
                    computation="Float64 residuals; 200-case streamed Gram blocks; original fixed split seed 20260902")
        print("POOL_MATRIX_RECONSTRUCTED", flush=True)
    for name, expected in identities.items():
        path = data/name if name in ("stress.npy", "idx_test.npy") else root/name
        if sha(path) != expected:
            raise RuntimeError("Input changed during metric verification: " + name)
    out=dict(units="fractions",seeds=list(range(10)),rows=rows,
             member_rows=members,pool=pool,
             input_sha256=identities,driver_sha256=sha(Path(__file__)),
             scope="Historical corrected pipeline and, when requested, its six retained members; quoted comparators are not rescored")
    args.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out))


if __name__=="__main__":
    main()
