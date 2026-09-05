"""Export small, source-bound images of recorded mechanics examples.

This is a deterministic data extraction, not a new training run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix(".json").exists():
        raise ValueError("Output exists; verify it rather than overwrite.")
    root = a.historical_root
    paths = {"loads": root / "data/structmech/loads.npy",
             "stress": root / "data/structmech/stress.npy",
             "indices": root / "data/structmech/idx_test.npy",
             "prediction": root / "seeds/sm_s0/runs/hpix_corr_pred_test.npy"}
    hashes = {k: sha(v) for k, v in paths.items()}
    arrays = {k: np.load(v, mmap_mode="r") for k, v in paths.items()}
    indices = arrays["indices"]
    if not np.array_equal(indices, np.arange(20000, 40000)):
        raise ValueError("Unexpected test ordering")
    errors = []
    for start in range(0, 20000, 100):
        y = np.asarray(arrays["stress"][indices[start:start+100]], dtype=np.float64).reshape(-1, 1681)
        pred = np.asarray(arrays["prediction"][start:start+100], dtype=np.float64)
        errors.extend(np.sqrt(np.sum((pred-y)**2, axis=1)/np.sum(y*y, axis=1)))
    order = np.argsort(np.asarray(errors), kind="stable")
    ranks = {"median": 10000, "p98": 19600, "first": None}
    payload = {}; cases = {}
    for name, rank in ranks.items():
        row = int(order[rank]) if rank is not None else 0
        index = int(indices[row])
        target = np.asarray(arrays["stress"][index], dtype=np.float64).reshape(41, 41)
        pred = np.asarray(arrays["prediction"][row], dtype=np.float64).reshape(41, 41)
        load = np.asarray(arrays["loads"][index], dtype=np.float64)
        if load.shape != (41,):
            raise ValueError(f"Unexpected load shape {load.shape}")
        payload.update({f"{name}_load": load, f"{name}_target": target,
                        f"{name}_prediction": pred, f"{name}_error": pred-target})
        # Independent scalar summation for each exported case.
        import math
        direct = math.sqrt(math.fsum(float(z)**2 for z in (pred-target).ravel()) /
                           math.fsum(float(z)**2 for z in target.ravel()))
        if abs(direct-errors[row]) > 1e-12:
            raise ValueError("Independent metric recomputation failed")
        cases[name] = {"test_row": row, "dataset_index": index,
                       "zero_based_sorted_error_rank": rank, "relative_error": direct}
    if hashes != {k: sha(v) for k, v in paths.items()}:
        raise ValueError("Source changed during extraction")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **payload)
    manifest = {"source_root": "/home/yitz/nmkc2", "sources": {
        k: {"path": str(v.relative_to(root)), "sha256": hashes[k]} for k, v in paths.items()},
        "seed": 0, "predictor": "historical six-member hpix_corr",
        "metric": "unweighted relative Euclidean grid norm",
        "orientation": "Stored axes are x1,x2; display array.T with origin=lower.",
        "selection": "Error-ranked examples for explanation; not a fresh evaluation.",
        "cases": cases, "npz_sha256": sha(a.out),
        "extractor_sha256": sha(Path(__file__))}
    a.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(a.out), "cases": cases, "sha256": manifest["npz_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
