"""Package the structural-mechanics test and validation residuals for release.

The campaign stores each member's prediction arrays in float32, 134 MB per member
and split on the test block; the network checkpoints are larger still. What the
paper's residual and selection analyses actually consume is the residual
f_m - y, and that is what this script exports: one compressed .npz per seed,
holding for every member of the published five-member configuration (and for
the reported pipeline, per-pixel stack plus kernel correction) the residual on
the 20000-sample test block and on that seed's 1000-sample validation split, in
float16, together with the per-sample target norms and the split indices. With
the target norms in the file, every per-sample relative error, the residual
second-moment matrix, the pairwise residual correlations, the equicorrelated
floor, the simplex weights and the conformal scores of the raw pipeline can be
recomputed without the dataset; diagnostics that need the prediction itself
(the per-pixel affine refit, the kernel posterior scale) need the distributed
dataset as well, since f_m = y + r_m.

Per array the manifest records shape, dtype, the sha256 of the array's C-order
little-endian bytes, the largest residual magnitude, and three relative errors:
from the stored float32 prediction, from the float16 residual, and the value the
campaign's own stage record carries (ens5_s*.json, pix5_s*.json). The distance
between the last two is the cost of the float16 rounding and is reported in the
paper.

Runs on the campaign tree: NMKC_ROOT points at it (seeds/sm_s*/runs, data/
structmech, code/common.py), NMKC_RELEASE_OUT at the output directory.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed/nmkc10seed")
OUT = os.environ.get("NMKC_RELEASE_OUT", os.path.join(ROOT, "release_residuals"))
SEEDS = [int(s) for s in os.environ.get("NMKC_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
N_VAL = 1000
MEMBERS = ["mlp", "mlpMSE", "mlpR", "fno", "krr"]
PIPELINE = "pix5"           # per-pixel stack of the five members plus kernel correction

sys.path.insert(0, os.path.join(ROOT, "code"))
os.environ.setdefault("NMKC_DATA", os.path.join(ROOT, "data", "structmech"))
from common import load_arrays, canonical_split, rel_l2  # noqa: E402

os.makedirs(OUT, exist_ok=True)


def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pred_path(rundir, member, split):
    """Locate a member's stored prediction for split in {'val','test'}."""
    if member == "krr":
        g = glob.glob(os.path.join(rundir, "krr_*_pred_%s.npy" % split))
    elif member == PIPELINE:
        g = glob.glob(os.path.join(rundir, "%s_corr_pred_%s.npy" % (PIPELINE, split)))
    else:
        suffix = "va" if split == "val" else "te"
        g = [p for p in glob.glob(os.path.join(rundir, "*_pred%s.npy" % suffix))
             if os.path.basename(p).split("_")[0] == member]
    if len(g) != 1:
        return None
    return g[0]


def recorded_error(seed, member, split):
    """The campaign's own stage record for this member and split."""
    if member == PIPELINE:
        if split != "test":
            return None
        p = os.path.join(ROOT, "pix5_s%d.json" % seed)
        with open(p) as fh:
            return float(json.load(fh)["final_test"])
    p = os.path.join(ROOT, "ens5_s%d.json" % seed)
    with open(p) as fh:
        rec = json.load(fh)
    return float(rec["per_member_%s" % split][member])


t_start = time.time()
loads, stress = load_arrays()
stress = stress.reshape(len(stress), -1)          # (40000, 1681) float32
manifest = dict(
    schema="nmkc-residual-release-v1",
    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    source_tree=ROOT,
    numpy=np.__version__,
    residual_definition="r = f - y, prediction minus target, on the 41x41 grid "
                        "flattened in C order (1681 values per sample); "
                        "predictions are the reflection-averaged arrays the "
                        "campaign scored",
    dtype="float16, little-endian ('<f2'); hashes are sha256 of the C-order bytes",
    test_block="samples 20000..39999 of the distributed file, fixed for every seed",
    val_split="n_val=1000 drawn from samples 0..19999 by common.canonical_split "
              "with NMKC_SPLIT_SEED equal to the seed; idx_val in each file",
    members=MEMBERS,
    pipeline=PIPELINE,
    files=[],
)
worst_round = 0.0
for seed in SEEDS:
    t_seed = time.time()
    rundir = os.path.join(ROOT, "seeds", "sm_s%d" % seed, "runs")
    os.environ["NMKC_SPLIT_SEED"] = str(seed)
    tr, va, te = canonical_split(n_val=N_VAL, seed=seed)
    Yva = stress[va].astype(np.float64)
    Yte = stress[te].astype(np.float64)
    nva = np.linalg.norm(Yva, axis=1)
    nte = np.linalg.norm(Yte, axis=1)
    payload = dict(
        idx_val=va.astype(np.int32), idx_test=te.astype(np.int32),
        norm_val=nva.astype(np.float32), norm_test=nte.astype(np.float32),
    )
    arrays = []
    missing = []
    for member in MEMBERS + [PIPELINE]:
        for split in ("val", "test"):
            if member == PIPELINE and split == "val":
                continue                    # the pipeline writes its test field only
            src = pred_path(rundir, member, split)
            if src is None:
                missing.append((member, split))
                continue
            P = np.load(src).astype(np.float64)
            Y = Yva if split == "val" else Yte
            nrm = nva if split == "val" else nte
            R = P - Y
            R16 = R.astype(np.float16)
            assert np.isfinite(R16).all(), (seed, member, split, "non-finite after float16 cast")
            e32 = rel_l2(P, Y)
            e16 = float(np.mean(np.linalg.norm(R16.astype(np.float64), axis=1) / nrm))
            rec = recorded_error(seed, member, split)
            key = "res_%s_%s" % (split, member)
            payload[key] = R16
            entry = dict(
                seed=seed, split_seed=seed, member=member, split=split, key=key,
                shape=list(R16.shape), dtype=str(R16.dtype),
                sha256=sha256_bytes(np.ascontiguousarray(R16).tobytes()),
                max_abs_residual=float(np.abs(R).max()),
                rel_l2_float32=float(e32), rel_l2_float16=e16,
                rel_l2_recorded=rec,
                float16_minus_recorded=(None if rec is None else float(e16 - rec)),
                float32_minus_recorded=(None if rec is None else float(e32 - rec)),
                source=os.path.basename(src), source_sha256=sha256_file(src),
            )
            if rec is not None:
                worst_round = max(worst_round, abs(e16 - rec))
            arrays.append(entry)
            print("s%d %-6s %-4s e32 %.6f e16 %.6f rec %s  max|r| %.1f" % (
                seed, member, split, e32, e16,
                "n/a" if rec is None else "%.6f" % rec, entry["max_abs_residual"]), flush=True)
            del P, R, R16
    if missing:
        print("s%d missing %s" % (seed, missing), flush=True)
    payload["meta"] = np.array(json.dumps(dict(
        schema=manifest["schema"], seed=seed, split_seed=seed, n_val=N_VAL,
        members=MEMBERS, pipeline=PIPELINE,
        keys=sorted(k for k in payload if k.startswith("res_")))))
    fname = "sm_residuals_s%d.npz" % seed
    fpath = os.path.join(OUT, fname)
    np.savez_compressed(fpath, **payload)
    size = os.path.getsize(fpath)
    manifest["files"].append(dict(
        file=fname, seed=seed, split_seed=seed, n_val=N_VAL,
        idx_val_sha256=sha256_bytes(payload["idx_val"].tobytes()),
        bytes=size, sha256=sha256_file(fpath), missing=missing, arrays=arrays))
    print("wrote %s  %.1f MB  in %.0fs" % (fname, size / 1e6, time.time() - t_seed), flush=True)

manifest["worst_float16_rounding_abs"] = worst_round
manifest["total_bytes"] = int(sum(f["bytes"] for f in manifest["files"]))
mpath = os.path.join(OUT, "residuals_manifest.json")
with open(mpath, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=1)
print("manifest %s  %d files  %.2f GB  worst float16 rounding %.2e  total %.0fs" % (
    mpath, len(manifest["files"]), manifest["total_bytes"] / 1e9, worst_round,
    time.time() - t_start), flush=True)
