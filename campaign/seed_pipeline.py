"""One full structural-mechanics pipeline at one campaign seed.

Runs the exact published chain -- KRR, five neural members, prediction dumps,
per-pixel affine stack, residual Matern correction, global convex stack, and
split-conformal UQ -- with the campaign seed threaded through the validation
split (NMKC_SPLIT_SEED), every member's training seed, the stack's half-split
(NMKC_PIPE_SEED) and the correction's tuning subsamples. Every step is skipped
if its outputs already exist, so a killed task resumes where it stopped.

Environment: NMKC_ROOT (campaign root), NMKC_SEED (int), TASK_ID,
NMKC_THREADS (default 6). Writes <root>/results/<TASK_ID>.json at the end.
"""
import contextlib
import json, os, pathlib, subprocess, sys, time
import fcntl
import numpy as np

ROOT = pathlib.Path(os.environ["NMKC_ROOT"])
SEED = int(os.environ["NMKC_SEED"])
TASK_ID = os.environ.get("TASK_ID", f"sm_seed_s{SEED}")
THREADS = int(os.environ.get("NMKC_THREADS", "6"))
SMOKE = os.environ.get("NMKC_SMOKE", "") == "1"
# smoke mode: two epochs everywhere, full data path -- exercises every stage
# The first campaign scheduled the FNO and the UNet for 200 epochs; the second
# campaign (the complete-schedule six-member run) set both to 100 through these
# two variables, so one file reproduces either.
EP = dict(mlp=400, mlpMSE=120, mlpR=100,
          fno=int(os.environ.get("NMKC_EP_FNO", "200")),
          unet=int(os.environ.get("NMKC_EP_UNET", "200"))) if not SMOKE else \
     dict(mlp=2, mlpMSE=2, mlpR=2, fno=2, unet=2)
CODE = ROOT / "code"
DATA = ROOT / "data" / "structmech"
SEED_DIR = ROOT / "seeds" / f"sm_s{SEED}"
RUNS = SEED_DIR / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PY = sys.executable

ENV = dict(os.environ,
           NMKC_DATA=str(DATA), NMKC_RUNS=str(RUNS),
           NMKC_SPLIT_SEED=str(SEED), NMKC_PIPE_SEED=str(SEED),
           OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
           MKL_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS))

NAMES = dict(
    mlp=f"mlp_s{SEED}_w1024_d4_n19000_mir",
    mlpMSE=f"mlpMSE_s{SEED}_w1024_d4_n19000_mir",
    mlpR=f"mlpR_s{SEED}_w1024_d4",
    fno=f"fno_s{SEED}_w64_m14_L4_mir",
    unet=f"unet_s{SEED}_w48_mir",
)
MEMBER_LIST = ",".join(NAMES[k] for k in ("mlp", "mlpMSE", "mlpR", "fno", "unet"))


@contextlib.contextmanager
def gram_lock():
    """At most ONE full-Gram stage per box, ever. Two concurrent 19000^2
    kernel solves OOM-killed a seed on the 31 GB box (rc=-9, 2026-08-02);
    every stage that factorizes a full Gram matrix must hold this lock."""
    lk = open(ROOT / ".gram.lock", "w")
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def prep_data_once():
    """Prepare loads/stress/idx arrays once per box, under a lock."""
    DATA.mkdir(parents=True, exist_ok=True)
    lock = open(DATA / ".prep.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if all((DATA / f).exists() for f in ("loads.npy", "stress.npy", "idx_train.npy", "idx_test.npy")):
            return
        env = dict(ENV, STRUCTMECH_SRC=str(ROOT / "data" / "suite"))
        run([PY, str(CODE / "prep_data.py")], env=env, log="prep")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def run(argv, env=None, log=""):
    t0 = time.time()
    print(f"[{TASK_ID}] step {log}: {' '.join(map(str, argv))}", flush=True)
    r = subprocess.run(list(map(str, argv)), env=env or ENV, cwd=str(CODE))
    if r.returncode != 0:
        raise SystemExit(f"step {log} failed rc={r.returncode}")
    print(f"[{TASK_ID}] step {log} done in {(time.time()-t0)/60:.1f} min", flush=True)


def step(outputs, argv, log):
    """Run argv unless every output already exists in RUNS."""
    if all((RUNS / o).exists() for o in outputs):
        print(f"[{TASK_ID}] step {log}: outputs present, skip", flush=True)
        return
    run(argv, log=log)


def main():
    t_start = time.time()
    prep_data_once()

    # 1. exact KRR member (tuned grid, full 19000 solve) + out-of-fold fields
    with gram_lock():
        step([f"krr_full_matern52_n19000.json", "krr_full_matern52_n19000_pred_val.npy",
              "krr_full_matern52_n19000_pred_test.npy"],
             [PY, CODE / "train_krr.py", "--tag", "krr_full"], "krr")
        step(["krr_oof_train.npy"], [PY, CODE / "krr_oof.py"], "krr_oof")

    # 2. neural members, configurations of the published table
    step([NAMES["mlp"] + ".json"],
         [PY, CODE / "train_mlp.py", "--seed", SEED, "--mirror", 1, "--epochs", EP["mlp"],
          "--threads", THREADS], "mlp")
    step([NAMES["mlpMSE"] + ".json"],
         [PY, CODE / "train_mlp.py", "--seed", SEED, "--mirror", 1, "--epochs", EP["mlpMSE"],
          "--mse", 1, "--tag", "mlpMSE", "--threads", THREADS], "mlpMSE")
    step([NAMES["mlpR"] + ".json"],
         [PY, CODE / "train_mlp_refine.py", "--seed", SEED, "--epochs", EP["mlpR"],
          "--threads", THREADS], "mlpR")
    step([NAMES["fno"] + ".json"],
         [PY, CODE / "train_fno.py", "--seed", SEED, "--mirror", 1, "--epochs", EP["fno"],
          "--tag", "fno", "--threads", THREADS], "fno")
    step([NAMES["unet"] + ".json"],
         [PY, CODE / "train_unet.py", "--seed", SEED, "--mirror", 1, "--epochs", EP["unet"],
          "--tag", "unet", "--threads", THREADS], "unet")

    # 3. reflection-averaged prediction dumps for the stack
    for k, name in NAMES.items():
        step([f"{name}_predte.npy"], [PY, CODE / "gen_preds.py", "--run", name, "--cpu"],
             f"preds_{k}")

    # 4. per-pixel affine stack (honesty protocol) + residual Matern correction
    step(["hpix.json", "hpix_stack_te.npy"],
         [PY, CODE / "stack_perpixel.py", "--members", MEMBER_LIST, "--krr", 1,
          "--tag", "hpix"], "hpix")
    with gram_lock():
        step(["hpix_corr.json", "hpix_corr_pred_test.npy"],
             [PY, CODE / "campaign" / "correct_stack.py", "--tag", "hpix"], "corr")

    # 5. global convex stack + its own correction (the simplex-theorem chain)
    with gram_lock():
        step(["hstk.json"],
             [PY, CODE / "stack_correct.py", "--members", MEMBER_LIST, "--krr", 1,
              "--tag", "hstk"], "hstk")

    # 6. split-conformal UQ with a calibration set never used for any selection
    step(["hpix_uq.json"],
         [PY, CODE / "campaign" / "uq_conformal.py", "--tag", "hpix"], "uq")

    # 7. summary: per-sample errors, paired bootstrap for the correction step
    sys.path.insert(0, str(CODE))
    os.environ.update(NMKC_DATA=str(DATA), NMKC_RUNS=str(RUNS), NMKC_SPLIT_SEED=str(SEED))
    from common import load_arrays, canonical_split, rel_l2
    loads, stress = load_arrays()
    tr, va, te = canonical_split(n_val=1000, seed=SEED)
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)
    S_te = np.load(RUNS / "hpix_stack_te.npy").astype(np.float64)
    C_te = np.load(RUNS / "hpix_corr_pred_test.npy").astype(np.float64)
    den = np.linalg.norm(Yte, axis=1)
    e_stack = np.linalg.norm(S_te - Yte, axis=1) / den
    e_corr = np.linalg.norm(C_te - Yte, axis=1) / den
    np.savez(SEED_DIR / "per_sample_errors.npz", stack=e_stack, corr=e_corr)

    B, n = 2000, len(Yte)
    rng = np.random.default_rng(100 + SEED)
    idx = rng.integers(0, n, size=(B, n))
    bs_stack = e_stack[idx].mean(1)
    bs_corr = e_corr[idx].mean(1)
    bs_delta = bs_corr - bs_stack                      # paired resample
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = dict(task_id=TASK_ID, kind="structmech_seed", seed=SEED,
               members={k: json.load(open(RUNS / (n_ + ".json")))["test"]
                        for k, n_ in NAMES.items()},
               krr=json.load(open(RUNS / "krr_full_matern52_n19000.json")).get("test"),
               hpix=json.load(open(RUNS / "hpix.json")),
               hpix_corr=json.load(open(RUNS / "hpix_corr.json")),
               hstk=json.load(open(RUNS / "hstk.json"))["report"],
               uq=json.load(open(RUNS / "hpix_uq.json")),
               stack_test=float(e_stack.mean()), corr_test=float(e_corr.mean()),
               delta_corr_minus_stack=float(e_corr.mean() - e_stack.mean()),
               boot_ci_stack=ci(bs_stack), boot_ci_corr=ci(bs_corr),
               boot_ci_delta=ci(bs_delta),
               minutes=(time.time() - t_start) / 60)
    res = ROOT / "results" / f"{TASK_ID}.json"
    res.parent.mkdir(exist_ok=True)
    tmp = res.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, res)
    print(f"[{TASK_ID}] complete: stack {e_stack.mean():.4f} corr {e_corr.mean():.4f} "
          f"delta CI {out['boot_ci_delta']}", flush=True)


if __name__ == "__main__":
    main()
