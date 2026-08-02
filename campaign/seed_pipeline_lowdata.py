"""Low-data (1250-sample) pipeline at one campaign seed.

The protocol fixes the data split (first 1250 samples, last 250 of them the
validation set), so the seed varies only the network training; the hybrid
stage's kernel corrections are deterministic given the trained member.

Environment: NMKC_ROOT, NMKC_SEED, TASK_ID, NMKC_THREADS.
"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(os.environ["NMKC_ROOT"])
SEED = int(os.environ["NMKC_SEED"])
TASK_ID = os.environ.get("TASK_ID", f"ld_seed_s{SEED}")
THREADS = int(os.environ.get("NMKC_THREADS", "5"))
CODE = ROOT / "code"
DATA = ROOT / "data" / "structmech"
SEED_DIR = ROOT / "seeds" / f"ld_s{SEED}"
RUNS = SEED_DIR / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PY = sys.executable

ENV = dict(os.environ,
           NMKC_DATA=str(DATA), NMKC_RUNS=str(RUNS), NMKC_SPLIT_SEED="0",
           OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
           MKL_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS))
NAME = f"mlpLD_s{SEED}_w1024_d4_n1000_mir"
EPOCHS = "2" if os.environ.get("NMKC_SMOKE", "") == "1" else "400"


def run(argv, log):
    t0 = time.time()
    print(f"[{TASK_ID}] step {log}", flush=True)
    r = subprocess.run(list(map(str, argv)), env=ENV, cwd=str(CODE))
    if r.returncode != 0:
        raise SystemExit(f"step {log} failed rc={r.returncode}")
    print(f"[{TASK_ID}] step {log} done in {(time.time()-t0)/60:.1f} min", flush=True)


if not (RUNS / (NAME + ".json")).exists():
    run([PY, CODE / "train_mlp.py", "--seed", SEED, "--mirror", 1, "--epochs", EPOCHS,
         "--batch", 64, "--ntrain", 1250, "--lowval", 250, "--tag", "mlpLD",
         "--threads", THREADS], "mlpLD")
if not (RUNS / "hybLD.json").exists():
    run([PY, CODE / "hybrid.py", "--runs", NAME, "--ntrain", 1250, "--lowval", 250,
         "--tag", "hybLD"], "hybLD")

member = json.load(open(RUNS / (NAME + ".json")))
hyb = json.load(open(RUNS / "hybLD.json"))
out = dict(task_id=TASK_ID, kind="lowdata_seed", seed=SEED,
           mlpLD_test_tta=member["test_tta"], mlpLD_test=member["test"],
           final_stage=hyb["report"]["final_stage"],
           final_test=hyb["report"]["final_test"])
res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, res)
print(f"[{TASK_ID}] complete: final {hyb['report']['final_test']:.4f}", flush=True)
