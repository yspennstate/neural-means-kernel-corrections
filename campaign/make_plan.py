"""Generate the nmkc10seed task queue for one research box.

Task JSONs land in <root>/tasks/; the supervisor (nmkc10_supervisor.py) claims
them in lexicographic order, so the id prefix encodes priority. A task is not
recreated if its id already exists in tasks/, active/, done/, failed/ or has a
result in results/.

    python make_plan.py --host box3 [--only sm,oco]  [--root /srv/aiwork/nmkc10seed]
"""
import argparse, json, os, pathlib

p = argparse.ArgumentParser()
p.add_argument("--host", required=True, choices=["box1", "box2", "box3"])
p.add_argument("--root", default="/srv/aiwork/nmkc10seed")
p.add_argument("--only", default="", help="comma filter: sm,oco,climsim,climsim_dl")
args = p.parse_args()

ROOT = pathlib.Path(args.root)
PYBIN = (ROOT / "PYBIN").read_text().strip()
only = set(f for f in args.only.split(",") if f)


def exists(task_id):
    for d in ("tasks", "active", "done", "failed"):
        if (ROOT / d / f"{task_id}.json").exists():
            return True
    return (ROOT / "results" / f"{task_id}.json").exists()


def emit(task_id, argv, env, threads, timeout_hours):
    if only and not any(task_id.split("_", 1)[1].startswith(f) for f in only):
        return
    if exists(task_id):
        print(f"skip {task_id} (exists)")
        return
    env = dict(env, NMKC_ROOT=str(ROOT), TASK_ID=task_id, NMKC_THREADS=str(threads))
    t = dict(task_id=task_id, argv=argv, env=env, threads=threads,
             timeout_hours=timeout_hours)
    f = ROOT / "tasks" / f"{task_id}.json"
    tmp = f.with_suffix(".tmp")
    json.dump(t, open(tmp, "w"), indent=1)
    os.replace(tmp, f)
    print(f"wrote {task_id}")


CODE = str(ROOT / "code")
SM_SEEDS = dict(box1=[0, 1, 2], box2=[3, 4, 5], box3=[6, 7, 8, 9])[args.host]
OCO = dict(box1=[], box2=["o2"], box3=["wco2", "sco2"])[args.host]

for s in SM_SEEDS:
    emit(f"a1_sm_s{s}", [PYBIN, f"{CODE}/campaign/seed_pipeline.py"],
         dict(NMKC_SEED=str(s)), threads=6, timeout_hours=36)

for band in OCO:
    for s in range(10):
        emit(f"a2_oco_{band}_s{s}",
             [PYBIN, f"{CODE}/campaign/jpl_seeded.py", "--band", band, "--seed", str(s)],
             dict(NMKC_JPL_DATA=str(ROOT / "data" / "jpl_oco2")),
             threads=5, timeout_hours=8)

if args.host == "box2":
    emit("a4_krr_multiscale_ld",
         [PYBIN, f"{CODE}/campaign/krr_multiscale_lowdata.py"], {},
         threads=5, timeout_hours=4)
    for s in range(10):
        emit(f"a4_ld_s{s}", [PYBIN, f"{CODE}/campaign/seed_pipeline_lowdata.py"],
             dict(NMKC_SEED=str(s)), threads=5, timeout_hours=6)

if args.host == "box3":
    emit("a0_climsim_dl", ["bash", f"{CODE}/campaign/get_climsim.sh"], {},
         threads=1, timeout_hours=6)
    points = ([(1000, range(10), 4)] + [(10000, range(10), 4)] +
              [(100000, range(10), 6)] + [(300000, range(7), 10)] +
              [(1000000, range(5), 20)] + [(2000000, range(3), 32)] +
              [(3500000, range(3), 48)])
    for n, seeds, hours in points:
        for s in seeds:
            emit(f"a3_climsim_train_n{n:07d}_s{s}",
                 [PYBIN, f"{CODE}/campaign/climsim_seeded.py", "--data", "train",
                  "--n", str(n), "--seed", str(s), "--kernel", "1"],
                 dict(NMKC_CLIMSIM=str(ROOT / "climsim")),
                 threads=6 if n >= 1000000 else 5, timeout_hours=hours)

print("plan generation complete for", args.host)
