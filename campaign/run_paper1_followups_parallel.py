"""Fixed single-worker follow-ups: ten mismatch probes, nine paired grids.

Runs the same frozen nineteen jobs in a separate eight-thread lane after
capacity becomes available. Admission reserves the centering lane even between
its stages. There is one follow-up controller and no runtime queue append.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time



def read_cpu():
    with open("/proc/stat", encoding="utf-8") as f:
        return list(map(int, f.readline().split()[1:9]))


def child_ticks(root):
    status = json.loads((root / "status.json").read_text())
    if status.get("status") == "COMPLETE":
        return {}, False
    pid = int((root / "controller.pid").read_text())
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    if b"run_centering_campaign.py" not in command:
        raise RuntimeError("Centering controller identity is not live")
    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    ticks = {}
    for child in children:
        try:
            # Split after comm, which may contain spaces or parentheses.
            fields = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
            ticks[int(child)] = int(fields[11]) + int(fields[12])
        except FileNotFoundError:
            pass
    return ticks, True


def required_idle(threads, main_core_use, main_pending):
    # One additional idle core is kept as headroom. If the centering process
    # is between stages, reserve its entire eight-thread demand in advance.
    return threads + (max(0.0, 8.0 - main_core_use) if main_pending else 0.0) + 1.0


def wait_capacity(root, threads):
    import shutil
    while True:
        a = read_cpu(); ca, pending = child_ticks(root); start = time.monotonic()
        time.sleep(3)
        b = read_cpu(); cb, pending_after = child_ticks(root)
        seconds = time.monotonic() - start
        delta = [y-x for x,y in zip(a,b)]
        idle = (delta[3]+delta[4]) / max(1,sum(delta)) * os.cpu_count()
        main_use = sum(max(0, cb.get(pid, before)-before) for pid,before in ca.items()) / os.sysconf("SC_CLK_TCK") / seconds
        needed = required_idle(threads, main_use, pending or pending_after)
        with open("/proc/meminfo", encoding="utf-8") as f:
            mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
        free_gib = shutil.disk_usage(root).free / 2**30
        rec = dict(observed_at=time.time(), idle_core_equivalents=idle,
                   main_core_equivalents=main_use, required_idle=needed,
                   available_gib=mem["MemAvailable"]/2**20, free_disk_gib=free_gib,
                   admission=idle >= needed and mem["MemAvailable"]/2**20 >= 32)
        with open(root/"followup_capacity.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec)+"\n")
        if free_gib < 40:
            raise RuntimeError("Campaign volume has less than 40 GiB free")
        if rec["admission"]:
            return
        print("CAPACITY_WAIT", json.dumps(rec), flush=True)
        time.sleep(15)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--historical-root",required=True,type=Path)
    p.add_argument("--jpl-data",required=True,type=Path)
    a=p.parse_args();root=a.root.resolve();code=root/"code";drivers=root/"diagnostics"
    sys.path.insert(0,str(code))
    from campaign.run_centering_campaign import digest,write_json
    env=dict(os.environ,NMKC_CODE=str(code),NMKC_DATA=str(a.historical_root/"data/structmech"),
             NMKC_JPL_DATA=str(a.jpl_data),NMKC_THREADS="8",CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[key]="8"
    jobs=[]
    for seed in range(10):
        out=root/"mismatch"/f"s{seed}"
        jobs.append(dict(id=f"mismatch_s{seed}",kind="mismatch",seed=seed,
                         argv=[sys.executable,"-B",str(drivers/"correction_mismatch.py"),"--seed",str(seed),
                               "--runs",str(a.historical_root/"seeds"/f"sm_s{seed}"/"runs"),
                               "--out",str(out),"--threads","8"],output=str(out/"summary.json")))
    for seed in range(3):
        for band in ("o2","wco2","sco2"):
            out=root/"oco_grid"/"seeds"/f"oco_{band}_s{seed}"/"grid_sensitivity.json"
            jobs.append(dict(id=f"grid_{band}_s{seed}",kind="grid",seed=seed,band=band,
                             argv=[sys.executable,"-B",str(drivers/"jpl_grid_sensitivity.py"),
                                   "--band",band,"--seed",str(seed),"--epochs","250"],output=str(out)))
    plan=dict(maximum_active_jobs=1,threads=8,jobs=jobs,
              driver_sha256={f.name:digest(f) for f in drivers.glob("*.py")},
              jpl_data_sha256={f.name:digest(f) for f in a.jpl_data.glob("*.jld")})
    plan_path=root/"followup_plan.json"
    with open(root/"followup_controller.lock","a",encoding="utf-8") as controller:
        try:fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("A follow-up controller is already active")
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8"))!=plan:
                raise RuntimeError("Follow-up plan or inputs changed")
        else:write_json(plan_path,plan)
        (root/"followup.pid").write_text(str(os.getpid())+"\n",encoding="utf-8")
        print("FIXED_QUEUE",len(jobs),"jobs; separate capacity-reserved lane",flush=True)
        with open(root/"followup_active.lock","a",encoding="utf-8") as active:
            fcntl.flock(active,fcntl.LOCK_EX)
            receipts=root/"followup_receipts";receipts.mkdir(exist_ok=True)
            for job in jobs:
                receipt=receipts/(job["id"]+".json");output=Path(job["output"])
                if receipt.exists():
                    prior=json.loads(receipt.read_text(encoding="utf-8"))
                    if prior["job"]!=job or not output.exists() or prior["output_sha256"]!=digest(output):
                        raise RuntimeError("Completed follow-up drift: "+job["id"])
                    print("VERIFIED_RESUME",job["id"],flush=True);continue
                if output.exists():raise RuntimeError("Unreceipted follow-up result: "+job["id"])
                if any(digest(drivers/f)!=s for f,s in plan["driver_sha256"].items()):
                    raise RuntimeError("Driver source changed after queue freeze")
                wait_capacity(root,8)
                specific=dict(env,NMKC_ROOT=str(root/"oco_grid") if job["kind"]=="grid" else str(root),
                              NMKC_SPLIT_SEED=str(job["seed"]),NMKC_PIPE_SEED=str(job["seed"]))
                t0=time.time()
                write_json(root/"followup_status.json",dict(status="RUNNING",job=job["id"],started_at=t0))
                print("START",job["id"],flush=True)
                try:
                    with open(root/"logs"/(job["id"]+".log"),"w",encoding="utf-8") as log:
                        result=subprocess.run(job["argv"],env=specific,cwd=code,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                except Exception as exc:
                    write_json(root/"followup_status.json",dict(status="FAILED",job=job["id"],error=repr(exc)))
                    raise
                if result.returncode or not output.exists():
                    write_json(root/"followup_status.json",dict(status="FAILED",job=job["id"],returncode=result.returncode))
                    raise RuntimeError("Follow-up failed: "+job["id"])
                write_json(receipt,dict(job=job,output_sha256=digest(output),seconds=time.time()-t0))
                print("DONE",job["id"],round(time.time()-t0,1),flush=True)
            write_json(root/"followup_status.json",dict(status="COMPLETE",jobs=len(jobs),finished_at=time.time()))
            print("FOLLOWUPS_COMPLETE",flush=True)


if __name__=="__main__":main()
