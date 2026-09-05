"""Fixed single-worker follow-ups: ten mismatch probes, nine paired grids.

Shares the centering campaign's active lock and eight-thread ceiling. No
task is appended at runtime. Results are reused only with matching receipts.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--historical-root",required=True,type=Path)
    p.add_argument("--jpl-data",required=True,type=Path)
    a=p.parse_args();root=a.root.resolve();code=root/"code";drivers=root/"diagnostics"
    sys.path.insert(0,str(code))
    from campaign.run_centering_campaign import digest,wait_capacity,write_json
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
        print("FIXED_QUEUE",len(jobs),"jobs; waiting for centering campaign",flush=True)
        with open(root/"active.lock","a",encoding="utf-8") as active:
            fcntl.flock(active,fcntl.LOCK_EX)
            centering=json.loads((root/"status.json").read_text(encoding="utf-8"))
            if centering.get("status")!="COMPLETE":
                raise RuntimeError("Centering campaign has not completed successfully")
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
