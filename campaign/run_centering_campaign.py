"""Finite, single-worker paired refiner-centering campaign on retained data.

Runs both pooled and fold-local arms at identical CPU/thread settings. Every
completed stage is pinned by its input and output hashes. A live flock admits
one stage for this campaign; a second controller exits. Existing campaigns
and their files are read-only inputs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def sample_cpu():
    with open("/proc/stat",encoding="utf-8") as f:
        return list(map(int,f.readline().split()[1:9]))


def wait_capacity(root,threads):
    while True:
        a=sample_cpu();time.sleep(3);b=sample_cpu()
        d=[y-x for x,y in zip(a,b)]
        idle=(d[3]+d[4])/max(1,sum(d))*os.cpu_count()
        with open("/proc/meminfo",encoding="utf-8") as f:
            mem={r.split(":")[0]:int(r.split()[1]) for r in f}
        free_gib=shutil.disk_usage(root).free/2**30
        rec=dict(observed_at=time.time(),idle_core_equivalents=idle,
                 available_gib=mem["MemAvailable"]/2**20,free_disk_gib=free_gib)
        with open(root/"capacity.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps(rec)+"\n")
        if free_gib < 40:
            raise RuntimeError("Campaign volume has less than 40 GiB free")
        if idle >= threads and rec["available_gib"]>=32:
            return
        print("CAPACITY_WAIT",json.dumps(rec),flush=True)
        time.sleep(15)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--historical-root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--seeds",default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    seeds=[int(s) for s in args.seeds.split(",")]
    if len(seeds)!=len(set(seeds)) or not seeds or args.threads<1 or args.threads>8:
        p.error("Unique seeds and a thread cap of 1..8 are required")
    root=args.root.resolve();code=root/"code"
    if root==args.historical_root.resolve() or root in args.historical_root.resolve().parents:
        p.error("Historical input and output roots must be disjoint")
    root.mkdir(parents=True,exist_ok=True)
    with open(root/"controller.lock","a",encoding="utf-8") as controller:
        try:fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("A controller already owns this campaign")
        with open(root/"active.lock","a",encoding="utf-8") as active:
            print("Waiting for any initial fold job to release the single active slot",flush=True)
            fcntl.flock(active,fcntl.LOCK_EX)
            (root/"controller.pid").write_text(str(os.getpid())+"\n",encoding="utf-8")
            manifest=dict(seeds=seeds,threads=args.threads,arms=["pooled","local"],
                          created_at=time.time(),maximum_active_jobs=1,
                          historical_root=str(args.historical_root),data=str(args.data),
                          data_sha256={p.name:digest(p) for p in sorted(args.data.glob("*.npy"))},
                          code_sha256={str(p.relative_to(code)):digest(p) for p in sorted(code.rglob("*.py"))},
                          retrospective=True)
            mf=root/"campaign_manifest.json"
            if mf.exists():
                prior=json.loads(mf.read_text(encoding="utf-8"))
                for key in ("seeds","threads","arms","historical_root","data","data_sha256","code_sha256"):
                    if prior[key]!=manifest[key]:raise RuntimeError("Resume manifest mismatch: "+key)
            else:write_json(mf,manifest)
            base_env=dict(os.environ,NMKC_ROOT=str(root),NMKC_DATA=str(args.data),
                          NMKC_CODE=str(code),NMKC_THREADS=str(args.threads),CUDA_VISIBLE_DEVICES="",
                          PYTHONDONTWRITEBYTECODE="1")
            for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                base_env[key]=str(args.threads)
            receipts=root/"receipts";receipts.mkdir(exist_ok=True)
            (root/"logs").mkdir(exist_ok=True)

            def step(label,argv,runs,seed,inputs,outputs,extra_outputs=()):
                receipt=receipts/(label+".json")
                deps={str(q):digest(q) for q in inputs}
                command=list(map(str,argv))
                if receipt.exists():
                    old=json.loads(receipt.read_text(encoding="utf-8"))
                    if old["argv"]!=command or old["inputs"]!=deps:
                        raise RuntimeError("Stage input drift: "+label)
                    if any(not q.exists() or digest(q)!=old["outputs"].get(str(q)) for q in outputs):
                        raise RuntimeError("Stage output drift: "+label)
                    if any(not Path(q).exists() or digest(q)!=s for q,s in old["outputs"].items()):
                        raise RuntimeError("Additional stage output drift: "+label)
                    print("VERIFIED_RESUME",label,flush=True);return
                if any(q.exists() for q in outputs):
                    raise RuntimeError("Unreceipted partial stage; inspect before resuming: "+label)
                wait_capacity(root,args.threads)
                env=dict(base_env,NMKC_RUNS=str(runs),NMKC_SPLIT_SEED=str(seed),NMKC_PIPE_SEED=str(seed))
                t0=time.time()
                write_json(root/"status.json",dict(stage=label,status="RUNNING",started_at=t0,argv=command))
                print("START",label,flush=True)
                with open(root/"logs"/(label+".log"),"w",encoding="utf-8") as log:
                    result=subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    write_json(root/"status.json",dict(stage=label,status="FAILED",returncode=result.returncode))
                    raise RuntimeError("Stage failed: "+label)
                if any(not q.exists() for q in outputs):
                    raise RuntimeError("Stage did not create required outputs: "+label)
                produced=list(outputs)+[q for q in extra_outputs if q.exists()]
                write_json(receipt,dict(argv=command,inputs=deps,
                           outputs={str(q):digest(q) for q in produced},seconds=time.time()-t0))
                print("DONE",label,round(time.time()-t0,1),flush=True)

            sys.path.insert(0,str(code))
            os.environ["NMKC_DATA"]=str(args.data)
            from common import load_arrays,canonical_split
            _,stress=load_arrays()
            for seed in seeds:
                historical=args.historical_root/"seeds"/f"sm_s{seed}"/"runs"
                seedroot=root/"seeds"/f"sm_s{seed}";local=seedroot/"runs"
                local.mkdir(parents=True,exist_ok=True)
                field=local/"krr_oof_train.npy";field_rec=local/"fold_centering.json"
                if field.exists() or field_rec.exists():
                    if not field.exists() or not field_rec.exists():raise RuntimeError("Partial initial fold job")
                    rec=json.loads(field_rec.read_text(encoding="utf-8"))
                    if rec["seed"]!=seed or rec["output_sha256"]!=digest(field) or rec["historical_sha256"]!=digest(historical/"krr_oof_train.npy") or rec["driver_sha256"]!=digest(code/"campaign/fold_centering_sensitivity.py"):
                        raise RuntimeError("Initial fold receipt does not match exact inputs")
                else:
                    step(f"s{seed}_fold",[sys.executable,"-B",code/"campaign/fold_centering_sensitivity.py","--historical",historical/"krr_oof_train.npy"],local,seed,[historical/"krr_oof_train.npy"],[field,field_rec])
                members=[f"mlp_s{seed}_w1024_d4_n19000_mir",f"mlpMSE_s{seed}_w1024_d4_n19000_mir",f"mlpR_s{seed}_w1024_d4",f"fno_s{seed}_w64_m14_L4_mir",f"unet_s{seed}_w48_mir"]
                for arm in ("pooled","local"):
                    runs=local if arm=="local" else seedroot/"pooled"
                    runs.mkdir(exist_ok=True)
                    reusable=[]
                    for name in members:
                        if name.startswith("mlpR_"):continue
                        reusable.extend(historical/(name+suf) for suf in (".json",".pt","_predtr.npy","_predva.npy","_predte.npy"))
                    reusable.extend(historical/f for f in ("krr_full_matern52_n19000.json","krr_full_matern52_n19000_pred_val.npy","krr_full_matern52_n19000_pred_test.npy"))
                    if arm=="pooled":reusable.append(historical/"krr_oof_train.npy")
                    for q in reusable:
                        if not q.is_file():raise FileNotFoundError(q)
                        dest=runs/q.name
                        if dest.is_symlink():
                            if dest.resolve()!=q.resolve():raise RuntimeError("Unexpected symlink: "+str(dest))
                        elif dest.exists():raise RuntimeError("Refusing to replace existing artifact: "+str(dest))
                        else:dest.symlink_to(q)
                    name=members[2]
                    kfields=[runs/f for f in ("krr_oof_train.npy","krr_full_matern52_n19000_pred_val.npy","krr_full_matern52_n19000_pred_test.npy")]
                    step(f"s{seed}_{arm}_refiner",[sys.executable,"-B",code/"train_mlp_refine.py","--seed",seed,"--epochs",100,"--threads",args.threads,"--tag","mlpR"],runs,seed,kfields,[runs/(name+".json"),runs/(name+".pt")])
                    preds=[runs/(name+f"_pred{s}.npy") for s in ("tr","va","te")]
                    step(f"s{seed}_{arm}_preds",[sys.executable,"-B",code/"gen_preds.py","--run",name,"--cpu"],runs,seed,[runs/(name+".json"),runs/(name+".pt"),runs/"krr_oof_train.npy"],preds)
                    memstr=",".join(members)
                    stack_inputs=[runs/(m+f"_pred{s}.npy") for m in members for s in ("tr","va","te")]+kfields
                    step(f"s{seed}_{arm}_pixel",[sys.executable,"-B",code/"stack_perpixel.py","--members",memstr,"--krr",1,"--tag","hpix"],runs,seed,stack_inputs,[runs/"hpix.json"],[runs/f"hpix_stack_{s}.npy" for s in ("tr","va","te")])
                    pixel=json.loads((runs/"hpix.json").read_text(encoding="utf-8"))
                    if pixel["used"]:
                        step(f"s{seed}_{arm}_correct",[sys.executable,"-B",code/"campaign/correct_stack.py","--tag","hpix"],runs,seed,[runs/f"hpix_stack_{s}.npy" for s in ("tr","va","te")],[runs/"hpix_corr.json",runs/"hpix_corr_pred_test.npy",runs/"hpix_corr_pred_val.npy"])
                    else:
                        step(f"s{seed}_{arm}_global",[sys.executable,"-B",code/"stack_correct.py","--members",memstr,"--krr",1,"--tag","hstk"],runs,seed,stack_inputs,[runs/"hstk.json",runs/"hstk_pred_test.npy"]+[runs/f"hstk_stack_{s}.npy" for s in ("tr","va","te")])
                        # Honest fallback, recorded explicitly and used consistently by UQ.
                        fallback=runs/"hpix_corr_pred_test.npy"
                        if not fallback.exists():fallback.symlink_to(runs/"hstk_pred_test.npy")
                    step(f"s{seed}_{arm}_uq",[sys.executable,"-B",code/"campaign/uq_conformal.py","--tag","hpix","--members",memstr+",krr"],runs,seed,[runs/"hpix_corr_pred_test.npy"],[runs/"hpix_uq.json",runs/"hpix_uq.npz"])
                os.environ["NMKC_SPLIT_SEED"]=str(seed)
                _,_,te=canonical_split(n_val=1000,seed=seed)
                y=stress[te].reshape(len(te),-1).astype(np.float64);den=np.linalg.norm(y,axis=1)
                metrics={};errs={}
                for arm,runs in (("pooled",seedroot/"pooled"),("local",local),("historical",historical)):
                    path=runs/"hpix_corr_pred_test.npy"
                    if not path.exists():path=runs/"hstk_pred_test.npy"
                    pred=np.load(path).astype(np.float64)
                    err=np.linalg.norm(pred-y,axis=1)/den
                    errs[arm]=err;metrics[arm]=dict(mean_relative_l2=float(err.mean()),rms_relative_l2=float(np.sqrt(np.mean(err*err))))
                delta=errs["local"]-errs["pooled"]
                np.savez(seedroot/"paired_errors.npz",**errs,local_minus_pooled=delta,test_indices=te)
                write_json(seedroot/"summary.json",dict(seed=seed,metrics=metrics,
                           local_minus_pooled_mean=float(delta.mean()),local_better_fraction=float(np.mean(delta<0)),
                           comparison="paired same host, 8-thread CPU training, fixed seed and schedule",
                           error_archive_sha256=digest(seedroot/"paired_errors.npz")))
                print("SEED_COMPLETE",seed,json.dumps(metrics),flush=True)
            write_json(root/"status.json",dict(status="COMPLETE",seeds=seeds,finished_at=time.time()))
            print("CAMPAIGN_COMPLETE",flush=True)


if __name__=="__main__":main()
