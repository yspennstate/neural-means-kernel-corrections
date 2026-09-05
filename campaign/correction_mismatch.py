"""Measure a frozen correction's training/query-mean discrepancy.

Rebuilds the exact archived per-pixel stack, evaluates its refiner and KRR
members on training rows with the full-data KRR channel, and compares the
old and inference-consistent residual fits at the same selected kernel.
No model training, kernel retuning, test-based selection, or eigensolve.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,os.environ.get("NMKC_CODE",str(Path(__file__).resolve().parents[1])))
from common import load_arrays,canonical_split


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def sqdist(a,b):
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)


def m52(d,s):
    r2=d/(s*s);a=np.sqrt(5)*np.sqrt(r2)
    return (1+a+(5/3)*r2)*np.exp(-a)


def median_scale(x,seed):
    sub=np.random.default_rng(seed).choice(len(x),min(2000,len(x)),replace=False)
    return float(np.sqrt(np.median(sqdist(x[sub],x[sub])[np.triu_indices(len(sub),1)])))


def relrows(p,y):return np.linalg.norm(p-y,axis=1)/np.linalg.norm(y,axis=1)


def describe(a):
    return dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


class RefMLP(nn.Module):
    def __init__(self,w,d):
        super().__init__();self.inp=nn.Linear(41+1681,w)
        self.hid=nn.ModuleList([nn.Linear(w,w) for _ in range(d-1)])
        self.out=nn.Linear(w,1681)
    def forward(self,x):
        h=F.silu(self.inp(x))
        for layer in self.hid:h=h+F.silu(layer(h))
        return self.out(h)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--chunk",type=int,default=1000)
    a=p.parse_args();start=time.time()
    if a.out.exists():raise SystemExit("Refusing to overwrite mismatch evidence")
    a.out.mkdir(parents=True)
    torch.set_num_threads(a.threads)
    os.environ["NMKC_SPLIT_SEED"]=str(a.seed)
    loads,stress=load_arrays();tr,va,te=canonical_split(n_val=1000,seed=a.seed)
    y=stress[tr].reshape(len(tr),-1).astype(np.float64)
    if not np.isfinite(y).all() or np.any(np.linalg.norm(y,axis=1)<=0):
        raise ValueError("Invalid training targets")
    x=loads[tr].astype(np.float64);mu=x.mean(0);sd=x.std(0)+1e-12
    xt=(x-mu)/sd;xv=(loads[va].astype(np.float64)-mu)/sd;xe=(loads[te].astype(np.float64)-mu)/sd
    n=len(tr);inputs=[]
    def read(name):
        path=a.runs/name;inputs.append(path)
        return json.loads(path.read_text(encoding="utf-8"))
    def array(name):
        path=a.runs/name;inputs.append(path)
        return np.load(path,mmap_mode="r")
    krec=read("krr_full_matern52_n19000.json")
    s=krec["best"]["smult"]*median_scale(xt,0);t=krec["best"]["lam"]*n
    k=m52(sqdist(xt,xt),s);k.flat[::n+1]+=t
    c=cho_factor(k,lower=True,overwrite_a=True,check_finite=False)
    alpha=cho_solve(c,y-y.mean(0),check_finite=False)
    # At training points, K alpha + mean = Y - n lambda alpha.
    full=y-t*alpha
    control=m52(sqdist(xt[:64],xt),s)@alpha+y.mean(0)
    full_identity_max=float(np.max(np.abs(control-full[:64])))
    kval=m52(sqdist(xv,xt),s)@alpha+y.mean(0)
    old_kval=array("krr_full_matern52_n19000_pred_val.npy")
    krr_val_reproduction=float(np.linalg.norm(kval-old_kval)/np.linalg.norm(old_kval))
    if krr_val_reproduction>1e-5 or full_identity_max>1e-5:
        raise RuntimeError("Full-data KRR reproduction failed")
    full=full.astype(np.float32)
    np.save(a.out/"krr_full_train.npy",full)
    del k,c,alpha,kval
    print("FULL_KRR_REPRODUCED",krr_val_reproduction,full_identity_max,flush=True)

    pixel=read("hpix.json")
    if not pixel["used"]:raise RuntimeError("This diagnostic requires the recorded per-pixel branch")
    names=pixel["members"]
    refname=f"mlpR_s{a.seed}_w1024_d4"
    if refname not in names or "krr" not in names:raise RuntimeError("Unexpected member set")
    cfg=read(refname+".json")["args"]
    net=RefMLP(cfg["width"],cfg["depth"])
    checkpoint=a.runs/(refname+".pt");inputs.append(checkpoint)
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True));net.eval()
    mux=float(loads[tr].mean());sdx=float(loads[tr].std())
    muy=torch.from_numpy(stress[tr].reshape(n,-1).mean(0,keepdims=True)).float()
    sdy=float(stress[tr].reshape(n,-1).std())
    mir=torch.arange(1681).reshape(41,41).flip(0).reshape(-1)
    def predict_ref(fields):
        outputs=[]
        with torch.no_grad():
            for off in range(0,n,a.chunk):
                idx=tr[off:off+a.chunk]
                xx=(torch.from_numpy(loads[idx]).float()-mux)/sdx
                ff=torch.from_numpy(np.array(fields[off:off+a.chunk],dtype=np.float32,copy=True))
                pp=net(torch.cat([xx,(ff-muy)/sdy],1))*sdy+muy
                pm=net(torch.cat([xx.flip(1),(ff[:,mir]-muy)/sdy],1))*sdy+muy
                outputs.append((.5*(pp+pm[:,mir])).numpy())
        return np.concatenate(outputs).astype(np.float32)
    pooled=array("krr_oof_train.npy")
    old_ref=array(refname+"_predtr.npy")
    reproduced_ref=predict_ref(pooled)
    ref_reproduction=float(np.linalg.norm(reproduced_ref-old_ref)/np.linalg.norm(old_ref))
    if ref_reproduction>1e-5:raise RuntimeError("Refiner inference reproduction failed")
    ref_full=predict_ref(full);np.save(a.out/"refiner_full_train.npy",ref_full)
    del reproduced_ref
    print("REFINER_REPRODUCED",ref_reproduction,flush=True)

    pval=[];train=[]
    for name in names:
        if name=="krr":
            pval.append(np.asarray(old_kval,dtype=np.float64));train.append(pooled)
        else:
            pval.append(np.asarray(array(name+"_predva.npy"),dtype=np.float64))
            train.append(array(name+"_predtr.npy"))
    pv=np.stack(pval);nv=pv.shape[1];dim=pv.shape[2]
    design=np.concatenate([pv,np.ones((1,nv,dim))],0)
    gram=np.einsum("mnd,knd->dmk",design,design)/nv
    rhs=np.einsum("mnd,nd->dm",design,stress[va].reshape(nv,-1).astype(np.float64))/nv
    gram+=pixel["ridge"]*np.eye(len(names)+1)[None]
    weights=np.linalg.solve(gram,rhs[...,None])[...,0]
    np.save(a.out/"frozen_pixel_weights.npy",weights)
    old_stack=array("hpix_stack_tr.npy")
    full_stack=np.empty_like(y,dtype=np.float32);stack_repro_max=0.0
    ir=names.index(refname);ik=names.index("krr")
    for off in range(0,n,a.chunk):
        end=min(n,off+a.chunk)
        pred=np.stack([np.asarray(v[off:end],dtype=np.float64) for v in train])
        design=np.concatenate([pred,np.ones((1,end-off,dim))],0)
        rebuilt=np.einsum("dm,mnd->nd",weights,design).astype(np.float32)
        stack_repro_max=max(stack_repro_max,float(np.max(np.abs(rebuilt-old_stack[off:end]))))
        pred[ir]=ref_full[off:end];pred[ik]=full[off:end]
        design=np.concatenate([pred,np.ones((1,end-off,dim))],0)
        full_stack[off:end]=np.einsum("dm,mnd->nd",weights,design).astype(np.float32)
    if stack_repro_max>1e-5:raise RuntimeError("Frozen stack reproduction failed")
    delta=np.asarray(old_stack,dtype=np.float64)-full_stack.astype(np.float64)
    np.save(a.out/"mean_delta_train.npy",delta.astype(np.float32))
    print("STACK_REPRODUCED",stack_repro_max,flush=True)

    crec=read("hpix_corr.json")["report"]
    s=crec["plus_corr"]["smult"]*median_scale(xt,a.seed)
    t=crec["plus_corr"]["lam"]*n
    k=m52(sqdist(xt,xt),s);k.flat[::n+1]+=t
    c=cho_factor(k,lower=True,overwrite_a=True,check_finite=False)
    residual=y-np.asarray(old_stack,dtype=np.float64)
    oldalpha=cho_solve(c,residual,check_finite=False)
    dalpha=cho_solve(c,delta,check_finite=False)
    coords=np.linspace(0,dim-1,8,dtype=int)
    direct=cho_solve(c,(y-full_stack)[:,coords],check_finite=False)
    coefficient_identity_max=float(np.max(np.abs(direct-(oldalpha+dalpha)[:,coords])))
    if coefficient_identity_max>1e-5:raise RuntimeError("Correction RHS identity failed")
    del k,c,direct
    metrics={};saved={}
    for label,xquery,idx,key in (("validation",xv,va,"va"),("test",xe,te,"te")):
        target=stress[idx].reshape(len(idx),-1).astype(np.float64)
        base=np.asarray(array("hpix_stack_"+key+".npy"),dtype=np.float64)
        old=np.empty_like(base);consistent=np.empty_like(base);termnorm=np.empty(len(idx))
        for off in range(0,len(idx),a.chunk):
            end=min(len(idx),off+a.chunk);kq=m52(sqdist(xquery[off:end],xt),s)
            term=kq@dalpha;old[off:end]=base[off:end]+kq@oldalpha
            consistent[off:end]=old[off:end]+term
            termnorm[off:end]=np.linalg.norm(term,axis=1)/np.linalg.norm(target[off:end],axis=1)
        eb=relrows(base,target);eo=relrows(old,target);ef=relrows(consistent,target)
        if not all(np.isfinite(v).all() for v in (eb,eo,ef,termnorm)):
            raise ValueError("Nonfinite correction comparison")
        metrics[label]=dict(base_mean=float(eb.mean()),historical_correction_mean=float(eo.mean()),
                            consistent_correction_mean=float(ef.mean()),propagated_mismatch=describe(termnorm))
        saved[label+"_base"]=eb;saved[label+"_historical"]=eo;saved[label+"_consistent"]=ef
        saved[label+"_propagated_mismatch"]=termnorm
        if abs(float(eo.mean())-crec["plus_corr"]["val" if key=="va" else "test"])>1e-7:
            raise RuntimeError("Recorded correction metric does not reproduce")
    val=metrics["validation"];test=metrics["test"]
    choose_new=val["consistent_correction_mean"]<val["base_mean"]
    metrics["inference_consistent_selected_stage"]="plus_corr" if choose_new else "stack"
    metrics["inference_consistent_selected_test"]=test["consistent_correction_mean"] if choose_new else test["base_mean"]
    metrics["training_mean_mismatch"]=describe(np.linalg.norm(delta,axis=1)/np.linalg.norm(y,axis=1))
    np.savez(a.out/"paired_errors.npz",**saved,validation_indices=va,test_indices=te)
    out=dict(seed=a.seed,threads=a.threads,kernel_frozen=True,weights_frozen=True,
             neural_retraining=False,metrics=metrics,seconds=time.time()-start,
             controls=dict(krr_validation_relative_frobenius=krr_val_reproduction,
                           full_krr_training_identity_max=full_identity_max,
                           refiner_training_relative_frobenius=ref_reproduction,
                           stack_training_max_absolute_error=stack_repro_max,
                           correction_coefficient_identity_max=coefficient_identity_max),
             source_sha256={str(q):sha(q) for q in sorted(set(inputs))},
             driver_sha256=sha(__file__),
             errors_sha256=sha(a.out/"paired_errors.npz"),
             interpretation="Finite observed-query diagnostic on the previously examined benchmark; no supremum certificate")
    (a.out/"summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("MISMATCH_COMPLETE",json.dumps(out),flush=True)


if __name__=="__main__":main()
