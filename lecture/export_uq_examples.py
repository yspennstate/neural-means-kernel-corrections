"""Extract historical seed-0 calibration data with independent error checks.

The retained power values were saved as float32. This extraction preserves
them and reports comparison tolerances; it does not refactor the kernel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--historical-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists():
        raise ValueError('Refuse to replace an existing extraction')
    root=args.historical_root
    runs=root/'seeds/sm_s0/runs'
    paths={'source_npz':runs/'hpix_uq.npz','raw_record':runs/'hpix_uq.json',
           'power_record':runs/'hpix_uq_plam.json','power':runs/'hpix_plam_test.npy',
           'prediction':runs/'hpix_corr_pred_test.npy','stress':root/'data/structmech/stress.npy',
           'indices':root/'data/structmech/idx_test.npy'}
    hashes={k:sha(p) for k,p in paths.items()}
    source=np.load(paths['source_npz'])
    error=np.asarray(source['err'],dtype=np.float64)
    disagreement=np.asarray(source['disagree'],dtype=np.float64)
    cal,ev=source['cal'],source['ev']
    perm=np.random.default_rng(0).permutation(20000)
    if not np.array_equal(cal,perm[:1000]) or not np.array_equal(ev,perm[1000:]):
        raise ValueError('Unexpected calibration/evaluation partition')
    power=np.asarray(np.load(paths['power']),dtype=np.float64)
    for a in (error,disagreement,power):
        if a.shape!=(20000,) or not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError('Invalid saved scores or scales')
    prediction=np.load(paths['prediction'],mmap_mode='r')
    stress=np.load(paths['stress'],mmap_mode='r')
    indices=np.load(paths['indices'])
    if not np.array_equal(indices,np.arange(20000,40000)):
        raise ValueError('Unexpected test ordering')
    recomputed=[]
    for start in range(0,20000,100):
        target=np.asarray(stress[indices[start:start+100]],dtype=np.float64).reshape(-1,1681)
        pred=np.asarray(prediction[start:start+100],dtype=np.float64)
        recomputed.extend(np.sqrt(np.sum((pred-target)**2,axis=1)))
    max_error_difference=float(np.max(np.abs(error-np.asarray(recomputed))))
    if not np.allclose(error,recomputed,rtol=1e-12,atol=1e-9):
        raise ValueError('Saved absolute errors do not reproduce from predictions')
    for row in (0,2517,10000,12320,19999):
        residual=np.asarray(prediction[row],dtype=np.float64)-np.asarray(stress[indices[row]],dtype=np.float64).ravel()
        direct=math.sqrt(math.fsum(float(v)*float(v) for v in residual))
        if abs(direct-error[row])>1e-9:raise ValueError('Scalar error recomputation failed')
    raw_record=json.loads(paths['raw_record'].read_text())
    power_record=json.loads(paths['power_record'].read_text())
    levels={}
    for alpha in (.1,.05):
        key=f'a{alpha:g}';k=math.ceil((1-alpha)*1001)
        rows={}
        for name,scale,record,record_name in (
                ('constant',np.ones(20000),raw_record,'raw'),
                ('disagreement',disagreement,raw_record,'scaled'),
                ('power',power,power_record,'plam')):
            scores=error/scale
            q=float(sorted(float(s) for s in scores[cal])[k-1])
            radius=q*scale[ev]
            coverage=sum(float(error[i])<=q*float(scale[i]) for i in ev)/len(ev)
            mean_radius=math.fsum(float(v) for v in radius)/len(ev)
            old=record[key][record_name]
            q_relative_difference=abs(q-old['q'])/old['q']
            tolerance=2e-6 if name=='power' else 1e-10
            if q_relative_difference>tolerance:raise ValueError('Quantile does not reproduce')
            if abs(coverage-old['coverage'])>1/len(ev)+1e-12:
                raise ValueError('Coverage differs beyond a single float32-boundary case')
            if abs(mean_radius-old['mean_width'])/old['mean_width']>tolerance:
                raise ValueError('Radius does not reproduce')
            rows[name]=dict(q=q,coverage=coverage,mean_radius=mean_radius,
                           recorded_coverage=old['coverage'],recorded_mean_radius=old['mean_width'],
                           q_relative_difference=q_relative_difference)
        levels[key]=dict(rank=k,rows=rows)
    # Freeze deciles by power on evaluation cases, then measure their local coverage.
    order=ev[np.argsort(power[ev],kind='stable')]
    q=levels['a0.1']['rows']['power']['q']
    deciles=[]
    for j,ids in enumerate(np.array_split(order,10),1):
        deciles.append(dict(decile=j,n=len(ids),mean_power=float(np.mean(power[ids])),
                            coverage=float(np.mean(error[ids]<=q*power[ids])),
                            mean_radius=float(np.mean(q*power[ids])),mean_error=float(np.mean(error[ids]))))
    if hashes!={k:sha(p) for k,p in paths.items()}:raise ValueError('Source changed during extraction')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,error=error,disagreement=disagreement,power=power,cal=cal,ev=ev)
    receipt=dict(seed=0,predictor='historical six-member hpix_corr',units='absolute Euclidean output-grid norms',
                 sources={k:dict(path=str(p.relative_to(root)),sha256=hashes[k]) for k,p in paths.items()},
                 source_power_precision='retained float32 values converted to float64; no kernel refactorization',
                 absolute_error_max_recomputation_difference=max_error_difference,levels=levels,
                 power_deciles=deciles,npz_sha256=sha(args.out),extractor_sha256=sha(Path(__file__)))
    args.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(levels=levels,power_deciles=deciles,npz_sha256=receipt['npz_sha256']),indent=2))


if __name__=='__main__':main()
