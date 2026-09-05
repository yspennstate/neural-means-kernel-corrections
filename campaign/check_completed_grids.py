"""Recompute completed OCO-2 grid scores directly from saved prediction vectors.

Uses a spectral Gram quadratic form rather than the producer's full-spectrum
norm path, then checks selected cases by scalar full-spectrum summation. It
also rebuilds both coordinate selectors from validation predictions. This can
check an explicitly named completed subset without claiming the campaign done.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--seeds',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refuse to replace an existing check')
    seeds=list(map(int,args.seeds.split(',')))
    if len(seeds)!=len(set(seeds)) or any(s not in range(3) for s in seeds):
        raise ValueError('Expected distinct named seeds in 0,1,2')
    fixed={args.data/'dimred_variables_4_mono.jld',args.data/'dimred_data_4_mono.jld'}
    fixed|={args.data/f'kf_results_{b}_4_mono.jld' for b in ('o2','wco2','sco2')}
    identities={str(p):sha(p) for p in fixed};rows=[]
    for band in ('o2','wco2','sco2'):
        with h5py.File(args.data/'dimred_variables_4_mono.jld','r') as h:
            pool=h[f'z_{band}'][:].astype(np.float64)
            target=h[f'z_{band}_test'][:].astype(np.float64)
        with h5py.File(args.data/'dimred_data_4_mono.jld','r') as h:
            basis=h[f'P_{band}'][:].astype(np.float64)
            mean=h[f'm_{band}'][:].astype(np.float64).ravel()
            zmean=h[f'm_z_{band}'][:].astype(np.float64).ravel()
            zscale=h[f's_z_{band}'][:].astype(np.float64).ravel()
        with h5py.File(args.data/f'kf_results_{band}_4_mono.jld','r') as h:
            kf=h['pred_zs'][:].astype(np.float64)
        if target.shape!=(2000,40) or basis.shape[0]!=40 or basis.shape[1]!=len(mean):
            raise ValueError('Unexpected source dimensions')
        gram=basis@basis.T
        physical=target*zscale+zmean
        denom2=np.einsum('ij,jk,ik->i',physical,gram,physical)+2*physical@(basis@mean)+mean@mean
        red_denom2=np.sum(target*target,axis=1)
        if np.any(denom2<=0) or np.any(red_denom2<=0):raise ValueError('Invalid norm denominator')
        for seed in seeds:
            folder=args.root/'oco_grid/seeds'/f'oco_{band}_s{seed}'
            receipt_path=args.root/'followup_receipts'/f'grid_{band}_s{seed}.json'
            paths=[receipt_path,folder/'grid_sensitivity.json',folder/'experiment_identity.json']
            paths += [folder/(scenario+suffix) for scenario in ('recorded_grid','expanded_grid')
                      for suffix in ('.json','_predictions.npz','_errors.npz')]
            identities.update({str(p):sha(p) for p in paths})
            receipt=read(receipt_path);summary=read(folder/'grid_sensitivity.json')
            if receipt['output_sha256']!=sha(folder/'grid_sensitivity.json'):
                raise ValueError('Unverified completed grid receipt')
            if summary['identity']!=read(folder/'experiment_identity.json'):
                raise ValueError('Experiment identity differs')
            if (summary['identity']['band'],summary['identity']['seed'])!=(band,seed):
                raise ValueError('Wrong grid identity')
            permutation=np.random.default_rng(seed).permutation(20000)
            source_val=pool[permutation[:2000]]
            controls={}
            for scenario in ('recorded_grid','expanded_grid'):
                report=read(folder/(scenario+'.json'))
                if report!=summary['results'][scenario] or report['errors_sha256']!=sha(folder/(scenario+'_errors.npz')):
                    raise ValueError('Grid report and error receipt differ')
                with np.load(folder/(scenario+'_predictions.npz')) as z:
                    if not np.array_equal(z['Yte'],target) or not np.array_equal(z['Yval'],source_val):
                        raise ValueError('Prediction targets differ from source and seeded split')
                    preds={k.removeprefix('test_'):z[k].copy() for k in z.files if k.startswith('test_')}
                    val={k.removeprefix('val_'):z[k].copy() for k in z.files if k.startswith('val_')}
                if not np.array_equal(preds['kernel_flow'],kf):raise ValueError('Source emulator prediction differs')
                names=[prefix+'_'+mode for mode in ('flat','wnum','radx') for prefix in ('mean','dkr')]
                selectors={}
                for label,candidates in (('combined',names),('combined_plus_ridge',names+['ridge_'+m for m in ('flat','wnum','radx')])):
                    winners=[]
                    for j in range(40):
                        losses=[math.fsum(float(v)**2 for v in val[name][:,j]-source_val[:,j])/len(source_val)
                                for name in candidates]
                        winner=candidates[min(range(len(losses)),key=losses.__getitem__)]
                        winners.append(winner)
                        if not np.array_equal(preds[label][:,j],preds[winner][:,j]):
                            raise ValueError('Combined prediction does not use its validation winner')
                    if winners!=report['winners'][label]:raise ValueError('Recorded coordinate winner differs')
                    selectors[label]=winners
                scores={};max_discrepancy=0.
                with np.load(folder/(scenario+'_errors.npz')) as err:
                    for model,pred in preds.items():
                        if pred.shape!=target.shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
                        delta=pred-target;scaled=delta*zscale
                        red=np.sqrt(np.sum(delta*delta,axis=1)/red_denom2)
                        rad=np.sqrt(np.einsum('ij,jk,ik->i',scaled,gram,scaled)/denom2)
                        for metric,values in (('reduced',red),('radiance',rad)):
                            saved=err[model+'_'+metric]
                            difference=float(np.max(np.abs(values-saved)))
                            max_discrepancy=max(max_discrepancy,difference)
                            if not np.allclose(values,saved,rtol=2e-9,atol=2e-12):
                                raise ValueError('Prediction-vector metric differs from reported array')
                            average=math.fsum(float(v) for v in values)/len(values)
                            if not math.isclose(average,report['metrics'][model][metric],rel_tol=2e-9,abs_tol=2e-12):
                                raise ValueError('Recomputed mean differs')
                            scores.setdefault(model,{})[metric]=average
                        for i in (0,999,1999):
                            true_spectrum=physical[i]@basis+mean
                            error_spectrum=scaled[i]@basis
                            direct=math.sqrt(math.fsum(float(v)**2 for v in error_spectrum)/
                                             math.fsum(float(v)**2 for v in true_spectrum))
                            if not math.isclose(direct,float(rad[i]),rel_tol=2e-10,abs_tol=1e-12):
                                raise ValueError('Scalar spectrum metric disagrees with quadratic form')
                        if model.startswith(('mean_','ridge_','kernel_flow')):
                            if scenario=='recorded_grid':controls[model]=pred.copy()
                            elif not np.array_equal(pred,controls[model]):raise ValueError('A frozen predictor changed between grids')
                rows.append(dict(band=band,seed=seed,scenario=scenario,models=len(preds),cases=2000,
                                 max_per_case_difference=max_discrepancy,metrics=scores,selectors=selectors))
                print('CHECKED',band,seed,scenario,'models',len(preds),'max_gap',max_discrepancy,flush=True)
    if identities!={p:sha(Path(p)) for p in identities}:raise ValueError('An input changed during the check')
    result=dict(kind='completed_grid_prediction_recomputation',seeds=seeds,bands=['o2','wco2','sco2'],
                campaign_completion_claim=False,rows=rows,input_sha256=identities,
                driver_sha256=sha(Path(__file__)),
                method='Spectral Gram quadratic form; scalar full-spectrum spot checks; independent validation-coordinate minimization')
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CHECK_COMPLETE',len(rows),'scenario results',flush=True)


if __name__=='__main__':main()
