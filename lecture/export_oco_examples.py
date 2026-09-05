"""Extract recorded OCO-2 spectra for the lecture without fitting any model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.json').exists():raise ValueError('Output already exists')
    paths=dict(variables=a.data/'dimred_variables_4_mono.jld',
               reconstruction=a.data/'dimred_data_4_mono.jld',
               predictions=a.result/'recorded_grid_predictions.npz',
               errors=a.result/'recorded_grid_errors.npz',
               summary=a.result/'grid_sensitivity.json')
    hashes={k:sha(v) for k,v in paths.items()}
    summary=json.loads(paths['summary'].read_text(encoding='utf-8'))
    if (summary['identity']['band'],summary['identity']['seed']) != ('o2',0):
        raise ValueError('Wrong band or seed')
    with h5py.File(paths['variables'],'r') as h:
        X=h['xr_o2_test'][:].astype(np.float64)
        Y=h['z_o2_test'][:].astype(np.float64)
    with h5py.File(paths['reconstruction'],'r') as h:
        P=h['P_o2'][:].astype(np.float64)
        mean=h['m_o2'][:].astype(np.float64).ravel()
        zmean=h['m_z_o2'][:].astype(np.float64).ravel()
        zscale=h['s_z_o2'][:].astype(np.float64).ravel()
    with np.load(paths['predictions']) as z:
        np.testing.assert_array_equal(Y,z['Yte'])
        predictions={k:z['test_'+k].copy() for k in ('mean_flat','dkr_flat','combined','kernel_flow')}
    with np.load(paths['errors']) as z:
        order=np.argsort(z['combined_radiance'],kind='stable')
        recorded={k:z[k].copy() for k in z.files}
    assert Y.shape==(2000,40) and P.shape[0]==40 and P.shape[1]==len(mean)
    if P.shape[1]>200000:raise ValueError('Unexpected spectrum size')
    payload={};cases={}
    for name,rank in (('median',1000),('p98',1960)):
        i=int(order[rank])
        target=(Y[i]*zscale+zmean)@P+mean
        # Independent scalar reconstruction at fixed channel locations.
        for channel in (0,len(mean)//2,len(mean)-1):
            scalar=math.fsum(float((Y[i,j]*zscale[j]+zmean[j])*P[j,channel]) for j in range(40))+float(mean[channel])
            np.testing.assert_allclose(scalar,target[channel],rtol=1e-12,atol=1e-12)
        payload[name+'_state']=X[i]
        payload[name+'_coefficients']=Y[i]
        payload[name+'_target_radiance']=target
        scores={}
        for model,pred in predictions.items():
            spectrum=(pred[i]*zscale+zmean)@P+mean
            payload[name+'_'+model+'_radiance']=spectrum
            e=math.sqrt(math.fsum(float(v)**2 for v in spectrum-target)/
                        math.fsum(float(v)**2 for v in target))
            np.testing.assert_allclose(e,recorded[model+'_radiance'][i],atol=1e-12,rtol=1e-10)
            scores[model]=e
        cases[name]=dict(test_row=i,zero_based_sorted_error_rank=rank,
                         selection_metric='combined_radiance',radiance_relative_errors=scores,
                         spectral_channels=len(target))
    if hashes != {k:sha(v) for k,v in paths.items()}:raise ValueError('Source drift during extraction')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**payload)
    manifest=dict(kind='source_bound_lecture_examples',band='o2',seed=0,
                  branch='recorded_grid in September5 sensitivity campaign',
                  sources={k:dict(file=v.name,sha256=hashes[k]) for k,v in paths.items()},
                  reconstruction='(z*s_z+m_z)@P+m; original stored channel order',
                  axis='Spectral channel index; no wavelength coordinates inferred',
                  cases=cases,npz_sha256=sha(a.out),extractor_sha256=sha(Path(__file__)),
                  scope='Selected case illustrations, not an aggregate result or fresh evaluation')
    a.out.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=cases,npz_sha256=manifest['npz_sha256']),indent=2))


if __name__=='__main__':main()
