"""Regenerate an explicitly specified Matérn spectrum from recorded load inputs.

This is a new subsample diagnostic, not a recovery of the missing historical
eigenvalue array and not the spectrum of the full deployed smoother.
Run with one BLAS thread and nice >= 12 on the research host.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from threadpoolctl import threadpool_info, threadpool_limits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_sample():
    def read():
        with open('/proc/stat') as f:
            return list(map(int, f.readline().split()[1:9]))
    a=read(); time.sleep(2); d=np.array(read())-a
    return float(100*(1-(d[3]+d[4])/d.sum()))


def squared_distances(x):
    v=np.sum(x*x,axis=1)
    return np.maximum(v[:,None]+v[None,:]-2*x@x.T,0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix' or os.getpriority(os.PRIO_PROCESS,0)<12:
        raise RuntimeError('Requires a low-priority research process')
    a.out.mkdir(exist_ok=False)
    begun=time.time()
    record=dict(kind='new_subsample_spectrum_diagnostic',
        started_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),n_training=19000,n_subsample=6000,
        split_seed=0,subsample_rng_seed=0,median_subset_size=2000,
        median_multiplier=4.,nugget=1e-5,
        kernel='Matern 5/2 on training-standardized recorded loads',
        parameter_status='Fixed diagnostic parameters matching runs/hyb_uq.json; no new selection',
        full_deployed_smoother=False,historical_array_recovered=False,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        stages=[])
    def checkpoint(stage):
        until=time.monotonic()+900
        while True:
            busy=cpu_sample()
            record['status']='RUNNING' if busy<90 else 'WAITING_FOR_HEADROOM'
            record['stages'].append(dict(stage=stage,cpu_busy_percent=busy,
                elapsed_seconds=time.time()-begun,at=datetime.now(timezone.utc).isoformat()))
            (a.out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
            if busy<90:return
            if time.monotonic()>=until:
                record['status']='STOPPED_FOR_PRESSURE'
                (a.out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
                raise RuntimeError('No headroom within 15 minutes before '+stage)
            time.sleep(15)
    checkpoint('input')
    paths={name:a.data/(name+'.npy') for name in ('loads','idx_train')}
    record['input_files']={name:dict(name=p.name,sha256=sha(p)) for name,p in paths.items()}
    loads=np.load(paths['loads'],allow_pickle=False)
    train=np.load(paths['idx_train'],allow_pickle=False)
    assert loads.shape==(40000,41) and train.shape==(20000,)
    assert np.isfinite(loads).all() and len(np.unique(train))==20000
    perm=np.random.default_rng(0).permutation(len(train))
    rows=train[perm[1000:]]
    x=loads[rows].astype(np.float64)
    mean=x.mean(0); scale=x.std(0)+1e-12
    z=(x-mean)/scale
    rng=np.random.default_rng(0)
    median_positions=rng.choice(len(z),2000,replace=False)
    with threadpool_limits(1):
        dmed=squared_distances(z[median_positions])
        median=float(np.sqrt(np.median(dmed[np.triu_indices(2000,1)])))
        del dmed
        positions=rng.choice(len(z),6000,replace=False)
        selected=z[positions]
        record['median_distance']=median
        record['length_scale']=4*median
        inputs=a.out/'design.npz'
        np.savez_compressed(inputs,loads_subsample=x[positions],standardized_loads=selected,
            training_mean=mean,training_scale=scale,training_rows=rows,
            subsample_positions=positions,subsample_rows=rows[positions],
            median_rows=rows[median_positions])
        record['design_sha256']=sha(inputs)
        checkpoint('Gram matrix')
        d2=squared_distances(selected)
        r2=d2/(4*median)**2
        del d2
        radial=np.sqrt(5*r2)
        kernel=(1+radial+(5/3)*r2)*np.exp(-radial)
        del radial,r2
        record['gram_trace']=float(np.trace(kernel))
        record['gram_frobenius_squared']=float(np.einsum('ij,ij->',kernel,kernel))
        checkpoint('eigenvalues')
        ev=eigvalsh(kernel,check_finite=True,driver='evd')
        assert ev[0]>=-1e-8 and np.isfinite(ev).all()
        np.testing.assert_allclose(ev.sum(),record['gram_trace'],rtol=1e-12)
        np.testing.assert_allclose(ev@ev,record['gram_frobenius_squared'],rtol=1e-12)
        lam=np.unique(np.r_[np.logspace(-9,-1,80),1e-5])
        deff=np.array([np.sum(ev/(ev+6000*l)) for l in lam])
        spectral=float(np.sum(ev/(ev+6000*1e-5)))
        checkpoint('independent Cholesky trace')
        system=kernel.copy()
        system.flat[::6001]+=6000*1e-5
        factor=cho_factor(system,lower=True,overwrite_a=True,check_finite=False)
        inverse_trace=0.
        for start in range(0,6000,256):
            stop=min(start+256,6000)
            rhs=np.zeros((6000,stop-start))
            rhs[np.arange(start,stop),np.arange(stop-start)]=1
            solved=cho_solve(factor,rhs,check_finite=False)
            inverse_trace+=float(solved[np.arange(start,stop),np.arange(stop-start)].sum())
        direct=6000-6000*1e-5*inverse_trace
        np.testing.assert_allclose(direct,spectral,rtol=1e-9,atol=1e-8)
        assert np.all(np.diff(deff)<=1e-8) and np.all((deff>=0)&(deff<=6000))
        np.savez(a.out/'spectrum.npz',eigenvalues=ev,lambda_grid=lam,effective_dimension=deff)
        record.update(eigenvalue_min=float(ev[0]),eigenvalue_max=float(ev[-1]),
            spectral_effective_dimension=spectral,direct_effective_dimension=direct,
            trace_check_absolute_difference=abs(direct-spectral),
            modes_for_99_percent_trace=int(np.searchsorted(np.cumsum(ev[::-1]),.99*ev.sum())+1),
            spectrum_sha256=sha(a.out/'spectrum.npz'),thread_pools=threadpool_info(),
            elapsed_seconds=time.time()-begun,finished_at=datetime.now(timezone.utc).isoformat(),
            status='COMPLETE')
        assert all(t['num_threads']<=1 for t in record['thread_pools'])
    (a.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
