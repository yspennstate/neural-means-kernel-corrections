"""Plot the pinned 6000-row diagnostic; refuse unverified numerical inputs.

python fig_spectrum_diagnostic.py --results campaign/collected/spectrum_20260906
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--figs',type=Path,default=Path(__file__).resolve().parent/'paper/figs')
    a=p.parse_args()
    record=json.loads((a.results/'result.json').read_text())
    assert record['status']=='COMPLETE' and record['full_deployed_smoother'] is False
    assert sha(a.results/'design.npz')==record['design_sha256']
    assert sha(a.results/'spectrum.npz')==record['spectrum_sha256']
    data=np.load(a.results/'spectrum.npz',allow_pickle=False)
    ev=data['eigenvalues'];lam=data['lambda_grid'];deff=data['effective_dimension']
    n=record['n_subsample'];fixed=record['nugget']
    assert n==6000 and len(ev)==n and np.all(np.diff(ev)>=0)
    assert np.all(np.diff(lam)>0) and np.all(np.diff(deff)<=1e-8)
    independent_curve=np.array([sum(float(v/(v+n*l)) for v in ev) for l in lam])
    np.testing.assert_allclose(independent_curve,deff,rtol=1e-12,atol=1e-8)
    spectral=sum(float(v/(v+n*fixed)) for v in ev)
    np.testing.assert_allclose(spectral,record['direct_effective_dimension'],rtol=1e-9,atol=1e-8)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.0))
    shown=np.maximum(ev[::-1]/n,1e-15)
    axes[0].semilogy(np.arange(1,n+1),shown,color='#3b5bdb',lw=1.2)
    axes[0].set(xlabel='eigenvalue index',ylabel=r'eigenvalue of $K_S/6000$',
                title='Matérn spectrum: 6000 selected rows',xlim=(1,n))
    axes[1].semilogx(lam,deff,color='#2b8a3e',lw=1.3)
    axes[1].axvline(fixed,color='#e8590c',ls='--',lw=1)
    axes[1].plot(fixed,spectral,'o',color='#e8590c',ms=4)
    axes[1].annotate(r'fixed $\lambda=10^{-5}$'+f'\n'+r'$d_{\rm eff}=$'+f'{spectral:.1f}',
        xy=(fixed,spectral),xytext=(.55,.7),textcoords='axes fraction',fontsize=8,
        arrowprops=dict(arrowstyle='-',lw=.7,color='#e8590c'))
    axes[1].set(xlabel=r'nugget $\lambda$',ylabel=r'$d_{\rm eff}(\lambda)$',
                title='Trace of the subsample smoother',ylim=(0,None))
    fig.tight_layout()
    a.figs.mkdir(parents=True,exist_ok=True)
    figure=a.figs/'spectra.pdf'
    fig.savefig(figure,metadata={'CreationDate':None,'ModDate':None})
    manifest=dict(kind='new_subsample_spectrum_figure',n_subsample=n,
        fixed_nugget=fixed,median_multiplier=record['median_multiplier'],
        effective_dimension=spectral,direct_effective_dimension=record['direct_effective_dimension'],
        result_sha256=sha(a.results/'result.json'),design_sha256=record['design_sha256'],
        spectrum_sha256=record['spectrum_sha256'],figure_sha256=sha(figure),
        producer_sha256=sha(Path(__file__)),clipped_eigenvalues_for_log_display=int(np.sum(ev/n<1e-15)),
        log_display_floor=1e-15,full_deployed_smoother=False)
    (a.figs/'spectra_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
