"""Scientific plots of saved calibration results and the exact reference law."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, betabinom

HERE=Path(__file__).resolve().parent
BG,INK,DIM='#111A23','#F2EEE7','#AEBBC9'
GOLD,BLUE,GREEN,RED='#E5B85C','#70B7DF','#94C9A9','#EC8C86'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def style(ax):
    ax.set_facecolor(BG);ax.tick_params(colors=DIM,labelsize=15)
    ax.grid(axis='y',alpha=.13,color=DIM)
    for spine in ax.spines.values():spine.set_color(DIM)
    ax.xaxis.label.set_color(INK);ax.yaxis.label.set_color(INK)


def main():
    source=HERE/'assets/uq_seed0.npz'
    data=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if sha(source)!=data['npz_sha256']:raise ValueError('UQ source receipt mismatch')
    figure_rows=[]
    models=('constant','disagreement','power');colors=(BLUE,GOLD,GREEN)
    fig,axs=plt.subplots(2,1,figsize=(10,7.2),layout='constrained',sharex=True)
    fig.set_facecolor(BG)
    values=data['levels']['a0.1']['rows']
    x=np.arange(3)
    for ax in axs:style(ax)
    radii=[values[m]['mean_radius'] for m in models]
    coverage=[100*values[m]['coverage'] for m in models]
    bars=axs[0].bar(x,radii,color=colors,width=.58)
    axs[0].bar_label(bars,labels=[f'{v:.1f}' for v in radii],color=INK,fontsize=18,padding=5)
    axs[0].set_ylim(0,480);axs[0].set_ylabel('Mean radius\n(absolute grid norm)',fontsize=17)
    axs[1].scatter(x,coverage,c=colors,s=100,zorder=3)
    for xx,v in zip(x,coverage):axs[1].text(xx,v+1.3,f'{v:.2f}%',color=INK,fontsize=18,ha='center')
    axs[1].axhline(90,color=DIM,ls='--',lw=1.2)
    axs[1].set_ylim(80,100);axs[1].set_ylabel('Empirical coverage (%)',fontsize=17)
    axs[1].set_xticks(x,('Constant','Disagreement','Power'),fontsize=18)
    axs[1].text(2.4,89.1,'Nominal 90%',color=DIM,fontsize=15,ha='right',va='top')
    path=HERE/'assets/uq_radii.png';fig.savefig(path,dpi=160,facecolor=BG);plt.close(fig)
    figure_rows.append(dict(kind='radii',path=path.name,sha256=sha(path)))
    deciles=data['power_deciles'];x=[d['decile'] for d in deciles]
    fig,axs=plt.subplots(2,1,figsize=(10,7.2),layout='constrained',sharex=True)
    fig.set_facecolor(BG)
    for ax in axs:style(ax)
    axs[0].plot(x,[100*d['coverage'] for d in deciles],'-o',color=GREEN,lw=2.8)
    axs[0].axhline(90,color=DIM,ls='--',lw=1.2)
    axs[0].set_ylim(60,102);axs[0].set_ylabel('Coverage (%)',fontsize=18)
    axs[1].plot(x,[d['mean_radius'] for d in deciles],'-o',color=GOLD,lw=2.8,label='Mean radius')
    axs[1].plot(x,[d['mean_error'] for d in deciles],'-o',color=BLUE,lw=2.8,label='Mean error')
    axs[1].set_ylim(0,650);axs[1].set_ylabel('Absolute grid norm',fontsize=18)
    axs[1].set_xlabel('Power-function decile (1 = smallest scale)',fontsize=18)
    axs[1].set_xticks(x)
    legend=axs[1].legend(loc='upper left',fontsize=16,facecolor=BG,edgecolor=DIM)
    for t in legend.get_texts():t.set_color(INK)
    path=HERE/'assets/uq_deciles.png';fig.savefig(path,dpi=160,facecolor=BG);plt.close(fig)
    figure_rows.append(dict(kind='deciles',path=path.name,sha256=sha(path)))
    # Rebuild the Beta-binomial mass independently from log-gamma arithmetic.
    n,a,b=19000,901,100
    def lb(x,y):return math.lgamma(x)+math.lgamma(y)-math.lgamma(x+y)
    mass=np.array([math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+
                           lb(j+a,n-j+b)-lb(a,b)) for j in range(n+1)])
    total=math.fsum(float(v) for v in mass)
    if abs(total-1)>1e-9:raise ValueError('Independent reference masses do not normalize')
    cdf=np.cumsum(mass)
    independent=[int(np.searchsorted(cdf,q)) for q in (.025,.975)]
    library=[int(betabinom.ppf(q,n,a,b)) for q in (.025,.975)]
    if independent!=library:raise ValueError('Reference interval methods disagree')
    p=np.linspace(.865,.937,361)
    ref=dict(m=1000,alpha=.1,k=a,n_eval=n,beta_parameters=[a,b],
             conditional_p=p.tolist(),conditional_density=beta.pdf(p,a,b).tolist(),
             interval_counts=independent,interval_fractions=[v/n for v in independent],
             independent_mass_sum=total,law='i.i.d. continuous scores, conditional on fixed fitted objects',
             scipy_version=__import__('scipy').__version__)
    (HERE/'assets/coverage_reference.json').write_text(json.dumps(ref,indent=2),encoding='utf-8')
    (HERE/'assets/uq_figures.json').write_text(json.dumps(dict(source_npz_sha256=data['npz_sha256'],
        source_manifest_sha256=sha(source.with_suffix('.json')),matplotlib_version=matplotlib.__version__,
        seed=0,nominal_coverage=.9,rows=figure_rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=figure_rows,reference_interval=ref['interval_fractions'],
                         mass_sum=total),indent=2))


if __name__=='__main__':main()
