"""Plot the completed sensitivity campaign for the closing lecture chapter.

The full evidence validator runs first. No partial-campaign plot is produced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'campaign'))
from aggregate_sensitivity import aggregate
from plot_uq_examples import style, BG, INK, DIM, BLUE, GOLD, GREEN


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence',type=Path,required=True)
    args=ap.parse_args()
    data=aggregate(args.evidence)
    rows=[]
    fig,axs=plt.subplots(2,1,figsize=(10,7.2),layout='constrained',sharex=True)
    fig.set_facecolor(BG)
    for ax in axs:style(ax)
    for ax,values,color,label in (
        (axs[0],data['centering']['aggregate']['local_minus_pooled']['per_seed'],BLUE,'Fold-local minus pooled'),
        (axs[1],data['mismatch']['aggregate']['test_consistent_minus_historical']['per_seed'],GOLD,'Consistent minus stored labels')):
        values=100*np.asarray(values)
        ax.axhline(0,color=DIM,lw=1.2)
        ax.vlines(np.arange(10),0,values,color=color,lw=2)
        ax.scatter(np.arange(10),values,color=color,s=55,zorder=3)
        ax.set_title(label,color=INK,fontsize=19)
        ax.set_ylabel('Difference\n(percentage points)',fontsize=16)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
        ax.yaxis.get_offset_text().set_color(DIM)
        ax.yaxis.get_offset_text().set_fontsize(15)
        limit=max(float(np.max(np.abs(values)))*1.25,1e-8)
        ax.set_ylim(-limit,limit)
    axs[1].set_xticks(range(10));axs[1].set_xlabel('Paired training seed',fontsize=18)
    path=HERE/'assets/sensitivity_centering.png'
    fig.savefig(path,dpi=160,facecolor=BG);plt.close(fig)
    rows.append(dict(kind='centering',path=path.name,sha256=sha(path)))
    for metric,title in (('reduced','Reduced-coordinate error'),('radiance','Reconstructed-radiance error')):
        fig,axs=plt.subplots(3,1,figsize=(10,7.2),layout='constrained',sharex=True)
        fig.set_facecolor(BG)
        for ax,band,band_title in zip(axs,('o2','wco2','sco2'),('O2','Weak CO2','Strong CO2')):
            style(ax)
            differences=data['oco_grids'][band]['differences']
            for i,(model,color) in enumerate((('kernel_raw',BLUE),('dkr_flat',GOLD),('combined',GREEN))):
                rec=differences[model+'_'+metric]
                values=100*np.asarray(rec['per_seed'])
                ax.scatter(i+np.array([-.12,0,.12]),values,c=color,s=40,zorder=3)
                ax.plot([i-.24,i+.24],[100*rec['mean']]*2,c=color,lw=3)
            ax.axhline(0,color=DIM,lw=1.1)
            ax.set_ylabel(band_title,fontsize=18)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
            ax.yaxis.get_offset_text().set_color(DIM)
        axs[0].set_title(title+'\nExpanded minus recorded grid (percentage points)',color=INK,fontsize=18)
        axs[2].set_xticks(range(3),('Raw-input kernel','Flat-feature head','Coordinate combination'),fontsize=15)
        axs[2].set_xlabel('Dots: three paired seeds; bars: their means',fontsize=16)
        path=HERE/f'assets/sensitivity_grid_{metric}.png'
        fig.savefig(path,dpi=160,facecolor=BG);plt.close(fig)
        rows.append(dict(kind='grid_'+metric,path=path.name,sha256=sha(path)))
    payload=dict(evidence_manifest_sha256=data['evidence_manifest_sha256'],
                 summary=data,rows=rows,driver_sha256=sha(Path(__file__)),
                 scope='Completed fixed campaign; descriptive paired variation on shared benchmark cases')
    (HERE/'assets/sensitivity_figures.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
