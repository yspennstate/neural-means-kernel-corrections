"""Scientific plots of actual reconstructed OCO-2 spectra for the lecture."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
BG,INK,DIM='#111A23','#F2EEE7','#AEBBC9'
COLORS={'combined':'#E5B85C','kernel_flow':'#70B7DF'}


def main():
    source=HERE/'assets/oco_o2_examples.npz'
    receipt=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=receipt['npz_sha256']:
        raise ValueError('Source receipt mismatch')
    rows=[]
    with np.load(source) as data:
        for case in ('median','p98'):
            target=data[case+'_target_radiance']
            scale=float(np.max(np.abs(target)))
            if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid display normalization')
            fig,axs=plt.subplots(2,1,figsize=(10,7.3),sharex=True,
                                gridspec_kw={'height_ratios':[1.35,1]},layout='constrained')
            fig.set_facecolor(BG)
            for ax in axs:
                ax.set_facecolor(BG)
                ax.tick_params(colors=DIM,labelsize=17)
                for spine in ax.spines.values():spine.set_color(DIM)
                ax.grid(alpha=.12,color=DIM)
                ax.yaxis.label.set_color(INK)
            channels=np.arange(len(target))
            axs[0].plot(channels,target/scale,color=INK,lw=1.2,label='Reference')
            for model,label in (('combined','Combined'),('kernel_flow','Source KF')):
                prediction=data[case+'_'+model+'_radiance']
                axs[0].plot(channels,prediction/scale,color=COLORS[model],lw=.8,alpha=.85,label=label)
                axs[1].plot(channels,(prediction-target)/scale,color=COLORS[model],lw=.9,label=label)
            axs[0].set_ylabel('Radiance\n/ reference peak',fontsize=16)
            axs[1].set_ylabel('Signed error\n/ reference peak',fontsize=16)
            axs[1].set_xlabel('Stored spectral channel index',fontsize=19,color=INK)
            axs[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
            axs[1].yaxis.get_offset_text().set_color(DIM)
            axs[1].yaxis.get_offset_text().set_fontsize(17)
            legend=axs[0].legend(loc='lower left',fontsize=17,facecolor=BG,edgecolor=DIM,ncol=3)
            for text in legend.get_texts():text.set_color(INK)
            path=HERE/'assets'/f'oco_o2_{case}_spectrum.png'
            fig.savefig(path,dpi=160,facecolor=BG)
            plt.close(fig)
            rows.append(dict(case=case,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             test_row=receipt['cases'][case]['test_row'],display_reference_peak=scale,
                             all_channels_plotted=len(target),
                             radiance_relative_errors=receipt['cases'][case]['radiance_relative_errors']))
    (HERE/'assets/oco_spectrum_figures.json').write_text(json.dumps(dict(
        source_npz_sha256=receipt['npz_sha256'],matplotlib_version=matplotlib.__version__,
        display='Every stored channel is plotted; common reference-peak scaling for each case.',
        rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
