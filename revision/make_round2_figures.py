"""Standalone scientific figures for the second reviewer-directed experiment batch."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'revision_outputs/reviewer_experiment_round2/figures'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    files=[BASE/'revision_outputs/lag_window_sensitivity/coefficient_centroids.csv',
           BASE/'revision_outputs/long_lag_benchmark/summary.csv',
           BASE/'revision_outputs/latitude_contrasts/latitude_contrasts.csv']
    real=pd.read_csv(files[0]);synthetic=pd.read_csv(files[1]);latitude=pd.read_csv(files[2])
    selected=real[real.period.eq('training')]
    summary=selected.groupby(['control_model','max_source_lag']).absolute_coefficient_centroid.median().unstack(0)
    summary.reset_index().to_csv(OUT/'window_centroid_plot_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
    colors=['#235f9b','#b85c16','#457443'];markers=['o','s','^']
    labels={'own3':'Own history 3','own12':'Own history 12','own3_physical_before12':'Own 3 + prior physical fields'}
    for model,color,marker in zip(labels,colors,markers):
        axes[0].plot(summary.index,summary[model],marker=marker,color=color,label=labels[model],lw=1.3)
    for true_lag,color,marker in zip((2,6,10),colors,markers):
        part=synthetic[synthetic.target_history.eq(3)&synthetic.true_lag.eq(true_lag)].sort_values('max_source_lag')
        axes[1].plot(part.max_source_lag,part.median_coefficient_centroid,marker=marker,color=color,
                     label=f'True source lag {true_lag}',lw=1.3)
        missing=part[~part.true_lag_in_window]
        axes[1].scatter(missing.max_source_lag,missing.median_coefficient_centroid,marker=marker,
                        facecolors='white',edgecolors=color,zorder=5,s=35)
    for ax in axes:
        ax.set_xticks([3,6,12]);ax.set_xlim(2,13);ax.set_ylim(0,12)
        ax.set_xlabel('Maximum source lag (6-hour steps)')
        ax.plot([3,6,12],[2,3.5,6.5],color='#555555',ls='--',lw=1,label='Equal-weight reference')
        ax.grid(axis='y',color='#dddddd',lw=.6);ax.legend(loc='upper left',frameon=False)
    axes[0].set_ylabel('Median absolute-coefficient lag centroid')
    axes[0].set_title('(a) ERA5: fixed 36 pairs, training fits')
    axes[1].set_title('(b) Known-lag benchmark: own history 3\n100 repetitions')
    fig.text(.5,.01,'Descriptive centroids, not travel times. Open markers: true lag lies outside fitted window.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(OUT/'source_window_centroids.png',dpi=220);fig.savefig(OUT/'source_window_centroids.pdf');plt.close(fig)

    frame=latitude[latitude.period.eq('evaluation')&latitude.view.eq('NH_minus_SH')&latitude.statistic.eq('linear_trend_per_10deg')]
    kinds=[('wind_to_humidity','Wind → humidity'),('humidity_to_cloud_cover','Humidity → cloud'),('wind_to_cloud_cover','Wind → cloud')]
    frame[frame.edge_type.isin([k[0] for k in kinds])].to_csv(OUT/'latitude_trend_plot_data.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.6),sharex=True,sharey=True)
    for ax,(kind,label) in zip(axes,kinds):
        part=frame[frame.edge_type.eq(kind)].sort_values('lag')
        for bandwidth,shift,color,marker in [(64,-.09,'#235f9b','o'),(128,.09,'#b85c16','s')]:
            x=part.effect.to_numpy();lo=part[f'ci95_low_hac{bandwidth}'].to_numpy();hi=part[f'ci95_high_hac{bandwidth}'].to_numpy()
            ax.errorbar(x,part.lag.to_numpy()+shift,xerr=np.vstack((x-lo,hi-x)),fmt=marker,
                        color=color,ms=4,lw=1,capsize=2,label=f'HAC{bandwidth}')
        ax.axvline(0,color='#555555',lw=.8);ax.set_title(label+'\n98 mirrored pairs');ax.set_yticks([1,2,3]);ax.set_ylim(3.5,.5)
        ax.grid(axis='x',color='#dddddd',lw=.6);ax.ticklabel_format(axis='x',style='plain')
    axes[0].set_ylabel('Source lag (6-hour steps)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    fig.supxlabel('NH−SH difference in signed seasonal-slope trend per 10° absolute latitude',fontsize=9,y=.10)
    fig.text(.5,.03,'2019–2025: DJF 2,528 and JJA 2,576 calendar rows; latitude-band pair counts 32/33/33.\nExploratory HAC intervals; target own-history controls. Three latitude rings; no area weighting.',ha='center',fontsize=7.2)
    fig.tight_layout(rect=(0,.17,1,.93));fig.savefig(OUT/'latitude_trends.png',dpi=220);fig.savefig(OUT/'latitude_trends.pdf');plt.close(fig)
    receipt={'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'figure_scope':'Primary W/H/C relations; all six relation families remain in the full latitude CSV. Centroids are descriptive medians, not inferred travel times.',
             'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.csv','.pdf','.png')}}
    (OUT/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
