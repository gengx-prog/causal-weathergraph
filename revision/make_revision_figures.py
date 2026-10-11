"""Standalone scientific figures from completed revision CSVs; no invented data."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from revision.prepare_inputs import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'figures';out.mkdir(exist_ok=True);receipts=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    blue='#236192';orange='#C76D25';dark='#333333';purple='#805A93';gray='#777777'
    def finish(fig,name,source_paths,note):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight')
        receipts.append({'figure':name,'sources':[{'path':str(f.relative_to(r)),'sha256':sha256(f)} for f in source_paths],'note':note})
        plt.close(fig)

    source=r/'simulations_post_diagnostic/all_null_summary.csv';df=pd.read_csv(source)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True)
    for i,scenario in enumerate(['ar1_homoskedastic','ar4_heteroskedastic']):
        for j,regime in enumerate(['all','external_calendar']):
            ax=axes[i,j]
            for method,color,marker,label in [('own_history_ols',blue,'o','OLS'),('own_history_hac64',orange,'s','HAC64')]:
                g=df[(df.scenario==scenario)&(df.regime==regime)&(df.method==method)].sort_values('sample_length')
                ax.errorbar(g.sample_length,g.empirical_fdr,yerr=[g.empirical_fdr-g.fdr_mc_ci95_low,g.fdr_mc_ci95_high-g.empirical_fdr],fmt=marker+'-',color=color,capsize=3,markersize=4,label=label,lw=1.1)
            ax.axhline(.05,color=dark,ls='--',lw=.9)
            ax.set_xscale('log',base=2);ax.set_xticks([2048,8192,16384],['2,048','8,192','16,384'])
            ax.set_xlim(1600,22000)
            ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2)
            ax.set_title(('AR1, constant variance' if i==0 else 'AR4, changing variance')+'\n'+('All time points' if j==0 else 'External calendar regime'))
            if j==0:ax.set_ylabel('Empirical FDR')
            if i==1:ax.set_xlabel('Generated time points')
    axes[0,0].legend(frameon=False,loc='upper right')
    fig.suptitle('Null calibration of the complete screening family',y=.995,fontsize=12)
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.text(.07,.012,'95% Monte Carlo intervals; 200 replicates at N=2,048, 100 at larger N.\nDashed: nominal 5%. Larger-N runs were added after the first diagnostic.',fontsize=8)
    finish(fig,'01_null_calibration',[source],'Only regression-null calibration rows, not outcome-selected stress tests. Intervals are Wilson intervals supplied by the simulation runner. Larger N is post-diagnostic.')

    source=r/'simulations/graph_summary.csv';df=pd.read_csv(source)
    scenarios=['weak_ar','strong_ar','observed_driver','hidden_driver']
    # Read scenario names from completed evidence; fail if the expected design changes.
    if not set(scenarios).issubset(set(df.scenario)):
        scenarios=['weak_ar','strong_ar','observed_common_driver','hidden_common_driver']
    if not set(scenarios).issubset(set(df.scenario)):
        scenarios=[s for s in df.scenario.unique() if 'switch' not in s]
    labels={scenarios[0]:'Weak persistence',scenarios[1]:'Strong persistence',scenarios[2]:'Observed common driver',scenarios[3]:'Hidden common driver'}
    methods=[('own_history_ols','Own-history OLS',gray,'o'),('own_history_hac64','Own-history HAC64',orange,'s'),('all_observed_history_ols','Multivariable OLS',blue,'D'),('pcmci_parcorr','PCMCI',purple,'^')]
    fig,ax=plt.subplots(figsize=(7.2,4.5));base=np.arange(len(scenarios))
    for k,(method,label,color,marker) in enumerate(methods):
        vals=[];errors=[]
        for scenario in scenarios:
            g=df[(df.scenario==scenario)&(df.method==method)]
            assert len(g)==1
            vals.append(float(g.mean_f1.iloc[0]));errors.append(1.96*float(g.sd_f1.iloc[0])/np.sqrt(float(g.repetitions.iloc[0])))
        ax.errorbar(vals,base+(k-1.5)*.15,xerr=errors,fmt=marker,color=color,capsize=2,markersize=5,label=label)
    ax.set_yticks(base,[labels[s] for s in scenarios]);ax.invert_yaxis();ax.set_xlim(0,1.04);ax.set_xlabel('Direct graph recovery F1');ax.grid(axis='x',alpha=.2)
    ax.set_title('Known-structure recovery under a common candidate universe')
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.62,.075),fontsize=8)
    fig.tight_layout(rect=(0,.18,1,1));fig.text(.05,.008,'50 replicates per scenario; bars are 95% Monte Carlo intervals.\nPredictive-screen edges and direct structural edges are different estimands.',fontsize=8)
    finish(fig,'02_known_graph_recovery',[source],'F1 requires exact directed source-target-lag matching. Score against the known direct graph; do not reinterpret these structural errors as regression-null p-value errors.')

    source=r/'full_var/own_history_vs_full_var_summary.csv';df=pd.read_csv(source)
    types=['wind_to_humidity','humidity_to_cloud_cover','wind_to_cloud_cover'];g=df.set_index('edge_type').loc[types]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True)
    for ax,period,title in zip(axes,['train','test'],['Discovery 1979–2018','Evaluation 2019–2025']):
        x=np.arange(3)
        for shift,model,label,color in [(-.18,'ownhistory','Target history',orange),(.18,'fullvar','All-region VAR',blue)]:
            counts=g[f'{period}_q05_{model}'].values
            ax.bar(x+shift,counts/g.n.values,width=.34,label=label,color=color)
            for xx,yy,count in zip(x+shift,counts/g.n.values,counts):ax.text(xx,yy+.015,str(count),ha='center',fontsize=8)
        ax.set_xticks(x,['W→H','H→C','W→C']);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Fraction passing exploratory HAC64/BH');axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1));fig.text(.08,.009,'594 candidate-lag tests per relation. Numbers above bars are counts.\nThese are sensitivity results; simulation does not establish universal 5% FDR control.',fontsize=8)
    finish(fig,'03_conditioning_sensitivity',[source],'Global BH across 2574 original candidate-lag tests within each period. Both models use lag3; fullVAR controls all264 regional variable histories.')

    source=r/'graph_nulls/candidate_null_enrichment.csv';df=pd.read_csv(source)
    fig,ax=plt.subplots(figsize=(7.2,3.3))
    g=df[(df.correction=='q_hac64_global')&(df.edge_type=='ALL')&(df.metric=='whc_lag_resolved_chains')].set_index('family').loc[['legacy858','symmetric_core']]
    y=np.arange(2)
    ax.errorbar(g.null_mean,y,xerr=[g.null_mean-g.null_q025,g.null_q975-g.null_mean],fmt='o',color=gray,capsize=5,label='Repeated candidate null, 95% range')
    ax.scatter(g.observed,y,marker='D',color=blue,s=45,label='Observed candidate graph',zorder=3)
    ax.set_yticks(y,['Original directional candidates','Symmetric spatial candidates']);ax.invert_yaxis();ax.set_xlabel('Lag-resolved WHC chain count');ax.set_title('Path counts relative to 100 distance-stratified candidate draws');ax.grid(axis='x',alpha=.2)
    ax.set_ylim(1.35,-.35)
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.62,.10))
    fig.tight_layout(rect=(0,.26,1,1));fig.text(.04,.008,'Null draws preserve variable-type and distance-bin counts, not node degrees.\nRanges show null variation, not confidence intervals for a causal effect.',fontsize=8)
    finish(fig,'04_path_count_null',[source],'Independent candidate randomization/refitting, distinct from degree-preserving topology MCMC whose mixing is inadequate. HAC64 remains exploratory.')
    (out/'figure_sources.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(f'Saved {len(receipts)} figures as PDF, SVG and PNG.')


if __name__=='__main__':main()
