"""Static, exportable figures from the derived CSVs (no model execution).

Run with system python3, which already provides matplotlib.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results'/'discovery';F=Path(__file__).resolve().parent/'figures';F.mkdir(exist_ok=True)
NAMES={'xlm-roberta-large':'XLM-R-large','bert-base-multilingual-cased':'mBERT','setu4993/LaBSE':'LaBSE',
       'google/muril-large-cased':'MuRIL-large','csebuetnlp/banglabert':'BanglaBERT','csebuetnlp/banglishbert':'BanglishBERT'}
ORDER=list(NAMES)
COLORS={'MLR':'#0072B2','LEACE':'#D55E00','COV':'#009E73','PCA':'#CC79A7','RAND':'#7A7A7A','AMBIENT':'#222222'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,
                     'savefig.dpi':220,'pdf.fonttype':42,'svg.fonttype':'none'})


def load(n):return pd.read_csv(D/n,low_memory=False)
def save(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight')
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    p=load('primary_confirmatory.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.3),sharey=True)
    for ax,concept,title in zip(axs,['F','S'],['F: machine figurative-gloss classes','S: supplied sentiment']):
        for i,m in enumerate(ORDER):
            r=p[(p.model==m)&(p.concept==concept)].iloc[0]
            color='#0072B2' if r.both_nulls_reject else '#666666'
            ax.errorbar(r.SSI,i,xerr=[[r.SSI-r.ci_lo],[r.ci_hi-r.SSI]],fmt='o',capsize=3,
                        color=color,mfc=color if r.both_nulls_reject else 'white',ms=7)
        ax.axvline(0,color='black',lw=.8);ax.axvline(1,color='#999999',lw=.7,ls=':')
        ax.set_title(title,fontsize=11);ax.set_xlabel('SSI with 95% component-bootstrap interval')
        ax.set_xlim(-.22,1.02);ax.grid(axis='x',alpha=.18)
    axs[0].set_yticks(range(6),[NAMES[m] for m in ORDER]);axs[0].invert_yaxis()
    fig.suptitle('Primary cross-lingual sharing at validation-selected layers',fontsize=13)
    fig.text(.5,-.035,'Filled: rejects both RAND and PERM after across-model Holm correction. One split/taxonomy refit.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'01_primary_ssi')

    fig,ax=plt.subplots(figsize=(10.5,4.1));x=np.arange(6)
    q=p[p.concept.eq('F')].set_index('model').loc[ORDER]
    for offset,col,label,color in [(-.24,'R_ep','English MLR','#0072B2'),(0,'PCA_R_mean','English PCA','#CC79A7'),(.24,'RAND_R_mean','English RAND mean','#888888')]:
        ax.bar(x+offset,100*(q[col]-q.R0),width=.23,label=label,color=color)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[NAMES[m] for m in ORDER],rotation=15,ha='right')
    ax.set_ylabel('Bangla F accuracy above majority floor (pp)')
    ax.set_title('Significant sharing need not outperform unsupervised PCA')
    ax.legend(ncol=3,frameon=False);fig.tight_layout();save(fig,'02_source_baselines')

    c=load('bn_fig_F_containment.csv')
    fig,axs=plt.subplots(2,3,figsize=(11,6.6),sharex=True,sharey=True)
    for ax,m in zip(axs.flat,ORDER):
        q=c[c.model.eq(m)&c.M.eq(11)]
        floor=c[c.model.eq(m)].R0.iloc[0];ambient=c[c.model.eq(m)].ambient_R.iloc[0]
        ax.axhline(100*floor,color='#888',ls=':',lw=1);ax.axvline(100*ambient,color='#aaa',ls='--',lw=1)
        for method in ['MLR','LEACE','COV','PCA','RAND']:
            r=q[q.method.eq(method)].iloc[0];ax.scatter(100*r.R,100*r.L,color=COLORS[method],s=48,label=method)
        ax.set_title(NAMES[m]);ax.grid(alpha=.12);ax.set_xlim(13,39);ax.set_ylim(14,32)
    for ax in axs[-1]:ax.set_xlabel('F retention accuracy (%)')
    for ax in axs[:,0]:ax.set_ylabel('F leakage accuracy (%)')
    handles=[Line2D([0],[0],marker='o',ls='',color=COLORS[m],label=m) for m in ['MLR','LEACE','COV','PCA','RAND']]
    fig.legend(handles=handles,loc='lower center',ncol=5,bbox_to_anchor=(.5,-.025),frameon=False)
    fig.suptitle('Native Bangla F, rank 11: recoverability and erasure depend on the estimator',fontsize=13)
    fig.tight_layout(rect=[0,.02,1,.96]);save(fig,'03_containment_and_leakage')

    q=load('matched_input_controls.csv');q=q[q.concept.eq('F')&q.control.eq('bn-context')].set_index('model').loc[ORDER]
    fig,ax=plt.subplots(figsize=(9,4))
    ax.errorbar(100*q.span_minus_control,np.arange(6),
        xerr=np.vstack([100*(q.span_minus_control-q.ci_lo),100*(q.ci_hi-q.span_minus_control)]),
        fmt='o',color='#0072B2',capsize=3)
    ax.set_yticks(np.arange(6),[NAMES[m] for m in ORDER]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('English-subspace accuracy: idiom span minus masked context (pp)')
    ax.set_title('Does the idiom span add a signal beyond its context?')
    ax.grid(axis='x',alpha=.2);fig.text(.5,-.02,'Post-hoc paired component bootstrap; unadjusted 95% intervals; same 2,166 TEST sense IDs.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'04_matched_context_control')

    q=load('identity_rank64_full_space.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,rep,title in zip(axs,['idiom_span','context_only_idiom_masked'],['Idiom spans (same lexical form)','Contexts with the idiom masked']):
        s=q[q.representation.eq(rep)]
        for off,method in [(-.23,'LEACE'),(0,'RAND')]:
            vals=s[s.method.eq(method)].set_index('model').loc[ORDER]
            ax.bar(x+off,100*vals.R,width=.23,color=COLORS[method],label=method+' rank 64')
        amb=s.drop_duplicates('model').set_index('model').loc[ORDER]
        ax.bar(x+.23,100*amb.ambient_R,width=.23,color='#222222',label='Ambient')
        ax.axhline(100*amb.chance.iloc[0],color='black',ls=':',lw=.8)
        ax.set_xticks(x,[NAMES[m].replace('-large','') for m in ORDER],rotation=35,ha='right')
        ax.set_ylabel('Two-context identity top-1 (%)');ax.set_title(title,fontsize=11)
    axs[0].set_ylim(0,100);axs[1].set_ylim(0,5)
    axs[0].legend(frameon=False,fontsize=8,ncol=3,loc='upper left')
    fig.suptitle('Lexical identity is easy; masked-context identity is much harder',fontsize=13)
    fig.text(.5,-.025,'Fixed M=64, N=3,125 SPACE sense pairs. 1,889 TEST sense pairs. Different y-axis scales; chance 0.053%.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'05_identity_lexical_control')

    q=load('open_class_curves.csv');q=q[q.concept.eq('F')]
    fig,axs=plt.subplots(2,3,figsize=(11,6.8),sharex=True,sharey=True)
    for ax,m in zip(axs.flat,ORDER):
        for method in ['MLR','LEACE','PCA','RAND']:
            s=q[q.model.eq(m)&q.method.eq(method)].sort_values('N')
            ax.plot(s.N,100*s.R,'o-',ms=3,lw=1.4,color=COLORS[method],label=method)
            sd=s.R_sd.fillna(0)
            ax.fill_between(s.N,100*(s.R-sd),100*(s.R+sd),color=COLORS[method],alpha=.1)
        ax.axhline(100*s.R0.iloc[0],color='#aaa',ls=':',lw=.7)
        ax.set_xscale('log');ax.set_title(NAMES[m]);ax.grid(alpha=.12);ax.set_ylim(13,39)
    for ax in axs[-1]:ax.set_xlabel('N SPACE idiom groups')
    for ax in axs[:,0]:ax.set_ylabel('Held-out F accuracy (%)')
    axs[0,0].legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle('Learning-curve conclusions depend on the estimator',fontsize=13)
    fig.text(.5,-.005,'Rank 11, bn-fig. Shading: SD over three subset draws, not inferential CIs. LEACE uses Ledoit–Wolf shrinkage throughout.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.02,1,.96]);save(fig,'06_open_class_curves')

    q=load('primary_layer_profiles.csv');q=q[q.concept.eq('F')]
    selected=load('selected_layers.csv');selected=selected[selected.phase.eq('primary')].set_index('model').layer.to_dict()
    fig,axs=plt.subplots(2,3,figsize=(11,6.6),sharey=True)
    for ax,m in zip(axs.flat,ORDER):
        s=q[q.model.eq(m)].sort_values('layer')
        ax.plot(s.layer,100*s.R_pp,'o-',label='Bangla-estimated',color='#0072B2',ms=3)
        ax.plot(s.layer,100*s.R,'s-',label='English-estimated',color='#D55E00',ms=3)
        ax.plot(s.layer,100*s.R0,':',label='Majority floor',color='#555555')
        ax.axvline(selected[m],ls='--',color='#999999',lw=.8);ax.set_title(NAMES[m]);ax.grid(alpha=.12)
    for ax in axs[-1]:ax.set_xlabel('Layer')
    for ax in axs[:,0]:ax.set_ylabel('F accuracy (%)')
    axs[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Task-relevant directions emerge beyond the embedding layer',fontsize=13)
    fig.text(.5,-.005,'All-layer curves are descriptive. Dashed vertical line: layer chosen using VALIDATION, not these TEST curves.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.02,1,.96]);save(fig,'07_layer_profiles')

    b=load('label_balance.csv');b=b[b.model.eq(ORDER[0])&b.split.eq('test')]
    fig,ax=plt.subplots(figsize=(9.5,3.6))
    for off,concept,color in [(-.18,'F','#0072B2'),(.18,'L','#D55E00')]:
        row=b[b.concept.eq(concept)].iloc[0];counts=json.loads(row.counts)
        values=[100*counts.get(str(i),0)/row.n for i in range(12)]
        ax.bar(np.arange(12)+off,values,width=.35,color=color,label=concept)
    ax.set_xticks(range(12));ax.set_xlabel('Code in the same SPACE-fitted taxonomy')
    ax.set_ylabel('Share of TEST labels (%)');ax.set_title('Literal-label concentration is a measurement caveat')
    ax.legend(frameon=False);fig.tight_layout();save(fig,'08_literal_taxonomy_balance')
    (F/'plotting_environment.json').write_text(json.dumps(dict(python=__import__('sys').executable,
        matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__),indent=2))
    print('Wrote eight PNG/PDF figures.')


if __name__=='__main__':main()
