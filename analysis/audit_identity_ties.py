"""Re-evaluate saved ambient identity similarities with explicit tie rules.

This audits a saved metric; it does not fit a model or alter campaign outputs.
The existing S4 identity code assigns an optimistic rank to ties. S3 retrieval
already uses midranks. We report the size of this difference on real features.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

C=Path(os.environ.get('IDIO_CAMPAIGN','campaign'))  # discovery campaign from run_research.py (read-only)
OUT=Path(__file__).resolve().parents[1]/'results'/'discovery'


def main():
    original=pd.read_csv(C/'exports/identity_tables/S4_identity.csv')
    results=[]
    for model in original.model.unique():
        slug=model.replace('/','_')
        worker=C/'identity'/slug
        prepared=worker/'split_0/labels_0'
        dest=prepared/slug
        meta=json.loads((dest/'model_revision.json').read_text())
        layer=json.loads((dest/'layer_selection.json').read_text())['layer']
        features=np.load(worker/'cache'/meta['fingerprint']/'features.npy',mmap_mode='r')
        items=pd.read_json(prepared/'items.json')
        pooling=pd.read_csv(dest/'pooling.csv',keep_default_na=False)
        valid=pooling.valid.astype(str).str.lower().eq('true').to_numpy()
        frame=items[valid].reset_index(drop=True)
        raw=np.asarray(features[valid,meta['layers'].index(layer),:],dtype=np.float64)
        with np.load(dest/f'preprocess_layer_{layer}.npz',allow_pickle=False) as d:
            z=(raw-d['mean'])/float(d['scale'])
        for first,second in [('bn-fig','bn-fig2'),('bn-context','bn-context2')]:
            def paired(split):
                a=frame[frame.condition.eq(first)&frame.split.eq(split)].reset_index(names='i1')[['i1','idiom_id']]
                b=frame[frame.condition.eq(second)&frame.split.eq(split)].reset_index(names='i2')[['i2','idiom_id']]
                return a.merge(b,on='idiom_id',validate='one_to_one')
            space,test=paired('space'),paired('test')
            center=z[np.r_[space.i1.to_numpy(),space.i2.to_numpy()]].mean(0)
            A,B=z[test.i1.to_numpy()]-center,z[test.i2.to_numpy()]-center
            A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-12)
            B=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-12)
            sims=B@A.T
            greater,equal=[],[]
            for mat in [sims,sims.T]:
                gold=np.diag(mat)
                greater.extend((mat>gold[:,None]+1e-12).sum(1))
                equal.extend((np.abs(mat-gold[:,None])<=1e-12).sum(1))
            greater,equal=np.asarray(greater),np.asarray(equal)
            saved=original[(original.model==model)&(original.condition==first)].ambient_R.iloc[0]
            optimistic=float(np.mean(greater==0))
            assert abs(saved-optimistic)<1e-10,(model,first,saved,optimistic)
            results.append(dict(model=model,condition=first,layer=layer,n_eval=len(test),
                saved_ambient_R=saved,reproduced_optimistic_R=optimistic,
                unique_gold_top1=float(np.mean((greater==0)&(equal==1))),
                random_tie_breaking_top1=float(np.mean(np.where(greater==0,1/equal,0))),
                tied_gold_at_top_fraction=float(np.mean((greater==0)&(equal>1))),
                max_gold_tie_size=int(equal.max()),
                optimistic_MRR=float(np.mean(1/(1+greater))),
                midrank_MRR=float(np.mean(1/(1+greater+.5*(equal-1))))))
        print('Audited',model,flush=True)
    pd.DataFrame(results).to_csv(OUT/'identity_ambient_tie_audit.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
