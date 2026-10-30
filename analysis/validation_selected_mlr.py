"""Exploratory baseline selection from saved validation predictions only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from idiom_subspace.studies import LayerStudy
from idiom_subspace.geometry import paired_cluster_bootstrap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--followup',required=True,type=Path)
    args=parser.parse_args()
    plan=REPO/'configs'/'plans'/'E2_VALIDATION_SELECTION_PLAN.md'
    stamp=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
               plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               status='post_hoc_exploratory_validation_only_selection')
    frozen=args.followup/'E2_VALIDATION_PLAN_FROZEN.json'
    if not frozen.exists():frozen.write_text(json.dumps(stamp,indent=2)+'\n')
    rows,validation_rows=[],[]
    for marker in sorted(args.followup.glob('*/complete.json')):
        directory=marker.parent
        model=json.loads(marker.read_text())['model']
        selected=json.loads((directory/'layer_selection.json').read_text())['layer']
        manifests=list((directory/'checkpoints'/str(selected)).glob('*/manifest.json'))
        assert len(manifests)==1
        checkpoint=manifests[0].parent
        p=pd.read_csv(directory/'item_predictions.csv.gz')
        for concept in ['F','S']:
            fits=[]
            for c in [.001,.01,.1,1.]:
                key=('custom',('E2',concept,c),'project')
                path=checkpoint/('eval_'+LayerStudy._key(key)+'.npz')
                with np.load(path,allow_pickle=False) as stored:
                    fit={k:stored[k] for k in stored.files}
                score=float(fit['validation_correct'].mean())
                fits.append((score,c,fit))
                validation_rows.append(dict(model=model,concept=concept,C=c,validation_accuracy=score,
                                             test_accuracy=float(fit['correct'].mean())))
            val,c,fit=sorted(fits,key=lambda x:(-x[0],x[1]))[0]
            cov=p[p.concept.eq(concept)&p.source.eq('en-idiom')&p.target.eq('bn-fig')&p.method.eq('COV')&p.representation.eq('project')]
            assert len(cov)==len(fit['correct'])
            assert np.array_equal(cov.y_true.to_numpy()==fit['predicted'],fit['correct'])
            contrast=cov.correct.to_numpy(float)-fit['correct'].astype(float)
            boot=paired_cluster_bootstrap(contrast,cov.component_id.to_numpy(),lambda a:a.mean(),n_boot=5000,seed=43)
            rows.append(dict(model=model,concept=concept,chosen_C=c,validation_accuracy=val,
                             tuned_MLR_accuracy=float(fit['correct'].mean()),COV_accuracy=float(cov.correct.mean()),
                             n=len(cov),**boot))
    out=REPO/'results'/'replication'
    pd.DataFrame(rows).to_csv(out/'validation_selected_mlr.csv',index=False)
    pd.DataFrame(validation_rows).to_csv(out/'regularization_validation_path.csv',index=False)
    (out/'E2_VALIDATION_PLAN_FROZEN.json').write_text(frozen.read_text())
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
