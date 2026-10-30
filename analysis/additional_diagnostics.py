"""Explicitly post-hoc diagnostics from saved, fixed TEST predictions.

Neither accuracy summaries nor complete-case GLMs replace registered results.
The original moderator files are preserved, including their failure records.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

C=Path(os.environ.get('IDIO_CAMPAIGN','campaign'))  # discovery campaign from run_research.py (read-only)
OUT=Path(__file__).resolve().parents[1]/'results'/'discovery'
SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from idiom_subspace.studies import moderator_analysis
from threadpoolctl import threadpool_limits


def main():
    df=pd.read_csv(C/'primary/predictions_all_jobs.csv',low_memory=False)
    primary=df[df.source_condition.eq('en-idiom')&df.condition.eq('bn-fig')].copy()
    corpus=pd.read_json(C/'primary/xlm-roberta-large/split_0/labels_0/corpus_prepared.json')
    labels=corpus[['idiom_id','F','S']]
    f=primary.merge(labels,on='idiom_id',validate='many_to_one')
    rows=[];by_class=[]
    for (model,concept),g in f.groupby(['model','concept']):
        class_rows=[]
        for code,gg in g.groupby(concept):
            row=dict(model=model,concept=concept,label=int(code),n=len(gg),cross_recall=gg.cross_correct.mean(),
                within_recall=gg.within_correct.mean(),floor_recall=gg.floor_correct.mean())
            class_rows.append(row);by_class.append(row)
        d=pd.DataFrame(class_rows)
        rows.append(dict(model=model,concept=concept,n=len(g),classes=len(d),
            cross_accuracy=g.cross_correct.mean(),within_accuracy=g.within_correct.mean(),
            cross_balanced_accuracy=d.cross_recall.mean(),within_balanced_accuracy=d.within_recall.mean(),
            floor_balanced_accuracy=d.floor_recall.mean(),note='post_hoc_macro_recall_on_fixed_TEST_predictions'))
    pd.DataFrame(rows).to_csv(OUT/'balanced_accuracy_diagnostic.csv',index=False)
    pd.DataFrame(by_class).to_csv(OUT/'class_recall_diagnostic.csv',index=False)

    original=pd.read_csv(OUT/'moderators.csv')
    primary_original=original[original.source_condition.eq('en-idiom')&original.condition.eq('bn-fig')]
    primary_original.to_csv(OUT/'primary_original_moderator_failures.csv',index=False)
    missing=[]
    for (model,concept),g in primary.groupby(['model','concept']):
        missing.append(dict(model=model,concept=concept,original_n=len(g),
            missing_cp_transparency=int(g.cp_transparency.isna().sum()),complete_case_n=int(g.cp_transparency.notna().sum()),
            missingness_definition='missing_counterpart_or_unavailable_MiniLM_cosine'))
    pd.DataFrame(missing).to_csv(OUT/'moderator_missingness.csv',index=False)
    complete=primary[primary.cp_transparency.notna()].copy()
    # Same original GLM, covariates and robust covariance; only rows with a
    # genuinely missing covariate are omitted. Constant anchored indicator is
    # dropped by the original helper. This estimates an anchored-only cohort.
    destination=OUT/'posthoc_complete_case_moderators'
    destination.mkdir(exist_ok=True)
    result=moderator_analysis(complete,destination)
    result['odds_ratio_per_sd']=np.exp(result.beta_standardized)
    result['odds_ratio_ci_lo']=np.exp(result.beta_standardized-1.96*result.se)
    result['odds_ratio_ci_hi']=np.exp(result.beta_standardized+1.96*result.se)
    result['analysis_scope']='post_hoc_complete_case_primary_direction_only_new_separate_family'
    result.to_csv(OUT/'posthoc_complete_case_moderators.csv',index=False)
    (destination/'DESIGN.json').write_text(json.dumps(dict(
        scope='post-hoc diagnostic; separate from primary RAND/PERM families',
        selection='en-idiom to bn-fig; remove missing cp_transparency; no outcome-based filtering',
        covariance='component-cluster-robust GLM; original helper; training/refit uncertainty not included',
        correction='Holm across every returned coefficient in this secondary 12-fit family',
        original_outputs_unchanged=True),indent=2))
    print('Balanced accuracy and complete-case moderator diagnostics written.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
