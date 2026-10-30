"""Read-only analysis of the frozen discovery campaign. No estimator/model refitting.

Run with the research environment: IDIO_CAMPAIGN=/path/to/campaign python analysis/analyze.py
(writes results/discovery/).
All extra confidence intervals are explicitly post-hoc descriptive diagnostics.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd

C = Path(os.environ.get('IDIO_CAMPAIGN', 'campaign'))  # discovery campaign from run_research.py (read-only)
OUT = Path(__file__).resolve().parents[1] / 'results'
DATA = OUT / 'discovery'
DATA.mkdir(parents=True, exist_ok=True)
MODELS = ['xlm-roberta-large', 'bert-base-multilingual-cased', 'setu4993/LaBSE',
          'google/muril-large-cased', 'csebuetnlp/banglabert', 'csebuetnlp/banglishbert']
SLUG = lambda m: m.replace('/', '_')


def table(phase, name):
    return pd.read_csv(C / 'exports' / f'{phase}_tables' / name, low_memory=False)


def save(name, data):
    data.to_csv(DATA / name, index=False)
    return data


def records(data):
    return json.loads(data.to_json(orient='records', double_precision=12))


def cluster_interval(values, groups, seed=20261021, draws=2000):
    # Mean of a paired correctness difference, resampling complete components.
    frame = pd.DataFrame({'value': values, 'group': groups})
    grouped = frame.groupby('group').value.agg(['sum', 'count'])
    sums, counts = grouped['sum'].to_numpy(), grouped['count'].to_numpy()
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(draws):
        ix = rng.integers(0, len(grouped), len(grouped))
        result.append(sums[ix].sum() / counts[ix].sum())
    return float(np.mean(values)), *np.quantile(result, [.025, .975]).tolist()


def main():
    summary = {}
    ledger = json.loads((C / 'phase_status.json').read_text())
    assert ledger['status'] == 'completed'
    assert all(p['exit_code'] == 0 for p in ledger['phases'].values())
    statuses = []
    for phase in ['primary', 'identity', 'decoder']:
        expected = 6 if phase != 'decoder' else 1
        paths = list((C / phase).glob('**/status.json'))
        assert len(paths) == expected, (phase, paths)
        for path in paths:
            d = json.loads(path.read_text())
            assert d['status'] == 'completed' and not d.get('analysis_pending')
            statuses.append(dict(phase=phase, source=str(path.relative_to(C)), **d))
        if phase != 'decoder':
            jobs = json.loads((C / phase / 'jobs.json').read_text())
            assert len(jobs) == expected and all(j['exit_code'] == 0 for j in jobs)
    save('job_status.csv', pd.DataFrame(statuses))
    integrity = []
    for line in (C / 'exports/SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = C / name
        h = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(8 * 2**20):
                h.update(chunk)
        integrity.append(dict(source=name, matches=h.hexdigest() == digest))
    assert all(r['matches'] for r in integrity), [r for r in integrity if not r['matches']]
    summary['integrity_files_verified'] = len(integrity)
    save('integrity_check.csv', pd.DataFrame(integrity))
    duration = []
    for phase, info in ledger['phases'].items():
        start, stop = [datetime.fromisoformat(info[k]) for k in ['started_at', 'finished_at']]
        duration.append(dict(phase=phase, hours=(stop-start).total_seconds()/3600,
                             started_utc=info['started_at'], finished_utc=info['finished_at']))
    save('phase_durations.csv', pd.DataFrame(duration))

    selections = []
    for phase in ['primary', 'identity', 'decoder']:
        for path in (C / phase).glob('**/layer_selection.json'):
            d = json.loads(path.read_text())
            model = json.loads((path.parent/'status.json').read_text())['model']
            selections.append(dict(phase=phase, model=model, layer=d['layer'], criterion=d['criterion'],
                                   validation_score=d['validation_score']))
    selections = save('selected_layers.csv', pd.DataFrame(selections))
    selected = selections[selections.phase.eq('primary')][['model','layer']]

    inf = pd.read_csv(C / 'primary/inference_all_jobs.csv')
    primary = inf[inf.primary_test].copy()
    assert len(primary) == 12 and primary.RAND_draws.eq(1999).all() and primary.PERM_draws.eq(1999).all()
    assert primary.PERM_inferential.all()
    primary['cross_gain_pp'] = 100*(primary.R_ep-primary.R0)
    primary['within_gain_pp'] = 100*(primary.R_pp-primary.R0)
    primary['gain_over_RAND_pp'] = 100*(primary.R_ep-primary.RAND_R_mean)
    primary['gain_over_PCA_pp'] = 100*(primary.R_ep-primary.PCA_R_mean)
    primary['gain_over_PERM_pp'] = 100*(primary.R_ep-primary.PERM_R_mean)
    primary['both_nulls_reject'] = (primary.RAND_p_holm_all_jobs.le(.05) & primary.PERM_p_holm_all_jobs.le(.05))
    expected = (primary.R_ep-primary.R0)/(primary.R_pp-primary.R0)
    assert np.allclose(expected,primary.SSI)
    save('primary_confirmatory.csv', primary)
    summary['primary'] = records(primary[['model','concept','layer','M','n','R_ep','R_pp','R0','SSI','ci_lo','ci_hi',
        'SSI_adjusted','RAND_p_holm_all_jobs','PERM_p_holm_all_jobs','both_nulls_reject','gain_over_PCA_pp']])
    summary['primary_null_rejections'] = dict(RAND=int(primary.RAND_p_holm_all_jobs.le(.05).sum()),
        PERM=int(primary.PERM_p_holm_all_jobs.le(.05).sum()), both=int(primary.both_nulls_reject.sum()))
    save('all_selected_layer_inference.csv', inf)

    grid = table('primary', 'S1_S3_grid.csv')
    full_grid = grid
    grid = grid[grid.group.eq('all')].copy()
    at_selected = grid.merge(selected,on=['model','layer'])
    self_grid = at_selected[at_selected.source_condition.eq(at_selected.condition)].copy()
    ambient = self_grid[self_grid.method.eq('AMBIENT')][['model','concept','condition','R','R0','M']].rename(
        columns={'R':'ambient_R','M':'ambient_D','R0':'ambient_floor'})
    containment = self_grid.merge(ambient,on=['model','concept','condition'],validate='many_to_one')
    containment['gain_capture'] = (containment.R-containment.R0)/(containment.ambient_R-containment.R0)
    containment['raw_accuracy_ratio'] = containment.R/containment.ambient_R
    containment['leak_above_floor_pp'] = 100*(containment.L-containment.R0)
    save('selected_layer_containment.csv', containment)
    save('bn_fig_F_containment.csv',containment[containment.condition.eq('bn-fig') & containment.concept.eq('F')])
    save('selected_layer_conditions.csv', self_grid[(self_grid.method.eq('MLR') &
         ((self_grid.concept.eq('F') & self_grid.M.eq(11)) | (self_grid.concept.eq('S') & self_grid.M.eq(3)))) |
         self_grid.method.eq('AMBIENT')])
    cross = at_selected[(at_selected.source_condition.eq('en-idiom')) & at_selected.condition.eq('bn-fig')]
    save('cross_language_estimators.csv',cross)
    layers = table('primary','ssi_all_layers.csv')
    layer_profile = layers[layers.group.eq('all') & layers.method.eq('MLR') & layers.source_condition.eq('en-idiom') &
        layers.condition.eq('bn-fig') & ((layers.concept.eq('F') & layers.M.eq(11)) | (layers.concept.eq('S') & layers.M.eq(3)))].copy()
    save('primary_layer_profiles.csv',layer_profile)

    convergence = []
    for (model,layer), frame in full_grid.groupby(['model','layer']):
        all_rows = frame[frame.group.eq('all')]
        for label, subset in [('all_groups',frame),('group_all_rows',all_rows)]:
            row = dict(model=model,layer=layer,population=label,rows=len(subset))
            for column in ['probe_converged','leak_probe_converged']:
                flags = subset[column].astype(str).str.lower()
                row[column+'_false'] = int(flags.eq('false').sum())
                row[column+'_true'] = int(flags.eq('true').sum())
                row[column+'_missing'] = int(flags.isin(['nan','none']).sum())
            convergence.append(row)
    save('convergence_audit.csv',pd.DataFrame(convergence))
    worst = at_selected[at_selected.method.ne('AMBIENT')].groupby(['model','concept','method']).spectral_gap.agg(['min','median']).reset_index()
    save('spectral_gaps.csv',worst)

    # Condition comparisons use exactly the same TEST sense IDs for each pair.
    preds = pd.read_csv(C/'primary/predictions_all_jobs.csv',low_memory=False)
    comparisons = []
    for (model,concept), frame in preds[preds.source_condition.eq('en-idiom')].groupby(['model','concept']):
        a = frame[frame.condition.eq('bn-fig')]
        for target in ['bn-context','bn-idiom','bn-sentence','bn-para']:
            b = frame[frame.condition.eq(target)]
            merged = a[['idiom_id','component_id','cross_correct','within_correct']].merge(
                b[['idiom_id','cross_correct','within_correct']],on='idiom_id',suffixes=('_span','_control'),validate='one_to_one')
            if merged.empty: continue
            delta = merged.cross_correct_span.astype(float)-merged.cross_correct_control.astype(float)
            est,lo,hi=cluster_interval(delta,merged.component_id)
            comparisons.append(dict(model=model,concept=concept,control=target,n=len(merged),
                n_components=merged.component_id.nunique(),cross_span=merged.cross_correct_span.mean(),
                cross_control=merged.cross_correct_control.mean(),span_minus_control=est,ci_lo=lo,ci_hi=hi,
                note='post_hoc_descriptive_paired_component_bootstrap_2000_no_multiplicity_claim'))
    save('matched_input_controls.csv',pd.DataFrame(comparisons))

    for name in ['S2_language.csv','S2_language_subspace.csv','S3_subspace_overlap.csv','S3_baselines.csv',
                 'S3_contrasts.csv','S3_asymmetry.csv','probe_robustness.csv','moderators.csv','bow_baselines.csv']:
        save(name,table('primary',name))
    overlap=table('primary','S3_subspace_overlap.csv')
    save('primary_subspace_overlap.csv',overlap[overlap.source_condition.eq('en-idiom') & overlap.condition.eq('bn-fig') &
         overlap.method.eq('MLR') & ((overlap.concept.eq('F') & overlap.M.eq(11)) | (overlap.concept.eq('S') & overlap.M.eq(3)))])
    base=table('primary','S3_baselines.csv')
    save('primary_global_geometry.csv',base[base.source_condition.eq('en-idiom') & base.condition.eq('bn-fig')])
    oc=table('primary','S4_openclass.csv')
    oc=oc[oc.condition.eq('bn-fig') & ((oc.concept.eq('F') & oc.M.eq(11)) | (oc.concept.eq('S') & oc.M.eq(3)))]
    curve=oc.groupby(['model','concept','method','M','N']).agg(R=('R','mean'),R_sd=('R','std'),
        L=('L','mean'),R0=('R0','first'),reps=('rep','nunique'),purity_error=('purity_err','mean'),
        purity_floor_error=('purity_floor_err','first')).reset_index()
    save('open_class_curves.csv',curve)

    identity=table('identity','S4_identity.csv')
    fixed=identity[identity.M.eq(64)].groupby(['model','representation','method','M','N']).agg(
        R=('R','mean'),R_sd=('R','std'),MRR=('MRR','mean'),L=('L','mean'),ambient_R=('ambient_R','first'),
        chance=('chance','first'),n_eval=('n_eval_idioms','first'),reps=('rep','nunique')).reset_index()
    save('identity_rank64_curves.csv',fixed)
    fixed_full=fixed[fixed.N.eq(3125)]
    save('identity_rank64_full_space.csv',fixed_full)

    label_summaries=[];fls_floors=[];validity=[];taxonomy=[]
    for model in MODELS:
        worker=C/'primary'/SLUG(model); prepared=worker/'split_0/labels_0';dest=prepared/SLUG(model)
        df=pd.read_json(prepared/'corpus_prepared.json')
        audit=json.loads((prepared/'label_audit.json').read_text())
        items=pd.read_json(prepared/'items.json')
        pool=pd.read_csv(dest/'pooling.csv',keep_default_na=False)
        valid=pool.valid.astype(str).str.lower().eq('true').to_numpy()
        assert len(items)==len(pool)
        frame=items[valid].reset_index(drop=True)
        for concept in ['F','L','S']:
            for split in ['space','probe','validation','test']:
                v=df.loc[df.split.eq(split)&df[concept].ge(0),concept]
                counts=v.value_counts()
                label_summaries.append(dict(model=model,concept=concept,split=split,n=len(v),
                    classes=v.nunique(),majority_class=counts.index[0],majority_share=counts.iloc[0]/len(v),
                    entropy_bits=float(-np.sum((counts/len(v))*np.log2(counts/len(v)))),counts=json.dumps(counts.to_dict())))
        for condition, all_items in items.groupby('condition'):
            p=pool.loc[all_items.index];flags=p.valid.astype(str).str.lower().eq('true')
            validity.append(dict(model=model,condition=condition,n=len(p),valid=int(flags.sum()),
                                  invalid=int((~flags).sum()),statuses=json.dumps(p.status.value_counts().to_dict())))
        for condition in frame.condition.unique():
            f=frame[frame.condition.eq(condition)&frame.F.ge(0)&frame.L.ge(0)&frame.F.ne(frame.L)]
            for concept in ['F','L']:
                train=f[f.split.eq('probe')][concept]
                test=f[f.split.eq('test')][concept]
                if train.empty or test.empty:continue
                majority=train.value_counts().index[0]
                fls_floors.append(dict(model=model,condition=condition,other=concept,n=len(test),
                    probe_majority_class=majority,floor_accuracy=test.eq(majority).mean(),floor_error=1-test.eq(majority).mean()))
        taxonomy.append(dict(model=model,F_SPACE_reseed_ARI=audit.get('F_SPACE_reseed_ARI'),
                             agreement=audit.get('agreement_diagnostics'),label_sources=audit.get('literal_gloss_sources')))
    save('label_balance.csv',pd.DataFrame(label_summaries))
    save('pooling_validity.csv',pd.DataFrame(validity))
    fls=table('primary','S4_FLS.csv').merge(pd.DataFrame(fls_floors),on=['model','condition','other','n'],validate='many_to_one')
    fls['other_accuracy_inside']=1-fls.purity_err
    fls['inside_gain_over_floor_pp']=100*(fls.floor_error-fls.purity_err)
    fls['outside_gain_over_floor_pp']=100*(fls.floor_error-fls.interference_err)
    save('FLS_with_conditional_floors.csv',fls)
    summary['taxonomy']=taxonomy
    dec=pd.read_csv(C/'decoder/inference_all.csv')
    save('decoder_inference.csv',dec)
    for name in ['S1_S3_grid.csv','S2_language_subspace.csv','S3_baselines.csv','S4_FLS.csv','S4_openclass.csv']:
        d=table('decoder',name)
        if 'group' in d:d=d[d.group.eq('all')]
        save('decoder_'+name,d)

    # Qualitative examples are selected deterministically from saved outcomes;
    # they are illustrations, not independent annotations or new observations.
    f=preds[preds.concept.eq('F')&preds.source_condition.eq('en-idiom')&preds.condition.eq('bn-fig')]
    consensus=f.groupby('idiom_id').cross_correct.agg(['sum','count']).reset_index()
    source=pd.read_json(C/'primary/xlm-roberta-large/split_0/labels_0/corpus_prepared.json')
    examples=consensus.merge(source,on='idiom_id')
    chosen=[]
    for success in [0,3,6]:
        candidate=examples[examples['sum'].eq(success)&examples['count'].eq(6)].sort_values('idiom_id').head(2).copy()
        candidate['selection_rule']=f'lexicographically_first_two_ids_with_{success}_of_6_correct'
        chosen.append(candidate)
    columns=[k for k in ['idiom_id','idiom','bn_ex','fig_en','lit_bn','lit_en','counterparts','F','L','S','sum','count','selection_rule'] if k in examples]
    save('illustrative_cases.csv',pd.concat(chosen,ignore_index=True)[columns])

    # Summaries never convert missing observations into zeros.
    summary['phase_durations']=duration
    summary['data_rows']={p.name:len(pd.read_csv(p,low_memory=False)) for p in DATA.glob('*.csv')}
    summary['analysis_time_utc']=datetime.now(timezone.utc).isoformat()
    summary['campaign']=str(C)
    summary['research_python']=sys.executable
    summary['analysis_packages']=dict(pandas=pd.__version__,numpy=np.__version__)
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Wrote',len(list(DATA.glob('*.csv'))),'derived tables.')
    print('Primary rejection counts:',summary['primary_null_rejections'])
    print('Integrity checks passed:',len(integrity))


if __name__=='__main__':
    main()
