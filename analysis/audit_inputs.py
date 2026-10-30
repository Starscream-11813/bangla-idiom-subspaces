"""Reproduce input/population diagnostics using saved artifacts and tokenizer.

No model weights, estimators, corpus records, or scientific outputs are changed.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

C=Path(os.environ.get('IDIO_CAMPAIGN','campaign'))  # discovery campaign from run_research.py (read-only)
OUT=Path(__file__).resolve().parents[1]/'results'/'discovery'


def main():
    rows=[]
    for phase in ['primary','identity','decoder']:
        for path in (C/phase).glob('**/coverage.json'):
            if not path.parent.parent.name.startswith('labels_'):continue
            for row in json.loads(path.read_text()):
                rows.append(dict(phase=phase,source_file=str(path.relative_to(C)),**row))
    pd.DataFrame(rows).to_csv(OUT/'coverage_by_job.csv',index=False)
    rows=[]
    for worker in sorted((C/'primary').iterdir()):
        if not (worker/'run_config.json').exists():continue
        prepared=worker/'split_0/labels_0';dest=prepared/worker.name
        pool=pd.read_csv(dest/'pooling.csv',keep_default_na=False)
        items=pd.read_json(prepared/'items.json')
        model=json.loads((dest/'status.json').read_text())['model']
        pool['condition']=items.condition
        valid=pool.valid.astype(str).str.lower().eq('true')
        for condition,g in pool[valid].groupby('condition'):
            rows.append(dict(model=model,condition=condition,n_valid=len(g),unknown_tokens=int(g.unk_count.sum()),
                input_tokens=int(g.full_token_count.sum()),unknown_token_fraction=float(g.unk_count.sum()/g.full_token_count.sum())))
    pd.DataFrame(rows).to_csv(OUT/'tokenization_diagnostics.csv',index=False)

    prepared=C/'identity/xlm-roberta-large/split_0/labels_0'
    items=pd.read_json(prepared/'items.json')
    pool=pd.read_csv(prepared/'xlm-roberta-large/pooling.csv',keep_default_na=False)
    frame=items[pool.valid.astype(str).str.lower().eq('true')].reset_index(drop=True)
    a=frame[frame.condition.eq('bn-fig')&frame.split.eq('test')]
    b=frame[frame.condition.eq('bn-fig2')&frame.split.eq('test')]
    pair=a.merge(b[['idiom_id']],on='idiom_id',validate='one_to_one')
    corpus=pd.read_json(C/'primary/xlm-roberta-large/split_0/labels_0/corpus_prepared.json')
    population=dict(identity_test_sense_pairs=len(pair),identity_test_idiom_groups=int(pair.idiom_group.nunique()),
        primary_test_records=2590,primary_exact_span_test_records=2166,
        source_idiom_groups=int(corpus.idiom_group.nunique()),distinct_original_idiom_strings=int(corpus.idiom.nunique()))
    (OUT/'population_diagnostics.json').write_text(json.dumps(population,indent=2))

    config=json.loads((C/'configs/full_3090.json').read_text())
    spec=next(m for m in config['models'] if m['name']=='bert-base-multilingual-cased')
    tok=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True,
                                    cache_dir=os.environ.get('HF_HUB_CACHE'))
    items=pd.read_json(C/'primary/bert-base-multilingual-cased/split_0/labels_0/items.json')
    anchors=items[items.condition.eq('en-idiom')]
    enc=tok(anchors.text.tolist(),return_offsets_mapping=True,truncation=True,max_length=128)
    total=target=width=0;substrings={}
    for row,ids,offsets in zip(anchors.itertuples(),enc['input_ids'],enc['offset_mapping']):
        start=row.text.find(row.span);end=start+len(row.span)
        for token,(a,b) in zip(ids,offsets):
            if b<=a:continue
            overlap=a<end and b>start;width+=int(overlap)
            if token==tok.unk_token_id:
                total+=1;target+=int(overlap);text=row.text[a:b];substrings[text]=substrings.get(text,0)+1
    attribution=dict(unknown_total=total,unknown_target=target,target_tokens=width,unknown_target_fraction=target/width,
        common_unknown_substrings=sorted(substrings.items(),key=lambda x:-x[1])[:10])
    (OUT/'mbert_unknown_token_attribution.json').write_text(json.dumps(attribution,indent=2,ensure_ascii=False))
    assert total==20476 and target==0

    predictions=pd.read_csv(C/'primary/predictions_all_jobs.csv',low_memory=False)
    predictions=predictions[predictions.source_condition.eq('en-idiom')&predictions.condition.eq('bn-fig')]
    rows=[]
    for (model,concept),g in predictions.groupby(['model','concept']):
        for group in ['transparent','opaque']:
            f=g[g[group]];cross=f.cross_correct.mean();within=f.within_correct.mean();floor=f.floor_correct.mean()
            denominator=within-floor
            rows.append(dict(model=model,concept=concept,group=group,n=len(f),R_ep=cross,R_pp=within,R0=floor,
                denominator=denominator,SSI=(cross-floor)/denominator if denominator>.02 else float('nan')))
    pd.DataFrame(rows).to_csv(OUT/'transparency_group_decomposition.csv',index=False)
    print('Input, population, canonical coverage and stratum diagnostics written.')


if __name__=='__main__':main()
