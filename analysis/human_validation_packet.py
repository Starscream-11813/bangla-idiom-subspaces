"""Sample the 300-item human-validation packet and write two blank, differently ordered sheets.

Never generates judgments. It samples one discovery TEST record per input-disjoint component, round-robin over
the F classes (25 per class). The blank sheets it wrote are data/human_validation/annotator_{A,B}.original.csv;
the volunteers' filled sheets are annotator_{A,B}.csv.

    python analysis/human_validation_packet.py --campaign CAMPAIGN --out NEW_FOLDER
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIELDS = ['f_plausibility', 'function', 'counterpart_type', 'counterpart_equivalence']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.glob('annotator_*.csv')):
        raise SystemExit('Annotation sheets already exist; refusing to overwrite human work.')
    args.out.mkdir(parents=True, exist_ok=True)
    source = next((args.campaign/'primary').glob('*/split_0/labels_0/corpus_prepared.json'))
    df = pd.read_json(source)
    test = df[df.split.eq('test') & df.F.ge(0)].copy()
    rng = np.random.default_rng(20261021)
    bins = {int(c): list(g.iloc[rng.permutation(len(g))].index) for c, g in test.groupby('F')}
    used, selected = set(), []
    while len(selected) < 300:
        changed = False
        for c in sorted(bins):
            while bins[c]:
                ix = bins[c].pop()
                comp = test.loc[ix, 'component_id']
                if comp not in used:
                    selected.append(ix)
                    used.add(comp)
                    changed = True
                    break
            if len(selected) == 300:
                break
        if not changed:
            raise ValueError('Fewer than 300 distinct held-out components available')
    sample = test.loc[selected].copy()
    taxonomy = pd.read_csv(REPO/'results/discovery/synthesis/F_taxonomy.csv')
    names = taxonomy.set_index('F').name.to_dict()
    rows = []
    for r in sample.itertuples():
        cp = r.sims[0] if isinstance(r.sims, list) and r.sims else ''
        rows.append(dict(sample_id='validation_'+hashlib.sha256(str(r.idiom_id).encode()).hexdigest()[:12],
                         idiom=r.idiom, bangla_example=r.bn_ex, english_example=r.en_ex,
                         bangla_figurative_gloss=r.fig_bn, english_figurative_gloss=r.fig_en,
                         english_counterpart=cp, assigned_F=int(r.F), assigned_F_name=names[int(r.F)],
                         **{field: '' for field in FIELDS}, rationale=''))
    sheet = pd.DataFrame(rows)
    for code, seed in [('A', 20261022), ('B', 20261023)]:
        sheet.sample(frac=1, random_state=seed).to_csv(args.out/f'annotator_{code}.csv', index=False,
                                                    encoding='utf-8-sig')
    sample[['idiom_id', 'component_id', 'F']].assign(sample_id=sheet.sample_id.to_numpy()).to_csv(
        args.out/'private_sample_key.csv', index=False)
    codebook = []
    for c, g in df[df.split.eq('space')].groupby('F'):
        examples = g.fig_en.sample(min(3, len(g)), random_state=20261021).tolist()
        codebook.append(dict(F=int(c), name=names[int(c)],
                             keywords=taxonomy.set_index('F').loc[c, 'keywords'],
                             space_examples=' | '.join(examples)))
    pd.DataFrame(codebook).to_csv(args.out/'space_only_codebook.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=['annotator_code', 'consent_recorded', 'review_or_exemption_reference',
                          'started_at', 'finished_at', 'hours', 'payment_amount', 'payment_currency',
                          'recruitment_description']).to_csv(args.out/'study_worklog.csv', index=False)
    manifest = dict(status='prepared_not_annotated', items=300, components=len(used),
                    discovery_test_only=True, sample_seed=20261021,
                    F_counts=sample.F.value_counts().sort_index().to_dict(),
                    selection='round-robin F strata, one record per component; descriptive stratified sample',
                    contains_model_predictions=False, human_judgments_generated=False,
                    dataset_license='CC BY-SA 4.0',
                    dataset_source='https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
