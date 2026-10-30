"""Prospective internal replication of E1--E4 on split seed 1 (frozen plan: configs/plans/FOLLOWUP_PLAN.md).

The discovery campaign's feature caches are opened read-only. Run with the research environment:

    python run_replication.py --campaign CAMPAIGN --out CAMPAIGN/paper_followup_seed1_20261021

Item-level evidence stays in --out; portable aggregates go to results/replication/.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
PLANS = REPO / 'configs' / 'plans'
sys.path.insert(0, str(REPO))
from idiom_subspace.config import Config, model_slug
from idiom_subspace.data import load_corpus, split_corpus
from idiom_subspace.labels import build_labels
from idiom_subspace.geometry import (Subspace, fit_estimator, fitted_preprocessor,
                                    holm, _symmetric_roots)
from idiom_subspace.studies import LayerStudy
from idiom_subspace.runner import write_json, environment
from idiom_subspace.progress import Progress

MODELS = ['csebuetnlp/banglabert', 'google/muril-large-cased', 'setu4993/LaBSE',
          'csebuetnlp/banglishbert', 'bert-base-multilingual-cased', 'xlm-roberta-large']
H1_MODELS = set(MODELS[:3])
METHODS = ['RAND', 'PCA', 'MLR', 'LDA', 'CPCA', 'COV', 'LEACE']


def stamp(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(values, groups, seed=41):
    """Conditional cluster bootstrap CI and one-sided component wild-sign p."""
    values = np.asarray(values, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=values, minlength=g)
    counts = np.bincount(inv, minlength=g)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1 / g), size=5000)
    boot = (weights @ sums) / (weights @ counts)
    observed = float(values.mean())
    exceed = 0
    for start in range(0, 19999, 1000):
        signs = 2 * rng.integers(0, 2, size=(min(1000, 19999-start), g)) - 1
        exceed += int(np.sum(signs @ sums >= values.sum() - 1e-12))
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(estimate=observed, ci_lo=float(lo), ci_hi=float(hi),
                p=(1+exceed)/20000, n=len(values), n_components=g,
                bootstrap_draws=5000, wild_sign_draws=19999)


def ratio_interval(ambient, english, native, groups):
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    e = np.bincount(inv, weights=ambient-english, minlength=g)
    b = np.bincount(inv, weights=ambient-native, minlength=g)
    w = np.random.default_rng(42).multinomial(g, np.full(g, 1/g), size=5000)
    den = w @ b
    ratios = np.divide(w @ e, den, out=np.full(5000, np.nan), where=den > 0)
    good = ratios[np.isfinite(ratios)]
    lo, hi = np.quantile(good, [.025, .975]) if len(good) else (np.nan, np.nan)
    return dict(erasure_fraction=float(e.sum()/b.sum()) if b.sum() > 0 else np.nan,
                fraction_ci_lo=float(lo), fraction_ci_hi=float(hi),
                fraction_valid_bootstraps=len(good),
                native_effect=float((ambient-native).mean()),
                english_effect=float((ambient-english).mean()))


def readout(study, concept, method, source, target, erase=False):
    rank = 11 if concept == 'F' else 3
    return study._evaluate_sub(target, concept, source, concept, method, rank,
                               representation='erase' if erase else 'project')


def records(study, ev, concept, method, source, target, representation):
    f = study.items.iloc[ev['test_idx']]
    d = f[['item_id', 'idiom_id', 'component_id']].copy()
    d['model'] = study.meta['model']
    d['concept'], d['method'] = concept, method
    d['source'], d['target'], d['representation'] = source, target, representation
    d['y_true'] = f[concept].to_numpy()
    d['y_pred'] = ev['predicted']
    d['correct'], d['floor_correct'] = ev['correct'], ev['floor']
    d['probe_converged'] = bool(ev.get('probe_converged', True))
    assert np.array_equal(d.y_true.to_numpy() == d.y_pred.to_numpy(), ev['correct'])
    return d


def e1_observations(study, dest):
    specs = []
    for concept in ['F', 'S']:
        rank = 11 if concept == 'F' else 3
        for method in METHODS:
            for source in ['en-idiom', 'bn-fig']:
                for rep in ['project', 'erase']:
                    specs.append(('bn-fig', concept, source, concept, method, rank,
                                  rep, False, 0, None, True))
        specs += [('bn-fig', concept, 'bn-fig', concept, 'AMBIENT', study.X.shape[1],
                   'project', False, 0, None, True)]
        for source in ['en-para', 'en-idiom', 'bn-para']:
            specs.append(('bn-para', concept, source, concept, 'COV', rank,
                          'project', False, 0, None, True))
    study._prefetch(specs, 'follow-up observed readouts')
    predictions, tests, scores = [], [], []
    for concept in ['F', 'S']:
        rank = 11 if concept == 'F' else 3
        for method in METHODS:
            for source in ['en-idiom', 'bn-fig']:
                ev = readout(study, concept, method, source, 'bn-fig')
                study.evidence[(concept, method, rank, source, 'bn-fig')] = ev
                for rep in ['project', 'erase']:
                    result = ev if rep == 'project' else readout(study, concept, method, source, 'bn-fig', True)
                    predictions.append(records(study, result, concept, method, source, 'bn-fig', rep))
                    scores.append(dict(model=study.meta['model'], concept=concept, method=method,
                                       source=source, target='bn-fig', representation=rep,
                                       accuracy=float(result['correct'].mean()),
                                       floor=float(result['floor'].mean()), n=len(result['correct']),
                                       converged=bool(result.get('probe_converged', True))))
        cov, mlr = [readout(study, concept, m, 'en-idiom', 'bn-fig') for m in ['COV', 'MLR']]
        f = study.items.iloc[cov['test_idx']]
        groups = f.component_id.to_numpy()
        tests.append(dict(model=study.meta['model'], concept=concept, hypothesis='H2_COV_minus_MLR',
                          **paired(cov['correct'].astype(float)-mlr['correct'], groups)))
        gloss, idiom = [readout(study, concept, 'COV', s, 'bn-para') for s in ['en-para', 'en-idiom']]
        for source, result in [('en-para', gloss), ('en-idiom', idiom)]:
            predictions.append(records(study, result, concept, 'COV', source, 'bn-para', 'project'))
        g = study.items.iloc[gloss['test_idx']].component_id.to_numpy()
        assert np.array_equal(gloss['test_idx'], idiom['test_idx'])
        tests.append(dict(model=study.meta['model'], concept=concept, hypothesis='H3_gloss_minus_counterpart',
                          **paired(gloss['correct'].astype(float)-idiom['correct'], g)))
        ambient = study._evaluate_sub('bn-fig', concept, 'bn-fig', concept, 'AMBIENT', study.X.shape[1])
        english, native = [readout(study, concept, 'LEACE', s, 'bn-fig', True) for s in ['en-idiom', 'bn-fig']]
        a0, ae, ab = [ev['correct'].astype(float) for ev in [ambient, english, native]]
        contrast = .25*(a0-ab)-(a0-ae)
        test = dict(model=study.meta['model'], concept=concept, hypothesis='H4_erasure_below_quarter',
                    **paired(contrast, groups), **ratio_interval(a0, ae, ab, groups))
        if test['native_effect'] <= 0:
            test['p'] = np.nan
        tests.append(test)
        predictions.append(records(study, ambient, concept, 'AMBIENT', 'bn-fig', 'bn-fig', 'project'))
    pd.concat(predictions, ignore_index=True).to_csv(dest/'item_predictions.csv.gz', index=False)
    return tests, scores


def mechanisms(study, dest):
    rows = []
    for concept in ['F', 'S']:
        rank = 11 if concept == 'F' else 3
        en = study.idx('en-idiom', 'space', concept)
        bn = study.idx('bn-fig', 'space', concept)
        y = study.items[concept].to_numpy()
        indices = [study.idx('bn-fig', s, concept) for s in ['probe', 'validation', 'test']]
        cov = study.sub('en-idiom', concept, 'COV', rank)
        for c in [.001, .01, .1, 1.]:
            estimator = fit_estimator(study.X[en], y[en], 'MLR', rank, C=c, seed=1,
                                      backend='torch', device='cuda', mlr_variant='reference', max_iter=3000)
            sub = estimator.at_rank(rank)
            ev = study._custom_readout(('E2', concept, c), sub, indices, y)
            rows.append(dict(experiment='E2', model=study.meta['model'], concept=concept,
                             setting=f'C={c:g}', accuracy=float(ev['correct'].mean()),
                             cov_shared_dimensions=float(np.linalg.norm(sub.basis.T@cov.basis, 'fro')**2),
                             probe_converged=bool(ev.get('probe_converged', True))))
        q = cov.basis
        Xe = study.X[en]-study.X[en].mean(0)
        Xb = study.X[bn]-study.X[bn].mean(0)
        se, sb = Xe.T@Xe/len(Xe), Xb.T@Xb/len(Xb)
        original = study.sub('en-idiom', concept, 'LEACE', rank)
        for label, sigma in [('english', se), ('pooled', .5*(se+sb)), ('target', sb)]:
            _, w = _symmetric_roots(sigma, 0.)
            invq = w@(w@q)
            factor = invq@np.linalg.pinv(q.T@invq, rcond=1e-12)
            sub = Subspace(cov.center, q, rank, 'LEACE', rank, factor)
            identity_error = float(np.linalg.norm(q.T@factor-np.eye(rank)))
            assert identity_error < 1e-5, (study.meta, concept, label, identity_error)
            operator_error = float(np.linalg.norm(sub.projection_operator-original.projection_operator,
                                                   'fro')/max(np.linalg.norm(original.projection_operator, 'fro'), 1e-12))
            if label == 'english':
                assert operator_error < 1e-4, ('source LEACE reconstruction', operator_error)
            for rep in ['project', 'erase']:
                ev = study._custom_readout(('E3', concept, label), sub, indices, y, representation=rep)
                rows.append(dict(experiment='E3', model=study.meta['model'], concept=concept,
                                 setting=label, representation=rep, accuracy=float(ev['correct'].mean()),
                                 idempotence_error=identity_error,
                                 source_operator_relative_error=operator_error,
                                 image_shared_dimensions=float(np.linalg.norm(q.T@original.basis, 'fro')**2),
                                 probe_converged=bool(ev.get('probe_converged', True))))
    pd.DataFrame(rows).to_csv(dest/'mechanisms.csv', index=False)
    return rows


def isotropy(study, dest):
    rows = []
    space = study.items.split.eq('space').to_numpy()
    pooled = np.concatenate([study.idx(c, 'space', 'F') for c in ['en-idiom', 'bn-fig']])
    mean = study.X[pooled].mean(0)
    pc = fit_estimator(study.X[pooled], None, 'PCA', 10, seed=1).basis
    sd = study.X[space].std(0)
    sd = np.maximum(sd, 1e-8)
    for setting in ['baseline', 'standardized', 'remove_pc_1', 'remove_pc_3', 'remove_pc_10']:
        if setting == 'baseline':
            z = study.X
        elif setting == 'standardized':
            z = study.X/sd
        else:
            k = int(setting.rsplit('_', 1)[1])
            zc = study.X-mean
            z = zc-(zc@pc[:, :k])@pc[:, :k].T
        centered = z[pooled]-z[pooled].mean(0)
        spectrum = np.maximum(np.linalg.eigvalsh(centered.T@centered/len(centered)), 0)
        participation = float(spectrum.sum()**2/(spectrum@spectrum))
        top = float(spectrum[-1]/spectrum.sum())
        alt = LayerStudy(z, study.items, study.cfg, study.meta,
                         checkpoint_dir=dest/'E4_checkpoints'/setting)
        for concept in ['F', 'S']:
            for method in ['COV', 'MLR']:
                for source in ['en-idiom', 'bn-fig']:
                    ev = readout(alt, concept, method, source, 'bn-fig')
                    rows.append(dict(experiment='E4', model=study.meta['model'], concept=concept,
                                     method=method, source=source, setting=setting,
                                     accuracy=float(ev['correct'].mean()),
                                     effective_dimension=participation, top_pc_variance_fraction=top,
                                     probe_converged=bool(ev.get('probe_converged', True))))
        del alt
    pd.DataFrame(rows).to_csv(dest/'isotropy.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    assert torch.cuda.is_available(), 'GPU required for this campaign'
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = PLANS/'FOLLOWUP_PLAN.md'
    sources = [Path(__file__), plan, REPO/'idiom_subspace/fastprobe.py',
               REPO/'idiom_subspace/studies.py',
               REPO/'idiom_subspace/geometry.py']
    freeze = dict(plan_sha256=stamp(plan), sources={str(p.relative_to(REPO)): stamp(p) for p in sources},
                  campaign=str(args.campaign.resolve()), hypotheses=48,
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(), environment=environment())
    freeze_path = out/'PLAN_FROZEN.json'
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text())
        assert old['sources'] == freeze['sources'], 'Frozen analysis sources changed: use fresh --out'
    else:
        write_json(freeze_path, freeze)
        shutil.copy2(plan, out/'FOLLOWUP_PLAN.md')
        shutil.copy2(Path(__file__), out/Path(__file__).name)
    cfg = Config.load(args.campaign/'configs/full_3090.json')
    cfg.seeds, cfg.label_seeds = [1], [0]
    cfg.inference_method, cfg.robustness = 'COV', False
    cfg.pairs = [['en-idiom', 'bn-fig']]
    cfg.compute_leakage, cfg.studies = False, ['S3']
    cfg.cache_only = True
    cfg.validate()
    write_json(out/'run_config.json', cfg.as_dict())
    all_tests, all_scores, all_mechanisms, all_isotropy = [], [], [], []
    with Progress(out/'progress.json', kind='paper_followup', echo=True) as progress:
        progress.update(stage='prepare', message='Seed-1 split and SPACE-only taxonomy; plan already frozen')
        prepared = out/'prepared'
        prepared.mkdir(exist_ok=True)
        if (prepared/'corpus_prepared.json').exists():
            df = pd.read_json(prepared/'corpus_prepared.json')
        else:
            raw, audit = load_corpus(args.campaign/'inputs/bagdhara_export_done_2026-03-01_010356.json')
            df = split_corpus(raw, seed=1, fractions=cfg.split_fractions, mode=cfg.split_mode)
            write_json(prepared/'split_audit.json', df.attrs['split_report'])
            df, label_report, artifacts = build_labels(df, seed=0, k=12, backend=cfg.label_backend,
                                                       encoder=cfg.label_encoder, revision=cfg.label_revision)
            write_json(prepared/'label_audit.json', label_report)
            np.save(prepared/'taxonomy_centers.npy', artifacts['kmeans'].cluster_centers_)
            df.to_json(prepared/'corpus_prepared.json', orient='records', force_ascii=False)
            del artifacts
        for number, name in enumerate(MODELS, 1):
            slug = model_slug(name)
            dest = out/slug
            dest.mkdir(exist_ok=True)
            marker = dest/'complete.json'
            if marker.exists():
                all_tests += pd.read_csv(dest/'hypotheses.csv').to_dict('records')
                all_scores += pd.read_csv(dest/'scores.csv').to_dict('records')
                all_mechanisms += pd.read_csv(dest/'mechanisms.csv').to_dict('records')
                if (dest/'isotropy.csv').exists():
                    all_isotropy += pd.read_csv(dest/'isotropy.csv').to_dict('records')
                continue
            progress.update(stage='load_cache', model=name, completed=number-1, total=6,
                            message='Read-only feature cache; real GPU readouts')
            old = args.campaign/'primary'/slug/'split_0/labels_0'
            items = pd.read_json(old/'items.json')
            columns = ['split', 'component_id', 'F', 'S', 'cp_transparency', 'transparent', 'opaque']
            lookup = df.set_index('idiom_id')
            assert set(items.idiom_id) <= set(lookup.index)
            for column in columns:
                items[column] = items.idiom_id.map(lookup[column])
            revision = json.loads((old/slug/'model_revision.json').read_text())
            leaf = Path(revision.get('cache_reused_from') or
                        str(args.campaign/'primary'/slug/'cache'/revision['fingerprint']))
            pooling = pd.read_csv(leaf/'pooling.csv')
            assert pooling.item_id.tolist() == items.item_id.tolist()
            manifest = json.loads((leaf/'manifest.json').read_text())
            features = np.load(leaf/'features.npy', mmap_mode='r', allow_pickle=False)
            assert list(features.shape) == manifest['shape']
            valid = pooling.valid.astype(str).str.lower().eq('true').to_numpy()
            frame = items.loc[valid].reset_index(drop=True)
            frame['token_count'] = pooling.loc[valid, 'token_count'].to_numpy()
            del items
            selection = []
            selected_file = dest/'layer_selection.json'
            if selected_file.exists():
                selected = json.loads(selected_file.read_text())['layer']
            else:
                for li, layer in enumerate(manifest['layers']):
                    progress.update(stage='validation_layer_selection', model=name, layer=layer,
                                    message='Select by native F COV validation accuracy only')
                    raw = np.asarray(features[valid, li, :], dtype=float)
                    assert np.isfinite(raw).all()
                    pre = fitted_preprocessor(raw[frame.split.eq('space')])
                    z = pre.transform(raw)
                    study = LayerStudy(z, frame, cfg, dict(model=name, seed=1, label_seed=0, layer=layer),
                                       checkpoint_dir=dest/'checkpoints'/str(layer))
                    vals = []
                    for rank in cfg.ranks:
                        ev = study._evaluate_sub('bn-fig', 'F', 'bn-fig', 'F', 'COV', rank)
                        if ev is not None:
                            vals.append(float(ev['validation_correct'].mean()))
                    selection.append((float(np.mean(vals)), layer))
                    del study, z, raw
                _, selected = sorted(selection, key=lambda x: (-x[0], x[1]))[0]
                write_json(selected_file, dict(layer=selected, candidates=selection,
                                              selected_using='validation_only', method='COV'))
            li = manifest['layers'].index(selected)
            raw = np.asarray(features[valid, li, :], dtype=float)
            pre = fitted_preprocessor(raw[frame.split.eq('space')])
            z = pre.transform(raw)
            del raw
            study = LayerStudy(z, frame, cfg, dict(model=name, seed=1, label_seed=0, layer=selected),
                               checkpoint_dir=dest/'checkpoints'/str(selected), progress=progress)
            tests, scores = e1_observations(study, dest)
            if name in H1_MODELS:
                inf, _ = study.inference()
                inf.to_csv(dest/'COV_nulls.csv', index=False)
                for row in inf.itertuples():
                    assert row.RAND_draws == 1999 and row.PERM_draws == 1999
                    for null in ['RAND', 'PERM']:
                        tests.append(dict(model=name, concept=row.concept, hypothesis='H1_'+null,
                                          estimate=row.SSI, ci_lo=row.ci_lo, ci_hi=row.ci_hi,
                                          p=getattr(row, null+'_p'), n=row.n, n_components=row.n_components,
                                          cross_accuracy=row.R_ep, native_accuracy=row.R_pp, floor=row.R0,
                                          null_mean_accuracy=getattr(row, null+'_R_mean'),
                                          movable_fraction=row.permutation_movable_fraction,
                                          null_draws=getattr(row, null+'_draws')))
            progress.update(stage='mechanism_checks', model=name, message='E2 regularization and E3 fixed-image covariance')
            mech = mechanisms(study, dest)
            iso = isotropy(study, dest) if name == 'xlm-roberta-large' else []
            pd.DataFrame(tests).to_csv(dest/'hypotheses.csv', index=False)
            pd.DataFrame(scores).to_csv(dest/'scores.csv', index=False)
            write_json(marker, dict(model=name, layer=selected, status='completed',
                                   completed_utc=datetime.now(timezone.utc).isoformat()))
            all_tests += tests
            all_scores += scores
            all_mechanisms += mech
            all_isotropy += iso
            pd.DataFrame(all_tests).to_csv(out/'hypotheses_partial.csv', index=False)
            progress.update(stage='model_complete', model=name, completed=number, total=6,
                            message='Item predictions, hypotheses and mechanism diagnostics saved')
            del study, z, frame, features
        tests = pd.DataFrame(all_tests)
        assert len(tests) == 48, len(tests)
        tests['p_holm_48'] = holm(tests.p.to_numpy(), family_size=48)
        tests['reject_holm_05'] = tests.p_holm_48.le(.05)
        tests.to_csv(out/'hypotheses.csv', index=False)
        pd.DataFrame(all_scores).to_csv(out/'scores.csv', index=False)
        pd.DataFrame(all_mechanisms).to_csv(out/'mechanisms.csv', index=False)
        pd.DataFrame(all_isotropy).to_csv(out/'isotropy.csv', index=False)
        portable = REPO/'results'/'replication'
        portable.mkdir(parents=True, exist_ok=True)
        for name in ['hypotheses.csv', 'scores.csv', 'mechanisms.csv', 'isotropy.csv', 'PLAN_FROZEN.json']:
            shutil.copy2(out/name, portable/name)
        for name in ['split_audit.json', 'label_audit.json']:
            shutil.copy2(prepared/name, portable/name)
        selections = {name: json.loads((out/model_slug(name)/'layer_selection.json').read_text()) for name in MODELS}
        write_json(portable/'layer_selection.json', selections)
        write_json(out/'completed.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                    hypotheses=48, family='one new Holm-48 internal-replication family',
                    converged_readouts=int(pd.DataFrame(all_scores).converged.sum())))


if __name__ == '__main__':
    main()
