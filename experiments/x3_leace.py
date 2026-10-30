"""X3: what drives LEACE's cross-lingual retention deficit (shrinkage, sample size, probe penalty).

Motivation: the paper rules out a simple English/Bangla covariance mismatch
(E3) but cannot separate (i) the covariance-dependent probe metric,
(ii) finite-sample error in the whitening Sigma^+ and (iii) other covariance
structure. Three ablations, all with English-fitted (en-idiom SPACE) erasers
read and erased on Bangla idioms in context (bn-fig), Bangla-fitted ones as the
native reference, and English-on-English as the in-distribution check:

ridge     LEACE whitening ridge from 0 (Moore-Penrose, published) to 100 x trace
          scale, plus Ledoit-Wolf; as ridge grows LEACE tends to the orthogonal
          COV projection, so the path interpolates oblique -> orthogonal.
sigma_n   the English COV image Q is held fixed (as in E3) while Sigma comes from
          25% / 50% (3 component subsamples each) or 100% of English SPACE, or an
          extended English sample (SPACE + PROBE + VALIDATION inputs, labels
          unused). If the deficit shrinks as Sigma is estimated from more rows,
          finite-sample whitening error contributes; if it does not, the oblique
          metric itself is the likelier cause.
probe_C   fixed probe penalties C in {0.01 ... 100} for COV and LEACE reading,
          to separate the penalty part of the metric from the function class.

At full supported rank (M = K-1) whitening by the pooled within-class covariance
gives the same LEACE operator (Woodbury), so it is checked numerically, not as an
ablation (``within_class_operator_difference``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import Subspace, _symmetric_roots, fit_estimator, ledoit_wolf_ridge

from .common import (Log, cell_tables, compact_readouts, concat, erase_readouts, finish, labels, load_cell,
                     make_study, rank_for)
from .metrics import paired

RIDGES = [0., 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., "ledoit_wolf"]
FRACTIONS = [.25, .5]
PROBE_CS = [.01, .1, 1., 10., 100.]


def _rows(study, condition, concept):
    return tuple(study.idx(condition, s, concept) for s in ["probe", "validation", "test"])


def _read(study, subs, rows, y, Cs=None):
    tr, va, te = rows
    return compact_readouts(study, [s.project(study.X[tr]) for s in subs], [s.project(study.X[va]) for s in subs],
                            [s.project(study.X[te]) for s in subs], y, rows, Cs)


def _erase(study, subs, rows, y, Cs=None):
    tr, va, te = rows
    return erase_readouts(study, study.X[tr], study.X[va], study.X[te], [s.input_factor for s in subs],
                          [s.basis for s in subs], y, rows, Cs)


def _fixed_image(cov, q, rows_X, ridge):
    """LEACE operator with image Q and the covariance of ``rows_X`` (E3 construction)."""
    Xc = rows_X - rows_X.mean(0)
    sigma = Xc.T @ Xc / len(Xc)
    value = ledoit_wolf_ridge(Xc) if ridge == "ledoit_wolf" else float(ridge)
    _, w = _symmetric_roots(sigma, value)
    invq = w @ (w @ q)
    factor = invq @ np.linalg.pinv(q.T @ invq, rcond=1e-12)
    identity_error = float(np.linalg.norm(q.T @ factor - np.eye(q.shape[1])))
    return Subspace(cov.center, q, q.shape[1], "LEACE", q.shape[1], factor), value, identity_error


def _operator_distance(sub, reference):
    P = sub.input_factor @ sub.basis.T
    R = reference.input_factor @ reference.basis.T
    return float(np.linalg.norm(P - R) / max(np.linalg.norm(R), 1e-12))


def _subsample(study, condition, concept, fraction, rep):
    ix = study.idx(condition, "space", concept)
    components = study.items.iloc[ix].component_id.astype(str).to_numpy()
    units = np.unique(components)
    rng = np.random.default_rng(7919 * int(study.meta["seed"]) + 101 * rep + int(fraction * 1000))
    chosen = rng.choice(units, size=max(2, int(round(fraction * len(units)))), replace=False)
    return ix[np.isin(components, chosen)]


def run_cell(ctx, study, log):
    out, items = [], {}
    options = study._estimator_options()
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        y = labels(study, concept)
        bn, en = _rows(study, "bn-fig", concept), _rows(study, "en-idiom", concept)
        en_space, bn_space = study.idx("en-idiom", "space", concept), study.idx("bn-fig", "space", concept)
        k_space = int(len(np.unique(y[en_space])))
        base = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                    concept=concept, M=rank, K_space=k_space, full_rank=rank == k_space - 1,
                    n_target_test=len(bn[2]))
        test_y = y[bn[2]]
        groups = study.items.iloc[bn[2]].component_id.astype(str).to_numpy()
        ambient = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, "AMBIENT", study.X.shape[1])
        items[(concept, "AMBIENT_native")] = ambient["predicted"]
        cov_en = study.sub("en-idiom", concept, "COV", rank)
        cov_bn = study.sub("bn-fig", concept, "COV", rank)

        def record(block, setting, estimator, source, target, representation, ev, **extra):
            yy = y[ev["test_idx"]]
            row = dict(base, block=block, setting=str(setting), estimator=estimator, source=source, target=target,
                       representation=representation, accuracy=float(np.mean(ev["predicted"] == yy)),
                       floor=float(np.mean(ev["floor"])), n=len(yy), C=ev.get("C"),
                       probe_converged=bool(ev.get("probe_converged", True)), **extra)
            out.append(row)
            if target == "bn-fig":
                items[(concept, f"{block}|{setting}|{estimator}|{source}|{representation}")] = ev["predicted"]

        # COV references (the ridge -> infinity limit of LEACE).
        for (source, sub) in [("en-idiom", cov_en), ("bn-fig", cov_bn)]:
            r, e = _read(study, [sub], bn, y)[0], _erase(study, [sub], bn, y)[0]
            record("reference", "COV", "COV", source, "bn-fig", "project", r)
            record("reference", "COV", "COV", source, "bn-fig", "erase", e)
        record("reference", "COV", "COV", "en-idiom", "en-idiom", "project", _read(study, [cov_en], en, y)[0])
        record("reference", "COV", "COV", "en-idiom", "en-idiom", "erase", _erase(study, [cov_en], en, y)[0])
        record("reference", "AMBIENT", "AMBIENT", "bn-fig", "bn-fig", "project", ambient)

        # (a) Ridge path for LEACE whitening, English- and Bangla-fitted.
        log(f"{study.meta['model']} {concept}: LEACE ridge path")
        fitted = {}
        for ridge in RIDGES:
            for source, ix in [("en-idiom", en_space), ("bn-fig", bn_space)]:
                est = fit_estimator(study.X[ix], y[ix], "LEACE", rank, seed=study.meta["seed"],
                                    **dict(options, ridge=ridge))
                fitted[(ridge, source)] = est.at_rank(rank)
        settings = [(r, s) for r in RIDGES for s in ["en-idiom", "bn-fig"]]
        subs = [fitted[k] for k in settings]
        reads, erases = _read(study, subs, bn, y), _erase(study, subs, bn, y)
        for (ridge, source), sub, r, e in zip(settings, subs, reads, erases):
            extra = dict(operator_distance_to_COV=_operator_distance(sub, cov_en if source == "en-idiom" else cov_bn))
            record("ridge", ridge, "LEACE", source, "bn-fig", "project", r, **extra)
            record("ridge", ridge, "LEACE", source, "bn-fig", "erase", e, **extra)
        english = [fitted[(r, "en-idiom")] for r in RIDGES]
        for ridge, r, e in zip(RIDGES, _read(study, english, en, y), _erase(study, english, en, y)):
            record("ridge", ridge, "LEACE", "en-idiom", "en-idiom", "project", r)
            record("ridge", ridge, "LEACE", "en-idiom", "en-idiom", "erase", e)

        # Within-class whitening gives the same full-rank operator (numerical check only).
        Xs, ys = study.X[en_space], y[en_space]
        classes, codes = np.unique(ys, return_inverse=True)
        means = np.stack([Xs[codes == k].mean(0) for k in range(len(classes))])
        residual = Xs - means[codes]
        within = _fixed_image(cov_en, cov_en.basis, Xs, 0.)[0]
        sw = residual.T @ residual / len(residual)
        _, w = _symmetric_roots(sw, 0.)
        invq = w @ (w @ cov_en.basis)
        factor_w = invq @ np.linalg.pinv(cov_en.basis.T @ invq, rcond=1e-12)
        sub_w = Subspace(cov_en.center, cov_en.basis, rank, "LEACE", rank, factor_w)
        within_difference = _operator_distance(sub_w, within)

        # (b) Fixed English image, Sigma from more or fewer English rows.
        log(f"{study.meta['model']} {concept}: fixed-image covariance sample-size curve")
        q = cov_en.basis
        extended = np.concatenate([en_space, en[0], en[1]])
        samples = [(f"space_{f:.2f}_rep{rep}", _subsample(study, "en-idiom", concept, f, rep))
                   for f in FRACTIONS for rep in range(3)]
        samples += [("space_1.00", en_space), ("space+probe+validation", extended)]
        variants = []
        for label, ix in samples:
            for ridge in [0., "ledoit_wolf"]:
                sub, value, identity = _fixed_image(cov_en, q, study.X[ix], ridge)
                variants.append((label, ridge, len(ix), value, identity, sub))
        reference_factor = next(v[5] for v in variants if v[0] == "space+probe+validation" and v[1] == 0.).input_factor
        subs = [v[5] for v in variants]
        for (label, ridge, n_rows, value, identity, sub), r, e in zip(variants, _read(study, subs, bn, y),
                                                                       _erase(study, subs, bn, y)):
            extra = dict(sigma_rows=n_rows, ridge_value=value, idempotence_error=identity, ridge=str(ridge),
                         factor_difference_to_extended=float(np.linalg.norm(sub.input_factor - reference_factor)
                                                             / max(np.linalg.norm(reference_factor), 1e-12)),
                         within_class_operator_difference=within_difference)
            record("sigma_n", f"{label}|{ridge}", "LEACE_fixed_image", "en-idiom", "bn-fig", "project", r, **extra)
            record("sigma_n", f"{label}|{ridge}", "LEACE_fixed_image", "en-idiom", "bn-fig", "erase", e, **extra)

        # (c) Probe penalty sweep for reading through COV and LEACE (ridge 0).
        log(f"{study.meta['model']} {concept}: probe-penalty sweep")
        pairs = [("COV", "en-idiom", cov_en), ("COV", "bn-fig", cov_bn),
                 ("LEACE", "en-idiom", fitted[(0., "en-idiom")]), ("LEACE", "bn-fig", fitted[(0., "bn-fig")])]
        for C in PROBE_CS:
            results = _read(study, [p[2] for p in pairs], bn, y, Cs=[C])
            for (estimator, source, _), r in zip(pairs, results):
                record("probe_C", C, estimator, source, "bn-fig", "project", r)

        # Paired contrasts on the shared Bangla TEST items.
        def correct(key):
            return (np.asarray(items[(concept, key)]) == test_y).astype(float)

        for ridge in RIDGES:
            k = lambda est, src, rep: f"ridge|{ridge}|{est}|{src}|{rep}"
            cross = correct(k("LEACE", "en-idiom", "project")) - correct("reference|COV|COV|en-idiom|project")
            native = correct(k("LEACE", "bn-fig", "project")) - correct("reference|COV|COV|bn-fig|project")
            test = paired(cross - native, groups)
            out.append(dict(base, block="contrast", setting=str(ridge), estimator="LEACE-COV",
                            source="cross_minus_native", target="bn-fig", representation="project",
                            deficit_pp=100 * test["estimate"], deficit_ci_lo_pp=100 * test["ci_lo"],
                            deficit_ci_hi_pp=100 * test["ci_hi"], n=len(test_y)))
        for label, ridge, *_ in variants:
            cross = correct(f"sigma_n|{label}|{ridge}|LEACE_fixed_image|en-idiom|project") - \
                correct("reference|COV|COV|en-idiom|project")
            test = paired(cross, groups)
            out.append(dict(base, block="contrast", setting=f"{label}|{ridge}", estimator="LEACE_fixed_image-COV",
                            source="en-idiom", target="bn-fig", representation="project",
                            deficit_pp=100 * test["estimate"], deficit_ci_lo_pp=100 * test["ci_lo"],
                            deficit_ci_hi_pp=100 * test["ci_hi"], n=len(test_y)))
    tables = []
    for concept in ["F", "S"]:
        keys = [k for k in items if k[0] == concept]
        if not keys:
            continue
        ix = study.idx("bn-fig", "test", concept)
        t = study.items.iloc[ix][["item_id", "idiom_id", "component_id"]].reset_index(drop=True)
        t.insert(0, "concept", concept)
        t["y_true"] = labels(study, concept)[ix]
        for key in keys:
            t[key[1]] = np.asarray(items[key])
        tables.append(t)
    return pd.DataFrame(out), concat(tables)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    summaries = []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=["en-idiom", "bn-fig"])
                study = make_study(ctx, cell, "leace")
                summary, items = run_cell(ctx, study, log)
                summary["selected_layer"] = cell.selected_layer
                saved = cell_tables(ctx, key, {"summary.csv": summary, "items.csv.gz": items})
                del study, cell
            else:
                log(f"restored {key}")
            summaries.append(saved["summary.csv"])
    finish(ctx, {"summary.csv": concat(summaries)}, dict(ridges=[str(r) for r in RIDGES], fractions=FRACTIONS,
                                                         probe_Cs=PROBE_CS))
    log("finished")
