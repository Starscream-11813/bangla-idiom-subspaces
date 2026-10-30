"""X1: strict zero-shot readout -- no Bangla labels anywhere in the pipeline.

Motivation: the published readouts train the probe on labelled Bangla PROBE
items, so they show that English directions carry Bangla class information, not
that an English classifier transfers. Here the subspace is fitted on English
SPACE (as before), the probe is trained on *English* PROBE items in that
subspace (C tuned on English VALIDATION), and it is applied unchanged to Bangla
TEST items. Two centrings are reported: ``source`` (Bangla inputs centred with
the English estimator's mean; nothing Bangla is used) and ``per_condition``
(each condition centred with its own unlabelled SPACE mean, a standard
label-free adaptation). Zero-shot SSI divides the zero-shot gain over the floor
by the supervised native gain (the published denominator); nulls are RAND
subspaces and permuted-label estimators read by the same English-trained probe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug

from .common import (Log, cell_tables, classes_in, compact_readouts, concat, erase_readouts, finish, labels,
                     load_cell, make_study, rank_for)
from .metrics import METRICS, metric_table, metric_values, paired, ssi
from .nulls import majority_label

PAIRS = [("en-idiom", "bn-fig"), ("en-para", "bn-para"), ("en-trans", "bn-fig")]
METHODS = ["MLR", "COV", "LEACE", "PCA"]
NULL_METHODS = ["MLR", "COV", "LEACE"]


def _shift(study, source, target, centering):
    """Offset subtracted from target rows; per_condition moves the target SPACE mean onto the source's."""
    if centering == "source":
        return np.zeros(study.X.shape[1])
    mean = lambda c: study.X[(study._condition == c) & (study._split == "space")].mean(0)
    return mean(target) - mean(source)


def _zero_shot(study, subs, rows, y, shift, Cs):
    tr, va, te = rows
    Xtr, Xva, Xte = study.X[tr], study.X[va], study.X[te] - shift
    keep = [j for j, s in enumerate(subs) if s is not None]
    results = compact_readouts(study, [subs[j].project(Xtr) for j in keep], [subs[j].project(Xva) for j in keep],
                               [subs[j].project(Xte) for j in keep], y, rows, Cs) if keep else []
    out = [None] * len(subs)
    for j, r in zip(keep, results):
        out[j] = r
    return out


def _ambient(study, rows, y, shift, Cs):
    tr, va, te = rows
    D = study.X.shape[1]
    return erase_readouts(study, study.X[tr], study.X[va], study.X[te] - shift, [np.zeros((D, 0))],
                          [np.zeros((D, 0))], y, rows, Cs)[0]


def _null_predictions(study, source, concept, method, rank, draws, permutation, rows, y, shift, Cs, chunk=128):
    """Zero-shot predictions of RAND (``method='RAND'``) or permuted-label estimators."""
    out = []
    for start in range(0, len(draws), chunk):
        part = draws[start:start + chunk]
        if permutation and method == "MLR":
            study._prefetch_permutation_estimators([(source, concept, method, rank, d) for d in part])
        subs = [study.sub(source, concept, method, rank, permutation=permutation, draw=d) for d in part]
        out += [r["predicted"] for r in _zero_shot(study, subs, rows, y, shift, Cs) if r is not None]
        for d in part:
            study._forget_draw(source, concept, method, rank, permutation, d, None)
    return np.asarray(out)


def run_cell(ctx, study, log):
    rand = int(ctx.option("zs_rand_draws", 999))
    perm = int(ctx.option("zs_perm_draws", 999))
    # Permuted-label COV fits are closed form; MLR refits (and LEACE's D x D whitening
    # per draw) are slow, so they are opt-in. RAND covers every method.
    perm_methods = {"COV"} | ({"MLR"} if ctx.option("perm_mlr", False) else set())
    Cs = study.cfg.probe_Cs
    rows_out, items_out = [], []
    for source, target in PAIRS:
        if not {source, target} <= set(study._condition):
            continue
        planned = (source, target) == ("en-idiom", "bn-fig")
        for concept in ["F", "S"]:
            rank = rank_for(study, concept)
            y_all = labels(study, concept)
            zs_rows = (study.idx(source, "probe", concept), study.idx(source, "validation", concept),
                       study.idx(target, "test", concept))
            en_rows = (study.idx(source, "probe", concept), study.idx(source, "validation", concept),
                       study.idx(source, "test", concept))
            if min(map(len, zs_rows)) == 0 or rank is None:
                continue
            test = zs_rows[2]
            y = y_all[test]
            classes = classes_in(y)
            groups = study.items.iloc[test].component_id.astype(str).to_numpy()
            specs = []
            for method in METHODS:
                specs += [(target, concept, source, concept, method, rank, "project", False, 0, None, True),
                          (target, concept, target, concept, method, rank, "project", False, 0, None, True)]
            specs.append((target, concept, target, concept, "AMBIENT", study.X.shape[1], "project", False, 0,
                          None, True))
            study._prefetch(specs, f"X1 supervised {source}->{target} {concept}")
            bn_floor = np.full(len(y), majority_label(study, target, concept))
            en_floor_label = majority_label(study, source, concept)
            floor_m = metric_values(y, bn_floor, classes)
            predictions = {"floor": bn_floor}
            for centering in ["source", "per_condition"]:
                shift = _shift(study, source, target, centering)
                nulls = {}
                if planned and rand > 0:
                    log(f"{study.meta['model']} {concept} {centering}: {rand} RAND zero-shot nulls")
                    nulls["RAND"] = _null_predictions(study, source, concept, "RAND", rank, list(range(1, rand + 1)),
                                                      False, zs_rows, y_all, shift, Cs)
                for method in METHODS + ["AMBIENT"]:
                    if method == "AMBIENT":
                        zs = _ambient(study, zs_rows, y_all, shift, Cs)
                        en = _ambient(study, en_rows, y_all, 0., Cs) if centering == "source" else None
                        native = study._evaluate_sub(target, concept, target, concept, "AMBIENT", study.X.shape[1])
                        cross = None
                    else:
                        sub = study.sub(source, concept, method, rank)
                        if sub is None:
                            continue
                        zs = _zero_shot(study, [sub], zs_rows, y_all, shift, Cs)[0]
                        en = _zero_shot(study, [sub], en_rows, y_all, 0., Cs)[0] if centering == "source" else None
                        cross = study._evaluate_sub(target, concept, source, concept, method, rank)
                        native = study._evaluate_sub(target, concept, target, concept, method, rank)
                    assert np.array_equal(zs["test_idx"], native["test_idx"])
                    native_m = metric_values(y, native["predicted"], classes)
                    zs_m = metric_values(y, zs["predicted"], classes)
                    row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                               source=source, target=target, concept=concept, method=method, centering=centering,
                               M=rank if method != "AMBIENT" else study.X.shape[1], n=len(y),
                               n_components=len(np.unique(groups)), K_test=len(classes),
                               english_majority_label=int(en_floor_label),
                               english_majority_accuracy_on_target=float(np.mean(y == en_floor_label)),
                               zs_probe_converged=bool(zs.get("probe_converged", True)))
                    cross_m = metric_values(y, cross["predicted"], classes) if cross is not None else None
                    for m in METRICS:
                        row[f"zs_{m}"] = zs_m[m]
                        row[f"native_{m}"] = native_m[m]
                        row[f"floor_{m}"] = floor_m[m]
                        row[f"SSI_zs_{m}"] = float(ssi(zs_m[m], native_m[m], floor_m[m], study.cfg.min_den))
                        if cross_m is not None:
                            row[f"supervised_cross_{m}"] = cross_m[m]
                            row[f"SSI_supervised_{m}"] = float(ssi(cross_m[m], native_m[m], floor_m[m],
                                                                   study.cfg.min_den))
                    if en is not None:
                        en_y = y_all[en["test_idx"]]
                        row["english_in_language_acc"] = float(np.mean(en["predicted"] == en_y))
                        row["english_in_language_floor"] = float(np.mean(en_y == en_floor_label))
                    above = paired(np.asarray(zs["predicted"] == y, float) - (bn_floor == y), groups)
                    row.update(above_floor_pp=100 * above["estimate"], above_floor_ci_lo_pp=100 * above["ci_lo"],
                               above_floor_ci_hi_pp=100 * above["ci_hi"], above_floor_p=above["p"])
                    if cross is not None:
                        gap = paired(np.asarray(cross["predicted"] == y, float) - (zs["predicted"] == y), groups)
                        row.update(supervised_minus_zs_pp=100 * gap["estimate"],
                                   supervised_minus_zs_ci_lo_pp=100 * gap["ci_lo"],
                                   supervised_minus_zs_ci_hi_pp=100 * gap["ci_hi"])
                    if planned and method in NULL_METHODS:
                        if method in perm_methods and perm > 0:
                            log(f"{study.meta['model']} {concept} {method} {centering}: {perm} PERM zero-shot nulls")
                            nulls["PERM_" + method] = _null_predictions(study, source, concept, method, rank,
                                                                        list(range(1, perm + 1)), True, zs_rows,
                                                                        y_all, shift, Cs)
                        for kind, preds in [("RAND", nulls.get("RAND")), ("PERM", nulls.get("PERM_" + method))]:
                            if preds is None or not len(preds):
                                continue
                            table = metric_table(y, preds, classes)
                            for m in METRICS:
                                values = table[m][:, 0]
                                null = ssi(values, native_m[m], floor_m[m], study.cfg.min_den)
                                null = null[np.isfinite(null)]
                                obs = row[f"SSI_zs_{m}"]
                                row[f"{kind}_zs_p_{m}"] = (float((1 + np.sum(null >= obs)) / (1 + len(null)))
                                                           if len(null) and np.isfinite(obs) else np.nan)
                                row[f"{kind}_zs_SSI_mean_{m}"] = float(null.mean()) if len(null) else np.nan
                                row[f"{kind}_zs_{m}_mean"] = float(values.mean())
                            row[f"{kind}_zs_draws"] = int(len(preds))
                    predictions[f"{method}_{centering}_zs"] = zs["predicted"]
                    if centering == "source":
                        predictions[f"{method}_native"] = native["predicted"]
                        if cross is not None:
                            predictions[f"{method}_supervised_cross"] = cross["predicted"]
                    rows_out.append(row)
            items = study.items.iloc[test][["item_id", "idiom_id", "component_id"]].reset_index(drop=True)
            items.insert(0, "target", target)
            items.insert(0, "source", source)
            items.insert(0, "concept", concept)
            items["y_true"] = y
            for name, values in predictions.items():
                items[name] = np.asarray(values)
            items_out.append(items)
    return pd.DataFrame(rows_out), concat(items_out)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    tables = []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                conditions = sorted({c for pair in PAIRS for c in pair})
                cell = load_cell(ctx, name, split, conditions=conditions)
                study = make_study(ctx, cell, "zero_shot")
                summary, items = run_cell(ctx, study, log)
                summary["selected_layer"] = cell.selected_layer
                saved = cell_tables(ctx, key, {"summary.csv": summary, "items.csv.gz": items})
                del study, cell
            else:
                log(f"restored {key}")
            tables.append(saved["summary.csv"])
    finish(ctx, {"summary.csv": concat(tables)},
           dict(zs_rand_draws=int(ctx.option("zs_rand_draws", 999)), zs_perm_draws=int(ctx.option("zs_perm_draws", 999)),
                perm_mlr=bool(ctx.option("perm_mlr", False)), pairs=PAIRS, methods=METHODS + ["AMBIENT"]))
    log("finished")
