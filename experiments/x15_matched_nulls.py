"""X15: a covariance-matched null and a label-free PCA reference for the planned and class-mean families.

Motivation: RAND draws Haar-random subspaces, which in anisotropic spaces miss the high-variance directions, so
RAND sits near the floor; and the label-free English PCA subspace of the same rank reads Bangla F about as well as the
planned MLR subspace in four encoders. Two questions follow: do the English estimators read Bangla classes better than
(i) random directions with English's variance profile (RANDCOV, see matched.py) and (ii) English's top principal
directions (PCA)?

tests      for each split, target (bn-fig: idioms in context, the planned target; bn-idiom: the idiom alone),
           estimator (MLR at the default penalty, COV), concept (F at M = 11, S at M = 3) and metric (accuracy, balanced
           accuracy, macro-F1): the observed SSI against 1,999 RANDCOV draws (Monte Carlo p with +1, Holm within
           split x target x estimator x metric over 12 tests); the PCA reference's accuracy and SSI; and the paired
           estimator-minus-PCA difference on the same TEST items (accuracy: component bootstrap CI and one-sided
           component wild-sign p, Holm over the 12 tests; balanced accuracy: component bootstrap CI).
Gate       on split 0 and bn-fig, the MLR cross accuracy reproduces the published planned test and the COV and PCA cross
           accuracies reproduce the published estimator table (verify3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from experiments.common import Log, classes_in, concat, labels, load_cell, make_study, rank_for, readout_indices
from experiments.metrics import METRICS, component_weights, metric_table, metric_values, paired, ssi
from experiments.nulls import majority_label

from .common3 import SOURCE, cell_tables, finish
from .matched import base_seed, matched_null_predictions, matched_pvalues

TARGETS = ["bn-fig", "bn-idiom"]
METHODS = ["MLR", "COV"]


def balanced_difference(y, a, b, groups, classes, n_boot, seed):
    """Balanced accuracy of a minus b, with a component bootstrap interval."""
    point = metric_table(y, np.stack([a, b]), classes)["bal"][:, 0]
    weights = component_weights(groups, n_boot, seed)
    boot = metric_table(y, np.stack([a, b]), classes, weights)["bal"]
    diff = boot[0] - boot[1]
    lo, hi = np.quantile(diff, [.025, .975])
    return float(point[0] - point[1]), float(lo), float(hi)


def run_cell(ctx, study, log):
    draws = int(ctx.option("cov_draws", 1999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    seed = int(study.meta["seed"])
    rows = []
    for target in TARGETS:
        if target not in set(study._condition):
            continue
        for concept in ["F", "S"]:
            rank = rank_for(study, concept)
            if rank is None:
                continue
            specs = [(target, concept, SOURCE, concept, m, rank, "project", False, 0, None, True)
                     for m in METHODS + ["PCA"]]
            specs += [(target, concept, target, concept, m, rank, "project", False, 0, None, True) for m in METHODS]
            study._prefetch(specs, f"X15 observed {target} {concept}")
            idx = readout_indices(study, target, concept)
            y_all = labels(study, concept)
            test = np.asarray(idx[2])
            y = y_all[test]
            classes = classes_in(y)
            groups = study.items.iloc[test].component_id.astype(str).to_numpy()
            floor = np.full(len(y), majority_label(study, target, concept))
            floor_m = metric_values(y, floor, classes)
            log(f"{study.meta['model']} {target} {concept}: {draws} RANDCOV draws", target=target, concept=concept)
            null = matched_null_predictions(study, study.idx(SOURCE, "space", concept), rank, idx, y_all,
                                            range(1, draws + 1), base_seed(study, concept))
            pca = study._evaluate_sub(target, concept, SOURCE, concept, "PCA", rank)
            pca_pred = np.asarray(pca["predicted"]) if pca is not None else None
            pca_m = metric_values(y, pca_pred, classes) if pca_pred is not None else None
            for method in METHODS:
                cross = study._evaluate_sub(target, concept, SOURCE, concept, method, rank)
                native = study._evaluate_sub(target, concept, target, concept, method, rank)
                if cross is None or native is None:
                    continue
                assert np.array_equal(np.asarray(cross["test_idx"]), test)
                cross_pred, native_pred = np.asarray(cross["predicted"]), np.asarray(native["predicted"])
                cross_m, native_m = metric_values(y, cross_pred, classes), metric_values(y, native_pred, classes)
                observed = {m: float(ssi(cross_m[m], native_m[m], floor_m[m], study.cfg.min_den)) for m in METRICS}
                row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                           method=method, source=SOURCE, target=target, M=rank, n=len(y),
                           n_components=len(np.unique(groups)), K_test=len(classes),
                           probe_converged_cross=bool(cross.get("probe_converged", True)))
                for m in METRICS:
                    row[m] = cross_m[m]
                    row[f"{m}_native"] = native_m[m]
                    row[f"{m}_floor"] = floor_m[m]
                    row[f"SSI_{m}"] = observed[m]
                row.update(matched_pvalues(y, null, native_m, floor_m, observed, classes, study.cfg.min_den))
                if pca_m is not None:
                    for m in METRICS:
                        row[f"PCA_{m}"] = pca_m[m]
                        row[f"PCA_SSI_{m}"] = float(ssi(pca_m[m], native_m[m], floor_m[m], study.cfg.min_den))
                    test_acc = paired((cross_pred == y).astype(float) - (pca_pred == y), groups)
                    row.update(minus_PCA_acc_pp=100 * test_acc["estimate"], minus_PCA_acc_ci_lo_pp=100 * test_acc["ci_lo"],
                               minus_PCA_acc_ci_hi_pp=100 * test_acc["ci_hi"], minus_PCA_acc_p=test_acc["p"])
                    est, lo, hi = balanced_difference(y, cross_pred, pca_pred, groups, classes, n_boot, seed)
                    row.update(minus_PCA_bal_pp=100 * est, minus_PCA_bal_ci_lo_pp=100 * lo,
                               minus_PCA_bal_ci_hi_pp=100 * hi)
                rows.append(row)
    return {"tests.csv": pd.DataFrame(rows)}


def adjust(tests):
    """Holm within split x target x estimator (x metric) over the 12 model x concept tests."""
    tests = tests.copy()
    for _, index in tests.groupby(["split", "target", "method"]).groups.items():
        for m in METRICS:
            column = f"RANDCOV_p_{m}"
            if column in tests:
                tests.loc[index, column + "_holm"] = holm(tests.loc[index, column].to_numpy(), family_size=len(index))
        if "minus_PCA_acc_p" in tests:
            tests.loc[index, "minus_PCA_acc_p_holm"] = holm(tests.loc[index, "minus_PCA_acc_p"].to_numpy(),
                                                            family_size=len(index))
    return tests


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    parts = []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=[SOURCE] + TARGETS)
                study = make_study(ctx, cell, "matched_nulls")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log))
                del study, cell
            else:
                log(f"restored {key}")
            parts.append(saved.get("tests.csv"))
    finish(ctx, {"tests.csv": adjust(concat(parts))},
           dict(source=SOURCE, targets=TARGETS, methods=METHODS, cov_draws=int(ctx.option("cov_draws", 1999))))
    log("finished")
