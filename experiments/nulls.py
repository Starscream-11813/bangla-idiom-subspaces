"""Calibrated SSI tests that keep every null draw's predicted labels.

``LayerStudy.inference`` keeps only correctness. This module replays exactly its
null design -- RAND subspaces with seeds ``meta.seed + 1009*draw``, one PCA
baseline and component-bundle label permutations of the estimator's SPACE labels
(``_permuted_labels``) -- through the same batched, checkpointed readouts, and
scores accuracy, balanced accuracy and macro-F1 on every draw.
"""
from __future__ import annotations

import numpy as np

from idiom_subspace.geometry import paired_cluster_bootstrap, shared_subspace_index

from .common import classes_in, labels
from .metrics import METRICS, bootstrap_ssi, metric_values, null_pvalues, ssi


def null_specs(target, concept, source, method, rank, rand_draws, perm_draws, pca=True):
    specs = [(target, concept, source, concept, "RAND", rank, "project", False, d, None, False)
             for d in range(1, rand_draws + 1)]
    if pca:
        specs.append((target, concept, source, concept, "PCA", rank, "project", False, 1, None, False))
    specs += [(target, concept, source, concept, method, rank, "project", True, d, None, False)
              for d in range(1, perm_draws + 1)]
    return specs


def permutation_plan(study, source, concept, requested):
    """Number of PERM draws, following inference(): component units with enough movable mass."""
    space = study.idx(source, "space", concept)
    y = study.items[concept].to_numpy()[space]
    _, info = study._permuted_labels(source, concept, space, y, 0)
    minimum = getattr(study.cfg, "min_permutation_movable_fraction", .5)
    movable = info["permutation_movable_fraction"] >= minimum
    component = info["permutation_unit"] == "component_id"
    return (requested if movable and component else 0), dict(info, PERM_inferential=bool(movable and component))


def predictions(study, target, concept, source, method, rank, kind, draws):
    out = []
    for draw in draws:
        ev = study._evaluate_sub(target, concept, source, concept, method if kind == "PERM" else kind, rank,
                                 permutation=kind == "PERM", draw=draw, need_margin=False)
        if ev is not None:
            out.append(np.asarray(ev["predicted"]))
    return np.asarray(out) if out else np.empty((0, 0), dtype=int)


def majority_label(study, target, concept):
    y = labels(study, concept)
    train = study.idx(target, "probe", concept)
    values, counts = np.unique(y[train], return_counts=True)
    return values[int(np.argmax(counts))]


def observed_test(study, concept, method, source, target, rank, n_boot, seed):
    """Cross/native readouts with all three metrics' SSIs and bootstrap intervals (no nulls)."""
    study._prefetch([(target, concept, source, concept, method, rank, "project", False, 0, None, True),
                     (target, concept, target, concept, method, rank, "project", False, 0, None, True)],
                    f"{method} {source}->{target} {concept}")
    cross = study._evaluate_sub(target, concept, source, concept, method, rank)
    native = study._evaluate_sub(target, concept, target, concept, method, rank)
    if cross is None or native is None:
        return None, None
    test = np.asarray(cross["test_idx"])
    y = labels(study, concept)[test]
    classes = classes_in(y)
    floor = np.full(len(y), majority_label(study, target, concept))
    groups = study.items.iloc[test].component_id.astype(str).to_numpy()
    row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"], concept=concept,
               method=method, M=rank, source=source, target=target, n=len(y), n_components=len(np.unique(groups)),
               K_test=len(classes))
    row.update(bootstrap_ssi(y, cross["predicted"], native["predicted"], floor, groups, classes, n_boot, seed,
                             study.cfg.min_den))
    return row, dict(test_idx=test, y=y, groups=groups, cross=np.asarray(cross["predicted"]),
                     native=np.asarray(native["predicted"]), floor=floor)


def null_test(study, concept, method, source, target, rank, rand_draws, perm_draws, n_boot, seed,
              pca=True, label="null draws"):
    """Observed cross/native readouts and RAND/PCA/PERM nulls for one planned-style test.

    Returns (row, items, null_arrays): ``row`` holds all three metrics with SSIs,
    bootstrap intervals, null means and Monte Carlo p values; ``items`` the
    item-level predictions; ``null_arrays`` the per-draw metric values.
    """
    perm, info = permutation_plan(study, source, concept, perm_draws)
    observed = [(target, concept, source, concept, method, rank, "project", False, 0, None, True),
                (target, concept, target, concept, method, rank, "project", False, 0, None, True)]
    study._prefetch(observed + null_specs(target, concept, source, method, rank, rand_draws, perm, pca), label)
    cross = study._evaluate_sub(target, concept, source, concept, method, rank)
    native = study._evaluate_sub(target, concept, target, concept, method, rank)
    if cross is None or native is None:
        return None, None, None
    assert np.array_equal(cross["test_idx"], native["test_idx"])
    test = np.asarray(cross["test_idx"])
    y = labels(study, concept)[test]
    classes = classes_in(y)
    majority = majority_label(study, target, concept)
    floor = np.full(len(y), majority)
    assert np.array_equal(floor == y, np.asarray(cross["floor"])), "floor reconstruction failed"
    groups = study.items.iloc[test].component_id.astype(str).to_numpy()
    boot = bootstrap_ssi(y, cross["predicted"], native["predicted"], floor, groups, classes, n_boot, seed,
                         study.cfg.min_den)
    original = paired_cluster_bootstrap(
        dict(cross=np.asarray(cross["correct"]), within=np.asarray(native["correct"]), floor=np.asarray(cross["floor"])),
        groups, lambda a: shared_subspace_index(a["cross"].mean(), a["within"].mean(), a["floor"].mean(),
                                                study.cfg.min_den), n_boot, study.meta["seed"])
    native_m = {m: boot[m + "_native"] for m in METRICS}
    floor_m = {m: boot[m + "_floor"] for m in METRICS}
    observed_ssi = {m: boot["SSI_" + m] for m in METRICS}
    row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
               concept=concept, method=method, M=rank, source=source, target=target, n=len(y),
               n_components=len(np.unique(groups)), K_test=len(classes), majority_label=int(majority),
               probe_converged_cross=bool(cross.get("probe_converged", True)),
               probe_converged_native=bool(native.get("probe_converged", True)),
               SSI_acc_ci_lo_original_bootstrap=original["ci_lo"],
               SSI_acc_ci_hi_original_bootstrap=original["ci_hi"],
               **{k: v for k, v in info.items() if k.startswith("permutation") or k == "PERM_inferential"})
    row.update(boot)
    nulls = {}
    for kind, draws in [("RAND", range(1, rand_draws + 1)), ("PERM", range(1, perm + 1))]:
        preds = predictions(study, target, concept, source, method, rank, kind, draws)
        values = null_pvalues(y, preds if len(preds) else None, native_m, floor_m, observed_ssi, classes,
                              study.cfg.min_den)
        row.update({f"{kind}_{k}": v for k, v in values.items()})
        if len(preds):
            from .metrics import metric_table
            table = metric_table(y, preds, classes)
            nulls[kind] = {m: table[m][:, 0] for m in METRICS}
    if pca:
        preds = predictions(study, target, concept, source, method, rank, "PCA", [1])
        if len(preds):
            values = metric_values(y, preds[0], classes)
            for m in METRICS:
                row[f"PCA_{m}"] = values[m]
                row[f"PCA_SSI_{m}"] = float(ssi(values[m], native_m[m], floor_m[m], study.cfg.min_den))
    row["RAND_draws"] = int(rand_draws)
    row["PERM_draws"] = int(perm)
    items = study.items.iloc[test][["item_id", "idiom_id", "component_id"]].reset_index(drop=True)
    items.insert(0, "concept", concept)
    items.insert(1, "method", method)
    items["y_true"] = y
    items["cross_pred"] = np.asarray(cross["predicted"])
    items["native_pred"] = np.asarray(native["predicted"])
    items["floor_pred"] = floor
    return row, items, nulls
