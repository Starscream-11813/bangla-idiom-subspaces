"""X12: is "patterns transfer, filters do not" a dichotomy or a continuum?

Motivation: the transfer claim was tested only at MLR's default C = 1, yet
the strong-penalty limit says that a strongly penalized MLR converges to the COV image, and
validation-selected C closes most of the supervised COV-MLR gap. So a
zero-shot readout through MLR at a small C should transfer too. The same holds for
LEACE, which becomes the orthogonal COV projection as its whitening ridge grows.

fits      English (en-idiom) and Bangla (bn-fig) SPACE, F at M = 11 and S at M = 3:
          MLR at estimator C in C_GRID (the follow-up's E2 grid .001-1, extended both
          ways), LEACE at whitening ridges in RIDGES (x trace scale; large ridges give
          COV), and COV (the C -> 0 limit). The probe C stays the pipeline's.
readouts  supervised cross (Bangla-trained probe through English directions), native
          (Bangla directions; the SSI denominator of each setting), and zero-shot (an
          English-trained probe applied unchanged to Bangla TEST, with each condition
          centred on its own SPACE mean as in X1, and on the English mean), plus the
          English probe's accuracy on English VALIDATION and TEST.
geometry  shared dimensions of each English image with the English and the Bangla COV
          image (the strong-penalty limit: the MLR image approaches the COV image as C falls).
nulls     999 RAND zero-shot draws per encoder x concept x centring (estimator-free,
          so one null serves every setting); Monte Carlo p for each zero-shot SSI.
selection C (or ridge) chosen without Bangla labels by English VALIDATION accuracy of
          the English probe (zero-shot), and by Bangla VALIDATION accuracy of the
          supervised readout (the E2 rule), reported in selected.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import fit_estimator, holm

from experiments.common import Log, compact_readouts, concat, labels, load_cell, make_study, rank_for
from experiments.metrics import metric_values, ssi
from experiments.nulls import majority_label
from experiments.x1_zero_shot import _null_predictions, _shift, _zero_shot

from .alignment import shared_dims
from .common2 import cell_tables, finish

SOURCE, TARGET = "en-idiom", "bn-fig"
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10.)
RIDGES = (0., 1e-2, 1., 100.)
CENTRINGS = ("per_condition", "source")


def _fits(study, concept, rank, options):
    """[(estimator, setting, English subspace, Bangla subspace)] along the penalty and ridge paths, then COV."""
    y = labels(study, concept)
    out = []
    en, bn = study.idx(SOURCE, "space", concept), study.idx(TARGET, "space", concept)
    seed = int(study.meta["seed"])
    for C in C_GRID:
        subs = [fit_estimator(study.X[ix], y[ix], "MLR", rank, seed=seed, **dict(options, C=C)).at_rank(rank)
                for ix in (en, bn)]
        out.append(("MLR", f"C={C:g}", float(C), *subs))
    for ridge in RIDGES:
        subs = [fit_estimator(study.X[ix], y[ix], "LEACE", rank, seed=seed, **dict(options, ridge=ridge)).at_rank(rank)
                for ix in (en, bn)]
        out.append(("LEACE", f"ridge={ridge:g}", float(ridge), *subs))
    out.append(("COV", "COV", np.nan, study.sub(SOURCE, concept, "COV", rank), study.sub(TARGET, concept, "COV", rank)))
    return [f for f in out if f[3] is not None and f[4] is not None and f[3].effective_rank == rank
            and f[4].effective_rank == rank]


def run_cell(ctx, study, log):
    rand = int(ctx.option("zs_rand_draws", 999))
    Cs = study.cfg.probe_Cs
    options = study._estimator_options()
    seed = int(study.meta["seed"])
    rows = []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        y = labels(study, concept)
        bn = tuple(study.idx(TARGET, s, concept) for s in ("probe", "validation", "test"))
        en = tuple(study.idx(SOURCE, s, concept) for s in ("probe", "validation", "test"))
        zs_rows = (en[0], en[1], bn[2])
        y_te, y_en = y[bn[2]], y[en[2]]
        classes = np.unique(y_te)
        floor_pred = np.full(len(y_te), majority_label(study, TARGET, concept))
        floor_m = metric_values(y_te, floor_pred, classes)
        log(f"{study.meta['model']} {concept}: fitting {len(C_GRID)} MLR, {len(RIDGES)} LEACE and COV pairs")
        fits = _fits(study, concept, rank, options)
        cov_en, cov_bn = study.sub(SOURCE, concept, "COV", rank), study.sub(TARGET, concept, "COV", rank)
        supervised = compact_readouts(study, [f[3].project(study.X[bn[0]]) for f in fits],
                                      [f[3].project(study.X[bn[1]]) for f in fits],
                                      [f[3].project(study.X[bn[2]]) for f in fits], y, bn)
        native = compact_readouts(study, [f[4].project(study.X[bn[0]]) for f in fits],
                                  [f[4].project(study.X[bn[1]]) for f in fits],
                                  [f[4].project(study.X[bn[2]]) for f in fits], y, bn)
        english = compact_readouts(study, [f[3].project(study.X[en[0]]) for f in fits],
                                   [f[3].project(study.X[en[1]]) for f in fits],
                                   [f[3].project(study.X[en[2]]) for f in fits], y, en)
        cov_native = native[[i for i, f in enumerate(fits) if f[0] == "COV"][0]]
        cov_native_m = metric_values(y_te, cov_native["predicted"], classes)
        zero_shot, null_ssi = {}, {}
        for centring in CENTRINGS:
            shift = _shift(study, SOURCE, TARGET, centring)
            zero_shot[centring] = _zero_shot(study, [f[3] for f in fits], zs_rows, y, shift, Cs)
            if rand > 0:
                log(f"{study.meta['model']} {concept} {centring}: {rand} RAND zero-shot nulls")
                preds = _null_predictions(study, SOURCE, concept, "RAND", rank, list(range(1, rand + 1)), False,
                                          zs_rows, y, shift, Cs)
                null_ssi[centring] = np.asarray([metric_values(y_te, p, classes)["acc"] for p in preds])
        for i, (method, setting, value, en_sub, bn_sub) in enumerate(fits):
            nat_m = metric_values(y_te, native[i]["predicted"], classes)
            sup_m = metric_values(y_te, supervised[i]["predicted"], classes)
            row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                       method=method, setting=setting, C=value if method == "MLR" else np.nan,
                       ridge=value if method == "LEACE" else np.nan, M=rank, n=len(y_te),
                       shared_dims_with_english_cov=shared_dims(cov_en.basis, en_sub.basis),
                       shared_dims_with_bangla_cov=shared_dims(cov_bn.basis, en_sub.basis),
                       shared_dims_with_own_bangla=shared_dims(bn_sub.basis, en_sub.basis),
                       random_expected_shared_dims=rank ** 2 / study.X.shape[1],
                       floor_acc=floor_m["acc"], native_acc=nat_m["acc"], cross_acc=sup_m["acc"],
                       native_bal=nat_m["bal"], cross_bal=sup_m["bal"], floor_bal=floor_m["bal"],
                       cov_native_acc=cov_native_m["acc"],
                       SSI_supervised_acc=float(ssi(sup_m["acc"], nat_m["acc"], floor_m["acc"], study.cfg.min_den)),
                       SSI_supervised_bal=float(ssi(sup_m["bal"], nat_m["bal"], floor_m["bal"], study.cfg.min_den)),
                       SSI_supervised_vs_cov_acc=float(ssi(sup_m["acc"], cov_native_m["acc"], floor_m["acc"],
                                                           study.cfg.min_den)),
                       cross_above_floor_pp=100 * (sup_m["acc"] - floor_m["acc"]),
                       bangla_validation_acc=float(np.mean(supervised[i]["validation_correct"])),
                       english_validation_acc=float(np.mean(english[i]["validation_correct"])),
                       english_test_acc=float(np.mean(np.asarray(english[i]["predicted"]) == y_en)),
                       english_test_floor=float(np.mean(y_en == majority_label(study, SOURCE, concept))),
                       probe_converged=bool(supervised[i].get("probe_converged", True)))
            for centring in CENTRINGS:
                zs = zero_shot[centring][i]
                zs_m = metric_values(y_te, zs["predicted"], classes)
                tag = "" if centring == "per_condition" else "_source"
                row[f"zs_acc{tag}"] = zs_m["acc"]
                row[f"zs_bal{tag}"] = zs_m["bal"]
                row[f"SSI_zs_acc{tag}"] = float(ssi(zs_m["acc"], nat_m["acc"], floor_m["acc"], study.cfg.min_den))
                row[f"SSI_zs_bal{tag}"] = float(ssi(zs_m["bal"], nat_m["bal"], floor_m["bal"], study.cfg.min_den))
                row[f"SSI_zs_vs_cov_acc{tag}"] = float(ssi(zs_m["acc"], cov_native_m["acc"], floor_m["acc"],
                                                           study.cfg.min_den))
                row[f"zs_above_floor_pp{tag}"] = 100 * (zs_m["acc"] - floor_m["acc"])
                if centring in null_ssi and len(null_ssi[centring]):
                    values = null_ssi[centring]
                    row[f"RAND_zs_p_acc{tag}"] = float((1 + np.sum(values >= zs_m["acc"])) / (1 + len(values)))
                    row[f"RAND_zs_acc_mean{tag}"] = float(values.mean())
                    row[f"RAND_zs_draws{tag}"] = int(len(values))
            rows.append(row)
    return pd.DataFrame(rows)


def select(path):
    """Settings chosen without TEST: by English VALIDATION (zero-shot) and Bangla VALIDATION (supervised)."""
    out = []
    for keys, g in path[path.method.isin(["MLR", "LEACE"])].groupby(["model", "split", "concept", "method"]):
        for rule, column in [("english_validation", "english_validation_acc"),
                             ("bangla_validation", "bangla_validation_acc")]:
            # Ties favour the stronger penalty (smaller C, larger ridge), the pattern end of the path.
            order = g.sort_values([column, "C" if keys[3] == "MLR" else "ridge"],
                                  ascending=[False, keys[3] == "MLR"])
            best = order.iloc[0]
            out.append(dict(zip(["model", "split", "concept", "method"], keys), rule=rule, setting=best.setting,
                            **{c: best[c] for c in ["C", "ridge", "SSI_supervised_acc", "SSI_zs_acc", "SSI_zs_bal",
                                                    "zs_above_floor_pp", "RAND_zs_p_acc",
                                                    "shared_dims_with_english_cov", column]
                               if c in best}))
    selected = pd.DataFrame(out)
    if not selected.empty and "RAND_zs_p_acc" in selected:
        for _, index in selected.groupby(["split", "method", "rule"]).groups.items():
            selected.loc[index, "RAND_zs_p_acc_holm"] = holm(selected.loc[index, "RAND_zs_p_acc"].to_numpy(),
                                                             family_size=len(index))
    return selected


def summarize(path):
    """Per split x concept x method: rank correlation of overlap with zero-shot SSI along the path."""
    out = []
    for keys, g in path.groupby(["split", "concept", "method"]):
        if g.setting.nunique() < 3:
            continue
        rho = lambda a, b: float(g[a].rank().corr(g[b].rank()))  # noqa: E731
        out.append(dict(zip(["split", "concept", "method"], keys), cells=len(g),
                        spearman_overlap_vs_zs_SSI=rho("shared_dims_with_english_cov", "SSI_zs_acc"),
                        spearman_overlap_vs_supervised_SSI=rho("shared_dims_with_english_cov", "SSI_supervised_acc"),
                        median_SSI_zs_acc=float(g.SSI_zs_acc.median())))
    for keys, g in path[path.method.eq("MLR")].groupby(["split", "concept", "setting"]):
        out.append(dict(split=keys[0], concept=keys[1], method="MLR", setting=keys[2], cells=len(g),
                        median_SSI_zs_acc=float(g.SSI_zs_acc.median()),
                        median_SSI_supervised_acc=float(g.SSI_supervised_acc.median()),
                        median_shared_dims_with_english_cov=float(g.shared_dims_with_english_cov.median()),
                        rand_rejections=int(g.get("RAND_zs_p_acc", pd.Series(dtype=float)).le(.05).sum())))
    return pd.DataFrame(out)


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
                cell = load_cell(ctx, name, split, conditions=[SOURCE, TARGET])
                study = make_study(ctx, cell, "penalty")
                saved = cell_tables(ctx, key, {"path.csv": run_cell(ctx, study, log)})
                del study, cell
            else:
                log(f"restored {key}")
            tables.append(saved["path.csv"])
    path = concat(tables)
    finish(ctx, {"path.csv": path, "selected.csv": select(path), "summary.csv": summarize(path)},
           dict(C_grid=list(C_GRID), ridges=list(RIDGES), centrings=list(CENTRINGS),
                zs_rand_draws=int(ctx.option("zs_rand_draws", 999))))
    log("finished")
