"""X18: intervals that refit the subspaces and probes, not only resample TEST.

Motivation: the published component-bootstrap intervals hold the estimated subspaces, probes and taxonomy fixed
and resample TEST components only, so they omit estimation uncertainty; the re-seeded taxonomies and the second split
move SSIs more than those intervals suggest.

refit      for the planned pair (en-idiom -> bn-fig), F at M = 11 and S at M = 3, MLR (C = 1) and COV, split 0: each
           draw resamples components with replacement within SPACE (the same components for both languages), PROBE,
           VALIDATION and TEST, refits the English and the Bangla estimator on the resampled SPACE rows, refits both
           probes (C tuned on the resampled VALIDATION rows) and scores the resampled TEST rows. Draw 0 refits on the
           original rows: its predictions must agree with the published readouts (the gate in verify3).
summary    the point SSI, the published-style conditional interval (TEST components only), the refit percentile
           interval, their widths, and the share of refit draws with SSI above zero, for accuracy and balanced
           accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import fit_estimator

from experiments.common import Log, classes_in, compact_readouts, concat, labels, load_cell, make_study, rank_for
from experiments.metrics import bootstrap_ssi, metric_values, ssi

from .common3 import SOURCE, cell_tables, finish

TARGET = "bn-fig"
METHODS = ["MLR", "COV"]
SPLITS = ["space", "probe", "validation", "test"]


def component_rows(study, condition, split, concept):
    """component_id -> row indices of one condition and split (rows with a label only)."""
    ix = study.idx(condition, split, concept)
    comps = study.items.component_id.astype(str).to_numpy()[ix]
    return {c: ix[comps == c] for c in np.unique(comps)}


def draw_rows(groups, keys, picked):
    empty = np.empty(0, dtype=int)
    return np.concatenate([groups.get(keys[i], empty) for i in picked]) if len(picked) else empty


def refit_once(study, concept, rank, method, rows, y_all):
    """One refit of both estimators and both probes; returns (cross, native, floor) TEST predictions or None."""
    src_space, tgt_space, tr, va, te = rows
    options = study._estimator_options()
    top = max([rank] + list(study.cfg.ranks))
    subs = []
    for space in (src_space, tgt_space):
        if len(np.unique(y_all[space])) < rank + 1:
            return None
        sub = fit_estimator(study.X[space], y_all[space], method, top, seed=int(study.meta["seed"]),
                            **options).at_rank(rank)
        if sub.effective_rank != rank:
            return None
        subs.append(sub)
    if min(len(tr), len(va), len(te)) == 0 or len(np.unique(y_all[tr])) < 2:
        return None
    # As LayerStudy._prefetch builds the published readouts: uncentred inputs times the input factor (the probe's
    # intercept absorbs the centring), so draw 0 reproduces them exactly.
    results = compact_readouts(study, [study.X[tr] @ s.input_factor for s in subs],
                               [study.X[va] @ s.input_factor for s in subs],
                               [study.X[te] @ s.input_factor for s in subs], y_all, (tr, va, te))
    values, counts = np.unique(y_all[tr], return_counts=True)
    floor = np.full(len(te), values[int(np.argmax(counts))])
    return np.asarray(results[0]["predicted"]), np.asarray(results[1]["predicted"]), floor


def run_cell(ctx, study, log):
    draws = int(ctx.option("refit_draws", 200))
    seed = int(study.meta["seed"])
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    rows_out, summary = [], []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        y_all = labels(study, concept)
        groups = {("src", "space"): component_rows(study, SOURCE, "space", concept)}
        for split in SPLITS:
            groups[("tgt", split)] = component_rows(study, TARGET, split, concept)
        space_keys = sorted(set(groups[("src", "space")]) | set(groups[("tgt", "space")]))
        keys = {split: sorted(groups[("tgt", split)]) for split in SPLITS[1:]}
        study._prefetch([(TARGET, concept, s, concept, m, rank, "project", False, 0, None, True)
                         for m in METHODS for s in (SOURCE, TARGET)], f"X18 published readouts {concept}")
        for method in METHODS:
            log(f"{study.meta['model']} {concept} {method}: {draws} refit draws", concept=concept, method=method)
            published = (study._evaluate_sub(TARGET, concept, SOURCE, concept, method, rank),
                         study._evaluate_sub(TARGET, concept, TARGET, concept, method, rank))
            for draw in range(draws + 1):
                if draw == 0:
                    picked = {"space": np.arange(len(space_keys)), **{s: np.arange(len(keys[s])) for s in keys}}
                    rows = (study.idx(SOURCE, "space", concept), study.idx(TARGET, "space", concept),
                            *[study.idx(TARGET, s, concept) for s in SPLITS[1:]])
                else:
                    rng = np.random.default_rng([seed, rank, METHODS.index(method), draw])
                    picked = {"space": rng.integers(0, len(space_keys), len(space_keys)),
                              **{s: rng.integers(0, len(keys[s]), len(keys[s])) for s in keys}}
                    rows = (draw_rows(groups[("src", "space")], space_keys, picked["space"]),
                            draw_rows(groups[("tgt", "space")], space_keys, picked["space"]),
                            *[draw_rows(groups[("tgt", s)], keys[s], picked[s]) for s in SPLITS[1:]])
                fitted = refit_once(study, concept, rank, method, rows, y_all)
                if fitted is None:
                    rows_out.append(dict(model=study.meta["model"], split=seed, concept=concept, method=method,
                                         draw=draw, valid=False))
                    continue
                cross, native, floor = fitted
                y = y_all[rows[4]]
                classes = classes_in(y)
                row = dict(model=study.meta["model"], split=seed, concept=concept, method=method, draw=draw,
                           valid=True, n_test=len(y))
                for name, pred in [("cross", cross), ("native", native), ("floor", floor)]:
                    row.update({f"{name}_{m}": v for m, v in metric_values(y, pred, classes).items()})
                for m in ["acc", "bal"]:
                    row[f"SSI_{m}"] = float(ssi(row[f"cross_{m}"], row[f"native_{m}"], row[f"floor_{m}"],
                                                study.cfg.min_den))
                if draw == 0 and all(p is not None for p in published):
                    row["agreement_cross"] = float(np.mean(cross == np.asarray(published[0]["predicted"])))
                    row["agreement_native"] = float(np.mean(native == np.asarray(published[1]["predicted"])))
                    test = rows[4]
                    conditional = bootstrap_ssi(y, cross, native, floor,
                                                study.items.iloc[test].component_id.astype(str).to_numpy(), classes,
                                                n_boot, seed, study.cfg.min_den)
                    for m in ["acc", "bal"]:
                        row[f"conditional_ci_lo_{m}"] = conditional[f"SSI_{m}_ci_lo"]
                        row[f"conditional_ci_hi_{m}"] = conditional[f"SSI_{m}_ci_hi"]
                rows_out.append(row)
    draws_table = pd.DataFrame(rows_out)
    return {"draws.csv": draws_table}


def summarize(draws_table):
    out = []
    for (model, split, concept, method), g in draws_table.groupby(["model", "split", "concept", "method"]):
        point = g[g.draw.eq(0)]
        refit = g[g.draw.gt(0) & g.valid.astype(bool)]
        row = dict(model=model, split=split, concept=concept, method=method, draws=int(g.draw.gt(0).sum()),
                   valid_draws=int(len(refit)))
        for column in ["agreement_cross", "agreement_native"]:
            row[column] = float(point[column].iloc[0]) if column in point and len(point) else np.nan
        for m in ["acc", "bal"]:
            values = refit[f"SSI_{m}"].astype(float)
            values = values[np.isfinite(values)]
            row[f"SSI_{m}"] = float(point[f"SSI_{m}"].iloc[0]) if len(point) else np.nan
            lo, hi = np.quantile(values, [.025, .975]) if len(values) else (np.nan, np.nan)
            row[f"refit_ci_lo_{m}"], row[f"refit_ci_hi_{m}"] = float(lo), float(hi)
            row[f"refit_share_above_zero_{m}"] = float(np.mean(values > 0)) if len(values) else np.nan
            for side in ["lo", "hi"]:
                column = f"conditional_ci_{side}_{m}"
                row[column] = float(point[column].iloc[0]) if column in point and len(point) else np.nan
            row[f"conditional_width_{m}"] = row[f"conditional_ci_hi_{m}"] - row[f"conditional_ci_lo_{m}"]
            row[f"refit_width_{m}"] = row[f"refit_ci_hi_{m}"] - row[f"refit_ci_lo_{m}"]
            row[f"width_ratio_{m}"] = (row[f"refit_width_{m}"] / row[f"conditional_width_{m}"]
                                       if row[f"conditional_width_{m}"] > 0 else np.nan)
        out.append(row)
    return pd.DataFrame(out)


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
                cell = load_cell(ctx, name, split, conditions=[SOURCE, TARGET])
                study = make_study(ctx, cell, "refit_bootstrap")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log))
                del study, cell
            else:
                log(f"restored {key}")
            parts.append(saved.get("draws.csv"))
    draws_table = concat(parts)
    finish(ctx, {"draws.csv": draws_table, "summary.csv": summarize(draws_table)},
           dict(source=SOURCE, target=TARGET, methods=METHODS, refit_draws=int(ctx.option("refit_draws", 200))))
    log("finished")
