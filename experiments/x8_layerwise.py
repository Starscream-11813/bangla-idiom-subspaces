"""X8: the planned readouts at every cached layer, with nulls, for COV, MLR and LEACE.

Motivation: one layer per model was selected on native accuracy, so the
conclusions might depend on that choice. For every cached layer (every third
layer plus the last) this reads the planned pair (en-idiom -> bn-fig) for F and
S with COV, MLR and LEACE, reports accuracy and balanced-accuracy SSI with
component-bootstrap intervals, 199 RAND draws per layer (all estimators) and 199
PERM draws per layer for COV (MLR with --perm-mlr), and the paired COV - MLR
accuracy difference per layer. The validation-selected layer is flagged; the
layer that maximises test SSI is reported only as a post hoc oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug

from .common import Log, cache_leaf, cell_tables, concat, finish, load_cell, make_study, rank_for, read_json
from .metrics import paired_two_sided
from .nulls import null_test

METHODS = ["COV", "MLR", "LEACE"]


def run_layer(ctx, name, split, layer, log):
    rand = int(ctx.option("layer_rand_draws", 199))
    perm = int(ctx.option("layer_perm_draws", 199))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    perm_methods = {"COV"} | ({"MLR"} if ctx.option("perm_mlr", False) else set())
    cell = load_cell(ctx, name, split, layer=layer, conditions=["en-idiom", "bn-fig"])
    study = make_study(ctx, cell, "layerwise")
    rows, contrasts = [], []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        kept = {}
        for method in METHODS:
            log(f"{name} split{split} layer{layer} {concept} {method}")
            row, items, _ = null_test(study, concept, method, "en-idiom", "bn-fig", rank, rand,
                                      perm if method in perm_methods else 0, n_boot, cell.meta["seed"],
                                      pca=False, label=f"X8 layer {layer} {method} {concept}")
            if row is None:
                continue
            native_validation = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, method, rank)
            row.update(selected_layer=cell.selected_layer, is_selected=layer == cell.selected_layer,
                       native_validation_acc=float(np.mean(native_validation["validation_correct"])))
            rows.append(row)
            kept[method] = items
        if {"COV", "MLR"} <= set(kept):
            a, b = kept["COV"], kept["MLR"]
            assert a.item_id.tolist() == b.item_id.tolist()
            values = (a.cross_pred == a.y_true).astype(float).to_numpy() - (b.cross_pred == b.y_true).to_numpy()
            contrasts.append(dict(model=name, split=split, layer=layer, concept=concept, contrast="COV_minus_MLR_cross",
                                  is_selected=layer == cell.selected_layer,
                                  **paired_two_sided(values, a.component_id.astype(str).to_numpy())))
    del study, cell
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    layers_all, contrasts_all = [], []
    for split in ctx.splits:
        for name in ctx.models:
            leaf, _ = cache_leaf(ctx.campaign, name)
            layers = read_json(leaf / "manifest.json")["layers"]
            requested = ctx.option("layers")
            if requested:
                layers = [l for l in layers if str(l) in requested.split(",")]
            for layer in layers:
                key = f"{model_slug(name)}_split{split}_layer{layer}"
                saved = cell_tables(ctx, key)
                if saved is None:
                    r, c = run_layer(ctx, name, split, layer, log)
                    saved = cell_tables(ctx, key, {"layers.csv": r, "contrasts.csv": c})
                else:
                    log(f"restored {key}")
                layers_all.append(saved["layers.csv"])
                contrasts_all.append(saved["contrasts.csv"])
    table = concat(layers_all)
    summary = []
    if not table.empty:
        for (model, split, concept, method), g in table.groupby(["model", "split", "concept", "method"]):
            g = g.sort_values("layer")
            chosen = g[g.is_selected]
            best = g.loc[g.SSI_acc.idxmax()] if g.SSI_acc.notna().any() else None
            summary.append(dict(model=model, split=split, concept=concept, method=method, layers=len(g),
                                selected_layer=int(g.selected_layer.iloc[0]),
                                SSI_acc_selected=float(chosen.SSI_acc.iloc[0]) if len(chosen) else np.nan,
                                SSI_acc_max=float(g.SSI_acc.max()), oracle_layer_post_hoc=int(best.layer) if best is not None else None,
                                SSI_acc_median=float(g.SSI_acc.median()),
                                layers_RAND_p_below_05=int(g.RAND_p_acc.le(.05).sum()),
                                layers_PERM_p_below_05=int(g.PERM_p_acc.le(.05).sum()) if "PERM_p_acc" in g else None,
                                SSI_bal_selected=float(chosen.SSI_bal.iloc[0]) if len(chosen) else np.nan,
                                SSI_bal_max=float(g.SSI_bal.max())))
    finish(ctx, {"layers.csv": table, "contrasts.csv": concat(contrasts_all), "summary.csv": pd.DataFrame(summary)},
           dict(layer_rand_draws=int(ctx.option("layer_rand_draws", 199)),
                layer_perm_draws=int(ctx.option("layer_perm_draws", 199))))
    log("finished")
