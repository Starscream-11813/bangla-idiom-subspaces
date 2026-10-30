"""X13: the idiom without its context (bn-idiom) as a second planned family.

Motivation: the corpus's example sentences were written to illustrate their
idioms, so masking the idiom barely hurts, and the planned test (en-idiom ->
bn-fig) largely measures whether English directions read Bangla sentences that
gloss the idiom. bn-idiom, the idiom alone in a neutral frame ("Here the
expression ... is used."), isolates it. The discovery tables already show higher SSI there
in all six encoders, partly because its native denominator is smaller.

tests      the planned design for en-idiom -> bn-idiom: MLR and COV, F at M = 11
           and S at M = 3, 1,999 RAND + 1,999 PERM draws, accuracy, balanced
           accuracy and macro-F1, Holm within each estimator x metric family of 12
           (X5's null_test). The family was specified after its point estimates
           were known but before any of its nulls were computed.
zero-shot  an English-trained probe through English COV and MLR directions,
           applied unchanged to bn-idiom TEST (X1's protocol, both centrings), with
           999 RAND and 999 PERM (COV) zero-shot nulls; Holm within estimator x
           centring.
absolute   for bn-idiom and bn-fig: native and English-direction gains over the
           floor in points, the published same-estimator SSI, and SSI against the
           full-space probe, a denominator shared by every estimator.
paired     on records with both inputs: bn-idiom minus bn-fig accuracy through
           English directions and through Bangla's own, with component bootstrap
           intervals and two-sided component wild-sign p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from experiments.common import Log, classes_in, concat, labels, load_cell, make_study, rank_for
from experiments.metrics import METRICS, metric_table, metric_values, paired_two_sided, ssi
from experiments.nulls import majority_label, null_test, observed_test
from experiments.x1_zero_shot import _null_predictions, _shift, _zero_shot

from .common2 import cell_tables, finish

SOURCE = "en-idiom"
TARGET, CONTEXT = "bn-idiom", "bn-fig"
METHODS = ["MLR", "COV"]
CENTRINGS = ("per_condition", "source")


def _correct(frame, prediction):
    return (frame[prediction].to_numpy() == frame.y_true.to_numpy()).astype(float)


def absolute_rows(study, concept, method, rank, target, cross, native, floor, y, ambient):
    """Gains over the floor in points, and SSI against the same estimator's and the full-space native probe."""
    classes = classes_in(y)
    m = {name: metric_values(y, pred, classes) for name, pred in
         [("cross", cross), ("native", native), ("floor", floor), ("ambient", ambient)]}
    row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"], concept=concept,
               method=method, M=rank, target=target, n=len(y))
    for k in ["acc", "bal"]:
        row.update({f"{name}_{k}": m[name][k] for name in m})
        row[f"cross_gain_{k}_pp"] = 100 * (m["cross"][k] - m["floor"][k])
        row[f"native_gain_{k}_pp"] = 100 * (m["native"][k] - m["floor"][k])
        row[f"ambient_gain_{k}_pp"] = 100 * (m["ambient"][k] - m["floor"][k])
        row[f"SSI_{k}"] = float(ssi(m["cross"][k], m["native"][k], m["floor"][k], study.cfg.min_den))
        row[f"SSI_ambient_{k}"] = float(ssi(m["cross"][k], m["ambient"][k], m["floor"][k], study.cfg.min_den))
    return row


def zero_shot_rows(ctx, study, concept, rank, log, target=TARGET):
    """X1's zero-shot protocol on ``target``, with RAND (every estimator) and PERM (COV) zero-shot nulls."""
    rand = int(ctx.option("zs_rand_draws", 999))
    perm = int(ctx.option("zs_perm_draws", 999))
    Cs = study.cfg.probe_Cs
    y_all = labels(study, concept)
    zs_rows = (study.idx(SOURCE, "probe", concept), study.idx(SOURCE, "validation", concept),
               study.idx(target, "test", concept))
    y = y_all[zs_rows[2]]
    classes = classes_in(y)
    floor = np.full(len(y), majority_label(study, target, concept))
    floor_m = metric_values(y, floor, classes)
    study._prefetch([(target, concept, target, concept, method, rank, "project", False, 0, None, True)
                     for method in METHODS], f"zero-shot natives {concept}")
    rows = []
    for centring in CENTRINGS:
        shift = _shift(study, SOURCE, target, centring)
        log(f"{study.meta['model']} {concept} {centring}: {rand} RAND zero-shot nulls on {target}")
        rand_preds = (_null_predictions(study, SOURCE, concept, "RAND", rank, list(range(1, rand + 1)), False, zs_rows,
                                        y_all, shift, Cs) if rand > 0 else np.empty((0, 0)))
        for method in METHODS:
            sub = study.sub(SOURCE, concept, method, rank)
            native = study._evaluate_sub(target, concept, target, concept, method, rank)
            zs = _zero_shot(study, [sub], zs_rows, y_all, shift, Cs)[0]
            native_m, zs_m = metric_values(y, native["predicted"], classes), metric_values(y, zs["predicted"], classes)
            row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                       concept=concept, method=method, centering=centring, target=target, M=rank, n=len(y),
                       zs_probe_converged=bool(zs.get("probe_converged", True)))
            for k in METRICS:
                row[f"zs_{k}"], row[f"native_{k}"], row[f"floor_{k}"] = zs_m[k], native_m[k], floor_m[k]
                row[f"SSI_zs_{k}"] = float(ssi(zs_m[k], native_m[k], floor_m[k], study.cfg.min_den))
            nulls = {"RAND": rand_preds}
            if method == "COV" and perm > 0:
                log(f"{study.meta['model']} {concept} {method} {centring}: {perm} PERM zero-shot nulls on {target}")
                nulls["PERM"] = _null_predictions(study, SOURCE, concept, method, rank, list(range(1, perm + 1)), True,
                                                  zs_rows, y_all, shift, Cs)
            for kind, preds in nulls.items():
                if preds is None or not len(preds):
                    continue
                table = metric_table(y, preds, classes)
                for k in METRICS:
                    null = ssi(table[k][:, 0], native_m[k], floor_m[k], study.cfg.min_den)
                    null = null[np.isfinite(null)]
                    obs = row[f"SSI_zs_{k}"]
                    row[f"{kind}_zs_p_{k}"] = (float((1 + np.sum(null >= obs)) / (1 + len(null)))
                                               if len(null) and np.isfinite(obs) else np.nan)
                row[f"{kind}_zs_draws"] = int(len(preds))
            rows.append(row)
    return rows


def run_cell(ctx, study, log):
    rand = int(ctx.option("null_draws", 1999))
    perm = int(ctx.option("perm_draws", 1999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    seed = int(study.meta["seed"])
    D = study.X.shape[1]
    tests, absolute, paired_out, zero_shot = [], [], [], []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        study._prefetch([(t, concept, t, concept, "AMBIENT", D, "project", False, 0, None, True)
                         for t in (TARGET, CONTEXT)], f"X13 ambient {concept}")
        ambient = {t: study._evaluate_sub(t, concept, t, concept, "AMBIENT", D) for t in (TARGET, CONTEXT)}
        for method in METHODS:
            log(f"{study.meta['model']} {concept} {method} -> {TARGET}: {rand} RAND + {perm} PERM draws",
                concept=concept, method=method)
            row, items, _ = null_test(study, concept, method, SOURCE, TARGET, rank, rand, perm, n_boot, seed,
                                      label=f"X13 {method} {concept}")
            if row is None:
                continue
            tests.append(row)
            _, context = observed_test(study, concept, method, SOURCE, CONTEXT, rank, n_boot, seed)
            frames = {TARGET: items.assign(cross_correct=_correct(items, "cross_pred"),
                                           native_correct=_correct(items, "native_pred"))}
            amb_target = np.asarray(ambient[TARGET]["predicted"])
            absolute.append(absolute_rows(study, concept, method, rank, TARGET, items.cross_pred.to_numpy(),
                                          items.native_pred.to_numpy(), items.floor_pred.to_numpy(),
                                          items.y_true.to_numpy(), amb_target))
            if context is not None:
                absolute.append(absolute_rows(study, concept, method, rank, CONTEXT, context["cross"], context["native"],
                                              context["floor"], context["y"],
                                              np.asarray(ambient[CONTEXT]["predicted"])))
                ctx_items = study.items.iloc[context["test_idx"]][["idiom_id", "component_id"]].reset_index(drop=True)
                ctx_items["cross_correct"] = (context["cross"] == context["y"]).astype(float)
                ctx_items["native_correct"] = (context["native"] == context["y"]).astype(float)
                frames[CONTEXT] = ctx_items
                both = frames[TARGET].merge(frames[CONTEXT], on="idiom_id", suffixes=("_idiom", "_context"))
                for kind in ["cross", "native"]:
                    values = both[f"{kind}_correct_idiom"] - both[f"{kind}_correct_context"]
                    test = paired_two_sided(values.to_numpy(), both.component_id_idiom.astype(str).to_numpy())
                    paired_out.append(dict(model=study.meta["model"], split=seed, concept=concept, method=method,
                                           readout=kind, contrast=f"{TARGET} minus {CONTEXT}",
                                           idiom_acc=float(both[f"{kind}_correct_idiom"].mean()),
                                           context_acc=float(both[f"{kind}_correct_context"].mean()),
                                           estimate_pp=100 * test["estimate"], ci_lo_pp=100 * test["ci_lo"],
                                           ci_hi_pp=100 * test["ci_hi"], p_two_sided=test["p_two_sided"],
                                           n=test["n"], n_components=test["n_components"]))
        zero_shot += zero_shot_rows(ctx, study, concept, rank, log)
    return {"tests.csv": pd.DataFrame(tests), "absolute.csv": pd.DataFrame(absolute),
            "paired.csv": pd.DataFrame(paired_out), "zero_shot.csv": pd.DataFrame(zero_shot)}


def adjust(tests, zero_shot):
    """Holm within estimator x metric (planned-style tests) and estimator x centring (zero-shot) over 12 tests."""
    tests, zero_shot = tests.copy(), zero_shot.copy()
    for _, index in tests.groupby(["split", "method"]).groups.items():
        for null in ["RAND", "PERM"]:
            for m in METRICS:
                column = f"{null}_p_{m}"
                if column in tests:
                    tests.loc[index, column + "_holm"] = holm(tests.loc[index, column].to_numpy(),
                                                             family_size=len(index))
    if not zero_shot.empty:
        for _, index in zero_shot.groupby(["split", "method", "centering"]).groups.items():
            for null in ["RAND", "PERM"]:
                for m in METRICS:
                    column = f"{null}_zs_p_{m}"
                    if column in zero_shot and zero_shot.loc[index, column].notna().any():
                        zero_shot.loc[index, column + "_holm"] = holm(zero_shot.loc[index, column].to_numpy(),
                                                                     family_size=len(index))
    return tests, zero_shot


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    parts = {k: [] for k in ["tests.csv", "absolute.csv", "paired.csv", "zero_shot.csv"]}
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=[SOURCE, TARGET, CONTEXT])
                study = make_study(ctx, cell, "idiom_isolated")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log))
                del study, cell
            else:
                log(f"restored {key}")
            for k in parts:
                parts[k].append(saved.get(k))
    tables = {k: concat(v) for k, v in parts.items()}
    tables["tests.csv"], tables["zero_shot.csv"] = adjust(tables["tests.csv"], tables["zero_shot.csv"])
    finish(ctx, tables, dict(source=SOURCE, target=TARGET, context=CONTEXT, methods=METHODS,
                             null_draws=int(ctx.option("null_draws", 1999)),
                             perm_draws=int(ctx.option("perm_draws", 1999)),
                             zs_rand_draws=int(ctx.option("zs_rand_draws", 999)),
                             zs_perm_draws=int(ctx.option("zs_perm_draws", 999))))
    log("finished")
