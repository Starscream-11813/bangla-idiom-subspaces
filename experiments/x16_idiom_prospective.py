"""X16: the frozen-plan, prospective test of the idiom without its context on the follow-up split.

The design is fixed in configs/plans/PLAN_X16.json, whose SHA-256 is pinned below: the run refuses to start if the
plan changed. Split 1 (the follow-up) has never been examined for the target bn-idiom; X13 tested it post hoc on the
discovery split only.

primary    COV, en-idiom -> bn-idiom, F at M = 11 and S at M = 3, six encoders: the SSI under balanced accuracy against
           1,999 RAND, 1,999 PERM and 1,999 RANDCOV draws, Holm within each null over the 12 tests; a test passes
           when all three Holm-adjusted p are at most .05.
secondary  COV under accuracy; MLR (C = 1) under balanced accuracy and accuracy; the zero-shot English classifier
           through COV (per-condition centring) against 999 RAND and 999 PERM zero-shot nulls; absolute gains over the
           floor in points.
gate       COV and MLR English-direction accuracy on bn-fig on split 1 reproduces the follow-up's scores.csv (verify3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from experiments.common import Log, classes_in, concat, labels, load_cell, make_study, rank_for, readout_indices
from experiments.metrics import METRICS, metric_table, metric_values, ssi
from experiments.nulls import majority_label, null_test
from experiments.x1_zero_shot import _null_predictions, _shift, _zero_shot

from .common3 import PLAN_X16, SOURCE, cell_tables, common, finish
from .matched import base_seed, matched_null_predictions, matched_pvalues

PLAN_SHA256 = "e423d9f21f2f718a94745e885851dec7c4a96eeccca8898b809fc6af1199fd5c"
TARGET, GATE = "bn-idiom", "bn-fig"
METHODS = ["COV", "MLR"]
NULLS = ["RAND", "PERM", "RANDCOV"]
ALPHA = .05


def check_plan():
    digest = common.text_sha256(PLAN_X16)
    if digest != PLAN_SHA256:
        raise RuntimeError(f"{PLAN_X16} differs from the frozen plan (sha256 {digest}, expected {PLAN_SHA256}); "
                           "a frozen plan must not change after it is committed")
    return digest


def zero_shot_row(ctx, study, concept, rank, log):
    """The plan's zero-shot secondary: English-trained probe through English COV, per-condition centring."""
    rand = int(ctx.option("zs_rand_draws", 999))
    perm = int(ctx.option("zs_perm_draws", 999))
    Cs = study.cfg.probe_Cs
    y_all = labels(study, concept)
    rows = (study.idx(SOURCE, "probe", concept), study.idx(SOURCE, "validation", concept),
            study.idx(TARGET, "test", concept))
    y = y_all[rows[2]]
    classes = classes_in(y)
    floor_m = metric_values(y, np.full(len(y), majority_label(study, TARGET, concept)), classes)
    shift = _shift(study, SOURCE, TARGET, "per_condition")
    sub = study.sub(SOURCE, concept, "COV", rank)
    native = study._evaluate_sub(TARGET, concept, TARGET, concept, "COV", rank)
    zs = _zero_shot(study, [sub], rows, y_all, shift, Cs)[0]
    if sub is None or native is None or zs is None:
        return None
    native_m, zs_m = metric_values(y, native["predicted"], classes), metric_values(y, zs["predicted"], classes)
    row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"], concept=concept,
               method="COV", centering="per_condition", target=TARGET, M=rank, n=len(y),
               zs_probe_converged=bool(zs.get("probe_converged", True)))
    for m in METRICS:
        row[f"zs_{m}"], row[f"native_{m}"], row[f"floor_{m}"] = zs_m[m], native_m[m], floor_m[m]
        row[f"SSI_zs_{m}"] = float(ssi(zs_m[m], native_m[m], floor_m[m], study.cfg.min_den))
    log(f"{study.meta['model']} {concept}: {rand} RAND and {perm} PERM zero-shot nulls on {TARGET}")
    nulls = {"RAND": _null_predictions(study, SOURCE, concept, "RAND", rank, list(range(1, rand + 1)), False, rows,
                                       y_all, shift, Cs) if rand > 0 else None,
             "PERM": _null_predictions(study, SOURCE, concept, "COV", rank, list(range(1, perm + 1)), True, rows,
                                       y_all, shift, Cs) if perm > 0 else None}
    for kind, preds in nulls.items():
        if preds is None or not len(preds):
            continue
        table = metric_table(y, preds, classes)
        for m in METRICS:
            null = ssi(table[m][:, 0], native_m[m], floor_m[m], study.cfg.min_den)
            null = null[np.isfinite(null)]
            obs = row[f"SSI_zs_{m}"]
            row[f"{kind}_zs_p_{m}"] = (float((1 + np.sum(null >= obs)) / (1 + len(null)))
                                       if len(null) and np.isfinite(obs) else np.nan)
        row[f"{kind}_zs_draws"] = int(len(preds))
    return row


def run_cell(ctx, study, log):
    rand = int(ctx.option("null_draws", 1999))
    perm = int(ctx.option("perm_draws", 1999))
    covd = int(ctx.option("cov_draws", 1999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    seed = int(study.meta["seed"])
    tests, gate, zero_shot = [], [], []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        study._prefetch([(GATE, concept, SOURCE, concept, m, rank, "project", False, 0, None, True) for m in METHODS],
                        f"X16 gate {concept}")
        for method in METHODS:
            g = study._evaluate_sub(GATE, concept, SOURCE, concept, method, rank)
            if g is not None:
                gate.append(dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                                 method=method, source=SOURCE, target=GATE, cross_acc=float(np.mean(g["correct"])),
                                 n=int(len(g["correct"]))))
        idx = readout_indices(study, TARGET, concept)
        y_all = labels(study, concept)
        log(f"{study.meta['model']} {concept}: {covd} RANDCOV draws on {TARGET}", concept=concept)
        null = matched_null_predictions(study, study.idx(SOURCE, "space", concept), rank, idx, y_all,
                                        range(1, covd + 1), base_seed(study, concept))
        for method in METHODS:
            log(f"{study.meta['model']} {concept} {method} -> {TARGET}: {rand} RAND + {perm} PERM draws",
                concept=concept, method=method)
            row, items, _ = null_test(study, concept, method, SOURCE, TARGET, rank, rand, perm, n_boot, seed,
                                      label=f"X16 {method} {concept}")
            if row is None:
                continue
            y = items.y_true.to_numpy()
            native_m = {m: row[f"{m}_native"] for m in METRICS}
            floor_m = {m: row[f"{m}_floor"] for m in METRICS}
            observed = {m: row[f"SSI_{m}"] for m in METRICS}
            row.update(matched_pvalues(y, null, native_m, floor_m, observed, classes_in(y), study.cfg.min_den))
            for m in METRICS:
                row[f"cross_gain_{m}_pp"] = 100 * (row[m] - row[f"{m}_floor"])
                row[f"native_gain_{m}_pp"] = 100 * (row[f"{m}_native"] - row[f"{m}_floor"])
            tests.append(row)
        zero_shot.append(zero_shot_row(ctx, study, concept, rank, log))
    return {"tests.csv": pd.DataFrame(tests), "gate.csv": pd.DataFrame(gate),
            "zero_shot.csv": pd.DataFrame([r for r in zero_shot if r is not None])}


def adjust(tests, zero_shot):
    """Holm within estimator x null x metric over the 12 tests, and the plan's decision rule."""
    tests, zero_shot = tests.copy(), zero_shot.copy()
    for _, index in tests.groupby(["split", "method"]).groups.items():
        for null in NULLS:
            for m in METRICS:
                column = f"{null}_p_{m}"
                if column in tests:
                    tests.loc[index, column + "_holm"] = holm(tests.loc[index, column].to_numpy(),
                                                             family_size=len(index))
    for m in METRICS:
        columns = [f"{null}_p_{m}_holm" for null in NULLS]
        if all(c in tests for c in columns):
            tests[f"pass_{m}"] = tests[columns].le(ALPHA).all(axis=1)
    if "pass_bal" in tests:
        tests["primary"] = tests.method.eq("COV")
        tests["primary_pass"] = tests.primary & tests.pass_bal
    if not zero_shot.empty:
        for _, index in zero_shot.groupby(["split"]).groups.items():
            for null in ["RAND", "PERM"]:
                for m in METRICS:
                    column = f"{null}_zs_p_{m}"
                    if column in zero_shot and zero_shot.loc[index, column].notna().any():
                        zero_shot.loc[index, column + "_holm"] = holm(zero_shot.loc[index, column].to_numpy(),
                                                                     family_size=len(index))
        for m in METRICS:
            columns = [f"{null}_zs_p_{m}_holm" for null in ["RAND", "PERM"]]
            if all(c in zero_shot for c in columns):
                zero_shot[f"pass_{m}"] = zero_shot[columns].le(ALPHA).all(axis=1)
    return tests, zero_shot


def run(ctx):
    plan = check_plan()
    log = Log(ctx.work, ctx.experiment)
    log(f"frozen plan {PLAN_X16.name} sha256 {plan}")
    parts = {k: [] for k in ["tests.csv", "gate.csv", "zero_shot.csv"]}
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=[SOURCE, TARGET, GATE])
                study = make_study(ctx, cell, "idiom_prospective")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log))
                del study, cell
            else:
                log(f"restored {key}")
            for k in parts:
                parts[k].append(saved.get(k))
    tables = {k: concat(v) for k, v in parts.items()}
    tables["tests.csv"], tables["zero_shot.csv"] = adjust(tables["tests.csv"], tables["zero_shot.csv"])
    finish(ctx, tables, dict(plan=PLAN_X16.name, plan_sha256=plan, source=SOURCE, target=TARGET, gate_target=GATE,
                             methods=METHODS, nulls=NULLS, null_draws=int(ctx.option("null_draws", 1999)),
                             perm_draws=int(ctx.option("perm_draws", 1999)),
                             cov_draws=int(ctx.option("cov_draws", 1999)),
                             zs_rand_draws=int(ctx.option("zs_rand_draws", 999)),
                             zs_perm_draws=int(ctx.option("zs_perm_draws", 999))))
    log("finished")
