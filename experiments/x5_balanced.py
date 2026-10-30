"""X5: balanced accuracy and macro-F1 for the planned tests, scored on every null draw.

Motivation: SSI is an accuracy ratio over imbalanced multi-class labels, and
the published nulls were scored by accuracy only. This replays the planned
discovery family (MLR, en-idiom -> bn-fig, F at M=11 and S at M=3, six
encoders; 1,999 RAND and 1,999 PERM draws) and the follow-up H1 design (COV,
split 1), keeping predicted labels, and recomputes SSI, null p values and Holm
adjustments with balanced accuracy and macro-F1 next to accuracy. The
accuracy-based numbers must reproduce the published ones (``ref_*`` columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from . import reference
from .common import Log, cell_tables, concat, finish, load_cell, make_study, rank_for
from .metrics import METRICS
from .nulls import null_test

DEFAULT_METHODS = {0: ["MLR", "COV"], 1: ["COV"]}
H1_MODELS = {"csebuetnlp/banglabert", "google/muril-large-cased", "setu4993/LaBSE"}


def family_flags(tests):
    """Mark the published planned families: discovery MLR (12 tests) and follow-up H1 COV (6 tests)."""
    planned = (tests.split.eq(0) & tests.method.eq("MLR")) | (
        tests.split.eq(1) & tests.method.eq("COV") & tests.model.isin(H1_MODELS))
    return planned


def adjust(tests):
    """Holm within each (split, method) family, separately for RAND/PERM and for each metric."""
    tests = tests.copy()
    for (split, method), index in tests.groupby(["split", "method"]).groups.items():
        for null in ["RAND", "PERM"]:
            for m in METRICS:
                column = f"{null}_p_{m}"
                if column in tests:
                    tests.loc[index, f"{null}_p_{m}_holm"] = holm(tests.loc[index, column].to_numpy(),
                                                                 family_size=len(index))
    return tests


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    rand = int(ctx.option("null_draws", ctx.cfg.null_draws))
    perm = int(ctx.option("perm_draws", ctx.cfg.perm_null_draws or ctx.cfg.null_draws))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    tests, nulls = [], []
    for split in ctx.splits:
        methods = ctx.option(f"methods_split{split}") or DEFAULT_METHODS[split]
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is not None:
                tests.append(saved["tests.csv"])
                nulls.append(saved["null_metrics.csv.gz"])
                log(f"restored {key}")
                continue
            log(f"loading {name} split {split}", model=name, split=split, completed=number - 1, total=len(ctx.models))
            cell = load_cell(ctx, name, split, conditions=["en-idiom", "bn-fig"])
            study = make_study(ctx, cell, "planned")
            rows, items, draws = [], [], []
            for method in methods:
                for concept in ["F", "S"]:
                    rank = rank_for(study, concept)
                    log(f"{name} split{split} {method} {concept} M={rank}: {rand} RAND + {perm} PERM draws",
                        model=name, split=split, method=method, concept=concept)
                    row, table, null = null_test(study, concept, method, "en-idiom", "bn-fig", rank, rand, perm,
                                                 n_boot, cell.meta["seed"], label=f"X5 {method} {concept}")
                    if row is None:
                        continue
                    row["selected_layer"] = cell.selected_layer
                    rows.append(row)
                    table.insert(0, "model", name)
                    table.insert(1, "split", split)
                    items.append(table)
                    for kind, values in null.items():
                        frame = pd.DataFrame({m: values[m] for m in METRICS})
                        frame.insert(0, "draw", np.arange(1, len(frame) + 1))
                        frame.insert(0, "null", kind)
                        frame.insert(0, "concept", concept)
                        frame.insert(0, "method", method)
                        frame.insert(0, "split", split)
                        frame.insert(0, "model", name)
                        draws.append(frame)
            saved = cell_tables(ctx, key, {"tests.csv": pd.DataFrame(rows), "items.csv.gz": concat(items),
                                           "null_metrics.csv.gz": concat(draws)})
            tests.append(saved["tests.csv"])
            nulls.append(saved["null_metrics.csv.gz"])
            del study, cell
    tests = concat(tests)
    if tests.empty:
        raise RuntimeError("X5 produced no tests")
    tests = adjust(tests)
    tests["published_family"] = family_flags(tests)
    checked = concat([reference.compare(tests[tests.split.eq(s)], s) for s in sorted(tests.split.unique())])
    agreement = []
    for (split, method), g in checked.groupby(["split", "method"]):
        # Rank correlations need at least three tests with all three SSIs defined.
        enough = g[["SSI_acc", "SSI_bal", "SSI_f1"]].notna().all(axis=1).sum() >= 3
        spearman = lambda other: float(g.SSI_acc.rank().corr(g[other].rank())) if enough else np.nan
        agreement.append(dict(split=split, method=method, tests=len(g),
                              spearman_SSI_acc_bal=spearman("SSI_bal"), spearman_SSI_acc_f1=spearman("SSI_f1"),
                              **{f"rejections_{null}_{m}": int(g[f"{null}_p_{m}_holm"].le(.05).sum())
                                 for null in ["RAND", "PERM"] for m in METRICS}))
    finish(ctx, {"tests.csv": checked, "agreement.csv": pd.DataFrame(agreement),
                 "null_metrics.csv.gz": concat(nulls)},
           dict(null_draws=rand, perm_draws=perm, bootstrap=n_boot))
    log("finished", completed=len(ctx.models), total=len(ctx.models))
