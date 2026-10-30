"""X19: do idiom-like English counterparts carry the shared signal, or do paraphrase-like ones?

Motivation: about a fifth of the English counterparts are literal paraphrases rather than idioms (the volunteers'
judgments), so English "idiom" directions may owe their sharing to counterparts that simply restate the meaning.
The pipeline's counterpart-transparency covariate (cosine between the counterpart and the English figurative gloss,
tertiles fitted on SPACE; labels.counterpart_transparency) splits SPACE into opaque counterparts (lowest tertile:
lexically and semantically distant, usually conventional English idioms) and transparent ones (highest tertile:
paraphrase-like). X17 checks this covariate against the volunteers' counterpart judgments.

subsets    the English subspace is estimated from the opaque SPACE records only, from the transparent ones only, and
           from 20 random subsets of the same size drawn from all SPACE records with a counterpart (the size-matched
           reference). The Bangla side is unchanged.
readout    for idioms in context (bn-fig) and the idiom alone (bn-idiom), F at M = 11 and S at M = 3, MLR and COV,
           split 0: the Bangla-trained probe through each English subspace; SSI with the full native denominator, for
           accuracy and balanced accuracy.
summary    opaque and transparent SSIs against the random subsets: differences from their mean, and two-sided
           empirical p (rank among the random draws).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug

from experiments.common import Log, classes_in, concat, labels, load_cell, make_study, rank_for, readout_indices
from experiments.metrics import metric_values, ssi
from experiments.nulls import majority_label

from .common3 import SOURCE, cell_tables, finish

TARGETS = ["bn-fig", "bn-idiom"]
METHODS = ["COV", "MLR"]


def _flag(items, column):
    if column not in items:
        return np.zeros(len(items), dtype=bool)
    return items[column].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()


def source_subsets(study, concept, draws, seed):
    """opaque, transparent and size-matched random SPACE subsets of the source condition (sorted row indices)."""
    space = study.idx(SOURCE, "space", concept)
    items = study.items.iloc[space]
    score = pd.to_numeric(items.get("cp_transparency"), errors="coerce").to_numpy(float) \
        if "cp_transparency" in items else np.full(len(space), np.nan)
    opaque, transparent = space[_flag(items, "opaque")], space[_flag(items, "transparent")]
    eligible = space[np.isfinite(score)]
    n = int(min(len(opaque), len(transparent)))
    if n < 10:
        return None, dict(opaque=len(opaque), transparent=len(transparent), eligible=len(eligible))
    rng = np.random.default_rng(seed)
    subsets = {"opaque": np.sort(rng.choice(opaque, n, replace=False)),
               "transparent": np.sort(rng.choice(transparent, n, replace=False))}
    for r in range(1, draws + 1):
        subsets[f"random{r}"] = np.sort(rng.choice(eligible, n, replace=False))
    return subsets, dict(opaque=len(opaque), transparent=len(transparent), eligible=len(eligible), size=n)


def run_cell(ctx, study, log):
    draws = int(ctx.option("subset_draws", 20))
    seed = int(study.meta["seed"])
    rows = []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        subsets, sizes = source_subsets(study, concept, draws, 20261027 + 31 * seed + rank)
        if subsets is None:
            log(f"{study.meta['model']} {concept}: too few opaque/transparent SPACE records {sizes}")
            continue
        for target in TARGETS:
            if target not in set(study._condition):
                continue
            specs = [(target, concept, target, concept, m, rank, "project", False, 0, None, True) for m in METHODS]
            specs += [(target, concept, SOURCE, concept, m, rank, "project", False, 0, None, True) for m in METHODS]
            specs += [(target, concept, SOURCE, concept, m, rank, "project", False, 0, ix, True)
                      for m in METHODS for ix in subsets.values()]
            log(f"{study.meta['model']} {concept} {target}: {len(subsets)} source subsets of {sizes['size']} records")
            study._prefetch(specs, f"X19 {target} {concept}")
            test = np.asarray(readout_indices(study, target, concept)[2])
            y = labels(study, concept)[test]
            classes = classes_in(y)
            floor_m = metric_values(y, np.full(len(y), majority_label(study, target, concept)), classes)
            for method in METHODS:
                native = study._evaluate_sub(target, concept, target, concept, method, rank)
                if native is None:
                    continue
                native_m = metric_values(y, native["predicted"], classes)
                for name, ix in [("all", None)] + list(subsets.items()):
                    # None when a subset's labels cannot support rank M (an estimator never pads with null directions)
                    ev = study._evaluate_sub(target, concept, SOURCE, concept, method, rank, subset=ix)
                    source_rows = ix if ix is not None else study.idx(SOURCE, "space", concept)
                    row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                               method=method, target=target, M=rank, subset=name.rstrip("0123456789"),
                               draw=int(name[6:]) if name.startswith("random") else 0, n_source=int(len(source_rows)),
                               source_classes=int(len(np.unique(labels(study, concept)[source_rows]))),
                               n_opaque=sizes["opaque"], n_transparent=sizes["transparent"], n_test=len(y),
                               valid=ev is not None)
                    if ev is not None:
                        cross_m = metric_values(y, ev["predicted"], classes)
                        for m in ["acc", "bal"]:
                            row[f"cross_{m}"], row[f"native_{m}"], row[f"floor_{m}"] = (cross_m[m], native_m[m],
                                                                                       floor_m[m])
                            row[f"SSI_{m}"] = float(ssi(cross_m[m], native_m[m], floor_m[m], study.cfg.min_den))
                    rows.append(row)
    return {"subsets.csv": pd.DataFrame(rows)}


def summarize(table):
    out = []
    keys = ["model", "split", "concept", "method", "target"]
    for key, g in table.groupby(keys):
        row = dict(zip(keys, key))
        row["invalid_subsets"] = int((~g.valid.astype(bool)).sum()) if "valid" in g else 0
        g = g[g.valid.astype(bool)] if "valid" in g else g
        for m in ["acc", "bal"]:
            random = g[g.subset.eq("random")][f"SSI_{m}"].astype(float)
            random = random[np.isfinite(random)]
            row[f"SSI_all_{m}"] = float(g[g.subset.eq("all")][f"SSI_{m}"].iloc[0]) if g.subset.eq("all").any() else np.nan
            row[f"random_mean_{m}"] = float(random.mean()) if len(random) else np.nan
            row[f"random_sd_{m}"] = float(random.std(ddof=1)) if len(random) > 1 else np.nan
            for kind in ["opaque", "transparent"]:
                value = g[g.subset.eq(kind)][f"SSI_{m}"]
                value = float(value.iloc[0]) if len(value) else np.nan
                row[f"SSI_{kind}_{m}"] = value
                row[f"{kind}_minus_random_{m}"] = value - row[f"random_mean_{m}"] if len(random) else np.nan
                if len(random) and np.isfinite(value):
                    upper = (1 + np.sum(random >= value)) / (1 + len(random))
                    lower = (1 + np.sum(random <= value)) / (1 + len(random))
                    row[f"{kind}_p_two_sided_{m}"] = float(min(1., 2 * min(upper, lower)))
                else:
                    row[f"{kind}_p_two_sided_{m}"] = np.nan
        row["random_draws"] = int(g.subset.eq("random").sum())
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
                cell = load_cell(ctx, name, split, conditions=[SOURCE] + TARGETS)
                study = make_study(ctx, cell, "counterpart_type")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log))
                del study, cell
            else:
                log(f"restored {key}")
            parts.append(saved.get("subsets.csv"))
    table = concat(parts)
    finish(ctx, {"subsets.csv": table, "summary.csv": summarize(table) if len(table) else pd.DataFrame()},
           dict(source=SOURCE, targets=TARGETS, methods=METHODS, subset_draws=int(ctx.option("subset_draws", 20))))
    log("finished")
