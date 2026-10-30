"""X2: the planned test for two concepts that are not built from English glosses.

Motivation: F clusters *English* figurative glosses, so it may favour
English-side semantics and gloss transfer by construction. FBN rebuilds the
taxonomy from the Bangla glosses; DOM is the corpus's usage-domain metadata
(see concepts.py). For each, the planned design is repeated exactly -- English
counterparts (en-idiom) -> Bangla idioms in context (bn-fig), MLR and COV at
M = K-1, RAND and component-permutation nulls, Holm within each concept x
estimator family (one test per encoder) -- and the published input ladder is
read without nulls, including the H3-style gloss-minus-counterpart contrast on
the Bangla gloss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from .common import (Log, cell_tables, cells_root, concat, finish, load_cell, make_study, rank_for,
                     split_corpus_frame, write_csv, write_json)
from .concepts import bangla_gloss_labels, usage_domain_labels
from .metrics import METRICS, paired
from .nulls import null_test, observed_test

LADDER = [("en-para", "bn-para"), ("en-idiom", "bn-para"), ("en-trans", "bn-fig"), ("en-idiom", "bn-context"),
          ("en-idiom", "bn-idiom"), ("en-idiom", "bn-sentence"), ("bn-fig", "en-idiom")]
CONDITIONS = sorted({c for pair in LADDER for c in pair} | {"en-idiom", "bn-fig"})


def build_labels(ctx, split, log):
    """Concept labels for one split (shared by all models), cached with the run's finished cells."""
    backend = ctx.option("fbn_backend", "nllb")
    folder = cells_root(ctx) / "labels" / f"split{split}"
    path = folder / "labels.csv"
    if path.exists():
        return pd.read_csv(path)
    name = ctx.models[0]
    corpus = split_corpus_frame(ctx, name, split)
    concepts = ctx.option("concepts", "FBN,DOM").split(",")
    table = corpus[["idiom_id"]].copy()
    reports = {}
    if "DOM" in concepts:
        dom, reports["DOM"] = usage_domain_labels(corpus, k=int(ctx.option("dom_k", 12)))
        table["DOM"] = dom.DOM.to_numpy()
    if "FBN" in concepts:
        log(f"building FBN for split {split} with {backend}")
        fbn, reports["FBN"] = bangla_gloss_labels(corpus, backend=backend, seed=0, k=int(ctx.option("fbn_k", 12)),
                                                  model=ctx.option("fbn_model", "facebook/nllb-200-distilled-600M"))
        table["FBN"] = fbn.FBN.to_numpy()
    write_csv(path, table)
    write_json(folder / "label_report.json", reports)
    write_json(ctx.portable / f"label_report_split{split}.json", reports)
    write_csv(ctx.portable / f"labels_split{split}.csv.gz", table)
    return table


def ladder_rows(study, concept, method, rank, n_boot, seed):
    rows, kept = [], {}
    for source, target in LADDER:
        if not {source, target} <= set(study._condition):
            continue
        row, kept[(source, target)] = observed_test(study, concept, method, source, target, rank, n_boot, seed)
        if row is not None:
            rows.append(row)
    contrasts = []
    gloss, counterpart = kept.get(("en-para", "bn-para")), kept.get(("en-idiom", "bn-para"))
    if gloss and counterpart:
        assert np.array_equal(gloss["test_idx"], counterpart["test_idx"])
        values = (gloss["cross"] == gloss["y"]).astype(float) - (counterpart["cross"] == counterpart["y"])
        contrasts.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), concept=concept,
                              method=method, contrast="H3_gloss_minus_counterpart_on_bn-para",
                              **paired(values, gloss["groups"])))
    return rows, contrasts


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    rand = int(ctx.option("null_draws", 999))
    perm = int(ctx.option("perm_draws", 999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    methods = ctx.option("methods", "MLR,COV").split(",")
    tests, ladders, contrasts = [], [], []
    for split in ctx.splits:
        table = build_labels(ctx, split, log)
        concepts = [c for c in ["FBN", "DOM"] if c in table]
        lookup = table.set_index("idiom_id")
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}_{ctx.option('fbn_backend', 'nllb')}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=CONDITIONS)
                for concept in concepts:
                    cell.frame[concept] = cell.frame.idiom_id.map(lookup[concept]).fillna(-1).astype(int)
                study = make_study(ctx, cell, "concepts", extra_columns=concepts)
                rows, ladder, contrast = [], [], []
                for concept in concepts:
                    rank = rank_for(study, concept)
                    if rank is None:
                        continue
                    for method in methods:
                        log(f"{name} split{split} {concept} {method} M={rank}: {rand} RAND + {perm} PERM",
                            model=name, concept=concept, method=method)
                        row, _, _ = null_test(study, concept, method, "en-idiom", "bn-fig", rank, rand, perm, n_boot,
                                              cell.meta["seed"], label=f"X2 {concept} {method}")
                        if row is not None:
                            row["selected_layer"] = cell.selected_layer
                            rows.append(row)
                        part, c = ladder_rows(study, concept, method, rank, n_boot, cell.meta["seed"])
                        ladder += part
                        contrast += c
                saved = cell_tables(ctx, key, {"tests.csv": pd.DataFrame(rows), "ladder.csv": pd.DataFrame(ladder),
                                               "contrasts.csv": pd.DataFrame(contrast)})
                del study, cell
            else:
                log(f"restored {key}")
            tests.append(saved["tests.csv"])
            ladders.append(saved["ladder.csv"])
            contrasts.append(saved["contrasts.csv"])
    tests = concat(tests)
    for (split, concept, method), index in tests.groupby(["split", "concept", "method"]).groups.items():
        for null in ["RAND", "PERM"]:
            for m in METRICS:
                column = f"{null}_p_{m}"
                tests.loc[index, column + "_holm"] = holm(tests.loc[index, column].to_numpy(), family_size=len(index))
    contrasts = concat(contrasts)
    if not contrasts.empty:
        for (split, concept, method), index in contrasts.groupby(["split", "concept", "method"]).groups.items():
            contrasts.loc[index, "p_holm"] = holm(contrasts.loc[index, "p"].to_numpy(), family_size=len(index))
    finish(ctx, {"tests.csv": tests, "ladder.csv": concat(ladders), "contrasts.csv": contrasts},
           dict(null_draws=rand, perm_draws=perm, bootstrap=n_boot, methods=methods,
                fbn_backend=ctx.option("fbn_backend", "nllb")))
    log("finished")
