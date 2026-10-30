"""X14: the decoder arm with a powered planned test (optional; Qwen3-8B).

Motivation: the decoder arm cannot reject anything (3--9 null draws, minimum
Holm-adjusted p of 0.2), and it read F at rank 3 rather than the encoders' 11.
Here every decoder of the campaign's decoder tree (configs/decoder_3090.json;
Qwen3-8B) is read at its validation-selected layer with the encoders' planned
design and analysis settings (configs/full_3090.json): MLR and COV, F at M = 11
and S at M = 3, 1,999 RAND + 1,999 PERM draws, accuracy, balanced accuracy and
macro-F1, Holm within each estimator x metric over the decoder's tests. The
zero-shot readout through COV and MLR (X1's protocol) gets 999 RAND and 999 PERM
(COV) zero-shot nulls.

The decoder tree has the primary tree's layout (``decoder/<slug>/split_0/labels_0``,
written by the same runner); the first round's loaders are pointed at it while a
decoder is loaded. Split 0 only: the decoder was not part of the follow-up.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pandas as pd

from idiom_subspace.config import model_slug

from experiments import common
from experiments.common import Log, concat, load_cell, make_study, rank_for
from experiments.nulls import null_test

from .common2 import cell_tables, finish
from .x13_idiom_isolated import SOURCE, adjust, zero_shot_rows

ROOT = "decoder"
TARGET = "bn-fig"
METHODS = ["MLR", "COV"]


def decoder_models(campaign):
    """Decoder model names from the campaign's decoder configuration (default: Qwen/Qwen3-8B)."""
    path = Path(campaign) / "configs" / "decoder_3090.json"
    if path.exists():
        names = [m["name"] for m in json.loads(path.read_text(encoding="utf-8")).get("models", [])]
        if names:
            return names
    return ["Qwen/Qwen3-8B"]


def decoder_paths(campaign, name, root=ROOT):
    prepared = Path(campaign) / root / model_slug(name) / "split_0" / "labels_0"
    return prepared, prepared / model_slug(name)


@contextmanager
def decoder_tree(root=ROOT):
    """Point the first round's loaders at ``campaign/<root>`` instead of ``campaign/primary``."""
    original_paths, original_leaf = common.discovery_paths, common.cache_leaf

    def cache_leaf(campaign, name):
        _, dest = decoder_paths(campaign, name, root)
        revision = common.read_json(dest / "model_revision.json")
        leaf = revision.get("cache_reused_from") or str(Path(campaign) / root / model_slug(name) / "cache"
                                                        / revision["fingerprint"])
        return Path(leaf), revision

    common.discovery_paths = lambda campaign, name: decoder_paths(campaign, name, root)
    common.cache_leaf = cache_leaf
    common._ITEMS.clear()
    try:
        yield
    finally:
        common.discovery_paths, common.cache_leaf = original_paths, original_leaf
        common._ITEMS.clear()


def run_cell(ctx, study, log):
    rand = int(ctx.option("null_draws", 1999))
    perm = int(ctx.option("perm_draws", 1999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    tests, zero_shot = [], []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        for method in METHODS:
            log(f"{study.meta['model']} {concept} {method}: {rand} RAND + {perm} PERM draws", concept=concept,
                method=method)
            row, _, _ = null_test(study, concept, method, SOURCE, TARGET, rank, rand, perm, n_boot,
                                  int(study.meta["seed"]), label=f"X14 {method} {concept}")
            if row is not None:
                tests.append(row)
        zero_shot += zero_shot_rows(ctx, study, concept, rank, log, target=TARGET)
    return pd.DataFrame(tests), pd.DataFrame(zero_shot)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    models = decoder_models(ctx.campaign)
    tests, zs = [], []
    for number, name in enumerate(models, 1):
        key = f"{model_slug(name)}_split0"
        saved = cell_tables(ctx, key)
        if saved is None:
            log(f"loading decoder {name}", model=name, completed=number - 1, total=len(models))
            with decoder_tree():
                cell = load_cell(ctx, name, 0, conditions=[SOURCE, TARGET])
            study = make_study(ctx, cell, "decoder")
            table, zero = run_cell(ctx, study, log)
            table["selected_layer"] = cell.selected_layer
            saved = cell_tables(ctx, key, {"tests.csv": table, "zero_shot.csv": zero})
            del study, cell
        else:
            log(f"restored {key}")
        tests.append(saved["tests.csv"])
        zs.append(saved["zero_shot.csv"])
    tests, zero = adjust(concat(tests), concat(zs))
    ctx.models = models
    ctx.splits = [0]
    finish(ctx, {"tests.csv": tests, "zero_shot.csv": zero},
           dict(decoder_models=models, root=ROOT, null_draws=int(ctx.option("null_draws", 1999)),
                perm_draws=int(ctx.option("perm_draws", 1999))))
    log("finished")
