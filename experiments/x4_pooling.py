"""X4: a pooling-matched 2 x 2 design that separates the input from the pooled tokens.

Motivation: bn-context differs from bn-fig both in its input (idiom masked)
and in the tokens that are pooled (context instead of span), so "masked contexts
rival the idioms" confounds the two. Two new conditions complete a factorial
design on the same example sentence (new forward passes, GPU):

                         pool = span position        pool = context tokens
  input = idiom present  bn-fig      (published)     bn-figctx   (new)
  input = idiom masked   bn-maskslot (new)           bn-context  (published)

bn-figctx keeps the idiom in the sentence but pools every token outside it;
bn-maskslot keeps the published masked sentence but pools the "[MASK]" span
(a real mask token for the BERT-family encoders; plain text for XLM-R, whose
mask token is "<mask>"). All five conditions (the four cells and the en-idiom
source) are re-extracted together with the original pinned model revision,
layers, maximum length and normalization, and checked against the original
cache. Readouts use the original preprocessor and are restricted to records
valid in all four Bangla cells, so every contrast is paired on the same records.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.data import _hash

from .common import (Log, cell_tables, concat, discovery_items, finish, load_cell, make_study, model_spec,
                     original_cache, original_preprocessor, rank_for, read_json, selected_layer,
                     split_corpus_frame, LABEL_COLUMNS, write_csv, write_json, labels)
from .metrics import bootstrap_ssi, paired_two_sided
from .nulls import majority_label

BANGLA = ["bn-fig", "bn-figctx", "bn-maskslot", "bn-context"]
NEW = {"bn-figctx": ("bn-fig", "context"), "bn-maskslot": ("bn-context", "span")}
CONTRASTS = [("input_at_context_pool", "bn-figctx", "bn-context"),
             ("input_at_span_pool", "bn-fig", "bn-maskslot"),
             ("pool_with_idiom", "bn-fig", "bn-figctx"),
             ("pool_with_mask", "bn-maskslot", "bn-context")]


def pooling_items(ctx, name):
    """Original en-idiom/bn-fig/bn-context items plus the two new pooling cells."""
    original = discovery_items(ctx.campaign, name)
    base = original[original.condition.isin(["en-idiom", "bn-fig", "bn-context"])].copy()
    new = []
    for condition, (parent, pool) in NEW.items():
        rows = original[original.condition.eq(parent)].copy()
        rows["condition"], rows["pool_target"] = condition, pool
        rows["item_id"] = [f"item_{_hash([r.idiom_id, condition, r.lang, r.text, r.span, pool])}"
                           for r in rows.itertuples()]
        new.append(rows)
    items = pd.concat([base] + new, ignore_index=True)
    assert items.item_id.is_unique
    return items


def extract(ctx, name, log):
    """New forward passes for the five conditions; returns (items, features, valid, manifest)."""
    from idiom_subspace.extraction import extract_features
    slug = model_slug(name)
    folder = ctx.work / "extraction" / slug
    items = pooling_items(ctx, name)
    _, _, _, original_manifest, _ = original_cache(ctx, name, 0)
    spec = model_spec(ctx.cfg, name)
    if spec.get("normalization", "none") != original_manifest.get("normalization", "none"):
        raise ValueError(f"{name}: configured normalization differs from the cached features")
    if not Path(spec["name"]).exists():
        spec["revision"] = original_manifest["resolved_revision"]
    cfg = copy.deepcopy(ctx.cfg)
    cfg.cache_only = False
    cfg.feature_cache_dirs = []
    cfg.layers = list(original_manifest["layers"])
    cfg.max_length = int(original_manifest["max_length"])
    cfg.device = ctx.option("device", "auto")
    log(f"extracting {len(items)} items for {name} at revision {spec.get('revision')}", model=name)
    features, pooling, manifest = extract_features(items[["item_id", "condition", "text", "span", "pool_target"]],
                                                   spec, cfg, folder / "cache")
    assert pooling.item_id.astype(str).tolist() == items.item_id.astype(str).tolist()
    valid = pooling.valid.astype(str).str.lower().eq("true").to_numpy()
    items = items.copy()
    items["token_count"] = pooling.token_count.to_numpy()
    items["pool_status"] = pooling.status.to_numpy()
    write_json(folder / "extraction_manifest.json", manifest)
    return items, features, valid, manifest


def extraction_check(ctx, name, items, features, valid, manifest):
    """Re-extracted original conditions against the original cache, at every layer."""
    original, ofeat, ovalid, omanifest, _ = original_cache(ctx, name, 0)
    position = {iid: j for j, iid in enumerate(original.item_id.astype(str))}
    rows = []
    for condition in ["en-idiom", "bn-fig", "bn-context"]:
        mine = np.flatnonzero(items.condition.eq(condition).to_numpy())
        theirs = np.asarray([position[i] for i in items.item_id.iloc[mine]])
        both = valid[mine] & ovalid[theirs]
        status_agree = float(np.mean(valid[mine] == ovalid[theirs]))
        for li, layer in enumerate(manifest["layers"]):
            a = np.asarray(features[mine[both], li, :], dtype=float)
            b = np.asarray(ofeat[theirs[both], omanifest["layers"].index(layer), :], dtype=float)
            cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            rows.append(dict(model=name, condition=condition, layer=layer, items=int(both.sum()),
                             validity_agreement=status_agree, max_abs_diff=float(np.abs(a - b).max()),
                             relative_rms_diff=float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-12)),
                             min_cosine=float(cos.min()), median_cosine=float(np.median(cos))))
    return pd.DataFrame(rows)


def with_split_labels(ctx, name, items, split):
    if split == 0:
        return items
    corpus = split_corpus_frame(ctx, name, 1).set_index("idiom_id")
    items = items.copy()
    for column in LABEL_COLUMNS:
        items[column] = items.idiom_id.map(corpus[column])
    return items


def analyse(ctx, name, split, substitute, log):
    layer = selected_layer(ctx, name, split)
    pre = original_preprocessor(ctx, name, split, layer)
    items, features, valid, manifest = substitute
    cell = load_cell(ctx, name, split, substitute=(with_split_labels(ctx, name, items, split), features, valid, manifest),
                     preprocessor=pre)
    f = cell.frame
    counts = f[f.condition.isin(BANGLA)].groupby("idiom_id").condition.nunique()
    matched = set(counts[counts.eq(len(BANGLA))].index)
    keep = (f.condition.eq("en-idiom") | f.idiom_id.isin(matched)).to_numpy()
    cell.frame, cell.X = f[keep].reset_index(drop=True), cell.X[keep]
    study = make_study(ctx, cell, "pooling")
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    cells, contrasts, tokens = [], [], []
    for condition in BANGLA + ["en-idiom"]:
        t = cell.frame[cell.frame.condition.eq(condition)]
        tokens.append(dict(model=name, split=split, condition=condition, items=len(t),
                           mean_pooled_tokens=float(t.token_count.mean()), median_pooled_tokens=float(t.token_count.median())))
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        D = study.X.shape[1]
        specs = [(c, concept, c, concept, "AMBIENT", D, "project", False, 0, None, True) for c in BANGLA]
        for method in ["COV", "MLR"]:
            for c in BANGLA:
                specs += [(c, concept, c, concept, method, rank, "project", False, 0, None, True),
                          (c, concept, "en-idiom", concept, method, rank, "project", False, 0, None, True)]
        study._prefetch(specs, f"X4 {concept}")
        y_all = labels(study, concept)
        by_cell = {}
        for c in BANGLA:
            ambient = study._evaluate_sub(c, concept, c, concept, "AMBIENT", D)
            test = np.asarray(ambient["test_idx"])
            frame = study.items.iloc[test]
            order = np.argsort(frame.idiom_id.to_numpy(), kind="stable")
            entry = dict(test=test[order], idiom=frame.idiom_id.to_numpy()[order], y=y_all[test][order],
                         groups=frame.component_id.astype(str).to_numpy()[order],
                         floor=np.full(len(test), majority_label(study, c, concept)), readouts={})
            entry["readouts"]["AMBIENT_native"] = np.asarray(ambient["predicted"])[order]
            for method in ["COV", "MLR"]:
                for direction, source in [("native", c), ("cross", "en-idiom")]:
                    ev = study._evaluate_sub(c, concept, source, concept, method, rank)
                    assert np.array_equal(np.asarray(ev["test_idx"]), test)
                    entry["readouts"][f"{method}_{direction}"] = np.asarray(ev["predicted"])[order]
            by_cell[c] = entry
        reference = by_cell[BANGLA[0]]
        for c in BANGLA[1:]:
            assert np.array_equal(by_cell[c]["idiom"], reference["idiom"]), "cells are not record-matched"
        for c, entry in by_cell.items():
            classes = np.unique(entry["y"])
            for method in ["COV", "MLR"]:
                row = dict(model=name, split=split, layer=layer, concept=concept, condition=c, method=method, M=rank,
                           n=len(entry["y"]), n_components=len(np.unique(entry["groups"])),
                           ambient_native_acc=float(np.mean(entry["readouts"]["AMBIENT_native"] == entry["y"])))
                row.update(bootstrap_ssi(entry["y"], entry["readouts"][f"{method}_cross"],
                                         entry["readouts"][f"{method}_native"], entry["floor"], entry["groups"],
                                         classes, n_boot, int(study.meta["seed"]), study.cfg.min_den))
                cells.append(row)
        for readout in ["AMBIENT_native", "COV_native", "COV_cross", "MLR_native", "MLR_cross"]:
            correct = {c: (by_cell[c]["readouts"][readout] == by_cell[c]["y"]).astype(float) for c in BANGLA}
            groups = reference["groups"]
            for label, a, b in CONTRASTS:
                test = paired_two_sided(correct[a] - correct[b], groups)
                contrasts.append(dict(model=name, split=split, concept=concept, readout=readout, contrast=label,
                                      first=a, second=b, **test))
            interaction = (correct["bn-fig"] - correct["bn-maskslot"]) - (correct["bn-figctx"] - correct["bn-context"])
            contrasts.append(dict(model=name, split=split, concept=concept, readout=readout,
                                  contrast="interaction_input_x_pool", first="(fig-maskslot)", second="(figctx-context)",
                                  **paired_two_sided(interaction, groups)))
    return pd.DataFrame(cells), pd.DataFrame(contrasts), pd.DataFrame(tokens)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    phase = ctx.option("phase", "all")
    checks, cells, contrasts, tokens = [], [], [], []
    for number, name in enumerate(ctx.models, 1):
        slug = model_slug(name)
        folder = ctx.work / "extraction" / slug
        log(f"{name}: phase {phase}", model=name, completed=number - 1, total=len(ctx.models))
        if phase == "analyse" and not (folder / "extraction_manifest.json").exists():
            raise FileNotFoundError(f"{folder} has no extraction; run --phase extract (or all) first")
        # Forward passes happen once; afterwards this is an exact fingerprint cache hit.
        substitute = extract(ctx, name, log)
        if not (folder / "extraction_check.csv").exists():
            write_csv(folder / "extraction_check.csv", extraction_check(ctx, name, *substitute))
        checks.append(pd.read_csv(folder / "extraction_check.csv"))
        if phase == "extract":
            continue
        for split in ctx.splits:
            key = f"{slug}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                c, k, t = analyse(ctx, name, split, substitute, log)
                saved = cell_tables(ctx, key, {"cells.csv": c, "contrasts.csv": k, "tokens.csv": t})
            cells.append(saved["cells.csv"])
            contrasts.append(saved["contrasts.csv"])
            tokens.append(saved["tokens.csv"])
    tables = {"extraction_check.csv": concat(checks)}
    if phase != "extract":
        tables.update({"cells.csv": concat(cells), "contrasts.csv": concat(contrasts), "tokens.csv": concat(tokens)})
    finish(ctx, tables, dict(phase=phase, conditions=BANGLA + ["en-idiom"]))
    log("finished")
