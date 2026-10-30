"""X7: hubness-corrected retrieval (CSLS) and a learned orthogonal (Procrustes) alignment.

Motivation: the paper's retrieval is nearest-neighbour cosine after
per-condition centring and compares neither hubness-corrected retrieval nor
learned alignment maps (Conneau et al., 2018; Artetxe et al., 2018).

retrieval  paired TEST items, both directions, per-condition SPACE centring:
           NN cosine (the published setting; checked against the published
           numbers), CSLS with k=10, and both after an orthogonal map W fitted
           on paired SPACE records (row-normalized), so no map sees a TEST pair.
           Two maps: full orthogonal Procrustes, and Procrustes restricted to
           the top-128 principal plane of the SPACE pairs (identity elsewhere),
           which guards against overfitting a D x D rotation to a few thousand
           pairs; SPACE and TEST pair cosines show which one generalizes.
aligned    the planned pair (en-idiom -> bn-fig): English subspaces carried into
           Bangla coordinates by W (basis W^T Q, LEACE input map W^T F), then
           read (Bangla-trained probe, the published protocol), erased, and read
           zero-shot by an English-trained probe; plus shared dimensions with the
           Bangla subspace before and after alignment. If alignment closes the
           read gap or makes English erasers work, the shared structure is a
           rotation; if not, the subspaces only partly overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import Subspace

from .common import (REPO, Log, cell_tables, compact_readouts, concat, erase_readouts, finish, labels, load_cell,
                     make_study, rank_for)
from .metrics import bootstrap_ssi, metric_values, ssi
from .nulls import majority_label

PAIRS = [("en-idiom", "bn-fig"), ("en-idiom", "bn-idiom"), ("en-trans", "bn-fig"), ("en-trans", "bn-sentence"),
         ("en-para", "bn-para"), ("en-idiom", "bn-context")]
CSLS_K = 10
REDUCED_K = 128
PUBLISHED = REPO / "results" / "discovery" / "S3_baselines.csv"


def paired_rows(study, source, target, split):
    items = study.items
    a = items[(items.condition == source) & (items.split == split)].reset_index(names="a")[["a", "idiom_id"]]
    b = items[(items.condition == target) & (items.split == split)].reset_index(names="b")[["b", "idiom_id"]]
    pairs = a.merge(b, on="idiom_id")
    assert not pairs.idiom_id.duplicated().any(), "retrieval pairs must be one-to-one"
    return pairs.a.to_numpy(), pairs.b.to_numpy()


def condition_mean(study, condition):
    return study.X[(study._condition == condition) & (study._split == "space")].mean(0)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def ranks_from(sim):
    """Midrank of the true partner (diagonal) among all candidates, as in the published baseline."""
    true = np.diag(sim)
    higher = (sim > true[:, None] + 1e-12).sum(1)
    ties = (np.abs(sim - true[:, None]) <= 1e-12).sum(1) - 1
    return 1 + higher + .5 * ties


def csls(sim, k=CSLS_K):
    k = min(k, sim.shape[1], sim.shape[0])
    r_query = np.sort(sim, axis=1)[:, -k:].mean(1)
    r_candidate = np.sort(sim, axis=0)[-k:, :].mean(0)
    return 2 * sim - r_query[:, None] - r_candidate[None, :]


def procrustes(A, B):
    """Orthogonal W minimizing ||A W - B||_F (rows already centred and normalized)."""
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def reduced_procrustes(A, B, k=REDUCED_K):
    """Rotation inside the top-k principal plane of the paired SPACE rows, identity outside.

    W = I + V (R - I) V^T is orthogonal; it has k(k-1)/2 instead of D(D-1)/2 free
    parameters, a guard against fitting a D x D rotation to a few thousand pairs.
    """
    k = min(k, A.shape[1], 2 * len(A) - 1)
    _, _, Vt = np.linalg.svd(np.vstack([A, B]), full_matrices=False)
    V = Vt[:k].T
    R = procrustes(A @ V, B @ V)
    return np.eye(A.shape[1]) + V @ (R - np.eye(k)) @ V.T


def retrieval(study, source, target):
    sa, sb = paired_rows(study, source, target, "space")
    ta, tb = paired_rows(study, source, target, "test")
    ca, cb = condition_mean(study, source), condition_mean(study, target)
    A_s, B_s = unit(study.X[sa] - ca), unit(study.X[sb] - cb)
    A_t, B_t = unit(study.X[ta] - ca), unit(study.X[tb] - cb)
    maps = {"procrustes": procrustes(A_s, B_s), f"procrustes_pca{REDUCED_K}": reduced_procrustes(A_s, B_s)}
    rows = []
    spaces = {"centred": (A_t, B_t)}
    spaces.update({name: (unit(A_t @ W), B_t) for name, W in maps.items()})
    for space, (P, Q) in spaces.items():
        sim = P @ Q.T
        for scoring, score in [("nn_cosine", sim), ("csls10", csls(sim))]:
            for direction, matrix in [(f"{source}->{target}", score), (f"{target}->{source}", score.T)]:
                ranks = ranks_from(matrix)
                rows.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                                 source=source, target=target, space=space, scoring=scoring, direction=direction,
                                 n=len(ranks), top1=float(np.mean(ranks == 1)), top10=float(np.mean(ranks <= 10)),
                                 MRR=float(np.mean(1 / ranks)), chance_top1=1 / len(ranks)))
    fits = []
    for name, W in [("centred", np.eye(A_s.shape[1]))] + list(maps.items()):
        fits.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), source=source, target=target,
                         space=name, space_pairs=len(sa), test_pairs=len(ta),
                         space_pair_cosine=float(np.mean(np.sum(unit(A_s @ W) * B_s, 1))),
                         test_pair_cosine=float(np.mean(np.sum(unit(A_t @ W) * B_t, 1)))))
    return rows, fits, maps


def aligned_part(study, maps, n_boot):
    rows = []
    cb = condition_mean(study, "bn-fig")
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        y = labels(study, concept)
        bn = tuple(study.idx("bn-fig", s, concept) for s in ["probe", "validation", "test"])
        en = (study.idx("en-idiom", "probe", concept), study.idx("en-idiom", "validation", concept), bn[2])
        tr, va, te = bn
        test_y = y[te]
        classes = np.unique(test_y)
        floor = np.full(len(test_y), majority_label(study, "bn-fig", concept))
        groups = study.items.iloc[te].component_id.astype(str).to_numpy()
        D = study.X.shape[1]
        specs = [("bn-fig", concept, "bn-fig", concept, "AMBIENT", D, "project", False, 0, None, True)]
        for method in ["COV", "MLR", "LEACE"]:
            specs += [("bn-fig", concept, "en-idiom", concept, method, rank, "project", False, 0, None, True),
                      ("bn-fig", concept, "bn-fig", concept, method, rank, "project", False, 0, None, True)]
        study._prefetch(specs, f"X7 published readouts {concept}")
        ambient = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, "AMBIENT", D)
        amb = float(np.mean(np.asarray(ambient["predicted"]) == test_y))
        floor_m = metric_values(test_y, floor, classes)
        for method in ["COV", "MLR", "LEACE"]:
            sub = study.sub("en-idiom", concept, method, rank)
            native_sub = study.sub("bn-fig", concept, method, rank)
            published = study._evaluate_sub("bn-fig", concept, "en-idiom", concept, method, rank)
            native = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, method, rank)
            native_m = metric_values(test_y, native["predicted"], classes)
            unaligned_erase, native_erase = erase_readouts(
                study, study.X[tr], study.X[va], study.X[te], [sub.input_factor, native_sub.input_factor],
                [sub.basis, native_sub.basis], y, bn)
            for alignment, W in maps.items():
                factor = None if sub.projection_factor is None else W.T @ sub.projection_factor
                aligned = Subspace(cb, W.T @ sub.basis, sub.effective_rank, method, rank, factor)
                read = compact_readouts(study, [aligned.project(study.X[tr])], [aligned.project(study.X[va])],
                                        [aligned.project(study.X[te])], y, bn)[0]
                erased = erase_readouts(study, study.X[tr], study.X[va], study.X[te], [aligned.input_factor],
                                        [aligned.basis], y, bn)[0]
                zs = compact_readouts(study, [sub.project(study.X[en[0]])], [sub.project(study.X[en[1]])],
                                      [aligned.project(study.X[te])], y, en)[0]
                row = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                           concept=concept, method=method, alignment=alignment, M=rank, n=len(test_y),
                           shared_dims_unaligned=float(np.linalg.norm(native_sub.basis.T @ sub.basis) ** 2),
                           shared_dims_aligned=float(np.linalg.norm(native_sub.basis.T @ aligned.basis) ** 2),
                           random_expected_shared_dims=rank ** 2 / D, ambient_acc=amb)
                for name, ev in [("published_cross", published), ("aligned_cross", read), ("aligned_zero_shot", zs)]:
                    m = metric_values(test_y, ev["predicted"], classes)
                    for k in ["acc", "bal", "f1"]:
                        row[f"{name}_{k}"] = m[k]
                        row[f"SSI_{name}_{k}"] = float(ssi(m[k], native_m[k], floor_m[k], study.cfg.min_den))
                for k in ["acc", "bal", "f1"]:
                    row[f"native_{k}"] = native_m[k]
                    row[f"floor_{k}"] = floor_m[k]
                boot = bootstrap_ssi(test_y, read["predicted"], native["predicted"], floor, groups, classes, n_boot,
                                     int(study.meta["seed"]), study.cfg.min_den)
                row.update({f"aligned_{k}": v for k, v in boot.items() if k.startswith("SSI_")})
                for name, ev in [("erase_aligned", erased), ("erase_unaligned", unaligned_erase),
                                 ("erase_native", native_erase)]:
                    row[f"{name}_acc"] = float(np.mean(np.asarray(ev["predicted"]) == test_y))
                    row[f"{name}_drop_pp"] = 100 * (amb - row[f"{name}_acc"])
                rows.append(row)
    return rows


def published_check(table):
    """Attach the published per-condition-centred NN retrieval (split 0) to the matching rows."""
    if not PUBLISHED.exists() or table.empty:
        return table
    ref = pd.read_csv(PUBLISHED)
    ref = ref[ref.centering.eq("per_condition_space")][["model", "source_condition", "condition", "n", "retr_top1",
                                                         "retr_MRR"]]
    ref = ref.rename(columns={"source_condition": "source", "condition": "target", "n": "published_n",
                              "retr_top1": "published_top1", "retr_MRR": "published_MRR"})
    ref["space"], ref["scoring"], ref["split"] = "centred", "nn_cosine", 0
    ref["direction"] = ref.source + "->" + ref.target
    merged = table.merge(ref, on=["model", "split", "source", "target", "space", "scoring", "direction"], how="left")
    merged["diff_top1_vs_published"] = merged.top1 - merged.published_top1
    merged["diff_MRR_vs_published"] = merged.MRR - merged.published_MRR
    return merged


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    retrieval_rows, fits, aligned = [], [], []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                conditions = sorted({c for p in PAIRS for c in p})
                cell = load_cell(ctx, name, split, conditions=conditions)
                study = make_study(ctx, cell, "alignment")
                r_rows, f_rows, planned = [], [], None
                for source, target in PAIRS:
                    if not {source, target} <= set(study._condition):
                        continue
                    log(f"{name}: retrieval {source} <-> {target}")
                    rows, fit, maps = retrieval(study, source, target)
                    r_rows += rows
                    f_rows += fit
                    if (source, target) == ("en-idiom", "bn-fig"):
                        planned = maps
                log(f"{name}: aligned readouts")
                a_rows = aligned_part(study, planned, n_boot) if planned is not None else []
                saved = cell_tables(ctx, key, {"retrieval.csv": pd.DataFrame(r_rows), "procrustes_fit.csv":
                                               pd.DataFrame(f_rows), "aligned.csv": pd.DataFrame(a_rows)})
                del study, cell
            else:
                log(f"restored {key}")
            retrieval_rows.append(saved["retrieval.csv"])
            fits.append(saved["procrustes_fit.csv"])
            aligned.append(saved["aligned.csv"])
    finish(ctx, {"retrieval.csv": published_check(concat(retrieval_rows)), "procrustes_fit.csv": concat(fits),
                 "aligned.csv": concat(aligned)}, dict(pairs=PAIRS, csls_k=CSLS_K, reduced_k=REDUCED_K))
    log("finished")
