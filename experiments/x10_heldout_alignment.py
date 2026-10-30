"""X10: does the rotation result survive a Procrustes map fitted on held-out pairs?

Motivation: X7's orthogonal map was fitted on paired SPACE records, which
also define the class means from which both languages' subspaces are
estimated, so the rotated read and erase shares (the paper's read/erase figure) could be
partly circular. Here the same maps are fitted on pairs that no estimator, class
mean or subspace ever sees.

fits       fit_on = space (X7's map, refitted: the reproduction gate), probe,
           validation, and probe_validation (primary: the most held-out pairs);
           each as the full Procrustes map and the top-128 principal-plane map;
           plus ``shuffle_draws`` maps fitted on randomly re-paired
           probe_validation rows (same marginals, no correspondence), a null
           for the gain of alignment. No label enters a map.
retrieval  X7's paired TEST retrieval (NN cosine and CSLS-10, both directions,
           per-condition SPACE centring) under every map, for X7's six pairs, and
           mean pair cosines on the fit pairs, on SPACE pairs and on TEST pairs.
aligned    the planned pair (en-idiom -> bn-fig): English COV, MLR and LEACE
           subspaces carried into Bangla coordinates (basis W^T Q, LEACE input map
           W^T F) and read by a Bangla-trained probe, erased, and read zero-shot by
           an English-trained probe, exactly as in X7, with shared dimensions and
           the read/erase bound check (erase share <= omega <= read share) per cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug

from experiments.common import Log, compact_readouts, concat, erase_readouts, labels, load_cell, make_study, rank_for
from experiments.metrics import bootstrap_ssi, metric_values, ssi
from experiments.nulls import majority_label
from experiments.x7_alignment import (CSLS_K, PAIRS, REDUCED_K, condition_mean, csls, paired_rows,
                                      published_check, ranks_from, unit)

from .alignment import FIT_SETS, MAPS, PRIMARY_FIT, fit_maps, pair_cosine, rotate, shared_dims
from .common2 import cell_tables, finish

METHODS = ["COV", "MLR", "LEACE"]
SHUFFLED = f"{PRIMARY_FIT}_shuffled"
MIN_NATIVE_DROP_PP = 2.0      # erase shares are reported only where the native eraser removes >= 2 points


def fitted_maps(study, source, target, fit_sets, shuffle_draws):
    """[(fit_on, draw, {name: W}, info)] for every requested pair set, plus shuffled controls."""
    out = []
    for fit_on in fit_sets:
        maps, info = fit_maps(study, source, target, fit_on)
        out.append((fit_on, 0, maps, info))
    for draw in range(1, shuffle_draws + 1):
        maps, info = fit_maps(study, source, target, PRIMARY_FIT, shuffle=7919 * int(study.meta["seed"]) + draw)
        out.append((SHUFFLED, draw, maps, info))
    return out


def retrieval(study, source, target, fitted):
    """Paired TEST retrieval and pair cosines under the unmapped space and every fitted map."""
    ta, tb = paired_rows(study, source, target, "test")
    ca, cb = condition_mean(study, source), condition_mean(study, target)
    A_t, B_t = unit(study.X[ta] - ca), unit(study.X[tb] - cb)
    meta = dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"], source=source,
                target=target)
    spaces = [("none", 0, "centred", None, None)]
    spaces += [(fit_on, draw, name, W, info) for fit_on, draw, maps, info in fitted if draw == 0
               for name, W in maps.items()]
    rows, fits = [], []
    for fit_on, draw, space, W, info in spaces:
        P = A_t if W is None else unit(A_t @ W)
        sim = P @ B_t.T
        for scoring, score in [("nn_cosine", sim), (f"csls{CSLS_K}", csls(sim))]:
            for direction, matrix in [(f"{source}->{target}", score), (f"{target}->{source}", score.T)]:
                ranks = ranks_from(matrix)
                rows.append(dict(meta, fit_on=fit_on, space=space, scoring=scoring, direction=direction,
                                 n=len(ranks), top1=float(np.mean(ranks == 1)), top10=float(np.mean(ranks <= 10)),
                                 MRR=float(np.mean(1 / ranks)), chance_top1=1 / len(ranks)))
        fit = dict(meta, fit_on=fit_on, space=space,
                   fit_pairs=None if info is None else info["fit_pairs"])
        for name, splits in [("space", ("space",)), ("probe_validation", FIT_SETS[PRIMARY_FIT]), ("test", ("test",))]:
            cosine, n = pair_cosine(study, source, target, splits, W)
            fit[f"{name}_pair_cosine"], fit[f"{name}_pairs"] = cosine, n
        if info is not None:
            fit["fit_pair_cosine"] = pair_cosine(study, source, target, FIT_SETS[fit_on], W)[0]
        fits.append(fit)
    return rows, fits


def _share(drop, native_drop):
    return float(drop / native_drop) if np.isfinite(native_drop) and native_drop >= MIN_NATIVE_DROP_PP else np.nan


def aligned_part(study, fitted, n_boot):
    """X7's aligned readouts for every fitted map (one row per map x concept x method)."""
    rows = []
    cb = condition_mean(study, "bn-fig")
    D = study.X.shape[1]
    seed = int(study.meta["seed"])
    maps = [(fit_on, draw, name, W, info) for fit_on, draw, ms, info in fitted for name, W in ms.items()]
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        y = labels(study, concept)
        bn = tuple(study.idx("bn-fig", s, concept) for s in ["probe", "validation", "test"])
        en = (study.idx("en-idiom", "probe", concept), study.idx("en-idiom", "validation", concept), bn[2])
        tr, va, te = bn
        test_y = y[te]
        classes = np.unique(test_y)
        floor = np.full(len(test_y), majority_label(study, "bn-fig", concept))
        groups = study.items.iloc[te].component_id.astype(str).to_numpy()
        specs = [("bn-fig", concept, "bn-fig", concept, "AMBIENT", D, "project", False, 0, None, True)]
        for method in METHODS:
            specs += [("bn-fig", concept, "en-idiom", concept, method, rank, "project", False, 0, None, True),
                      ("bn-fig", concept, "bn-fig", concept, method, rank, "project", False, 0, None, True)]
        study._prefetch(specs, f"X10 published readouts {concept}")
        ambient = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, "AMBIENT", D)
        amb = float(np.mean(np.asarray(ambient["predicted"]) == test_y))
        floor_m = metric_values(test_y, floor, classes)
        bangla_cov = study.sub("bn-fig", concept, "COV", rank)
        for method in METHODS:
            sub = study.sub("en-idiom", concept, method, rank)
            native_sub = study.sub("bn-fig", concept, method, rank)
            published = study._evaluate_sub("bn-fig", concept, "en-idiom", concept, method, rank)
            native = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, method, rank)
            native_m = metric_values(test_y, native["predicted"], classes)
            unaligned_erase, native_erase = erase_readouts(
                study, study.X[tr], study.X[va], study.X[te], [sub.input_factor, native_sub.input_factor],
                [sub.basis, native_sub.basis], y, bn)
            zs_train, zs_val = sub.project(study.X[en[0]]), sub.project(study.X[en[1]])
            for fit_on, draw, alignment, W, info in maps:
                # One map per call, exactly as in X7, so the SPACE refit reproduces X7's readouts.
                aligned = rotate(sub, W, cb)
                read = compact_readouts(study, [aligned.project(study.X[tr])], [aligned.project(study.X[va])],
                                        [aligned.project(study.X[te])], y, bn)[0]
                erased = erase_readouts(study, study.X[tr], study.X[va], study.X[te], [aligned.input_factor],
                                        [aligned.basis], y, bn)[0]
                zs = compact_readouts(study, [zs_train], [zs_val], [aligned.project(study.X[te])], y, en)[0]
                row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                           method=method, alignment=alignment, fit_on=fit_on, shuffle_draw=draw,
                           fit_pairs=info["fit_pairs"], M=rank, n=len(test_y),
                           shared_dims_unaligned=shared_dims(native_sub.basis, sub.basis),
                           shared_dims_aligned=shared_dims(native_sub.basis, aligned.basis),
                           shared_dims_cov_aligned=shared_dims(bangla_cov.basis, aligned.basis),
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
                                     seed, study.cfg.min_den)
                row.update({f"aligned_{k}": v for k, v in boot.items() if k.startswith("SSI_")})
                for name, ev in [("erase_aligned", erased), ("erase_unaligned", unaligned_erase),
                                 ("erase_native", native_erase)]:
                    row[f"{name}_acc"] = float(np.mean(np.asarray(ev["predicted"]) == test_y))
                    row[f"{name}_drop_pp"] = 100 * (amb - row[f"{name}_acc"])
                row["probe_converged_aligned"] = bool(read.get("probe_converged", True))
                row["probe_converged_zero_shot"] = bool(zs.get("probe_converged", True))
                # The read/erase bound on the same quantities as the paper's figure: omega = shared dims / M.
                for state in ["unaligned", "aligned"]:
                    omega = row[f"shared_dims_{state}"] / rank
                    read_share = row["SSI_published_cross_acc" if state == "unaligned" else "SSI_aligned_cross_acc"]
                    erase_share = _share(row[f"erase_{state}_drop_pp"], row["erase_native_drop_pp"])
                    row[f"omega_{state}"] = omega
                    row[f"read_share_{state}"] = read_share
                    row[f"erase_share_{state}"] = erase_share
                    row[f"read_ok_{state}"] = bool(read_share >= omega) if np.isfinite(read_share) else None
                    row[f"erase_ok_{state}"] = bool(erase_share <= omega) if np.isfinite(erase_share) else None
                rows.append(row)
    return rows


def summarize(aligned):
    """Per split x fit set x map x concept x method: medians and read/erase bound counts."""
    if aligned.empty:
        return pd.DataFrame()
    a = aligned.copy()
    a["prop3_ok"] = a.read_ok_aligned.astype("boolean") & a.erase_ok_aligned.astype("boolean")
    keys = ["split", "fit_on", "alignment", "concept", "method"]
    out = a.groupby(keys, dropna=False).agg(
        cells=("model", "size"), models=("model", "nunique"), fit_pairs=("fit_pairs", "median"),
        shared_dims_unaligned=("shared_dims_unaligned", "median"),
        shared_dims_aligned=("shared_dims_aligned", "median"),
        omega_aligned_min=("omega_aligned", "min"), omega_aligned_max=("omega_aligned", "max"),
        SSI_published_cross_acc=("SSI_published_cross_acc", "median"),
        SSI_aligned_cross_acc=("SSI_aligned_cross_acc", "median"),
        SSI_aligned_cross_acc_min=("SSI_aligned_cross_acc", "min"),
        SSI_aligned_cross_acc_max=("SSI_aligned_cross_acc", "max"),
        SSI_aligned_zero_shot_acc=("SSI_aligned_zero_shot_acc", "median"),
        erase_share_aligned=("erase_share_aligned", "median"),
        erase_aligned_drop_pp=("erase_aligned_drop_pp", "median"),
        erase_native_drop_pp=("erase_native_drop_pp", "median"),
        read_ok=("read_ok_aligned", lambda s: int(s.astype("boolean").fillna(False).sum())),
        erase_ok=("erase_ok_aligned", lambda s: int(s.astype("boolean").fillna(False).sum())),
        erase_defined=("erase_share_aligned", lambda s: int(np.isfinite(s).sum())),
        prop3_ok=("prop3_ok", lambda s: int(s.fillna(False).sum()))).reset_index()
    return out


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    fit_sets = [s for s in ctx.option("fit_sets", ",".join(FIT_SETS)).split(",") if s]
    unknown = sorted(set(fit_sets) - set(FIT_SETS))
    if unknown:
        raise ValueError(f"Unknown fit sets {unknown}; choose from {sorted(FIT_SETS)}")
    shuffle_draws = int(ctx.option("shuffle_draws", 3))
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
                    log(f"{name}: maps and retrieval {source} <-> {target}")
                    draws = shuffle_draws if (source, target) == ("en-idiom", "bn-fig") else 0
                    fitted = fitted_maps(study, source, target, fit_sets, draws)
                    rows, fit = retrieval(study, source, target, fitted)
                    r_rows += rows
                    f_rows += fit
                    if (source, target) == ("en-idiom", "bn-fig"):
                        planned = fitted
                log(f"{name}: aligned readouts under {sum(len(m) for _, _, m, _ in planned or [])} maps")
                a_rows = aligned_part(study, planned, n_boot) if planned is not None else []
                saved = cell_tables(ctx, key, {"retrieval.csv": pd.DataFrame(r_rows),
                                               "procrustes_fit.csv": pd.DataFrame(f_rows),
                                               "aligned.csv": pd.DataFrame(a_rows)})
                del study, cell
            else:
                log(f"restored {key}")
            retrieval_rows.append(saved["retrieval.csv"])
            fits.append(saved["procrustes_fit.csv"])
            aligned.append(saved["aligned.csv"])
    aligned_all = concat(aligned)
    finish(ctx, {"retrieval.csv": published_check(concat(retrieval_rows)), "procrustes_fit.csv": concat(fits),
                 "aligned.csv": aligned_all, "summary.csv": summarize(aligned_all)},
           dict(pairs=PAIRS, csls_k=CSLS_K, reduced_k=REDUCED_K, fit_sets=fit_sets, maps=list(MAPS),
                primary_fit=PRIMARY_FIT, shuffle_draws=shuffle_draws, min_native_drop_pp=MIN_NATIVE_DROP_PP))
    log("finished")
