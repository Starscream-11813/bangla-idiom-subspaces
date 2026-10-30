"""X17: do the volunteers' judgments predict which items transfer?

Motivation: the two volunteers' judgments are reported only class by class; whether evaluative force, class
aptness or counterpart quality relates to cross-lingual readout needs an item-level test that separates them from the
F class. The 300 judged items are discovery TEST records (25 per F class, one per held-out component); their packet
identifier is 'validation_' + sha256(idiom_id)[:12], so they are matched to the readouts without the private key.

items      split 0, F at M = 11, for the idiom alone (bn-idiom; all 300 items) and in context (bn-fig; the items with
           an exact Bangla span): correctness of the native COV and MLR readouts, of Bangla-trained probes through the
           English COV and MLR subspaces, of the zero-shot English classifier through COV (per-condition centring)
           and of the full-space native probe. Item-level evidence stays in OUT.
contrasts  for each target, outcome and judgment (both judged the class apt; both judged it good; the share judging
           the idiom evaluative; both judged the counterpart an idiom; both judged it equivalent): the coefficient of a
           linear probability model with F-class and encoder fixed effects, and, for the cross-lingual outcomes, the
           native readout's correctness as a covariate. Two-sided permutation p (the judgment permuted across items
           within F class), item bootstrap 95% interval (items resampled within class), Holm over the five judgments
           within target x outcome. Exploratory.
agreement  the counterpart-transparency covariate used by X19 (cosine of counterpart and gloss) against the volunteers'
           counterpart judgments: AUC for literal paraphrases against idioms, and Spearman with the equivalent share.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import holm

from experiments.common import Log, concat, labels, load_cell, make_study, rank_for, readout_indices
from experiments.x1_zero_shot import _shift, _zero_shot

from .common3 import HUMAN_JUDGMENTS, SOURCE, cell_tables, finish

TARGETS = ["bn-idiom", "bn-fig"]
CONCEPT = "F"
PREDICTORS = ["apt_both", "good_both", "evaluative", "idiom_cp_both", "equivalent_both"]
OUTCOMES = ["cross_cov", "zs_cov", "cross_mlr", "native_cov"]


def sample_id(idiom_id):
    return "validation_" + hashlib.sha256(str(idiom_id).encode()).hexdigest()[:12]


def human_features(path):
    """Item-level judgments of both volunteers (results/human_validation/judgments.csv, without corpus text)."""
    j = pd.read_csv(path, keep_default_na=False)
    apt = lambda v: v.isin(["good", "partial"])  # noqa: E731
    evaluative = lambda v: v.isin(["evaluative", "both"]).astype(float)  # noqa: E731
    return pd.DataFrame(dict(
        sample_id=j.sample_id, F_human=j.assigned_F.astype(int),
        apt_both=(apt(j.f_plausibility_A) & apt(j.f_plausibility_B)).astype(float),
        good_both=(j.f_plausibility_A.eq("good") & j.f_plausibility_B.eq("good")).astype(float),
        evaluative=(evaluative(j.function_A) + evaluative(j.function_B)) / 2,
        idiom_cp_both=(j.counterpart_type_A.eq("idiom") & j.counterpart_type_B.eq("idiom")).astype(float),
        paraphrase_cp_both=(j.counterpart_type_A.eq("literal_paraphrase")
                            & j.counterpart_type_B.eq("literal_paraphrase")).astype(float),
        equivalent_both=(j.counterpart_equivalence_A.eq("equivalent")
                         & j.counterpart_equivalence_B.eq("equivalent")).astype(float),
        equivalent_share=(j.counterpart_equivalence_A.eq("equivalent").astype(float)
                          + j.counterpart_equivalence_B.eq("equivalent").astype(float)) / 2))


def run_cell(ctx, study, log, wanted):
    D = study.X.shape[1]
    rank = rank_for(study, CONCEPT)
    out = []
    for target in TARGETS:
        if target not in set(study._condition) or rank is None:
            continue
        specs = [(target, CONCEPT, SOURCE, CONCEPT, m, rank, "project", False, 0, None, True) for m in ["COV", "MLR"]]
        specs += [(target, CONCEPT, target, CONCEPT, m, rank, "project", False, 0, None, True) for m in ["COV", "MLR"]]
        specs.append((target, CONCEPT, target, CONCEPT, "AMBIENT", D, "project", False, 0, None, True))
        study._prefetch(specs, f"X17 readouts {target}")
        y_all = labels(study, CONCEPT)
        test = np.asarray(readout_indices(study, target, CONCEPT)[2])
        y = y_all[test]
        evals = dict(cross_cov=study._evaluate_sub(target, CONCEPT, SOURCE, CONCEPT, "COV", rank),
                     cross_mlr=study._evaluate_sub(target, CONCEPT, SOURCE, CONCEPT, "MLR", rank),
                     native_cov=study._evaluate_sub(target, CONCEPT, target, CONCEPT, "COV", rank),
                     native_mlr=study._evaluate_sub(target, CONCEPT, target, CONCEPT, "MLR", rank),
                     ambient=study._evaluate_sub(target, CONCEPT, target, CONCEPT, "AMBIENT", D))
        zs_rows = (study.idx(SOURCE, "probe", CONCEPT), study.idx(SOURCE, "validation", CONCEPT), test)
        zs = _zero_shot(study, [study.sub(SOURCE, CONCEPT, "COV", rank)], zs_rows, y_all,
                        _shift(study, SOURCE, target, "per_condition"), study.cfg.probe_Cs)[0]
        frame = study.items.iloc[test].reset_index(drop=True)
        table = pd.DataFrame(dict(model=study.meta["model"], split=int(study.meta["seed"]), target=target,
                                  idiom_id=frame.idiom_id.astype(str), component_id=frame.component_id.astype(str),
                                  F=y))
        table["sample_id"] = table.idiom_id.map(sample_id)
        for column in ["cp_transparency", "opaque", "transparent"]:
            if column in frame:
                table[column] = frame[column].to_numpy()
        for name, ev in evals.items():
            if ev is not None:
                assert np.array_equal(np.asarray(ev["test_idx"]), test)
                table[name] = (np.asarray(ev["predicted"]) == y).astype(float)
        if zs is not None:
            table["zs_cov"] = (np.asarray(zs["predicted"]) == y).astype(float)
        keep = table.sample_id.isin(wanted)
        log(f"{study.meta['model']} {target}: {int(keep.sum())} of {len(wanted)} judged items in TEST")
        out.append(table[keep])
    return {"items.csv.gz": concat(out)}


def _design(d, predictor, outcome, covariate):
    columns = [np.ones(len(d)), d[predictor].to_numpy(float)]
    if covariate:
        columns.append(d["native_cov"].to_numpy(float))
    fixed = pd.get_dummies(d[["F", "model"]].astype(str), drop_first=True).to_numpy(float)
    return np.column_stack(columns + [fixed]), d[outcome].to_numpy(float)


def _coef(X, y):
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def contrast(d, predictor, outcome, n_perm, n_boot, seed):
    """Fixed-effects coefficient of one judgment, with within-class permutation p and item bootstrap interval."""
    d = d.dropna(subset=[predictor, outcome] + (["native_cov"] if outcome != "native_cov" else [])).reset_index(
        drop=True)
    covariate = outcome != "native_cov"
    X, y = _design(d, predictor, outcome, covariate)
    estimate = _coef(X, y)
    rng = np.random.default_rng(seed)
    item = d.groupby("sample_id").agg(F=("F", "first"), value=(predictor, "first"))
    position = {s: i for i, s in enumerate(item.index)}
    row_item = d.sample_id.map(position).to_numpy()
    rows_by_item = [np.flatnonzero(row_item == i) for i in range(len(item))]
    values = item.value.to_numpy(float)
    classes = [np.flatnonzero(item.F.to_numpy() == c) for c in np.unique(item.F.to_numpy())]
    exceed = 0
    Xp = X.copy()
    for _ in range(n_perm):
        permuted = values.copy()
        for members in classes:
            permuted[members] = values[rng.permutation(members)]
        Xp[:, 1] = permuted[row_item]
        exceed += abs(_coef(Xp, y)) >= abs(estimate) - 1e-12
    boot = []
    for _ in range(n_boot):
        picked = np.concatenate([rng.choice(members, size=len(members), replace=True) for members in classes])
        rows = np.concatenate([rows_by_item[i] for i in picked])
        Xb, yb = X[rows], y[rows]
        if np.ptp(Xb[:, 1]) > 0:
            boot.append(_coef(Xb, yb))
    lo, hi = np.quantile(boot, [.025, .975]) if boot else (np.nan, np.nan)
    high = d[predictor].ge(.5)
    return dict(estimate_pp=100 * estimate, ci_lo_pp=100 * lo, ci_hi_pp=100 * hi,
                p_perm=(1 + exceed) / (1 + n_perm), n_items=int(len(item)), n_rows=int(len(d)),
                items_high=int((values >= .5).sum()), items_low=int((values < .5).sum()),
                mean_high=float(d.loc[high, outcome].mean()) if high.any() else np.nan,
                mean_low=float(d.loc[~high, outcome].mean()) if (~high).any() else np.nan,
                native_covariate=covariate, permutation_draws=int(n_perm), bootstrap_draws=int(len(boot)))


def auc(scores, positive):
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def agreement(items, feats):
    """Counterpart transparency (X19's covariate) against the volunteers' counterpart judgments, one row per item."""
    per_item = items.drop_duplicates("sample_id").merge(feats, on="sample_id")
    if "cp_transparency" not in per_item:
        return pd.DataFrame()
    d = per_item[np.isfinite(per_item.cp_transparency.astype(float))]
    pair = d[d.idiom_cp_both.eq(1) | d.paraphrase_cp_both.eq(1)]
    rho = spearmanr(d.cp_transparency.astype(float), d.equivalent_share) if len(d) > 2 else None
    out = [dict(measure="AUC: transparency higher for literal paraphrases than idioms (both volunteers)",
                value=auc(pair.cp_transparency, pair.paraphrase_cp_both.eq(1)), n=int(len(pair))),
           dict(measure="Spearman: transparency with the share judging the counterpart equivalent",
                value=float(rho.statistic) if rho is not None else np.nan, n=int(len(d)),
                p=float(rho.pvalue) if rho is not None else np.nan)]
    for flag in ["opaque", "transparent"]:
        if flag in d:
            group = d[d[flag].astype(str).str.lower().isin(["true", "1", "1.0"])]
            out.append(dict(measure=f"share of {flag} items both judged an idiom counterpart",
                            value=float(group.idiom_cp_both.mean()) if len(group) else np.nan, n=int(len(group))))
            out.append(dict(measure=f"share of {flag} items both judged a literal paraphrase",
                            value=float(group.paraphrase_cp_both.mean()) if len(group) else np.nan, n=int(len(group))))
    return pd.DataFrame(out)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    path = ctx.option("human", None) or HUMAN_JUDGMENTS
    feats = human_features(path)
    wanted = set(feats.sample_id)
    parts = []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=[SOURCE] + TARGETS)
                study = make_study(ctx, cell, "human_items")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log, wanted))
                del study, cell
            else:
                log(f"restored {key}")
            parts.append(saved.get("items.csv.gz"))
    items = concat(parts)
    merged = items.merge(feats, on="sample_id", how="inner")
    coverage = pd.DataFrame([dict(model=model, split=split, target=target, items=int(g.sample_id.nunique()),
                                  packet_items=len(wanted),
                                  F_matches_packet=bool((g.F.astype(int) == g.F_human).all()))
                             for (model, split, target), g in merged.groupby(["model", "split", "target"])])
    n_perm = int(ctx.option("human_perm", 4999))
    n_boot = int(ctx.option("human_boot", 2000))
    rows = []
    for target in TARGETS:
        d = merged[merged.target.eq(target)]
        if d.empty:
            continue
        for outcome in OUTCOMES:
            if outcome not in d:
                continue
            block = []
            for k, predictor in enumerate(PREDICTORS):
                log(f"contrast {target} {outcome} ~ {predictor}")
                seed = 20261027 + 97 * k + 7 * OUTCOMES.index(outcome) + TARGETS.index(target)
                block.append(dict(target=target, outcome=outcome, predictor=predictor,
                                  **contrast(d, predictor, outcome, n_perm, n_boot, seed)))
            p = np.array([r["p_perm"] for r in block])
            for r, ph in zip(block, holm(p, family_size=len(p))):
                r["p_perm_holm"] = float(ph)
            rows += block
    accuracy = pd.DataFrame([dict(target=target, model=model, outcome=o, accuracy=float(g[o].mean()),
                                  n=int(g[o].notna().sum()))
                             for (target, model), g in merged.groupby(["target", "model"])
                             for o in OUTCOMES + ["native_mlr", "ambient"] if o in g])
    finish(ctx, {"contrasts.csv": pd.DataFrame(rows), "agreement.csv": agreement(items, feats),
                 "coverage.csv": coverage, "accuracy.csv": accuracy},
           dict(human=str(path).replace("\\", "/").split("/")[-1], targets=TARGETS, concept=CONCEPT,
                predictors=PREDICTORS, outcomes=OUTCOMES, human_perm=n_perm, human_boot=n_boot))
    log("finished")
