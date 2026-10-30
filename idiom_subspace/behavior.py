"""Optional, item-level geometry/behavior analysis using externally supplied ratings.

This module does not generate behavioral evidence. Scores must be collected on
the same idiom/model identities as the held-out geometry. A retrieval decision
for one item is a binary outcome, not an item-level subspace separation index.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCORE_KEYS = ["idiom_id", "model"]
OPTIONAL_SCORE_KEYS = ["condition", "replicate", "rater"]
DEFAULT_PREDICTORS = ("cross_correct", "within_correct", "retrieval_margin", "cross_margin", "within_margin",
                      "figurative_margin", "cosine_figurative", "cosine_literal_control")
DEFAULT_COVARIATES = ("frequency", "length", "idiom_len", "token_count", "cultural", "rural", "rare",
                      "religious", "historical", "anchored", "very_common", "very_rare")
STRATA = ("feature", "direction", "layer", "representation", "lang", "language", "seed", "label_seed",
          "concept", "method", "M", "e", "p", "source_condition", "control", "split_mode")


def _scores(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(
        value, dtype={"idiom_id": str, "model": str}
    )
    required = SCORE_KEYS + ["score"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Behavior scores are missing columns: {missing}")
    if frame.empty:
        raise ValueError("Behavior scores contain no item-level observations.")
    keys = SCORE_KEYS + [c for c in OPTIONAL_SCORE_KEYS if c in frame]
    for column in keys:
        if frame[column].isna().any():
            raise ValueError(f"Behavior key {column!r} contains missing values.")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Behavior key {column!r} contains empty values.")
    aggregate_names = {"all", "overall", "mean", "average", "aggregate", "total", "pooled", "*"}
    if frame.idiom_id.str.lower().isin(aggregate_names).any():
        raise ValueError("Aggregate scores cannot substitute for per-idiom behavior scores.")
    for column in ("level", "scope", "aggregation"):
        if column in frame and frame[column].astype(str).str.lower().isin(aggregate_names).any():
            raise ValueError(f"Aggregate score rows detected in {column!r}.")
    frame["score"] = pd.to_numeric(frame.score, errors="raise")
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 5).all():
        raise ValueError("Each behavior score must be finite and between 0 and 5 inclusive.")
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Behavior observation keys must be unique: {keys}; "
                         f"duplicate examples: {frame.loc[duplicates, keys].head(3).to_dict('records')}")
    return frame


def validate_behavior_scores(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read and validate a CSV with idiom_id, model, score and optional keys."""
    return _scores(path)


def write_score_template(items: pd.DataFrame, path: str | Path,
                         conditions: Sequence[str] | None = None) -> pd.DataFrame:
    """Export blank rows for real annotation; blank scores are deliberately invalid."""
    if not set(SCORE_KEYS).issubset(items):
        raise ValueError("Template source requires idiom_id and model columns.")
    if conditions is not None and (not conditions or len(set(conditions)) != len(conditions)):
        raise ValueError("Provide at least one distinct condition.")
    selected = [c for c in ("idiom_id", "model", "condition", "replicate", "source_id", "idiom", "prompt", "generation", "reference_meaning") if c in items]
    if "condition" in items:
        frame = items[selected].copy()
        if "status" in items:
            frame = frame.loc[items.status.eq("generated")]
        if conditions is not None:
            frame = frame.loc[frame.condition.isin(conditions)]
        keys = SCORE_KEYS + ["condition"] + (["replicate"] if "replicate" in frame else [])
        if frame.duplicated(keys).any():
            raise ValueError("Template source has repeated generation keys; provide an explicit replicate column.")
        frame = frame.assign(score="", rater="")
    else:
        frame = items[selected].copy()
        if frame.duplicated(SCORE_KEYS).any():
            raise ValueError("Template source has repeated idiom/model keys; supply condition or replicate metadata.")
        frame = pd.concat([frame.assign(condition=c, score="", rater="") for c in (conditions or ("baseline",))], ignore_index=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return frame


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    result = float(spearmanr(x, y).statistic)
    return result if np.isfinite(result) else None


def _json_finite(value):
    """Use JSON null for undefined inferential quantities, never NaN literals."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _cluster_correlation(frame: pd.DataFrame, predictor: str, iterations: int,
                         rng: np.random.Generator) -> dict:
    cluster_column = "component_id" if "component_id" in frame else "idiom_id"
    complete = frame[list(dict.fromkeys(["idiom_id", cluster_column, predictor, "score"]))].dropna()
    estimate = _finite_spearman(complete[predictor].to_numpy(), complete.score.to_numpy())
    clusters = [group for _, group in complete.groupby(cluster_column, sort=True)]
    samples = []
    if estimate is not None and len(clusters) >= 3:
        for _ in range(iterations):
            sampled = pd.concat([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
            rho = _finite_spearman(sampled[predictor].to_numpy(), sampled.score.to_numpy())
            if rho is not None:
                samples.append(rho)
    interval = np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) >= 20 else [None, None]
    return {"predictor": predictor, "n_observations": len(complete), "n_idioms": complete.idiom_id.nunique(),
            "n_components": complete.component_id.nunique() if "component_id" in complete else None,
            "n_bootstrap_clusters": len(clusters),
            "spearman_rho": estimate, "ci_95_low": interval[0], "ci_95_high": interval[1],
            "bootstrap_unit": cluster_column, "valid_bootstrap_samples": len(samples),
            "status": "ok" if estimate is not None else "insufficient_variation_or_observations"}


def _mixed_model(frame: pd.DataFrame, predictor: str, covariates: Sequence[str]) -> dict:
    """Crossed random intercepts, preserving explicit diagnostic failures."""
    columns = SCORE_KEYS + ["score", predictor] + list(covariates)
    if "condition" in frame:
        columns.append("condition")
    complete = frame[columns].dropna().copy()
    diagnostic = {"predictor": predictor, "n_observations": len(complete),
                  "n_idioms": complete.idiom_id.nunique(), "n_models": complete.model.nunique(),
                  "covariates": list(covariates), "random_effects": ["idiom_id", "model"]}
    if complete.idiom_id.nunique() < 3 or complete.model.nunique() < 2:
        return dict(diagnostic, status="not_fit", reason="Crossed effects require at least three idioms and two models.")
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        return dict(diagnostic, status="missing_dependency", reason=str(exc))
    terms = []
    dropped = []
    for column in [predictor] + list(covariates):
        if complete[column].nunique() < 2:
            dropped.append(column)
            continue
        # Standardize continuous predictors for a well-scaled optimizer. Binary
        # indicators retain their interpretable zero/one coding.
        if complete[column].nunique() > 2:
            complete[column] = (complete[column] - complete[column].mean()) / complete[column].std(ddof=0)
        terms.append(f'Q("{column}")')
    diagnostic["dropped_constant_predictors"] = dropped
    if predictor in dropped:
        return dict(diagnostic, status="not_fit", reason="The geometry predictor is constant.")
    if "condition" in complete and complete.condition.nunique() > 1:
        terms.append("C(condition)")
    if len(complete) <= len(terms) + 3:
        return dict(diagnostic, status="not_fit", reason="Too few complete observations for the specified fixed effects.")
    formula = "score ~ " + " + ".join(terms)
    diagnostic["formula"] = formula
    complete["_single_group"] = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.mixedlm(formula, complete, groups="_single_group", re_formula="0",
                                vc_formula={"idiom_id": "0 + C(idiom_id)", "model": "0 + C(model)"})
            result = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as exc:
            return dict(diagnostic, status="fit_failed", error_type=type(exc).__name__, reason=str(exc),
                        warnings=[str(w.message) for w in caught])
    coefficient_name = f'Q("{predictor}")'
    interval = result.conf_int().loc[coefficient_name]
    return dict(diagnostic, status="converged" if result.converged else "not_converged",
                warnings=[str(w.message) for w in caught],
                predictor_estimate=float(result.params[coefficient_name]),
                predictor_standard_error=float(result.bse[coefficient_name]),
                predictor_pvalue=float(result.pvalues[coefficient_name]),
                predictor_ci_95=[float(interval.iloc[0]), float(interval.iloc[1])],
                variance_components={name: float(value) for name, value in zip(model.exog_vc.names, result.vcomp)},
                interpretation="Association only; continuous predictors standardized; ordinal scores modeled as continuous.")


def geometry_behavior_analysis(geometry_df: pd.DataFrame, scores_df: pd.DataFrame,
                               out_dir: str | Path, *, predictors: Sequence[str] | None = None,
                               covariates: Sequence[str] | None = None,
                               bootstrap_iterations: int = 1000, seed: int = 0,
                               condition_matches_geometry: bool = False) -> dict:
    """Join explicit item/model IDs and report correlations and crossed effects.

    Ratings/replicates are averaged within idiom/model/condition before modeling.
    Bootstrap clusters use counterpart-connected component_id when supplied;
    otherwise they use idiom_id. With layer_role=validation_selected, each
    model's separately selected block is compared without stratifying by its
    absolute layer index; only one layer per model/configuration is permitted.
    Feature, direction and layer geometries are analyzed separately. Geometry
    ``condition`` denotes a representation (e.g., bn-fig), whereas a score's
    ``condition`` usually denotes baseline/erase-F. They remain distinct unless
    condition_matches_geometry=True explicitly requests a same-condition join.
    No encoder model is matched to a different decoder model's behavior.
    """
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative.")
    geometry = geometry_df.copy()
    if not set(SCORE_KEYS).issubset(geometry):
        raise ValueError("Geometry requires explicit idiom_id and model columns.")
    if any("ssi" in str(column).lower() for column in geometry.columns if column not in SCORE_KEYS):
        raise ValueError("Supply item-level retrieval outcomes or margins, not a quantity labeled per-idiom SSI.")
    for column in SCORE_KEYS:
        if geometry[column].isna().any():
            raise ValueError(f"Geometry key {column} contains missing values.")
        geometry[column] = geometry[column].astype(str).str.strip()
        if geometry[column].eq("").any():
            raise ValueError(f"Geometry key {column} contains empty values.")
    if "component_id" in geometry:
        if geometry.component_id.isna().any() or geometry.component_id.astype(str).str.strip().eq("").any():
            raise ValueError("component_id must be present for every geometry row when supplied.")
        geometry["component_id"] = geometry.component_id.astype(str)
        if geometry.groupby("idiom_id").component_id.nunique().gt(1).any():
            raise ValueError("An idiom_id must map to one counterpart component across models and geometries.")
    scores = _scores(scores_df)
    selected = list(predictors) if predictors is not None else [p for p in DEFAULT_PREDICTORS if p in geometry]
    if not selected or not set(selected).issubset(geometry):
        raise ValueError(f"Provide item-level geometry predictors; expected one of {DEFAULT_PREDICTORS}.")
    for column in selected:
        geometry[column] = pd.to_numeric(geometry[column], errors="raise").astype(float)
        if not np.isfinite(geometry[column].dropna()).all():
            raise ValueError(f"Geometry predictor {column} contains non-finite values.")
        if column.endswith("correct") and not geometry[column].dropna().isin([0, 1]).all():
            raise ValueError(f"Binary retrieval outcome {column} must contain only zero or one.")
    confounds = list(covariates) if covariates is not None else [c for c in DEFAULT_COVARIATES if c in geometry]
    if not set(confounds).issubset(geometry):
        raise ValueError("Requested covariates are absent from geometry.")
    if len(set(selected + confounds)) != len(selected + confounds):
        raise ValueError("Geometry predictors and covariates must be distinct column names.")
    if any(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", column) is None for column in selected + confounds):
        raise ValueError("Predictor/covariate names must be ordinary identifier column names.")
    for column in confounds:
        if geometry[column].dtype == object:
            geometry[column] = geometry[column].replace({"True": 1, "False": 0, "true": 1, "false": 0})
        geometry[column] = pd.to_numeric(geometry[column], errors="raise").astype(float)
        if not np.isfinite(geometry[column].dropna()).all():
            raise ValueError(f"Covariate {column} contains non-finite values.")
    if "condition" in geometry and not condition_matches_geometry:
        geometry = geometry.rename(columns={"condition": "geometry_condition"})
    if condition_matches_geometry and not ("condition" in geometry and "condition" in scores):
        raise ValueError("Same-condition matching requires condition in both geometry and scores.")
    strata = [s for s in STRATA + ("geometry_condition",) if s in geometry]
    if "layer_role" in geometry:
        if geometry.layer_role.isna().any() or geometry.layer_role.astype(str).str.strip().eq("").any():
            raise ValueError("layer_role must be nonempty for every geometry row.")
        geometry["layer_role"] = geometry.layer_role.astype(str)
        selected_layers = geometry.layer_role.eq("validation_selected")
        if selected_layers.any():
            if not selected_layers.all():
                raise ValueError("Analyze validation_selected layer rows separately from unselected layer grids.")
            if "layer" not in geometry or geometry.layer.isna().any():
                raise ValueError("validation_selected geometry must retain its actual layer column for audit.")
            strata = [s for s in strata if s != "layer"] + ["layer_role"]
            layer_keys = list(dict.fromkeys(["model"] + strata + (["condition"] if condition_matches_geometry else [])))
            if geometry.groupby(layer_keys, dropna=False).layer.nunique().gt(1).any():
                raise ValueError("validation_selected permits only one actual layer within each model/configuration stratum.")
        else:
            strata.append("layer_role")
    join_keys = SCORE_KEYS + (["condition"] if condition_matches_geometry else [])
    if "condition" in join_keys:
        geometry["condition"] = geometry.condition.astype(str)
    identity = list(dict.fromkeys(join_keys + strata))
    if geometry.duplicated(identity).any():
        raise ValueError(f"Geometry rows must be unique for {identity}; add explicit strata or choose one geometry.")
    score_keys = SCORE_KEYS + (["condition"] if "condition" in scores else [])
    averaged = scores.groupby(score_keys, as_index=False).agg(score=("score", "mean"), n_ratings=("score", "count"))
    matched = geometry.merge(averaged, on=join_keys, how="inner", validate="many_to_many")
    if matched.empty:
        raise ValueError("No exact idiom_id/model matches between geometry and behavior scores.")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output / "behavior_matched_rows.csv", index=False)
    unmatched = scores.merge(geometry[join_keys].drop_duplicates(), on=join_keys, how="left", indicator=True)
    unmatched = unmatched[unmatched._merge == "left_only"].drop(columns="_merge")
    unmatched.to_csv(output / "behavior_unmatched_scores.csv", index=False)
    correlations = []
    models = []
    rng = np.random.default_rng(seed)
    groups = matched.groupby(strata, dropna=False, sort=True) if strata else [((), matched)]
    for group_key, frame in groups:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        labels = {k: str(v) for k, v in zip(strata, values)}
        for predictor in selected:
            correlations.append(dict(labels, scope="pooled", **_cluster_correlation(frame, predictor, bootstrap_iterations, rng)))
            for model_id, model_frame in frame.groupby("model", sort=True):
                correlations.append(dict(labels, scope="within_model", model=str(model_id),
                                         **_cluster_correlation(model_frame, predictor, bootstrap_iterations, rng)))
            models.append(dict(labels, **_mixed_model(frame, predictor, confounds)))
    pd.DataFrame(correlations).to_csv(output / "behavior_correlations.csv", index=False)
    report = {"status": "completed", "score_rows": len(scores), "matched_analysis_rows": len(matched),
              "unmatched_score_rows": len(unmatched), "model_identity_matching": "exact",
              "condition_matches_geometry": condition_matches_geometry,
              "rating_handling": "mean within idiom/model/condition; n_ratings retained",
              "bootstrap_iterations": bootstrap_iterations, "seed": seed,
              "predictors": selected, "covariates": confounds,
              "missing_default_covariates": [c for c in DEFAULT_COVARIATES if c not in geometry],
              "strata": strata, "mixed_models": models,
              "bootstrap_unit": "component_id" if "component_id" in geometry else "idiom_id",
              "n_idioms": matched.idiom_id.nunique(),
              "n_components": matched.component_id.nunique() if "component_id" in matched else None,
              "caveats": ["Observational association does not establish causality.",
                          "Scores use an ordinal 0–5 rubric; the linear mixed model is an approximation.",
                          "Cluster intervals resample counterpart components when provided, otherwise idioms, preserving their model observations.",
                          "Multiple predictors/strata are exploratory; p-values are unadjusted."]}
    report = _json_finite(report)
    (output / "behavior_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def validate_rubric_concordance(automatic: str | Path | pd.DataFrame,
                                human: str | Path | pd.DataFrame,
                                out_dir: str | Path | None = None) -> dict:
    """Pair automatic and human rubric observations on explicit item/model keys.

    Multiple human raters are kept for quadratic-weighted kappa. Spearman uses
    the mean human score per item/model/condition/replicate. Fractional rubric
    values are supported for rank correlation, but never rounded for kappa.
    """
    auto, people = _scores(automatic), _scores(human)
    keys = SCORE_KEYS + [c for c in ("condition", "replicate") if c in auto and c in people]
    for column in ("condition", "replicate"):
        if (column in auto) != (column in people):
            raise ValueError(f"Both score sources must include {column} when either does.")
    if auto.duplicated(keys).any():
        raise ValueError("Automatic scores must have one observation per paired item/model/condition/replicate.")
    paired = people.merge(auto[keys + ["score"]], on=keys, suffixes=("_human", "_automatic"), how="inner", validate="many_to_one")
    if paired.empty:
        raise ValueError("No matching automatic/human score identities.")
    per_item = paired.groupby(keys, as_index=False).agg(score_human=("score_human", "mean"),
                                                      score_automatic=("score_automatic", "first"))
    report = {"matched_ratings": len(paired), "matched_items": len(per_item),
              "unmatched_human_rows": len(people) - len(paired),
              "unmatched_automatic_rows": len(auto) - len(per_item),
              "spearman_rho": _finite_spearman(per_item.score_human.to_numpy(), per_item.score_automatic.to_numpy()),
              "quadratic_weighted_kappa": None, "kappa_status": "requires_integer_rubric_scores"}
    human_values = paired.score_human.to_numpy()
    auto_values = paired.score_automatic.to_numpy()
    if np.equal(human_values, np.floor(human_values)).all() and np.equal(auto_values, np.floor(auto_values)).all():
        observed = np.zeros((6, 6), dtype=float)
        np.add.at(observed, (human_values.astype(int), auto_values.astype(int)), 1)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
        weight = (np.arange(6)[:, None] - np.arange(6)[None, :]) ** 2 / 25
        denominator = float((weight * expected).sum())
        report["quadratic_weighted_kappa"] = 1 - float((weight * observed).sum()) / denominator if denominator > 0 else None
        report["kappa_status"] = "ok" if denominator > 0 else "undefined_constant_ratings"
    if out_dir is not None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        paired.to_csv(output / "rubric_paired_ratings.csv", index=False)
        (output / "rubric_concordance.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def paired_intervention_effects(scores_df: str | Path | pd.DataFrame, out_dir: str | Path,
                                metadata: pd.DataFrame | None = None, *,
                                bootstrap_iterations: int = 1000, seed: int = 0) -> dict:
    """Estimate scored intervention-minus-baseline effects on exactly paired items.

    Rater and replicate scores are averaged within idiom/model/condition before
    pairing. Positive differences indicate improvement on the external 0–5
    rubric. Component clusters retain all their paired idioms in a bootstrap
    draw; absent/incomplete component metadata falls back explicitly to idioms.
    No score or significance probability is generated by this analysis.
    """
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative.")
    scores = _scores(scores_df)
    if "condition" not in scores or not scores.condition.eq("baseline").any():
        raise ValueError("Paired intervention analysis requires a baseline condition.")
    keys = SCORE_KEYS + ["condition"]
    means = scores.groupby(keys, as_index=False).agg(score=("score", "mean"), n_ratings=("score", "count"))
    baseline = means.loc[means.condition.eq("baseline"), SCORE_KEYS + ["score", "n_ratings"]].rename(
        columns={"score": "baseline_score", "n_ratings": "baseline_n_ratings"})
    interventions = means.loc[~means.condition.eq("baseline")].copy()
    if interventions.empty:
        raise ValueError("No scored intervention conditions are available to pair with baseline.")
    merged = interventions.merge(baseline, on=SCORE_KEYS, how="left", validate="many_to_one", indicator=True)
    unpaired = merged.loc[merged._merge.eq("left_only")].drop(columns="_merge")
    paired = merged.loc[merged._merge.eq("both")].drop(columns="_merge").copy()
    if paired.empty:
        raise ValueError("No intervention scores have an exact idiom_id/model baseline match.")
    paired = paired.rename(columns={"score": "intervention_score", "n_ratings": "intervention_n_ratings"})
    paired["difference"] = paired.intervention_score - paired.baseline_score
    metadata_columns = ["component_id", "anchored", "cultural", "religious", "historical", "rural", "rare"]
    if metadata is not None:
        if not set(SCORE_KEYS).issubset(metadata):
            raise ValueError("Intervention metadata requires explicit idiom_id/model keys.")
        available = [column for column in metadata_columns if column in metadata]
        meta = metadata[SCORE_KEYS + available].copy()
        for column in SCORE_KEYS:
            if meta[column].isna().any() or meta[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Metadata key {column} contains missing values.")
            meta[column] = meta[column].astype(str).str.strip()
        for column in available:
            conflicts = meta.groupby(SCORE_KEYS, dropna=False)[column].nunique(dropna=False).gt(1)
            if conflicts.any():
                raise ValueError(f"Metadata {column} is inconsistent within idiom_id/model keys.")
        meta = meta.drop_duplicates(SCORE_KEYS)
        paired = paired.merge(meta, on=SCORE_KEYS, how="left", validate="many_to_one")
    component_complete = "component_id" in paired and not (
        paired.component_id.isna().any() or paired.component_id.astype(str).str.strip().eq("").any())
    cluster_column = "component_id" if component_complete else "idiom_id"
    fallback_reason = None if component_complete else "No complete, consistent component_id mapping for every paired item; clustered by idiom_id."
    if component_complete:
        paired["component_id"] = paired.component_id.astype(str)
    if "anchored" in paired:
        truth_values = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
        normalized = paired.anchored.astype(str).str.strip().str.lower()
        invalid = paired.anchored.notna() & ~normalized.isin(truth_values)
        if invalid.any():
            raise ValueError("Metadata anchored must contain boolean/zero-one flags or missing values.")
        paired["anchored"] = normalized.map(truth_values)
    rng = np.random.default_rng(seed)
    summaries = []
    for (model_id, condition), frame in paired.groupby(["model", "condition"], sort=True):
        groups = [("all", frame)]
        if "anchored" in frame:
            for flag, label in ((True, "anchored"), (False, "nonanchored")):
                subset = frame.loc[frame.anchored.eq(flag)]
                if subset[cluster_column].nunique() >= 3:
                    groups.append((label, subset))
        for label, subset in groups:
            clusters = [part.difference.to_numpy(dtype=float) for _, part in subset.groupby(cluster_column, sort=True)]
            samples = []
            if len(clusters) >= 3:
                for _ in range(bootstrap_iterations):
                    chosen = rng.integers(0, len(clusters), len(clusters))
                    samples.append(float(np.concatenate([clusters[index] for index in chosen]).mean()))
            interval = np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) >= 20 else [None, None]
            summaries.append({"model": model_id, "condition": condition, "group": label,
                              "n_pairs": len(subset), "n_idioms": subset.idiom_id.nunique(),
                              "n_components": subset.component_id.nunique() if component_complete else None,
                              "bootstrap_unit": cluster_column, "n_bootstrap_clusters": len(clusters),
                              "baseline_mean": float(subset.baseline_score.mean()),
                              "intervention_mean": float(subset.intervention_score.mean()),
                              "mean_difference": float(subset.difference.mean()),
                              "ci_95_low": interval[0], "ci_95_high": interval[1],
                              "valid_bootstrap_samples": len(samples),
                              "status": "ok" if interval[0] is not None else "insufficient_bootstrap_units_or_draws"})
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "intervention_paired_differences.csv", index=False)
    unpaired.to_csv(output / "intervention_unpaired_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "intervention_effects.csv", index=False)
    report = _json_finite({"status": "completed", "score_rows": len(scores), "paired_rows": len(paired),
                           "unpaired_intervention_rows": len(unpaired), "seed": seed,
                           "bootstrap_iterations": bootstrap_iterations, "bootstrap_unit": cluster_column,
                           "bootstrap_fallback_reason": fallback_reason,
                           "difference_convention": "intervention score minus baseline score; positive means improvement",
                           "rating_handling": "mean across raters/replicates per idiom/model/condition before exact baseline pairing",
                           "anchored_subgroups_minimum_units": 3, "effects": summaries,
                           "caveats": ["Only externally supplied scores are analyzed; missing ratings are not imputed.",
                                       "Intervals are descriptive cluster-bootstrap percentile intervals; no bootstrap p-values are reported.",
                                       "Multiple operations/subgroups are exploratory; intervals are unadjusted."]})
    (output / "intervention_effects.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report
