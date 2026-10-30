"""Post-run checks for X9--X14: aggregates exist, X7 is reproduced, every eraser is valid.

    python run_experiments.py verify [--results DIR] [--x7 DIR] [--check X10]

Writes results/experiments/VERIFY_X9-X14.json (VERIFY_<codes>.json with --check) and exits nonzero
when an error-level check fails. A missing experiment is reported as "not run".

Gates (errors):
  X10  the map refitted on SPACE pairs reproduces X7's aligned cells and
       retrieval (same maps, same readouts), and the unmapped NN retrieval
       reproduces the published S3 baselines;
  X9   its raw COV/LEACE/MLR erasure drops and ambient accuracies reproduce
       X7's, and every removed basis is orthonormal.
Warnings (reported, not failing): INLP or R-LACE fits that do not guard their
source language on VALIDATION, and probe convergence below 95%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.verify import _check, _read, check_x7, convergence

from .common2 import REPO, RESULTS, X7_RESULTS, common

REQUIRED = {
    "X10_heldout_alignment": ["retrieval.csv", "procrustes_fit.csv", "aligned.csv", "summary.csv"],
    "X9_erasers_nonlinear": ["erasers.csv", "path.csv", "linear.csv", "summary.csv"],
    "X11_taxonomy_robustness": ["tests.csv", "ladder.csv", "contrasts.csv"],
    "X12_penalty_path": ["path.csv", "selected.csv", "summary.csv"],
    "X13_idiom_isolated": ["tests.csv", "absolute.csv", "paired.csv", "zero_shot.csv"],
    "X14_decoder_powered": ["tests.csv", "zero_shot.csv"],
}
PUBLISHED_PAIRS = REPO / "results" / "discovery" / "all_selected_layer_inference.csv"
FOLLOWUP_MECHANISMS = REPO / "results" / "replication" / "mechanisms.csv"
ESTIMATORS = REPO / "results" / "discovery" / "synthesis" / "estimator_families.csv"
X10_GATE = [("shared_dims_unaligned", .001), ("shared_dims_aligned", .01), ("ambient_acc", .002),
            ("SSI_published_cross_acc", .03), ("aligned_cross_acc", .005), ("SSI_aligned_cross_acc", .03),
            ("aligned_zero_shot_acc", .005), ("erase_aligned_drop_pp", .5), ("erase_unaligned_drop_pp", .5),
            ("erase_native_drop_pp", .5)]
GUARD_GAP = .03


def check_x10(folder, x7):
    out = []
    aligned = _read(folder / "aligned.csv")
    reference = _read(x7 / "aligned.csv")
    if aligned.empty:
        return out
    if reference.empty:
        out.append(_check(False, "X10: X7 aligned.csv available for the reproduction gate", str(x7)))
        return out
    space = aligned[aligned.fit_on.eq("space") & aligned.shuffle_draw.eq(0)]
    keys = ["model", "split", "concept", "method", "alignment"]
    merged = space.merge(reference, on=keys, suffixes=("", "_x7"))
    if merged.empty:
        out.append(_check(False, "X10 gate: SPACE-fitted cells match X7 cells", "no common cells"))
    for column, tolerance in X10_GATE:
        if merged.empty or column not in merged or f"{column}_x7" not in merged:
            continue
        diff = float(np.nanmax(np.abs(merged[column] - merged[f"{column}_x7"])))
        out.append(_check(diff <= tolerance,
                          f"X10 gate: SPACE-fitted {column} reproduces X7 (max |diff| <= {tolerance})",
                          f"max |diff| = {diff:.4g} over {len(merged)} cells"))
    retrieval, ref = _read(folder / "retrieval.csv"), _read(x7 / "retrieval.csv")
    if not retrieval.empty and not ref.empty:
        keys = ["model", "split", "source", "target", "space", "scoring", "direction"]
        matched = retrieval[retrieval.fit_on.isin(["none", "space"])].merge(ref[keys + ["top1", "MRR"]], on=keys,
                                                                            suffixes=("", "_x7"))
        if len(matched):
            diff = float(np.abs(matched.top1 - matched.top1_x7).max())
            out.append(_check(diff <= .002, "X10 gate: unmapped and SPACE-mapped retrieval reproduce X7",
                              f"max |top1 diff| = {diff:.4f} over {len(matched)} rows"))
    out += check_x7(folder)
    return out


def check_x9(folder, x7):
    out = []
    erasers, linear = _read(folder / "erasers.csv"), _read(folder / "linear.csv")
    if not erasers.empty:
        worst = float(erasers.orthonormal_error.max())
        out.append(_check(worst <= 1e-6, "X9: every removed basis is orthonormal", f"max error {worst:.2e}"))
        inlp = erasers[erasers.eraser.eq("INLP")]
        if len(inlp):
            guarded = inlp.inlp_guarded.astype(bool)
            unguarded = inlp[~guarded][["model", "concept", "source"]].values.tolist()
            out.append(_check(guarded.mean() >= .9, "X9: INLP guards its source language on VALIDATION (>= 90% of fits)",
                              f"{guarded.mean():.2f} of {len(inlp)} fits; unguarded: {unguarded}", level="warning"))
        rl = erasers[erasers.eraser.eq("RLACE")]
        if len(rl):
            ok = rl.guard_gap.le(GUARD_GAP)
            out.append(_check(bool(ok.mean() >= .9),
                              f"X9: R-LACE guards its source language on VALIDATION "
                              f"(gap <= {GUARD_GAP}, >= 90% of fits)",
                              f"{ok.mean():.2f} of {len(rl)} fits; largest gap {rl.guard_gap.max():.3f}",
                              level="warning"))
    reference = _read(x7 / "aligned.csv")
    if linear.empty or reference.empty:
        if reference.empty:
            out.append(_check(False, "X9: X7 aligned.csv available for the reproduction gate", str(x7)))
        return out
    ref = reference.drop_duplicates(["model", "split", "concept", "method"])
    raw = linear[linear.target.eq("bn-fig") & linear.condition.eq("raw") & linear.eraser.isin(["COV", "LEACE", "MLR"])]
    on = dict(left_on=["model", "split", "concept", "eraser"], right_on=["model", "split", "concept", "method"])
    for source, column in [("en-idiom", "erase_unaligned_drop_pp"), ("bn-fig", "erase_native_drop_pp")]:
        merged = raw[raw.source.eq(source)].merge(ref, suffixes=("", "_x7"), **on)
        if merged.empty:
            out.append(_check(False, f"X9 gate: {source}-fitted COV/LEACE/MLR cells match X7 cells", "no common cells"))
            continue
        diff = float(np.abs(merged.drop_pp - merged[column]).max())
        out.append(_check(diff <= .5, f"X9 gate: {source}-fitted COV/LEACE/MLR erasure reproduces X7 {column} "
                                      "(max |diff| <= 0.5 pp)", f"max |diff| = {diff:.3f} pp over {len(merged)} cells"))
        diff = float(np.abs(merged.ambient_acc - merged.ambient_acc_x7).max())
        out.append(_check(diff <= .002, "X9 gate: ambient Bangla accuracy reproduces X7",
                          f"max |diff| = {diff:.4f}"))
    return out


def check_x11(folder):
    out = []
    for path in sorted(folder.glob("label_report_split*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        r = report.get("reproduction", {})
        agreement = float(r.get("agreement", np.nan))
        out.append(_check(agreement >= .99, f"X11 gate ({path.stem}): F rebuilt with seed 0 reproduces the campaign's F",
                          f"agreement {agreement:.4f}, ARI {r.get('ARI', np.nan):.4f} over {r.get('records')} records"))
        ari = {k: round(v.get("ARI_with_F_space", np.nan), 3) for k, v in report.get("reseeded", {}).items()}
        out.append(_check(True, f"X11 ({path.stem}): re-seeded taxonomies", f"ARI with F (SPACE): {ari}",
                          level="warning"))
        tag = report.get("TAG", {})
        labelled = tag.get("labelled_records", 0) / max(tag.get("records", 1), 1)
        out.append(_check(tag.get("k", 0) >= 6 and labelled >= .3,
                          f"X11 ({path.stem}): TAG has at least 6 classes and labels at least 30% of records",
                          f"k {tag.get('k')}, labelled share {labelled:.2f}, AMI with F {tag.get('AMI_with_F_space')}",
                          level="warning"))
    return out


def check_x12(folder):
    out = []
    path = _read(folder / "path.csv")
    if path.empty:
        return out
    mech = _read(FOLLOWUP_MECHANISMS)
    if not mech.empty:
        e2 = mech[mech.experiment.eq("E2")][["model", "concept", "setting", "accuracy"]]
        mine = path[path.split.eq(1) & path.method.eq("MLR")][["model", "concept", "setting", "cross_acc"]]
        merged = mine.merge(e2, on=["model", "concept", "setting"])
        if len(merged):
            diff = float(np.abs(merged.cross_acc - merged.accuracy).max())
            out.append(_check(diff <= .005, "X12 gate: seed-1 MLR accuracies along C reproduce the follow-up's E2 path",
                              f"max |diff| = {diff:.4f} over {len(merged)} cells"))
    est = _read(ESTIMATORS)
    if not est.empty:
        names = {"csebuetnlp/banglabert": "BanglaBERT", "csebuetnlp/banglishbert": "BanglishBERT",
                 "bert-base-multilingual-cased": "mBERT", "xlm-roberta-large": "XLM-R",
                 "google/muril-large-cased": "MuRIL", "setu4993/LaBSE": "LaBSE"}
        cov = path[path.split.eq(0) & path.method.eq("COV")].assign(name=lambda d: d.model.map(names))
        ref = est[est.method.eq("COV") & ((est.concept.eq("F") & est.M.eq(11)) | (est.concept.eq("S") & est.M.eq(3)))]
        merged = cov.merge(ref, left_on=["name", "concept"], right_on=["model", "concept"])
        if len(merged):
            diff = float(np.abs(merged.cross_acc - merged.cross_R).max())
            out.append(_check(diff <= .005, "X12: discovery COV readouts reproduce the published estimator table",
                              f"max |diff| = {diff:.4f} over {len(merged)} cells", level="warning"))
    mlr = path[path.method.eq("MLR")]
    if len(mlr):
        ends = mlr.groupby("C").shared_dims_with_english_cov.median()
        out.append(_check(ends.iloc[0] >= ends.loc[1.0] if 1.0 in ends.index else True,
                          "X12: the MLR image moves towards the COV image as C falls (the strong-penalty limit)",
                          f"median shared dimensions by C: {ends.round(2).to_dict()}", level="warning"))
    return out


def check_x13(folder):
    out = []
    tests, ref = _read(folder / "tests.csv"), _read(PUBLISHED_PAIRS)
    if tests.empty or ref.empty:
        return out
    ref = ref[ref.group.eq("all") & ref.method.eq("MLR") & ref.source_condition.eq("en-idiom")
              & ref.condition.eq("bn-idiom")]
    ref = ref[["model", "concept", "R_ep", "SSI"]].rename(columns={"R_ep": "published_cross_acc",
                                                                  "SSI": "published_SSI"})
    merged = tests[tests.split.eq(0) & tests.method.eq("MLR")].merge(ref, on=["model", "concept"])
    if merged.empty:
        return out
    d_acc = float(np.nanmax(np.abs(merged.acc - merged.published_cross_acc)))
    out.append(_check(d_acc <= .005, "X13 gate: MLR cross accuracy on bn-idiom reproduces the published pair table",
                      f"max |diff| = {d_acc:.4f} over {len(merged)} tests"))
    defined = merged[merged.published_SSI.notna() & merged.SSI_acc.notna()]
    if len(defined):
        d_ssi = float(np.abs(defined.SSI_acc - defined.published_SSI).max())
        out.append(_check(d_ssi <= .03, "X13 gate: MLR SSI on bn-idiom reproduces the published pair table",
                          f"max |diff| = {d_ssi:.4f} over {len(defined)} tests"))
    return out


def verify(results=None, out=None, only=None, x7=None):
    """Check every experiment folder, or only those whose code (X9, X10) is in ``only``."""
    results = Path(results or RESULTS)
    x7 = Path(x7 or X7_RESULTS)
    report = dict(created_utc=common.utc(), results=str(results), x7_reference=str(x7), experiments={})
    failed = False
    for name, files in REQUIRED.items():
        if only and name.split("_")[0] not in only:
            continue
        folder = results / name
        if not (folder / "MANIFEST.json").exists():
            report["experiments"][name] = dict(status="not run")
            continue
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        required = list(files)
        if name.startswith("X9_") and "mlp" in manifest.get("parts", ["mlp"]):
            required.append("mlp.csv")
        checks = []
        for file in required:
            rows = len(_read(folder / file))
            checks.append(_check(rows > 0, f"{name}/{file} has rows", f"{rows} rows"))
        try:
            if name.startswith("X10_"):
                checks += check_x10(folder, x7)
            if name.startswith("X9_"):
                checks += check_x9(folder, x7)
            if name.startswith("X11_"):
                checks += check_x11(folder)
            if name.startswith("X12_"):
                checks += check_x12(folder)
            if name.startswith("X13_"):
                checks += check_x13(folder)
            checks += convergence(folder, name)
        except Exception as error:  # report, do not hide
            checks.append(_check(False, f"{name}: verification raised", repr(error)))
        passed = all(c["passed"] or c["level"] == "warning" for c in checks)
        failed |= not passed
        report["experiments"][name] = dict(status="passed" if passed else "FAILED", created=manifest.get("created_utc"),
                                           models=manifest.get("models"), splits=manifest.get("splits"),
                                           checks=checks)
    common.write_json(results / ("VERIFY_X9-X14.json" if not only else f"VERIFY_{'_'.join(only)}.json"), report)
    for name, info in report["experiments"].items():
        print(f"{name:24s} {info['status']}")
        for c in info.get("checks", []):
            if not c["passed"]:
                print(f"    {'WARNING' if c['level'] == 'warning' else 'FAILED'}: {c['check']} ({c['detail']})")
    return 1 if failed else 0
