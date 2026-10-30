"""Post-run checks for X15--X19: aggregates exist, the gates reproduce published numbers, the plan was not changed.

    python run_experiments.py verify [--results DIR] [--check X15]

Writes results/experiments/VERIFY_X15-X19.json (VERIFY_<codes>.json with --check) and exits nonzero when an error-level check fails.
A missing experiment is reported as "not run".

Gates (errors):
  X15  on split 0 and bn-fig, the MLR cross accuracy reproduces the published planned test and the COV and PCA cross
       accuracies reproduce the published estimator table (within 0.005);
  X16  the manifest records the frozen plan's SHA-256; on split 1 and bn-fig, COV and MLR English-direction accuracy
       reproduces the follow-up's scores.csv (within 0.005);
  X17  every judged item carries the packet's F class;
  X18  draw 0 (a refit on the original rows) agrees with the published readouts on at least 98% of TEST items;
  X19  the opaque, transparent and random source subsets of a cell have one size.
Warnings (reported, not failing): fewer than 290 of the 300 judged items on bn-idiom, under 90% valid refit draws,
reference cells that could not be matched, and probe convergence below 95%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.verify import _check, _read, convergence

from .common3 import ESTIMATORS, FOLLOWUP_SCORES, PUBLISHED_PAIRS, RESULTS, common
from .x16_idiom_prospective import PLAN_SHA256

REQUIRED = {
    "X15_matched_nulls": ["tests.csv"],
    "X16_idiom_prospective": ["tests.csv", "gate.csv", "zero_shot.csv"],
    "X17_human_items": ["contrasts.csv", "coverage.csv", "accuracy.csv"],
    "X18_refit_bootstrap": ["draws.csv", "summary.csv"],
    "X19_counterpart_type": ["subsets.csv", "summary.csv"],
}
SHORT = {"csebuetnlp/banglabert": "BanglaBERT", "csebuetnlp/banglishbert": "BanglishBERT",
         "bert-base-multilingual-cased": "mBERT", "xlm-roberta-large": "XLM-R", "google/muril-large-cased": "MuRIL",
         "setu4993/LaBSE": "LaBSE"}


def _matched(out, merged, a, b, tolerance, name):
    if merged.empty:
        out.append(_check(False, f"{name}: reference cells matched", "no common cells", level="warning"))
        return
    diff = float(np.nanmax(np.abs(merged[a] - merged[b])))
    out.append(_check(diff <= tolerance, f"{name} (max |diff| <= {tolerance})",
                      f"max |diff| = {diff:.4g} over {len(merged)} cells"))


def check_x15(folder):
    out = []
    tests = _read(folder / "tests.csv")
    if tests.empty:
        return out
    fig = tests[tests.split.eq(0) & tests.target.eq("bn-fig")]
    pairs = _read(PUBLISHED_PAIRS)
    if not pairs.empty:
        ref = pairs[pairs.group.eq("all") & pairs.method.eq("MLR") & pairs.source_condition.eq("en-idiom")
                    & pairs.condition.eq("bn-fig")][["model", "concept", "R_ep"]]
        merged = fig[fig.method.eq("MLR")].merge(ref, on=["model", "concept"])
        _matched(out, merged, "acc", "R_ep", .005, "X15 gate: MLR cross accuracy on bn-fig reproduces the planned test")
    est = _read(ESTIMATORS)
    if not est.empty:
        ref = est[(est.concept.eq("F") & est.M.eq(11)) | (est.concept.eq("S") & est.M.eq(3))]
        mine = fig.assign(name=fig.model.map(SHORT))
        for method, column in [("COV", "acc"), ("PCA", "PCA_acc")]:
            merged = mine[mine.method.eq("COV")].merge(ref[ref.method.eq(method)][["model", "concept", "cross_R"]],
                                                       left_on=["name", "concept"], right_on=["model", "concept"])
            _matched(out, merged, column, "cross_R", .005,
                     f"X15 gate: {method} cross accuracy on bn-fig reproduces the estimator table")
    return out


def check_x16(folder):
    out = []
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    out.append(_check(manifest.get("plan_sha256") == PLAN_SHA256, "X16: run used the frozen plan",
                      f"manifest {manifest.get('plan_sha256')} vs pinned {PLAN_SHA256}"))
    gate, scores = _read(folder / "gate.csv"), _read(FOLLOWUP_SCORES)
    if not gate.empty and not scores.empty:
        ref = scores[scores.source.eq("en-idiom") & scores.target.eq("bn-fig") & scores.representation.eq("project")]
        merged = gate.merge(ref[["model", "concept", "method", "accuracy"]], on=["model", "concept", "method"])
        _matched(out, merged, "cross_acc", "accuracy", .005,
                 "X16 gate: split-1 COV and MLR accuracy on bn-fig reproduces the follow-up")
    tests = _read(folder / "tests.csv")
    if not tests.empty and "RANDCOV_draws" in tests:
        planned = min(int(manifest.get(k, 0)) for k in ["null_draws", "perm_draws", "cov_draws"])
        out.append(_check(planned >= 1999, "X16: the plan's 1,999 draws per null were used",
                          f"smallest budget {planned}", level="warning"))
    return out


def check_x17(folder):
    out = []
    coverage = _read(folder / "coverage.csv")
    if coverage.empty:
        return out
    out.append(_check(bool(coverage.F_matches_packet.astype(bool).all()), "X17: judged items carry the packet's F class",
                      f"{int((~coverage.F_matches_packet.astype(bool)).sum())} model x target cells differ"))
    idiom = coverage[coverage.target.eq("bn-idiom")]
    if len(idiom):
        low = int(idiom["items"].min())
        out.append(_check(low >= 290, "X17: at least 290 of the 300 judged items found on bn-idiom",
                          f"smallest count {low}", level="warning"))
    return out


def check_x18(folder):
    out = []
    summary = _read(folder / "summary.csv")
    if summary.empty:
        return out
    for column in ["agreement_cross", "agreement_native"]:
        if column in summary:
            low = float(summary[column].min())
            out.append(_check(low >= .98, f"X18 gate: draw 0 reproduces the published readouts ({column} >= 0.98)",
                              f"lowest {low:.4f}"))
    share = float((summary.valid_draws / summary.draws.clip(lower=1)).min())
    out.append(_check(share >= .9, "X18: at least 90% of refit draws valid", f"lowest share {share:.3f}",
                      level="warning"))
    return out


def check_x19(folder):
    out = []
    subsets = _read(folder / "subsets.csv")
    if subsets.empty:
        return out
    restricted = subsets[subsets.subset.ne("all")]
    sizes = restricted.groupby(["model", "split", "concept", "target", "method"]).n_source.nunique()
    out.append(_check(bool(sizes.le(1).all()), "X19: opaque, transparent and random subsets have one size per cell",
                      f"{int(sizes.gt(1).sum())} cells differ"))
    if "valid" in subsets:
        named = restricted[restricted.subset.isin(["opaque", "transparent"])]
        bad = named[~named.valid.astype(bool)][["model", "concept", "target", "method", "subset"]].values.tolist()
        out.append(_check(not bad, "X19: every opaque and transparent subset supports rank M",
                          f"{len(bad)} invalid: {bad[:6]}", level="warning"))
    return out


def verify(results=None, out=None, only=None):
    results = Path(results or RESULTS)
    report = dict(created_utc=common.utc(), results=str(results), experiments={})
    failed = False
    checkers = {"X15_": check_x15, "X16_": check_x16, "X17_": check_x17, "X18_": check_x18, "X19_": check_x19}
    for name, files in REQUIRED.items():
        if only and name.split("_")[0] not in only:
            continue
        folder = results / name
        if not (folder / "MANIFEST.json").exists():
            report["experiments"][name] = dict(status="not run")
            continue
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        checks = []
        for file in files:
            rows = len(_read(folder / file))
            checks.append(_check(rows > 0, f"{name}/{file} has rows", f"{rows} rows"))
        try:
            for prefix, checker in checkers.items():
                if name.startswith(prefix):
                    checks += checker(folder)
            checks += convergence(folder, name)
        except Exception as error:  # report, do not hide
            checks.append(_check(False, f"{name}: verification raised", repr(error)))
        passed = all(c["passed"] or c["level"] == "warning" for c in checks)
        failed |= not passed
        report["experiments"][name] = dict(status="passed" if passed else "FAILED", created=manifest.get("created_utc"),
                                           models=manifest.get("models"), splits=manifest.get("splits"),
                                           checks=checks)
    common.write_json(results / ("VERIFY_X15-X19.json" if not only else f"VERIFY_{'_'.join(only)}.json"), report)
    for name, info in report["experiments"].items():
        print(f"{name:24s} {info['status']}")
        for c in info.get("checks", []):
            if not c["passed"]:
                print(f"    {'WARNING' if c['level'] == 'warning' else 'FAILED'}: {c['check']} ({c['detail']})")
    return 1 if failed else 0
