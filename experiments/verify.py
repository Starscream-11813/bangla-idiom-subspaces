"""Post-run checks: every aggregate exists, published numbers are reproduced, extraction matches.

    python run_experiments.py verify [--results DIR] [--out OUT]

Writes results/experiments/VERIFY_X1-X8.json and returns a nonzero exit code when a check fails.
A missing experiment is reported as "not run", not as a failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RESULTS, utc, write_json
from . import reference

REQUIRED = {
    "X1_zero_shot": ["summary.csv"],
    "X2_concepts": ["tests.csv", "ladder.csv", "contrasts.csv"],
    "X3_leace_ablation": ["summary.csv"],
    "X4_pooling_matched": ["extraction_check.csv", "cells.csv", "contrasts.csv", "tokens.csv"],
    "X5_balanced_metrics": ["tests.csv", "agreement.csv", "null_metrics.csv.gz"],
    "X6_nonlinear_mdl": ["mlp.csv", "mdl.csv", "mdl_summary.csv"],
    "X7_alignment": ["retrieval.csv", "procrustes_fit.csv", "aligned.csv"],
    "X8_layerwise": ["layers.csv", "contrasts.csv", "summary.csv"],
}


def _check(ok, name, detail, level="error"):
    """A failed ``error`` check fails the experiment; a failed ``warning`` is only reported."""
    return dict(check=name, passed=bool(ok), detail=detail, level=level)


def _read(path):
    """A table, or an empty frame for a missing or column-less (empty) file."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def check_x5(folder):
    out = []
    tests = _read(folder / "tests.csv")
    if tests.empty:
        return out
    for split in sorted(tests.split.unique()):
        t = tests[tests.split.eq(split) & tests.get("ref_SSI", pd.Series(np.nan, index=tests.index)).notna()]
        if t.empty:
            continue
        d_acc = float(np.nanmax(np.abs(t.diff_cross_acc))) if "diff_cross_acc" in t else np.nan
        d_ssi = float(np.nanmax(np.abs(t.diff_SSI_acc))) if "diff_SSI_acc" in t else np.nan
        out.append(_check(d_acc <= .005, f"X5 split {split}: cross accuracy reproduces published (max |diff| <= 0.005)",
                          f"max |diff| = {d_acc:.4f} over {len(t)} tests"))
        out.append(_check(d_ssi <= .03, f"X5 split {split}: accuracy SSI reproduces published (max |diff| <= 0.03)",
                          f"max |diff| = {d_ssi:.4f}"))
        if "diff_RAND_p" in t:
            agree = (t.RAND_p_acc.le(.05) == t.ref_RAND_p.le(.05)).mean()
            out.append(_check(agree >= .9, f"X5 split {split}: RAND p<=.05 decisions agree with published",
                              f"agreement {agree:.2f}"))
    return out


def check_x4(folder):
    check = _read(folder / "extraction_check.csv")
    if check.empty:
        return []
    worst = check.median_cosine.min()
    validity = check.validity_agreement.min()
    return [_check(worst >= .999, "X4: re-extracted features match the original cache (median cosine >= 0.999)",
                   f"lowest median cosine {worst:.6f}; lowest minimum cosine {check.min_cosine.min():.6f}"),
            _check(validity >= .999, "X4: pooling validity agrees with the original cache", f"lowest {validity:.4f}")]


def check_x3(folder):
    out = []
    summary = _read(folder / "summary.csv")
    scores = reference.scores(1)
    if summary.empty:
        return out
    ref = summary[(summary.block == "reference") & summary.split.eq(1) & summary.target.eq("bn-fig")]
    if not scores.empty and not ref.empty:
        merged = ref.merge(scores.rename(columns={"accuracy": "published"}),
                           left_on=["model", "concept", "estimator", "source", "target", "representation"],
                           right_on=["model", "concept", "method", "source", "target", "representation"])
        if len(merged):
            diff = float(np.abs(merged.accuracy - merged.published).max())
            out.append(_check(diff <= .005, "X3 split 1: COV/AMBIENT readouts reproduce follow-up scores",
                              f"max |diff| = {diff:.4f} over {len(merged)} readouts"))
    return out


def check_x7(folder):
    table = _read(folder / "retrieval.csv")
    if "published_top1" not in table:
        return []
    t = table[table.published_top1.notna()]
    if t.empty:
        return []
    diff = float(np.abs(t.diff_top1_vs_published).max())
    same_n = bool((t.n == t.published_n).all())
    return [_check(diff <= .002 and same_n, "X7: centred NN retrieval reproduces the published S3 baselines",
                   f"max |top1 diff| = {diff:.4f} over {len(t)} rows; identical n: {same_n}")]


def convergence(folder, name):
    out = []
    for file in folder.glob("*.csv"):
        table = _read(file)
        for column in [c for c in table if "converged" in c]:
            values = table[column].dropna()
            if len(values):
                share = float(values.astype(bool).mean())
                out.append(_check(share >= .95, f"{name}/{file.name}: {column} >= 95%", f"{share:.3f}",
                                  level="warning"))
    return out


def verify(results=None, out=None, only=None):
    """Check every experiment folder (or only those whose names start with a code in ``only``)."""
    results = Path(results or RESULTS)
    report = dict(created_utc=utc(), results=str(results), experiments={})
    failed = False
    for name, files in REQUIRED.items():
        if only and not any(name.startswith(code) for code in only):
            continue
        folder = results / name
        if not (folder / "MANIFEST.json").exists():
            report["experiments"][name] = dict(status="not run")
            continue
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        checks = []
        for file in files:
            path = folder / file
            rows = len(_read(path))
            checks.append(_check(rows > 0, f"{name}/{file} has rows", f"{rows} rows"))
        try:
            if name == "X5_balanced_metrics":
                checks += check_x5(folder)
            if name == "X4_pooling_matched" and (folder / "extraction_check.csv").exists():
                checks += check_x4(folder)
            if name == "X3_leace_ablation":
                checks += check_x3(folder)
            if name == "X7_alignment":
                checks += check_x7(folder)
            checks += convergence(folder, name)
        except Exception as error:  # report, do not hide
            checks.append(_check(False, f"{name}: verification raised", repr(error)))
        passed = all(c["passed"] or c["level"] == "warning" for c in checks)
        failed |= not passed
        report["experiments"][name] = dict(status="passed" if passed else "FAILED", created=manifest.get("created_utc"),
                                           models=manifest.get("models"), splits=manifest.get("splits"),
                                           checks=checks)
    write_json(results / ("VERIFY_X1-X8.json" if not only else f"VERIFY_{'_'.join(only)}.json"), report)
    for name, info in report["experiments"].items():
        print(f"{name:24s} {info['status']}")
        for c in info.get("checks", []):
            if not c["passed"]:
                print(f"    {'WARNING' if c['level'] == 'warning' else 'FAILED'}: {c['check']} ({c['detail']})")
    return 1 if failed else 0
