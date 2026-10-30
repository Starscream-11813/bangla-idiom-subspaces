"""Covariance-matched random subspaces (RANDCOV): a null with the data's variance profile but no labels.

The published RAND null draws Haar-uniform rank-M subspaces. In strongly anisotropic encoder spaces (XLM-R's
participation ratio is about 15 of 1,024) such subspaces almost never meet the few high-variance directions, so a
readout through them sits near the majority floor, and beating RAND says little more than beating the floor. RANDCOV
draws each direction from N(0, S), where S is the covariance of the source condition's SPACE rows
(X_c^T g / sqrt(n) with g ~ N(0, I_n)), and orthonormalizes the M draws. The subspace is as likely as the data to
point along high-variance directions but ignores the labels, so an estimator that beats it reads Bangla classes
better than English's variance alone would.

RAND uses the seeds meta.seed + 1009 * draw; RANDCOV uses its own stride, so the two null streams never coincide.
"""
from __future__ import annotations

import numpy as np

from idiom_subspace.geometry import Subspace

from experiments.common import compact_readouts

SEED_STRIDE = 7919


def base_seed(study, concept):
    """One stream per split and concept, shared by every target and estimator of that concept."""
    return 10_007 * (int(study.meta["seed"]) + 1) + {"F": 101, "S": 211}.get(concept, 307)


def matched_basis(Xc, rank, seed):
    """Orthonormal basis of M directions drawn from N(0, Xc^T Xc / n)."""
    g = np.random.default_rng(seed).standard_normal((len(Xc), rank))
    basis = np.linalg.qr(Xc.T @ g / np.sqrt(len(Xc)), mode="reduced")[0]
    return basis


def matched_subspaces(X_space, rank, draws, seed0):
    """Subspace objects (centred on the SPACE mean) for the given draw numbers."""
    X_space = np.asarray(X_space, dtype=float)
    center = X_space.mean(0)
    Xc = X_space - center
    return [Subspace(center, matched_basis(Xc, rank, seed0 + SEED_STRIDE * int(d)), rank, "RANDCOV", rank)
            for d in draws]


def matched_null_predictions(study, space_rows, rank, rows, y, draws, seed0, shift=None, chunk=128, Cs=None):
    """Predicted TEST labels of readouts through covariance-matched subspaces of the source SPACE rows.

    ``rows`` = (train, validation, test) row indices. For a supervised readout all three come from the target; for a
    zero-shot readout train and validation come from the source and ``shift`` (X1's centring offset) is subtracted from
    the test rows. Returns an int array (len(draws), len(test)).
    """
    tr, va, te = rows
    X_space = np.asarray(study.X[space_rows], dtype=float)
    center = X_space.mean(0)
    Xc = X_space - center
    Xtr, Xva = study.X[tr] - center, study.X[va] - center
    Xte = study.X[te] - center - (0. if shift is None else shift)
    draws = list(draws)
    out = []
    for start in range(0, len(draws), chunk):
        part = draws[start:start + chunk]
        bases = [matched_basis(Xc, rank, seed0 + SEED_STRIDE * int(d)) for d in part]
        results = compact_readouts(study, [Xtr @ Q for Q in bases], [Xva @ Q for Q in bases],
                                   [Xte @ Q for Q in bases], y, rows, Cs)
        out += [np.asarray(r["predicted"]) for r in results]
    return np.asarray(out) if out else np.empty((0, len(te)), dtype=int)


def matched_pvalues(y, null_preds, native_m, floor_m, observed_ssi, classes, min_den):
    """Monte Carlo p (with +1), mean and 95th percentile of the RANDCOV SSI, per metric."""
    from experiments.metrics import METRICS, metric_table, ssi
    out = {}
    if null_preds is None or not len(null_preds):
        return out
    table = metric_table(y, null_preds, classes)
    for m in METRICS:
        values = table[m][:, 0]
        null = ssi(values, native_m[m], floor_m[m], min_den)
        finite = null[np.isfinite(null)]
        obs = observed_ssi[m]
        out[f"RANDCOV_p_{m}"] = (float((1 + np.sum(finite >= obs)) / (1 + len(finite)))
                                 if len(finite) and np.isfinite(obs) else np.nan)
        out[f"RANDCOV_SSI_{m}_mean"] = float(finite.mean()) if len(finite) else np.nan
        out[f"RANDCOV_SSI_{m}_q95"] = float(np.quantile(finite, .95)) if len(finite) else np.nan
        out[f"RANDCOV_{m}_mean"] = float(values.mean())
    out["RANDCOV_draws"] = int(len(null_preds))
    return out
