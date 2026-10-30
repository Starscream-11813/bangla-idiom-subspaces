"""Accuracy, balanced accuracy and macro-F1 for fixed TEST predictions, and their SSIs.

All functions take integer labels. ``labels`` is the class set over which macro
averages run; callers pass the classes present in the TEST labels, so a class
that the probe never sees in TEST cannot inflate a macro average.
"""
from __future__ import annotations

import numpy as np

METRICS = ("acc", "bal", "f1")


def confusion_counts(y_true, pred, labels, weights=None):
    """Per-class true positives, predicted counts and true counts.

    ``pred`` is (n,) or (B, n); ``weights`` is (n,) or (W, n) multiplicities.
    Returns arrays shaped (..., K) for tp, predicted and support.
    """
    y = np.asarray(y_true)
    P = np.atleast_2d(np.asarray(pred))
    labels = np.asarray(labels)
    if weights is None:
        w = np.ones((1, len(y)))
    else:
        w = np.atleast_2d(np.asarray(weights, dtype=float))
    tp = np.empty((P.shape[0], w.shape[0], len(labels)))
    predicted = np.empty_like(tp)
    support = np.empty((w.shape[0], len(labels)))
    for j, k in enumerate(labels):
        is_true = (y == k).astype(float)
        is_pred = (P == k).astype(float)
        support[:, j] = w @ is_true
        tp[:, :, j] = (is_pred * is_true) @ w.T
        predicted[:, :, j] = is_pred @ w.T
    return tp, predicted, support


def metric_table(y_true, pred, labels, weights=None):
    """Accuracy, balanced accuracy (macro recall) and macro-F1.

    Returns a dict of arrays shaped (B, W): B prediction vectors by W weightings
    (W=1 without bootstrap weights). Classes with zero support in a weighting are
    left out of that weighting's macro averages.
    """
    y = np.asarray(y_true)
    P = np.atleast_2d(np.asarray(pred))
    w = np.ones((1, len(y))) if weights is None else np.atleast_2d(np.asarray(weights, dtype=float))
    tp, predicted, support = confusion_counts(y, P, labels, w)
    total = w.sum(1)
    correct = (P[:, None, :] == y[None, None, :]) * w[None, :, :]
    acc = correct.sum(-1) / total[None, :]
    present = support[None, :, :] > 0
    recall = np.divide(tp, support[None, :, :], out=np.zeros_like(tp), where=present)
    bal = (recall * present).sum(-1) / np.maximum(present.sum(-1), 1)
    denominator = predicted + support[None, :, :]
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    macro_f1 = (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)
    return dict(acc=acc, bal=bal, f1=macro_f1)


def metric_values(y_true, pred, labels):
    """Scalar metrics for one prediction vector."""
    table = metric_table(y_true, np.asarray(pred)[None, :], labels)
    return {k: float(v[0, 0]) for k, v in table.items()}


def ssi(cross, native, floor, min_den=0.02):
    """Vectorized shared-subspace index with the pipeline's undefined-denominator rule."""
    cross, native, floor = np.broadcast_arrays(np.asarray(cross, float), np.asarray(native, float),
                                               np.asarray(floor, float))
    den = native - floor
    out = np.full(cross.shape, np.nan)
    ok = np.isfinite(cross) & np.isfinite(den) & (den > min_den)
    out[ok] = (cross[ok] - floor[ok]) / den[ok]
    return out


def component_weights(groups, n_boot, seed):
    """Multiplicity of every row under ``n_boot`` component bootstrap draws, (n_boot, n)."""
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(g, np.full(g, 1 / g), size=n_boot)
    return counts[:, inv].astype(float)


def bootstrap_ssi(y_true, cross, native, floor, groups, labels, n_boot=1000, seed=0, min_den=0.02):
    """Point SSI and 95% component-bootstrap intervals for all three metrics.

    The same component draw is applied to the cross, native and floor predictions,
    so the interval is paired, as in ``paired_cluster_bootstrap``.
    """
    preds = np.stack([np.asarray(cross), np.asarray(native), np.asarray(floor)])
    point = metric_table(y_true, preds, labels)
    weights = component_weights(groups, n_boot, seed)
    boot = metric_table(y_true, preds, labels, weights)
    out = {}
    for m in METRICS:
        c, n, f = point[m][:, 0]
        out[m] = c
        out[m + "_native"] = n
        out[m + "_floor"] = f
        estimate = float(ssi(c, n, f, min_den))
        values = ssi(boot[m][0], boot[m][1], boot[m][2], min_den)
        values = values[np.isfinite(values)]
        lo, hi = (np.quantile(values, [.025, .975]) if len(values) and np.isfinite(estimate)
                  else (np.nan, np.nan))
        out["SSI_" + m] = estimate
        out["SSI_" + m + "_ci_lo"] = float(lo)
        out["SSI_" + m + "_ci_hi"] = float(hi)
        out["SSI_" + m + "_boot_valid"] = int(len(values))
    return out


def null_pvalues(y_true, null_preds, native_metrics, floor_metrics, observed_ssi, labels, min_den=0.02):
    """Monte Carlo p values (with +1) of each metric's SSI against prediction nulls.

    ``null_preds`` is (B, n). ``native_metrics``/``floor_metrics``/``observed_ssi``
    map metric names to scalars. Returns p, null mean SSI, null mean metric, draws.
    """
    out = {}
    if null_preds is None or len(null_preds) == 0:
        for m in METRICS:
            out.update({f"p_{m}": np.nan, f"null_SSI_{m}_mean": np.nan, f"null_{m}_mean": np.nan,
                        f"draws_{m}": 0})
        return out
    table = metric_table(y_true, null_preds, labels)
    for m in METRICS:
        values = table[m][:, 0]
        null_ssi = ssi(values, native_metrics[m], floor_metrics[m], min_den)
        finite = null_ssi[np.isfinite(null_ssi)]
        obs = observed_ssi[m]
        if np.isfinite(obs) and len(finite):
            p = float((1 + np.sum(finite >= obs)) / (1 + len(finite)))
        else:
            p = np.nan
        out[f"p_{m}"] = p
        out[f"null_SSI_{m}_mean"] = float(finite.mean()) if len(finite) else np.nan
        out[f"null_{m}_mean"] = float(values.mean())
        out[f"draws_{m}"] = int(len(finite))
    return out


def paired(values, groups, seed=41, n_boot=5000, sign_draws=19999):
    """Conditional cluster bootstrap CI and one-sided component wild-sign p (run_followup.paired)."""
    values = np.asarray(values, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=values, minlength=g)
    counts = np.bincount(inv, minlength=g)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1 / g), size=n_boot)
    boot = (weights @ sums) / (weights @ counts)
    observed = float(values.mean())
    exceed = 0
    for start in range(0, sign_draws, 1000):
        signs = 2 * rng.integers(0, 2, size=(min(1000, sign_draws - start), g)) - 1
        exceed += int(np.sum(signs @ sums >= values.sum() - 1e-12))
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(estimate=observed, ci_lo=float(lo), ci_hi=float(hi),
                p=(1 + exceed) / (sign_draws + 1), n=len(values), n_components=g,
                bootstrap_draws=n_boot, wild_sign_draws=sign_draws)


def paired_two_sided(values, groups, seed=41, n_boot=5000, sign_draws=19999):
    """As ``paired`` but with a two-sided wild-sign p (for contrasts without a predicted sign)."""
    values = np.asarray(values, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=values, minlength=g)
    counts = np.bincount(inv, minlength=g)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1 / g), size=n_boot)
    boot = (weights @ sums) / (weights @ counts)
    total = abs(values.sum())
    exceed = 0
    for start in range(0, sign_draws, 1000):
        signs = 2 * rng.integers(0, 2, size=(min(1000, sign_draws - start), g)) - 1
        exceed += int(np.sum(np.abs(signs @ sums) >= total - 1e-12))
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(estimate=float(values.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_two_sided=(1 + exceed) / (sign_draws + 1), n=len(values), n_components=g,
                bootstrap_draws=n_boot, wild_sign_draws=sign_draws)
