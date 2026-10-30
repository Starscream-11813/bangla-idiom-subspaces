"""Batched L2-regularized logistic regression on GPU (or CPU) with PyTorch.

Every problem in a batch is solved independently: the batched L-BFGS below
keeps per-problem curvature pairs, step sizes and convergence flags, so a
batch of B probes gives the same solutions as B separate fits.

The objective is exactly scikit-learn's ``LogisticRegression`` with an L2
penalty (lbfgs solver), rescaled by 1/(C n) so that the gradient tolerance is
comparable to scikit-learn's ``tol``::

    multinomial (K > 2):  mean_i CE(x_i W + b, y_i) + ||W||^2 / (2 C n)
    binary      (K == 2): mean_i log(1 + exp(-s_i (x_i w + b))) + ||w||^2 / (2 C n)

The intercept is never penalized. Fits are solved to a gradient tolerance
(``LOGISTIC_TOL``, shared with the scikit-learn backend) rather than stopped
after a fixed number of iterations, so they do not depend on optimizer details.
A fit also stops when an accepted step no longer decreases the objective
(scipy's REL_REDUCTION_OF_F test with scikit-learn's ``ftol``); in float32 on
GPU this is the usual stop near the optimum, at gradients of about 1e-5 to
2e-4. Both stops count as ``success``, as they do in scipy and in
scikit-learn's convergence warning; iteration caps and line-search failures
do not. Two feature maps are supported without materializing B copies of a
large design matrix:

* compact: per-problem features ``Z_b`` (padded with zero columns; zero
  features keep zero weights under the L2 penalty, so padding is exact);
* erase: ``X - (X F_b) G_b^T`` for one shared ``X`` and low-rank ``F_b, G_b``
  (orthogonal or oblique erasure); an empty ``F_b`` is the ambient space.

Importing this module does not initialize CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Gradient tolerance of every logistic fit in both backends (scikit-learn's
# ``tol``; its default 1e-4 leaves low-rank MLR directions solver-dependent).
LOGISTIC_TOL = 1e-5
# scikit-learn's L-BFGS-B ``ftol`` (scipy's factr * epsmch).
SKLEARN_FTOL = 64 * np.finfo(np.float64).eps


def _torch():
    import torch
    return torch


def resolve_device(preference="auto"):
    torch = _torch()
    if preference in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def default_dtype(device):
    torch = _torch()
    # Consumer GPUs have very slow float64; float32 is ample for a gradient
    # tolerance of 1e-5 on a mean loss. CPU fits use float64.
    return torch.float32 if device.type == "cuda" else torch.float64


def lbfgs_batched(fg, theta, *, max_iter=1000, tol=LOGISTIC_TOL, history=10, max_ls=40, ftol=SKLEARN_FTOL):
    """Minimize B independent smooth objectives with per-problem L-BFGS.

    ``fg(theta) -> (f, g)`` with theta of shape (B, P), f (B,), g (B, P).
    Returns (theta, info) with per-problem iteration counts; ``converged``
    marks max|g| <= tol, ``flat`` a stop on no further decrease of f, and
    ``success`` either of the two.
    """
    torch = _torch()
    B, P = theta.shape
    f, g = fg(theta)
    S = theta.new_zeros(history, B, P)
    Y = theta.new_zeros(history, B, P)
    rho = theta.new_zeros(history, B)
    gamma = 1.0 / g.abs().sum(1).clamp_min(1.0)
    stored = 0
    iterations = torch.zeros(B, dtype=torch.long, device=theta.device)
    converged = g.abs().amax(1) <= tol
    failed = torch.zeros(B, dtype=torch.bool, device=theta.device)
    flat = torch.zeros(B, dtype=torch.bool, device=theta.device)
    active = ~converged
    for _ in range(max_iter):
        if not bool(active.any()):
            break
        q = g.clone()
        order = [(stored - 1 - i) % history for i in range(min(stored, history))]
        alphas = {}
        for j in order:
            alphas[j] = rho[j] * (S[j] * q).sum(1)
            q = q - alphas[j][:, None] * Y[j]
        r = gamma[:, None] * q
        for j in reversed(order):
            beta = rho[j] * (Y[j] * r).sum(1)
            r = r + S[j] * (alphas[j] - beta)[:, None]
        d = -r
        gd = (g * d).sum(1)
        ascent = gd >= 0
        if bool(ascent.any()):
            d = torch.where(ascent[:, None], -g, d)
            gd = (g * d).sum(1)
        d = torch.where(active[:, None], d, torch.zeros_like(d))
        gd = torch.where(active, gd, torch.zeros_like(gd))
        t = torch.ones(B, dtype=theta.dtype, device=theta.device)
        accepted = ~active
        theta_new, f_new, g_new = theta.clone(), f.clone(), g.clone()
        for _ls in range(max_ls):
            trial = theta + t[:, None] * d
            ft, gt = fg(trial)
            ok = (~accepted) & torch.isfinite(ft) & (ft <= f + 1e-4 * t * gd)
            if bool(ok.any()):
                theta_new = torch.where(ok[:, None], trial, theta_new)
                f_new = torch.where(ok, ft, f_new)
                g_new = torch.where(ok[:, None], gt, g_new)
                accepted = accepted | ok
            if bool(accepted.all()):
                break
            t = torch.where(accepted, t, t * .5)
        stalled = active & ~accepted
        s, y = theta_new - theta, g_new - g
        sy = (s * y).sum(1)
        valid = active & accepted & (sy > 1e-12 * (s * s).sum(1).clamp_min(1e-30).sqrt() * (y * y).sum(1).clamp_min(1e-30).sqrt())
        slot = stored % history
        S[slot] = torch.where(valid[:, None], s, torch.zeros_like(s))
        Y[slot] = torch.where(valid[:, None], y, torch.zeros_like(y))
        rho[slot] = torch.where(valid, 1.0 / sy.clamp_min(1e-30), torch.zeros_like(sy))
        yy = (y * y).sum(1)
        gamma = torch.where(valid & (yy > 0), sy / yy.clamp_min(1e-30), gamma)
        stored += 1
        decrease = (f - f_new).abs() <= ftol * torch.maximum(f.abs(), torch.ones_like(f))
        theta, f, g = theta_new, f_new, g_new
        iterations = iterations + active.long()
        done = g.abs().amax(1) <= tol
        converged = converged | (active & done)
        failed = failed | stalled
        flat = flat | (active & ~done & ~stalled & decrease & accepted)
        active = active & ~done & ~stalled & ~(decrease & accepted)
    info = dict(iterations=iterations.cpu().numpy(), converged=converged.cpu().numpy(),
                flat=flat.cpu().numpy(), success=(converged | flat).cpu().numpy(),
                line_search_failed=failed.cpu().numpy(), grad_max=g.abs().amax(1).cpu().numpy())
    return theta, info


@dataclass
class CompactFeatures:
    """Per-problem materialized features; ``train`` has shape (B, n, r)."""
    train: object
    val: object
    test: object


@dataclass
class EraseFeatures:
    """Shared X with per-problem low-rank erasure X - (X F) G^T."""
    train: object            # (n, D)
    val: object
    test: object
    F: object                # (B, D, r) input factors, r may be 0
    G: object                # (B, D, r) image bases


def _logits(features, part, W, b, binary):
    torch = _torch()
    if isinstance(features, CompactFeatures):
        Z = getattr(features, part)
        out = torch.einsum("bnd,bdk->bnk", Z, W)
    else:
        X = getattr(features, part)
        out = torch.einsum("nd,bdk->bnk", X, W)
        if features.F.shape[-1]:
            XF = torch.einsum("nd,bdr->bnr", X, features.F)
            GW = torch.einsum("bdr,bdk->brk", features.G, W)
            out = out - torch.einsum("bnr,brk->bnk", XF, GW)
    out = out + b[:, None, :]
    return out[..., 0] if binary else out


def fit_batched(features, y_train, n_classes, Cs, *, fit_intercept=True, device=None, dtype=None,
                max_iter=1000, tol=LOGISTIC_TOL):
    """Fit one L2 logistic regression per problem.

    ``y_train`` holds class indices in ``range(n_classes)``: shape (n,) shared
    by all problems or (B, n). ``Cs`` has one inverse penalty per problem.
    Returns (W, b, info); W is (B, d, K') with K'=1 for binary problems.
    """
    torch = _torch()
    binary = n_classes == 2
    Kp = 1 if binary else n_classes
    X = features.train
    B = len(Cs)
    d = X.shape[-1]
    n = X.shape[-2]
    yt = torch.as_tensor(np.asarray(y_train), device=X.device, dtype=torch.long)
    if yt.ndim == 1:
        yt = yt[None].expand(B, n)
    Ct = torch.as_tensor(np.asarray(Cs, dtype=float), device=X.device, dtype=X.dtype)
    penalty = 1.0 / (2.0 * Ct * n)                              # (B,)
    if binary:
        sign = (yt == 1).to(X.dtype) * 2 - 1                    # (B, n)
    P_w = d * Kp

    def unpack(theta):
        W = theta[:, :P_w].reshape(B, d, Kp)
        bias = theta[:, P_w:] if fit_intercept else torch.zeros(B, Kp, device=theta.device, dtype=theta.dtype)
        return W, bias

    def fg(theta):
        theta = theta.detach().requires_grad_(True)
        W, bias = unpack(theta)
        logits = _logits(features, "train", W, bias, binary)
        if binary:
            loss = torch.nn.functional.softplus(-sign * logits).mean(1)
        else:
            lse = torch.logsumexp(logits, dim=-1)
            picked = logits.gather(-1, yt[..., None])[..., 0]
            loss = (lse - picked).mean(1)
        f = loss + penalty * (W * W).sum((1, 2))
        (g,) = torch.autograd.grad(f.sum(), theta)
        return f.detach(), g.detach()

    theta0 = torch.zeros(B, P_w + (Kp if fit_intercept else 0), device=X.device, dtype=X.dtype)
    theta, info = lbfgs_batched(fg, theta0, max_iter=max_iter, tol=tol)
    W, bias = unpack(theta)
    return W.detach(), bias.detach(), info


def _predict(features, part, W, b, classes):
    """Probabilities over ``classes`` for every problem: (B, n, K)."""
    torch = _torch()
    binary = len(classes) == 2
    logits = _logits(features, part, W, b, binary)
    if binary:
        p1 = torch.sigmoid(logits)
        return torch.stack([1 - p1, p1], dim=-1)
    return torch.softmax(logits, dim=-1)


def _to_tensor(value, device, dtype):
    torch = _torch()
    return torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=dtype)


def compact_features(train_list, val_list, test_list, device, dtype):
    """Pad per-problem feature matrices (n, r_b) into (B, n, max r)."""
    torch = _torch()
    width = max([1] + [np.asarray(a).shape[1] for a in train_list])
    def stack(arrays):
        n = np.asarray(arrays[0]).shape[0]
        out = torch.zeros(len(arrays), n, width, device=device, dtype=dtype)
        for j, a in enumerate(arrays):
            a = np.asarray(a)
            if a.shape[1]:
                out[j, :, :a.shape[1]] = _to_tensor(a, device, dtype)
        return out
    return CompactFeatures(stack(train_list), stack(val_list), stack(test_list))


def erase_features(Xtr, Xva, Xte, factors, bases, device, dtype):
    """Shared design matrices with per-problem (F, G) erasure factors."""
    torch = _torch()
    rank = max([0] + [np.asarray(f).shape[1] for f in factors])
    D = np.asarray(Xtr).shape[1]
    F = torch.zeros(len(factors), D, rank, device=device, dtype=dtype)
    G = torch.zeros(len(factors), D, rank, device=device, dtype=dtype)
    for j, (f, gb) in enumerate(zip(factors, bases)):
        f, gb = np.asarray(f), np.asarray(gb)
        if f.shape[1]:
            F[j, :, :f.shape[1]] = _to_tensor(f, device, dtype)
            G[j, :, :gb.shape[1]] = _to_tensor(gb, device, dtype)
    return EraseFeatures(_to_tensor(Xtr, device, dtype), _to_tensor(Xva, device, dtype),
                         _to_tensor(Xte, device, dtype), F, G)


def _margins(proba, y_idx):
    """True-class probability minus the best competitor; unseen labels -> -max."""
    n = proba.shape[0]
    result = -proba.max(axis=1)
    known = y_idx >= 0
    if known.any():
        rows = np.flatnonzero(known)
        competitors = proba[known].copy()
        competitors[np.arange(len(rows)), y_idx[known]] = -np.inf
        result[known] = proba[rows, y_idx[known]] - competitors.max(axis=1)
    if proba.shape[1] == 1:
        result = np.where(known, 1., -1.)
    return result[:n]


def tune_readouts(features, y, indices, Cs, *, device=None, dtype=None, need_margin=True,
                  max_iter=1000, tol=LOGISTIC_TOL, chunk=None):
    """Fit, tune C on validation and score test for every problem in ``features``.

    ``y`` is the full label vector; ``indices`` = (train, val, test) row indices
    into it. Problems are expanded over ``Cs``; ties favour the smaller C, as in
    ``geometry.tune_probe``. Returns one dict per problem with the same fields
    as ``LayerStudy._fit_parts``.
    """
    torch = _torch()
    tr, va, te = (np.asarray(i) for i in indices)
    y = np.asarray(y)
    classes, ytr = np.unique(y[tr], return_inverse=True)
    lookup = {c: j for j, c in enumerate(classes.tolist())}
    yva = np.asarray([lookup.get(v, -1) for v in y[va].tolist()])
    yte = np.asarray([lookup.get(v, -1) for v in y[te].tolist()])
    counts = np.bincount(ytr, minlength=len(classes))
    majority = classes[int(np.argmax(counts))]
    floor = y[te] == majority
    unseen = int((yte < 0).sum())
    Cs = sorted(set(float(c) for c in Cs))
    B = features.train.shape[0] if isinstance(features, CompactFeatures) else features.F.shape[0]
    results = [None] * B
    if len(classes) < 2:
        # Constant training labels: the empirical-prior classifier.
        pred_va = np.full(len(va), classes[0]); pred_te = np.full(len(te), classes[0])
        margin = np.where(yte == 0, 1., -1.) if need_margin else np.empty(0)
        for j in range(B):
            results[j] = dict(correct=pred_te == y[te], validation_correct=pred_va == y[va], floor=floor,
                              predicted=pred_te, validation_predicted=pred_va,
                              margin=margin, test_idx=te, val_idx=va, C=Cs[0], unseen_test_classes=unseen)
        return results
    best_score = np.full(B, -np.inf)
    best = [None] * B
    for C in Cs:
        W, b, info = fit_batched(features, ytr, len(classes), [C] * B, max_iter=max_iter, tol=tol)
        with torch.no_grad():
            pva = _predict(features, "val", W, b, classes).float().cpu().numpy()
            pte = _predict(features, "test", W, b, classes).float().cpu().numpy()
        for j in range(B):
            score = float(np.mean(classes[pva[j].argmax(1)] == y[va])) if len(va) else 0.
            if score > best_score[j]:
                best_score[j] = score
                best[j] = (C, pva[j], pte[j], bool(info["success"][j]))
    for j in range(B):
        C, pva, pte, converged = best[j]
        results[j] = dict(correct=classes[pte.argmax(1)] == y[te],
                          validation_correct=classes[pva.argmax(1)] == y[va], floor=floor,
                          predicted=classes[pte.argmax(1)], validation_predicted=classes[pva.argmax(1)],
                          margin=_margins(pte, yte) if need_margin else np.empty(0),
                          test_idx=te, val_idx=va, C=C, unseen_test_classes=unseen,
                          probe_backend="torch", probe_converged=converged)
    return results


def fit_logistic_weights(X, y_list, *, C=1.0, fit_intercept=True, device=None, dtype=None,
                         max_iter=1000, tol=LOGISTIC_TOL):
    """Fit one logistic regression per label vector on a shared X.

    Used for MLR subspace estimation, including batched permutation draws.
    Returns a list of (classes, coef) with coef shaped like scikit-learn's
    ``coef_``: (1, D) for binary and (K, D) otherwise.
    """
    torch = _torch()
    device = resolve_device(device)
    dtype = dtype or default_dtype(device)
    X = np.asarray(X)
    out = [None] * len(y_list)
    groups = {}
    for j, labels in enumerate(y_list):
        classes, codes = np.unique(np.asarray(labels), return_inverse=True)
        groups.setdefault(len(classes), []).append((j, classes, codes))
    Xt = _to_tensor(X, device, dtype)
    empty = torch.zeros(1, X.shape[1], 0, device=device, dtype=dtype)
    for k, members in groups.items():
        if k < 2:
            for j, classes, _ in members:
                out[j] = (classes, np.zeros((0, X.shape[1])))
            continue
        feats = EraseFeatures(Xt, Xt[:1], Xt[:1], empty.expand(len(members), -1, -1),
                              empty.expand(len(members), -1, -1))
        codes = np.stack([c for _, _, c in members])
        W, _, _ = fit_batched(feats, codes, k, [C] * len(members), fit_intercept=fit_intercept,
                              max_iter=max_iter, tol=tol)
        W = W.double().cpu().numpy()
        for (j, classes, _), w in zip(members, W):
            out[j] = (classes, w.T)   # (K', D) like coef_
    return out
