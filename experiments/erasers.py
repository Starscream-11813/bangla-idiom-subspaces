"""INLP and R-LACE concept erasers, fitted on one condition's SPACE rows (X9).

Both return an orthonormal basis B (D x R) of the removed subspace, so the
erased representation is x - (x B) B^T and the frozen pipeline's ``Subspace``
(basis = input factor = B) carries them through the same batched readouts as
the COV, MLR and LEACE erasers.

INLP   iterative nullspace projection (Ravfogel et al., 2020). At step t a
       multinomial L2 logistic regression (C = 1, intercept) is fitted on the
       source SPACE rows with the directions found so far projected out, and
       the row space of its class weights (K-1 directions: the L2 penalty
       centres multinomial weights) is added to B. The accuracy of each step's
       classifier on the source VALIDATION rows is INLP's stopping statistic;
       the operating point is the first step after which it is within ``eps``
       of the majority rate (the source language is linearly guarded on
       held-out rows). The whole path is kept, up to ``max_rank`` directions.
R-LACE relaxed linear adversarial concept erasure (Ravfogel et al., 2022). A
       rank-k projection P is optimized over the Fantope {0 <= P <= I,
       tr P = k} against a linear softmax adversary that reads x (I - P):
       alternating minibatch SGD steps for the adversary and projected
       gradient ascent for P, with the Fantope projection computed by
       eigendecomposition and bisection on the eigenvalue shift (the reference
       implementation's ``solve_constraint``). Every ``eval_every`` steps P is
       rounded to its top-k eigenvectors and scored by a freshly fitted
       logistic probe on the source VALIDATION rows; the rounded P with the
       highest validation loss is kept, and the run stops once that probe is
       within ``eps`` of the majority rate. The SGD learning rate is chosen
       per fit by a short pilot over a small grid, scored the same way.

Nothing here sees the target language or any TEST row.
"""
from __future__ import annotations

import time

import numpy as np

EPS = 0.01                 # guard tolerance: within one point of the source VALIDATION majority rate
INLP_C = 1.0               # the pipeline's estimator_C, as for MLR
INLP_MAX_RANK = 264        # 24 steps for F (11 directions each), 88 for S (3 each)
# Reference settings (SGD, weight decay 1e-4, minibatches of 256, P initialized at 0.1 N(0, 1)). The
# game's speed depends on the data's spectrum (on synthetic 12-class data, isotropic or with a power-law
# spectrum whose top eigenvalue is 40 times the mean, 0.01 and 0.03 guard within 2,000 steps and the
# reference 0.003 does not within 4,000-5,000; on inputs scaled 30x, 0.01 stalls), so the rate is
# chosen per fit from RLACE_LRS by a pilot (see ``rlace``).
RLACE = dict(steps=20000, batch=256, pilot_steps=1500, weight_decay=1e-4, eval_every=250, init_scale=0.1)
RLACE_LRS = (0.003, 0.01, 0.03, 0.1)


def _device_dtype(device=None):
    from idiom_subspace.fastprobe import default_dtype, resolve_device
    device = resolve_device(device)
    return device, default_dtype(device)


def class_codes(y, classes):
    lookup = {c: i for i, c in enumerate(np.asarray(classes).tolist())}
    return np.asarray([lookup.get(v, -1) for v in np.asarray(y).tolist()], dtype=int)


def orthonormal_error(B):
    B = np.asarray(B, dtype=float)
    return float(np.abs(B.T @ B - np.eye(B.shape[1])).max()) if B.shape[1] else 0.


def logistic_on_erased(X, codes, K, X_eval, factor=None, image=None, C=1.0, device=None, max_iter=1000):
    """Multinomial L2 logistic regression on X - (X F) G^T (F = G = 0: the full space).

    Returns (W (D, K'), b, probabilities on X_eval (n_eval, K), success); K' = 1 for two classes.
    """
    import torch
    from idiom_subspace.fastprobe import EraseFeatures, _predict, _to_tensor, fit_batched
    device, dtype = _device_dtype(device)
    D = X.shape[1]
    factor = np.zeros((D, 0)) if factor is None else np.asarray(factor)
    image = factor if image is None else np.asarray(image)
    r = factor.shape[1]
    F = torch.zeros(1, D, r, device=device, dtype=dtype)
    G = torch.zeros(1, D, r, device=device, dtype=dtype)
    if r:
        F[0] = _to_tensor(factor, device, dtype)
        G[0] = _to_tensor(image, device, dtype)
    evaluation = _to_tensor(X_eval, device, dtype)
    feats = EraseFeatures(_to_tensor(X, device, dtype), evaluation, evaluation, F, G)
    W, b, info = fit_batched(feats, np.asarray(codes), K, [C], max_iter=max_iter)
    with torch.no_grad():
        proba = _predict(feats, "val", W, b, np.arange(K))[0].double().cpu().numpy()
    return W[0].double().cpu().numpy(), b[0].double().cpu().numpy(), proba, bool(info["success"][0])


def scores(proba, codes):
    """Accuracy and mean log-loss of class probabilities against integer codes."""
    codes = np.asarray(codes)
    picked = proba[np.arange(len(codes)), codes]
    return float(np.mean(proba.argmax(1) == codes)), float(-np.mean(np.log(np.maximum(picked, 1e-12))))


def _prepare(X, y, X_val, y_val):
    """Centre on the SPACE mean; keep VALIDATION rows whose class occurs in SPACE."""
    classes = np.unique(y)
    codes = class_codes(y, classes)
    keep = np.isin(y_val, classes)
    vcodes = class_codes(np.asarray(y_val)[keep], classes)
    center = np.asarray(X, float).mean(0)
    majority_code = int(np.bincount(codes, minlength=len(classes)).argmax())
    majority = float(np.mean(vcodes == majority_code)) if len(vcodes) else np.nan
    return (classes, codes, np.asarray(X, float) - center, vcodes, np.asarray(X_val, float)[keep] - center,
            center, majority)


def guard(X, y, X_val, y_val, factor=None, image=None, C=1.0, device=None):
    """Source-language guard statistic of any eraser: a probe fitted on erased SPACE, scored on VALIDATION."""
    classes, codes, Xc, vcodes, Vc, _, majority = _prepare(X, y, X_val, y_val)
    _, _, proba, ok = logistic_on_erased(Xc, codes, len(classes), Vc, factor, image, C, device)
    acc, loss = scores(proba, vcodes)
    return dict(guard_val_acc=acc, guard_val_loss=loss, guard_val_majority=majority, guard_gap=acc - majority,
                guard_probe_converged=ok, guard_val_n=int(len(vcodes)))


def inlp(X, y, X_val, y_val, max_rank=INLP_MAX_RANK, C=INLP_C, eps=EPS, device=None, tol=1e-8):
    """Iterative nullspace projection on SPACE rows, with VALIDATION accuracy along the path.

    ``path[t]`` describes a classifier fitted after t removal steps; ``ranks[t]`` is
    the removed rank after t steps; ``basis[:, :ranks[t]]`` spans the removed subspace.
    """
    started = time.time()
    classes, codes, Xc, vcodes, Vc, center, majority = _prepare(X, y, X_val, y_val)
    K, D = len(classes), Xc.shape[1]
    per_step = 1 if K == 2 else K - 1
    steps = max(1, int(np.ceil(min(max_rank, D - per_step) / per_step)))
    B = np.zeros((D, 0))
    ranks, path = [0], []
    for t in range(steps + 1):
        W, _, proba, ok = logistic_on_erased(Xc, codes, K, Vc, B, B, C, device)
        acc, loss = scores(proba, vcodes)
        path.append(dict(step=t, removed_rank=int(B.shape[1]), val_acc=acc, val_loss=loss, val_majority=majority,
                         guarded=bool(acc <= majority + eps), probe_converged=ok))
        if t == steps:
            break
        # The fitted weights already lie in the complement of B; project anyway for numerical safety.
        W = W - B @ (B.T @ W)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        if not len(s) or s[0] <= 0:
            break
        new = U[:, s > tol * s[0]][:, :per_step]
        for _ in range(2):
            new = new - B @ (B.T @ new)
        new = np.linalg.qr(new)[0]
        B = np.hstack([B, new])
        ranks.append(int(B.shape[1]))
    guarded = [p["step"] for p in path if p["step"] >= 1 and p["guarded"]]
    stop = guarded[0] if guarded else path[-1]["step"]
    return dict(center=center, basis=B, ranks=ranks, path=path, stop=int(stop), guarded=bool(guarded),
                majority=majority, per_step=per_step, seconds=time.time() - started,
                orthonormal_error=orthonormal_error(B))


def fantope_project(P, rank, iters=60):
    """Euclidean projection of a symmetric matrix onto {0 <= P <= I, tr P = rank}.

    Returns (projection, eigenvectors, clipped eigenvalues); eigenvalues ascend.
    The shift theta solves sum(clip(lambda - theta, 0, 1)) = rank by bisection,
    done with tensor ops only so that no GPU synchronization is needed.
    """
    import torch
    try:
        evals, evecs = torch.linalg.eigh(P)
    except RuntimeError:
        # cuSOLVER can fail on clustered spectra (clipped Fantope iterates); LAPACK in float64 does not.
        evals, evecs = (t.to(P.device, P.dtype) for t in torch.linalg.eigh(P.detach().double().cpu()))
    lo, hi = evals.min() - 1, evals.max()
    for _ in range(iters):
        mid = (lo + hi) / 2
        above = torch.clamp(evals - mid, 0, 1).sum() > rank
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
    lam = torch.clamp(evals - (lo + hi) / 2, 0, 1)
    return (evecs * lam) @ evecs.T, evecs, lam


def rlace(X, y, X_val, y_val, rank, steps=RLACE["steps"], batch=RLACE["batch"], lrs=RLACE_LRS,
          pilot_steps=RLACE["pilot_steps"], weight_decay=RLACE["weight_decay"], eval_every=RLACE["eval_every"],
          init_scale=RLACE["init_scale"], eps=EPS, seed=0, device=None, progress=None):
    """Relaxed linear adversarial concept erasure of ``rank`` directions (see the module docstring).

    The SGD game's speed depends on the spectrum of the data, so the learning rate is
    chosen per fit: a pilot of ``pilot_steps`` steps per rate in ``lrs``, scored like the
    rounded P itself (highest source-VALIDATION loss of a fresh probe). A pilot that
    already guards is the result; otherwise the chosen rate runs the full ``steps``
    budget from the same seed. Diverging rates (non-finite P or losses) are skipped.
    """
    started = time.time()
    prepared = _prepare(X, y, X_val, y_val)
    pilots = []
    for lr in lrs:
        try:
            out = _rlace_game(prepared, rank, min(pilot_steps, steps), batch, lr, weight_decay, eval_every, init_scale,
                              eps, seed, device, progress, f"pilot lr {lr:g}")
        except FloatingPointError as error:
            pilots.append(dict(lr=lr, diverged=True, error=str(error)[:200]))
            continue
        pilots.append(dict(lr=lr, diverged=False, result=out))
    ranked = sorted([p for p in pilots if not p["diverged"]], key=lambda p: -p["result"]["best_val_loss"])
    final, chosen = None, None
    for candidate in ranked:
        if candidate["result"]["stopped_early"] or pilot_steps >= steps:
            final, chosen = candidate["result"], candidate["lr"]
            break
        try:
            final = _rlace_game(prepared, rank, steps, batch, candidate["lr"], weight_decay, eval_every, init_scale,
                                eps, seed, device, progress, f"lr {candidate['lr']:g}")
            chosen = candidate["lr"]
            break
        except FloatingPointError:
            continue
    if final is None:
        raise FloatingPointError(f"R-LACE diverged at every learning rate in {list(lrs)}")
    summary = []
    for p in pilots:
        r = p.get("result", {})
        summary.append(dict(lr=p["lr"], diverged=p["diverged"], best_val_loss=r.get("best_val_loss"),
                            best_val_acc=r.get("best_val_acc"), steps_run=r.get("steps_run"),
                            stopped_early=r.get("stopped_early")))
    final = dict(final, lr_used=chosen, pilot=summary, seconds=time.time() - started)
    return final


def _rlace_game(prepared, rank, steps, batch, lr, weight_decay, eval_every, init_scale, eps, seed, device, progress,
                phase=""):
    import torch
    device, dtype = _device_dtype(device)
    classes, codes, Xc, vcodes, Vc, center, majority = prepared
    K, D, n = len(classes), Xc.shape[1], len(Xc)
    binary = K == 2
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    Xt = torch.as_tensor(Xc, device=device, dtype=dtype)
    yt = torch.as_tensor(codes, device=device, dtype=torch.long)
    out = 1 if binary else K
    predictor = torch.nn.Linear(D, out).to(device=device, dtype=dtype)
    with torch.no_grad():
        bound = 1 / np.sqrt(D)
        predictor.weight.copy_(((torch.rand(out, D, generator=gen) * 2 - 1) * bound).to(device, dtype))
        predictor.bias.copy_(((torch.rand(out, generator=gen) * 2 - 1) * bound).to(device, dtype))
        start = init_scale * torch.randn(D, D, generator=gen)
    P = ((start + start.T) / 2).to(device=device, dtype=dtype).requires_grad_(True)
    opt_P = torch.optim.SGD([P], lr=lr, weight_decay=weight_decay)
    opt_c = torch.optim.SGD(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    eye = torch.eye(D, device=device, dtype=dtype)

    def loss_fn(logits, target):
        if binary:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target.to(dtype))
        return torch.nn.functional.cross_entropy(logits, target)

    def rows():
        return torch.randperm(n, generator=gen)[:min(batch, n)].to(device)

    def project(check):
        # The finiteness check synchronizes the GPU, so it runs only at evaluation steps.
        with torch.no_grad():
            if check and not bool(torch.isfinite(P).all()):
                raise FloatingPointError("non-finite P")
            try:
                projected, evecs, lam = fantope_project((P + P.T) / 2, rank)
            except RuntimeError as error:
                raise FloatingPointError(str(error)) from error
            P.copy_(projected)
        return evecs, lam

    evecs, lam = project(True)
    best, history, stopped = None, [], False
    step = 0
    for step in range(int(steps)):
        r = rows()
        loss_P = -loss_fn(predictor(Xt[r] @ (eye - (P + P.T) / 2)), yt[r])
        opt_P.zero_grad()
        loss_P.backward()
        opt_P.step()
        evaluate = step % eval_every == 0 or step == steps - 1
        evecs, lam = project(evaluate)
        r = rows()
        loss_c = loss_fn(predictor(Xt[r] @ (eye - P.detach())), yt[r])
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        if evaluate:
            if not np.isfinite(float(loss_c.detach().cpu())) or not bool(torch.isfinite(evecs).all()):
                raise FloatingPointError("non-finite adversary loss or eigenvectors")
            basis = np.linalg.qr(evecs[:, -rank:].double().cpu().numpy())[0]
            _, _, proba, ok = logistic_on_erased(Xc, codes, K, Vc, basis, basis, 1.0, device)
            acc, loss = scores(proba, vcodes)
            history.append(dict(step=int(step), val_acc=acc, val_loss=loss, val_majority=majority,
                                adversary_loss=float(loss_c.detach().cpu()), probe_converged=ok,
                                relaxed_mass_top_rank=float(lam[-rank:].sum().cpu()) / rank))
            if best is None or loss > best["val_loss"]:
                best = dict(step=int(step), val_loss=loss, val_acc=acc, basis=basis)
            if progress is not None:
                progress(phase, step, acc, majority)
            if step > 0 and abs(acc - majority) < eps:
                stopped = True
                break
    return dict(center=center, basis=best["basis"], history=history, best_step=best["step"],
                best_val_acc=best["val_acc"], best_val_loss=best["val_loss"], majority=majority,
                steps_run=int(step) + 1, stopped_early=stopped, orthonormal_error=orthonormal_error(best["basis"]))
