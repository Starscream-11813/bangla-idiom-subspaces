"""Train-only geometry and inference helpers for concept-subspace experiments.

Rows are observations. ``Subspace.project`` returns compact, isometric
coordinates of the estimator's projected observations. For LEACE the primary
projection and its complementary eraser are oblique, as in the reference
implementation; other estimators use orthogonal projections. Orthogonal image
diagnostics are explicitly named and never substituted for the LEACE operator.

No function in this module fits on a test set or downloads model weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import warnings
from functools import cached_property
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .fastprobe import LOGISTIC_TOL


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite two-dimensional matrix")
    return X


def _known_labels(y):
    """None, NaN and the numeric sentinel -1 denote unavailable labels."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    def known(v):
        if v is None:
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(v) and v != -1)
        return True
    return y, np.asarray([known(v) for v in y], dtype=bool)


def _labeled(X, y):
    X = _matrix(X)
    y, mask = _known_labels(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if not mask.any():
        raise ValueError("At least one observation with a known label is required")
    # Recover a concrete numeric/string dtype after removing None from an
    # object array; sklearn otherwise rejects ordinary integer labels stored
    # in an object array as an unknown classification target.
    return X[mask], np.asarray(y[mask].tolist())


def _svd_basis(A, requested, spectrum=None):
    """Return only numerically supported left singular directions.

    When ``spectrum`` is a list, the singular values are appended to it so that
    callers can report how well each truncated prefix is determined.
    """
    if requested == 0 or min(A.shape) == 0:
        return np.empty((A.shape[0], 0), dtype=np.float64)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(np.float64).eps * (s[0] if len(s) else 0)
    r = min(requested, int(np.sum(s > tol)))
    if spectrum is not None:
        spectrum.extend(np.where(s > tol, s, 0.).tolist())
    return U[:, :r]


def spectral_gap(spectrum, rank):
    """Relative gap (s_M - s_{M+1}) / s_M after the rank-M prefix.

    A gap near zero means the M-dimensional prefix is not well determined:
    nearly tied directions can rotate between fits (e.g. two solvers).
    """
    if spectrum is None or rank < 1 or len(spectrum) < rank:
        return np.nan
    s = np.asarray(spectrum, dtype=float)
    nxt = s[rank] if len(s) > rank else 0.
    return float((s[rank - 1] - nxt) / s[rank - 1]) if s[rank - 1] > 0 else np.nan


def _ordered_qr(A):
    """Orthonormalize an independent, ordered spectral prefix without mixing it.

    An SVD of the unwhitened Fisher/LEACE columns would reorder their prefixes.
    A reduced QR preserves every leading column span instead.
    """
    if A.shape[1] == 0:
        return A.copy(), np.empty((0, 0))
    Q, R = np.linalg.qr(A, mode="reduced")
    return Q, R


def ledoit_wolf_ridge(Xc):
    """Tikhonov ridge equivalent to Ledoit--Wolf shrinkage of the covariance of centred rows.

    LW uses (1-a) S + a mu I with mu = tr(S)/d. Whitening is invariant to a
    global scale, so this equals S + (a / (1 - a)) mu I, i.e. ridge a/(1-a)
    in ``_symmetric_roots``. Needed when there are fewer rows than dimensions.
    """
    from sklearn.covariance import ledoit_wolf_shrinkage
    if len(Xc) < 2:
        return 1.0
    alpha = float(ledoit_wolf_shrinkage(np.asarray(Xc, dtype=np.float64), assume_centered=True))
    alpha = min(max(alpha, 0.), .999)
    return alpha / (1 - alpha)


def _symmetric_roots(S, ridge):
    """Stable covariance roots with a scale-relative Tikhonov ridge."""
    d = S.shape[0]
    scale = float(np.trace(S) / max(d, 1))
    if scale <= 0:
        return np.zeros_like(S), np.zeros_like(S)
    e, V = np.linalg.eigh((S + S.T) * .5)
    e = np.maximum(e, 0) + ridge * scale
    tol = max(d, 1) * np.finfo(float).eps * max(float(e.max()), scale)
    # Use the same support for both roots (Moore--Penrose whitening at ridge=0).
    root = np.where(e > tol, np.sqrt(np.maximum(e, 0)), 0.)
    inv = np.divide(1, root, out=np.zeros_like(root), where=e > tol)
    return (V * root) @ V.T, (V * inv) @ V.T


@dataclass(frozen=True)
class Subspace:
    """A centered projection with compact, metric-preserving coordinates.

    ``basis`` has shape (ambient dimensions, effective rank), is orthonormal,
    and spans the projection's IMAGE. ``projection_factor`` is the input map;
    for orthogonal estimators it equals ``basis``. For LEACE these differ.
    The row-vector projection matrix is ``projection_factor @ basis.T``.

    Returning compact coordinates rather than an ambient rank-deficient vector
    preserves inner products and the L2 geometry of a probe on the reference
    projection, since reconstruction multiplies by an orthonormal matrix.
    """
    center: np.ndarray
    basis: np.ndarray
    effective_rank: int
    method: str
    requested_rank: int
    projection_factor: Optional[np.ndarray] = None
    spectral_gap: float = np.nan

    def __post_init__(self):
        if (self.center.ndim != 1 or self.basis.shape != (len(self.center), self.effective_rank)
                or self.effective_rank < 0 or self.effective_rank > self.requested_rank):
            raise ValueError("Invalid subspace dimensions")
        if not np.isfinite(self.center).all() or not np.isfinite(self.basis).all():
            raise ValueError("Subspace parameters must be finite")
        if not np.allclose(self.basis.T @ self.basis, np.eye(self.effective_rank), atol=1e-8):
            raise ValueError("Subspace basis must be orthonormal")
        if self.projection_factor is not None:
            if (self.projection_factor.shape != self.basis.shape
                    or not np.isfinite(self.projection_factor).all()):
                raise ValueError("Invalid projection factor")

    @property
    def input_factor(self):
        return self.basis if self.projection_factor is None else self.projection_factor

    @property
    def projection_operator(self):
        """Ambient row-vector projection; materialized only when requested."""
        return self.input_factor @ self.basis.T

    @property
    def erase_operator(self):
        """Ambient complementary operator; not retained per null draw."""
        return np.eye(len(self.center)) - self.projection_operator

    def _centered(self, X):
        X = _matrix(X)
        if X.shape[1] != len(self.center):
            raise ValueError("X has the wrong number of features")
        return X - self.center

    def project(self, X):
        """Isometric coordinates of the primary projection, shape (n, rank)."""
        return self._centered(X) @ self.input_factor

    def erase(self, X):
        """Centered primary residual: oblique LEACE, orthogonal otherwise."""
        Xc = self._centered(X)
        return Xc - (Xc @ self.input_factor) @ self.basis.T

    def formal_erase(self, X):
        """Compatibility alias of the primary eraser (including LEACE)."""
        return self.erase(X)

    def orthogonal_project(self, X):
        """Explicit diagnostic: orthogonal coordinates of the image span."""
        return self._centered(X) @ self.basis

    def orthogonal_erase(self, X):
        """Explicit diagnostic: erase the orthogonal image projection."""
        Xc = self._centered(X)
        return Xc - (Xc @ self.basis) @ self.basis.T

    @cached_property
    def complement_basis(self):
        """Orthonormal coordinates for the image of the primary eraser.

        This requires a complete QR, so callers should usually keep ambient
        residuals when rank is small compared with the ambient dimension.
        """
        d, r = len(self.center), self.effective_rank
        if r == 0:
            return np.eye(d)
        return np.linalg.qr(self.input_factor, mode="complete")[0][:, r:]

    def erase_coordinates(self, X):
        """Isometric compact coordinates of the primary complementary output."""
        return self.erase(X) @ self.complement_basis


@dataclass(frozen=True)
class FittedEstimator:
    """One estimator fit whose ordered spectral prefixes can be reused.

    All ranks share one MLR fit or spectral decomposition. ``basis`` stores
    orthonormal image prefixes. LEACE also stores input factors encoding its
    oblique projection; no dense D-by-D operator is retained per rank.
    """
    center: np.ndarray
    basis: np.ndarray
    method: str
    max_rank: int
    projection_inputs: Optional[np.ndarray] = None
    image_triangular: Optional[np.ndarray] = None
    spectrum: Optional[tuple] = None

    @property
    def effective_rank(self):
        return self.basis.shape[1]

    def at_rank(self, rank):
        if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= self.max_rank:
            raise ValueError("rank must be an integer between zero and the fitted max_rank")
        r = min(int(rank), self.effective_rank)
        factor = None
        if self.projection_inputs is not None:
            # P = A B.T, B=QR. Compact coordinates X A R.T reconstruct X P
            # through Q.T and exactly preserve its inner products/L2 metric.
            factor = self.projection_inputs[:, :r] @ self.image_triangular[:r, :r].T
        return Subspace(self.center, self.basis[:, :r], r, self.method, int(rank), factor,
                        spectral_gap(self.spectrum, r))


MLR_VARIANTS = {"reference", "centered"}


def mlr_coefficients(X, y_codes, C=1.0, *, variant="reference", seed=0, backend="sklearn",
                     max_iter=5000, device=None):
    """Logistic-regression weights used by the MLR estimator, shape (K', D).

    ``reference`` follows concept_space (``get_Linear_model``): no intercept,
    rows of ``coef_`` normalized to unit length before the SVD (``norm=True``).
    ``centered`` fits an intercept and centers the weights across classes.
    ``y_codes`` may be a list of label vectors (batched permutation draws).
    Both backends solve the objective to ``LOGISTIC_TOL`` (the reference used
    scikit-learn's default 1e-4, which leaves rank-1 directions solver-dependent).
    """
    if variant not in MLR_VARIANTS:
        raise ValueError(f"Unknown MLR variant {variant!r}; use one of {sorted(MLR_VARIANTS)}")
    batched = isinstance(y_codes, list)
    labels = y_codes if batched else [y_codes]
    fit_intercept = variant == "centered"
    if backend == "torch":
        from .fastprobe import fit_logistic_weights
        fitted = [coef for _, coef in fit_logistic_weights(X, labels, C=C, fit_intercept=fit_intercept,
                                                             device=device, max_iter=max(max_iter, 1000),
                                                             tol=LOGISTIC_TOL)]
    else:
        # One estimator fit per condition/concept: allow ample iterations, keep logs quiet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitted = [LogisticRegression(C=C, fit_intercept=fit_intercept, max_iter=max(max_iter, 5000),
                                         tol=LOGISTIC_TOL, random_state=seed).fit(X, codes).coef_
                      for codes in labels]
    out = []
    for coef in fitted:
        coef = np.asarray(coef, dtype=np.float64)
        if variant == "reference":
            norms = np.linalg.norm(coef, axis=1, keepdims=True)
            coef = coef / np.where(norms > 0, norms, 1.)
        elif coef.shape[0] > 2:
            coef = coef - coef.mean(0, keepdims=True)
        out.append(coef)
    return out if batched else out[0]


def fit_estimator(X, y, method, max_rank, seed=0, ridge=0.0, C=1.0, *, mlr_variant="reference",
                  backend="sklearn", max_iter=5000, device=None, mlr_coef=None):
    """Fit one ranked estimator on estimation rows, then reuse ``at_rank``.

    RAND: random orthonormal directions. PCA: principal directions of X.
    MLR: row space of logistic-regression weights; the default ``reference``
    variant reproduces concept_space (no intercept, unit-norm class rows before
    the SVD). LDA: Fisher directions. CPCA: PCA of equally weighted class
    centroids centered at their centroid mean, matching the reference. COV:
    feature/one-hot-label covariance SVD. LEACE: the reference oblique
    projection, using Moore--Penrose covariance whitening by default. Explicit
    ridge>0 gives a regularized variant; at full supported cross-covariance
    rank its training covariance is still erased, while truncation removes
    only the selected directions.

    Supervised methods have rank at most K-1 and return their effective
    numerical rank rather than padding with arbitrary null directions. Missing
    labels are excluded; constant labels give an empty supervised subspace.
    ``mlr_coef`` supplies precomputed MLR weights (batched permutation fits).
    """
    X = _matrix(X)
    if len(X) == 0:
        raise ValueError("At least one estimation row is required")
    if not isinstance(max_rank, (int, np.integer)) or max_rank < 0:
        raise ValueError("max_rank must be a nonnegative integer")
    shrink = isinstance(ridge, str)
    if shrink and ridge != "ledoit_wolf":
        raise ValueError("ridge must be a nonnegative number or 'ledoit_wolf'")
    if not shrink and (not np.isfinite(ridge) or ridge < 0):
        raise ValueError("ridge must be finite and nonnegative")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    method = method.upper()
    if method not in {"RAND", "PCA", "MLR", "LDA", "CPCA", "COV", "LEACE"}:
        raise ValueError(f"Unknown subspace estimator: {method}")
    if method not in {"RAND", "PCA"}:
        X, y = _labeled(X, y)
    center, d = X.mean(0), X.shape[1]
    Xc = X - center
    requested, limit = int(max_rank), min(int(max_rank), d)
    inputs, triangular = None, None
    spectrum = []
    if method == "RAND":
        basis = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, limit)), mode="reduced")[0]
    elif method == "PCA":
        # Fit the largest needed prefix once. Fixed-seed randomized SVD avoids
        # computing every principal direction in a large encoder space.
        # One extra component so the spectral gap after the last requested rank is known.
        components, pca = _pca_prefix(X, limit + 1 if limit < min(X.shape) else limit, seed)
        basis = components[:limit].T
        if pca is not None:
            spectrum.extend(pca.singular_values_.tolist())
    else:
        classes, yi = np.unique(y, return_inverse=True)
        k, n = len(classes), len(X)
        limit = min(limit, k - 1)
        Y = np.eye(k)[yi]
        count = Y.sum(0)
        means = (Y.T @ X) / count[:, None]
        mean_offsets = means - center
        Cxy = Xc.T @ (Y - Y.mean(0)) / n
        if limit == 0:
            basis = np.empty((d, 0))
            if method == "LEACE":
                inputs, triangular = np.empty((d, 0)), np.empty((0, 0))
        elif method == "MLR":
            coef = mlr_coef if mlr_coef is not None else mlr_coefficients(
                X, yi, C, variant=mlr_variant, seed=seed, backend=backend, max_iter=max_iter, device=device)
            basis = _svd_basis(np.asarray(coef, dtype=np.float64).T, limit, spectrum)
        elif method == "CPCA":
            basis = _svd_basis((means - means.mean(0)).T, limit, spectrum)
        elif method == "COV":
            basis = _svd_basis(Cxy, limit, spectrum)
        elif method == "LDA":
            residual = X - means[yi]
            Sw = residual.T @ residual / n
            if np.trace(Sw) == 0:
                # Class-constant rows are perfectly discriminative, rather
                # than uninformative: every between-class direction has an
                # infinite unregularized Fisher ratio.
                basis = _svd_basis(mean_offsets.T * np.sqrt(count / n), limit, spectrum)
            else:
                _, whitening = _symmetric_roots(Sw, ledoit_wolf_ridge(residual) if shrink else ridge)
                whitened_means = whitening @ (mean_offsets.T * np.sqrt(count / n))
                U = _svd_basis(whitened_means, limit, spectrum)
                basis, _ = _ordered_qr(whitening @ U)
        else:  # regularized LEACE
            unwhitening, whitening = _symmetric_roots(Xc.T @ Xc / n, ledoit_wolf_ridge(Xc) if shrink else ridge)
            U = _svd_basis(whitening @ Cxy, limit, spectrum)
            # Column-vector removed component: S^1/2 U U^T S^-1/2.
            # Its image is span(S^1/2 U); row vectors use the transpose.
            basis, triangular = _ordered_qr(unwhitening @ U)
            inputs = whitening @ U
    return FittedEstimator(center, basis, method, requested, inputs, triangular,
                           tuple(spectrum) if spectrum else None)


def fit_subspace(X, y, method, rank, seed=0, ridge=0.0, C=1.0, **kwargs):
    """Convenience one-rank wrapper; use ``fit_estimator`` for rank sweeps."""
    return fit_estimator(X, y, method, rank, seed=seed, ridge=ridge, C=C, **kwargs).at_rank(rank)


def _pca_prefix(X, requested, seed):
    """Deterministic/seeded PCA prefix without a full large-space SVD."""
    Xc = X - X.mean(0)
    limit = min(requested, *X.shape)
    if limit == 0 or not np.any(Xc):
        return np.empty((0, X.shape[1])), None
    smallest = min(X.shape)
    solver = "randomized" if smallest > 256 and limit < .8 * smallest else "full"
    pca = PCA(n_components=limit, svd_solver=solver, random_state=seed,
              iterated_power=7)
    pca.fit(X)
    singular = pca.singular_values_
    tol = max(X.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
    keep = int(np.sum(singular > tol))
    pca.components_ = pca.components_[:keep]
    pca.n_components_ = keep
    pca.singular_values_ = pca.singular_values_[:keep]
    pca.explained_variance_ = pca.explained_variance_[:keep]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:keep]
    return pca.components_, pca


def fit_probe(X, y, seed=0, kind="linear", C=1.0, max_iter=5000):
    """Fit a classifier only on the supplied training rows.

    Zero-feature and constant-label tasks use an empirical-prior classifier.
    Nonlinear alternatives are an RBF SVM (``rbf``) and one-hidden-layer MLP
    (``mlp``); ``C`` is inverse L2 strength for the MLP as for the linear probe.
    """
    X, y = _labeled(X, y)
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    kind = kind.lower()
    if kind not in {"linear", "mlp", "rbf"}:
        raise ValueError(f"Unknown probe kind: {kind}")
    if X.shape[1] == 0 or len(np.unique(y)) == 1:
        model = DummyClassifier(strategy="prior")
    elif kind == "linear":
        model = LogisticRegression(C=C, max_iter=max_iter, tol=LOGISTIC_TOL, random_state=seed)
    elif kind == "rbf":
        model = SVC(C=C, kernel="rbf", probability=True, random_state=seed)
    else:
        model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1 / C,
                              max_iter=1000, random_state=seed)
    # Convergence is recorded on the model (and in the outputs) instead of
    # printing a multi-line warning for every capped fit.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    model.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return model


def tune_probe(Xtrain, ytrain, Xval, yval, Cs, seed=0, kind="linear", max_iter=5000):
    """Choose C on validation accuracy, returning the train-only fitted model.

    Ties favor the smaller C. Unknown validation labels are ignored; a label
    unseen during training is a legitimate validation error, not a crash.
    """
    Xval, yval = _labeled(Xval, yval)
    Cs = sorted(set(float(c) for c in Cs))
    if not Cs:
        raise ValueError("At least one candidate C is required")
    best, best_c, best_score = None, None, -np.inf
    for c in Cs:
        model = fit_probe(Xtrain, ytrain, seed=seed, kind=kind, C=c, max_iter=max_iter)
        score = float(np.mean(model.predict(Xval) == yval))
        if score > best_score:
            best, best_c, best_score = model, c, score
    return best, best_c


def resolve_probe_backend(cfg):
    """'torch' when requested (or 'auto' with CUDA available), else 'sklearn'."""
    requested = str(getattr(cfg, "probe_backend", "sklearn") or "sklearn").lower()
    if requested not in {"auto", "torch", "sklearn"}:
        raise ValueError("probe_backend must be auto, torch or sklearn")
    if requested == "auto":
        try:
            import torch
            return "torch" if torch.cuda.is_available() else "sklearn"
        except ImportError:
            return "sklearn"
    return requested


@dataclass
class _Preprocessor:
    mean_: np.ndarray
    scale_: float
    pca: Optional[PCA]

    def transform(self, X):
        X = _matrix(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("X has the wrong number of features")
        out = X - self.mean_
        if self.pca is not None:
            out = out @ self.pca.components_.T
        return out / self.scale_

    @property
    def effective_rank(self):
        return len(self.mean_) if self.pca is None else self.pca.n_components_


def fitted_preprocessor(Xspace, pca_dim=None, seed=0):
    """Fit centering, optional reproducible PCA, and global RMS on space only.

    PCA computes only the requested prefix (fixed-seed randomized SVD for large
    reductions, exact SVD otherwise), excluding unsupported zero-variance
    directions. The fitted object never updates parameters in ``transform``.
    """
    Xspace = _matrix(Xspace)
    if len(Xspace) == 0:
        raise ValueError("At least one estimation row is required")
    mean = Xspace.mean(0)
    Xc = Xspace - mean
    pca = None
    if pca_dim is not None:
        if not isinstance(pca_dim, (int, np.integer)) or pca_dim < 1:
            raise ValueError("pca_dim must be a positive integer or None")
        components, pca = _pca_prefix(Xspace, int(pca_dim), seed)
        if pca is None:
            # Avoid undefined explained-variance ratios for constant inputs.
            pca = PCA(n_components=0, svd_solver="full", random_state=seed)
            pca.components_ = np.empty((0, Xspace.shape[1]))
            pca.n_components_ = 0
            pca.singular_values_ = np.empty(0)
            pca.explained_variance_ = np.empty(0)
            pca.explained_variance_ratio_ = np.empty(0)
        Xc = Xc @ pca.components_.T
    scale = float(np.sqrt(np.mean(Xc ** 2))) if Xc.size else 1.0
    return _Preprocessor(mean_=mean, scale_=scale if scale > 0 else 1.0, pca=pca)


def shared_subspace_index(cross, within, floor, min_den=0.02):
    """Chance-adjusted SSI, undefined for a weak/invalid within reference."""
    if not np.isfinite([cross, within, floor]).all() or within - floor <= min_den:
        return np.nan
    return float((cross - floor) / (within - floor))


def null_adjusted_index(cross, within, null, min_den=0.02):
    """Normalize transfer above a matched null rather than a majority floor."""
    return shared_subspace_index(cross, within, null, min_den=min_den)


def paired_cluster_bootstrap(values, groups, statistic, n_boot=500, seed=0):
    """Resample groups, preserving paired observations and whole clusters.

    ``values`` is a row-aligned ndarray, tuple of arrays, or mapping of arrays.
    ``statistic`` accepts that structure and returns a scalar. Invalid bootstrap
    statistics are excluded from intervals and counted in ``n_valid``; an
    undefined point estimate yields an undefined interval as well.
    """
    if isinstance(values, Mapping):
        arrays = {key: np.asarray(value) for key, value in values.items()}
        lengths = [len(v) for v in arrays.values()]
        take = lambda ix: {key: value[ix] for key, value in arrays.items()}
    elif isinstance(values, tuple):
        arrays = tuple(np.asarray(value) for value in values)
        lengths = [len(v) for v in arrays]
        take = lambda ix: tuple(value[ix] for value in arrays)
    else:
        arrays = np.asarray(values)
        lengths = [len(arrays)]
        take = lambda ix: arrays[ix]
    groups = np.asarray(groups)
    if groups.ndim != 1 or not lengths or any(n != len(groups) for n in lengths):
        raise ValueError("Values and one-dimensional groups must be row-aligned")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    empty = dict(estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_valid=0)
    if len(groups) == 0:
        return empty
    _, inv = np.unique(groups, return_inverse=True)
    clusters = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    estimate = float(statistic(take(np.arange(len(groups)))))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        ix = np.concatenate([clusters[k] for k in rng.integers(0, len(clusters), len(clusters))])
        value = float(statistic(take(ix)))
        if np.isfinite(value):
            boot.append(value)
    lo, hi = np.quantile(boot, [.025, .975]) if boot and np.isfinite(estimate) else (np.nan, np.nan)
    return dict(estimate=estimate, ci_lo=float(lo), ci_hi=float(hi), n_valid=len(boot))


def monte_carlo_pvalue(obs, null, tail="greater"):
    """Finite-null Monte Carlo p value with a +1 correction.

    The two-sided alternative compares absolute magnitudes, appropriate for a
    null centered at zero. Undefined observations never receive a p value.
    """
    if tail not in {"greater", "less", "two-sided"}:
        raise ValueError("tail must be greater, less, or two-sided")
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    exceed = (null >= obs if tail == "greater" else
              null <= obs if tail == "less" else np.abs(null) >= abs(obs))
    return float((1 + exceed.sum()) / (1 + len(null)))


def holm(pvals, family_size=None):
    """Holm step-down adjustment, optionally preserving a planned family size.

    Undefined results remain NaN. Without ``family_size``, only finite tests
    define the family (the historical exploratory behavior). For a predeclared
    family, supply its full size: missing/uncomputed hypotheses conservatively
    occupy the remaining ranks as p=1 instead of reducing multiplicity.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p.ravel()))
    if np.isinf(p).any():
        raise ValueError("p values must be finite or NaN")
    if np.any((p.ravel()[valid] < 0) | (p.ravel()[valid] > 1)):
        raise ValueError("p values must lie in [0, 1]")
    if family_size is None:
        family_size = len(valid)
    if (isinstance(family_size, (bool, np.bool_)) or not isinstance(family_size, (int, np.integer))
            or family_size < len(valid)):
        raise ValueError("family_size must be an integer at least the number of finite p values")
    order = valid[np.argsort(p.ravel()[valid], kind="stable")]
    multipliers = family_size - np.arange(len(order))
    adjusted = np.minimum(1, np.maximum.accumulate(p.ravel()[order] * multipliers))
    out.ravel()[order] = adjusted
    return out
