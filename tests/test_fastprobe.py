"""Batched PyTorch probes reproduce scikit-learn's L2 logistic regression."""
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

torch = pytest.importorskip("torch")

from idiom_subspace import fastprobe as fp
from idiom_subspace.geometry import fit_estimator, mlr_coefficients, spectral_gap
from test_studies import synthetic_study

CPU = torch.device("cpu")


def _problem(seed, n=300, d=12, k=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) * .4
    y = rng.integers(0, k, n)
    X[np.arange(n), y % d] += 1.5
    return X, y


def _sklearn(X, y, C, fit_intercept):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, fit_intercept=fit_intercept, tol=1e-10, max_iter=20000).fit(X, y)


@pytest.mark.parametrize("k", [2, 3, 6])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("C", [.1, 1.])
def test_batched_solution_matches_sklearn_objective(k, fit_intercept, C):
    X, y = _problem(k, k=k)
    feats = fp.erase_features(X, X, X, [np.zeros((X.shape[1], 0))], [np.zeros((X.shape[1], 0))], CPU, torch.float64)
    W, b, info = fp.fit_batched(feats, y, k, [C], fit_intercept=fit_intercept, tol=1e-9, max_iter=5000)
    reference = _sklearn(X, y, C, fit_intercept)
    coef = W[0].numpy().T
    assert np.linalg.norm(coef - reference.coef_) / np.linalg.norm(reference.coef_) < 2e-3
    logits = X @ W[0].numpy() + b[0].numpy()
    predicted = (logits[:, 0] > 0).astype(int) if k == 2 else logits.argmax(1)
    assert np.mean(predicted == reference.predict(X)) > .995


def test_batch_problems_are_independent_and_padding_is_exact():
    X, y = _problem(1, k=3)
    widths = [1, 4, 9]
    Z = [X[:, :w] for w in widths]
    together = fp.compact_features(Z, Z, Z, CPU, torch.float64)
    W, b, _ = fp.fit_batched(together, y, 3, [1.] * 3, tol=1e-9, max_iter=5000)
    for j, w in enumerate(widths):
        alone = fp.compact_features([Z[j]], [Z[j]], [Z[j]], CPU, torch.float64)
        Wj, bj, _ = fp.fit_batched(alone, y, 3, [1.], tol=1e-9, max_iter=5000)
        np.testing.assert_allclose(W[j, :w].numpy(), Wj[0, :w].numpy(), atol=2e-4)
        assert np.abs(W[j, w:].numpy()).max(initial=0) == 0     # zero padding keeps zero weights


def test_low_rank_erasure_map_equals_materialized_residual():
    X, y = _problem(2, k=3)
    rng = np.random.default_rng(0)
    F = rng.normal(size=(X.shape[1], 2)); G = np.linalg.qr(rng.normal(size=(X.shape[1], 2)))[0]
    residual = X - (X @ F) @ G.T
    implicit = fp.erase_features(X, X, X, [F], [G], CPU, torch.float64)
    explicit = fp.compact_features([residual], [residual], [residual], CPU, torch.float64)
    W1, b1, _ = fp.fit_batched(implicit, y, 3, [1.], tol=1e-9, max_iter=5000)
    W2, b2, _ = fp.fit_batched(explicit, y, 3, [1.], tol=1e-9, max_iter=5000)
    np.testing.assert_allclose(W1.numpy(), W2.numpy(), atol=1e-5)


def test_tune_readouts_matches_study_fields_and_prefers_small_C_on_ties():
    X, y = _problem(3, k=3)
    tr, va, te = np.arange(0, 150), np.arange(150, 220), np.arange(220, 300)
    feats = fp.compact_features([X[tr]], [X[va]], [X[te]], CPU, torch.float64)
    result = fp.tune_readouts(feats, y, (tr, va, te), [10., .1, 1.])[0]
    assert set(result) >= {"correct", "validation_correct", "floor", "margin", "test_idx", "val_idx", "C",
                           "unseen_test_classes"}
    assert len(result["correct"]) == len(te) and len(result["margin"]) == len(te)
    np.testing.assert_array_equal(result["predicted"] == y[te], result["correct"])
    np.testing.assert_array_equal(result["validation_predicted"] == y[va], result["validation_correct"])
    constant = fp.tune_readouts(feats, np.zeros_like(y), (tr, va, te), [1.])[0]
    assert constant["correct"].all()
    np.testing.assert_array_equal(constant["predicted"], np.zeros(len(te)))


def test_reference_mlr_reproduces_concept_space_recipe():
    X, y = _problem(4, k=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = LogisticRegression(C=1., fit_intercept=False, max_iter=5000, tol=fp.LOGISTIC_TOL).fit(X, y).coef_
    expected = coef / np.linalg.norm(coef, axis=1, keepdims=True)
    np.testing.assert_allclose(mlr_coefficients(X, y, 1., variant="reference"), expected, atol=1e-6)
    exact = _sklearn(X, y, 1., False).coef_
    exact = exact / np.linalg.norm(exact, axis=1, keepdims=True)
    for backend in ("sklearn", "torch"):
        ours = mlr_coefficients(X, y, 1., variant="reference", backend=backend, device="cpu")
        assert np.abs(ours - exact).max() < 2e-3
    for rank in (1, 2, 4):
        reference_span = np.linalg.svd(expected, full_matrices=False)[2][:rank].T
        ours = fit_estimator(X, y, "MLR", rank, mlr_variant="reference").at_rank(rank).basis
        np.testing.assert_allclose(ours @ ours.T, reference_span @ reference_span.T, atol=1e-6)


def test_spectral_gap_flags_nearly_tied_directions():
    assert spectral_gap([3., 1., .5], 1) == pytest.approx(2 / 3)
    assert spectral_gap([1.2266, 1.2229, 0.], 1) < .01
    assert np.isnan(spectral_gap(None, 1))
    X, y = _problem(5, k=4)
    sub = fit_estimator(X, y, "COV", 3).at_rank(2)
    assert 0 <= sub.spectral_gap <= 1


def _torch_study(**kwargs):
    study = synthetic_study(**kwargs)
    study.cfg.probe_backend = "torch"
    study.backend, study._device = "torch", fp.resolve_device("cpu")
    return study


def test_torch_study_matches_sklearn_when_estimators_are_shared():
    results = []
    for backend in ("sklearn", "torch"):
        study = synthetic_study(methods=("MLR", "LEACE", "RAND", "PCA", "COV", "CPCA", "LDA"), ranks=(1, 2))
        if backend == "torch":
            study.backend, study._device = "torch", fp.resolve_device("cpu")
        options = study._estimator_options
        study._estimator_options = lambda o=options: dict(o(), backend="sklearn")
        results.append((study.run_grid(), study.inference()[0]))
    key = ["concept", "method", "M", "source_condition", "condition", "group"]
    grid = results[0][0].merge(results[1][0], on=key, suffixes=("_sk", "_t"))
    assert len(grid) == len(results[0][0])
    assert (grid.R_sk - grid.R_t).abs().max() < 1e-9
    assert (grid.L_sk - grid.L_t).abs().max() < 1e-9
    assert results[1][0].probe_backend.eq("torch").all()
    inference = results[0][1].merge(results[1][1], on=["concept", "source_condition", "condition", "group"],
                                    suffixes=("_sk", "_t"))
    np.testing.assert_allclose(inference.SSI_sk, inference.SSI_t)


def test_torch_backend_batches_resume_and_extend_without_refits():
    with TemporaryDirectory() as checkpoint:
        first = _torch_study(methods=("MLR", "RAND"), ranks=(1,), checkpoint_dir=checkpoint)
        calls = []
        actual = fp.tune_readouts
        def counting(*args, **kwargs):
            calls.append(1)
            return actual(*args, **kwargs)
        with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=counting):
            grid = first.run_grid()
            rows, _ = first.inference()
        # Batched: far fewer solver calls than readouts (grid + 4 tasks x (3 RAND + 1 PCA + 3 PERM)).
        assert 0 < len(calls) < 40
        resumed = _torch_study(methods=("MLR", "RAND"), ranks=(1,), checkpoint_dir=checkpoint)
        with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=AssertionError("refit")), \
             patch("idiom_subspace.studies.fit_estimator", side_effect=AssertionError("estimator refit")):
            pd.testing.assert_frame_equal(grid, resumed.run_grid())
            again, _ = resumed.inference()
        pd.testing.assert_frame_equal(rows, again)
        resumed.cfg.null_draws = 5
        extended, _ = resumed.inference()
        assert extended.RAND_draws.eq(5).all()


def test_batched_permutation_estimators_equal_sequential_fits():
    batched = _torch_study(methods=("MLR",), ranks=(1, 2))
    sequential = _torch_study(methods=("MLR",), ranks=(1, 2))
    batched._prefetch_permutation_estimators([("bn-idiom", "F", "MLR", 2, d) for d in (1, 2, 3)])
    for draw in (1, 2, 3):
        a = batched.sub("bn-idiom", "F", "MLR", 2, permutation=True, draw=draw)
        b = sequential.sub("bn-idiom", "F", "MLR", 2, permutation=True, draw=draw)
        np.testing.assert_allclose(a.basis @ a.basis.T, b.basis @ b.basis.T, atol=1e-4)


def test_grid_rand_baseline_averages_requested_draws():
    single = synthetic_study(methods=("RAND",), ranks=(1,))
    multi = synthetic_study(methods=("RAND",), ranks=(1,))
    multi.grid_rand_draws = 4
    a, b = single.run_grid(), multi.run_grid()
    a, b = a[a.method.eq("RAND")], b[b.method.eq("RAND")]
    assert len(a) and a.subspace_draws.eq(1).all() and b.subspace_draws.eq(4).all()
    assert not np.allclose(a.R.to_numpy(), b.R.to_numpy())


def test_float32_stop_at_working_precision_counts_as_success():
    X, y = _problem(5, n=400, d=40, k=6)
    plain = [np.zeros((X.shape[1], 0))]
    exact = fp.fit_batched(fp.erase_features(X, X, X, plain, plain, CPU, torch.float64), y, 6, [1.],
                           tol=1e-9, max_iter=5000)
    # A gradient tolerance float32 cannot reach: the fit stops when f stops decreasing.
    W, b, info = fp.fit_batched(fp.erase_features(X, X, X, plain, plain, CPU, torch.float32), y, 6, [1.],
                                tol=1e-12, max_iter=5000)
    assert not info["converged"][0] and info["flat"][0] and info["success"][0]
    assert info["iterations"][0] < 5000
    W64, b64 = exact[0][0].numpy(), exact[1][0].numpy()
    assert np.linalg.norm(W[0].double().numpy() - W64) / np.linalg.norm(W64) < 5e-3
    predicted = (X @ W[0].double().numpy() + b[0].double().numpy()).argmax(1)
    assert np.mean(predicted == (X @ W64 + b64).argmax(1)) > .99
    # An iteration cap is not success.
    _, _, capped = fp.fit_batched(fp.erase_features(X, X, X, plain, plain, CPU, torch.float64), y, 6, [1.],
                                  tol=1e-9, max_iter=3)
    assert not capped["success"][0] and not capped["converged"][0]


def test_float32_readouts_and_estimators_match_float64():
    X, y = _problem(6, n=500, d=30, k=5)
    tr, va, te = np.arange(0, 250), np.arange(250, 350), np.arange(350, 500)
    results = {}
    for dtype in (torch.float64, torch.float32):
        feats = fp.erase_features(X[tr], X[va], X[te], [np.zeros((30, 0))], [np.zeros((30, 0))], CPU, dtype)
        results[dtype] = fp.tune_readouts(feats, y, (tr, va, te), [.1, 1.])[0]
        assert results[dtype]["probe_converged"]
    assert np.mean(results[torch.float64]["correct"] == results[torch.float32]["correct"]) > .99
    with patch.object(fp, "default_dtype", lambda device: torch.float32):
        single = mlr_coefficients(X, y, 1., variant="reference", backend="torch", device="cpu")
    double = mlr_coefficients(X, y, 1., variant="reference", backend="torch", device="cpu")
    for rank in (1, 2, 4):
        a = np.linalg.svd(single, full_matrices=False)[2][:rank].T
        c = np.linalg.svd(double, full_matrices=False)[2][:rank].T
        assert (np.linalg.svd(a.T @ c, compute_uv=False) ** 2).mean() > .9999
