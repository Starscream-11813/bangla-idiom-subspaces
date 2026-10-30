"""Offline checks for geometry invariants and inference edge cases."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from idiom_subspace.geometry import (
    fit_estimator, fit_probe, fit_subspace, fitted_preprocessor, holm, monte_carlo_pvalue,
    null_adjusted_index, paired_cluster_bootstrap, shared_subspace_index,
    tune_probe,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.y = np.repeat(np.arange(3), 80)
        self.X = rng.normal(size=(len(self.y), 8))
        self.X[:, 0] += 4 * self.y
        self.X[:, 1] += 2 * (self.y == 1)

    def test_all_estimators_have_idempotent_complementary_projections(self):
        for method in ("RAND", "PCA", "MLR", "LDA", "CPCA", "COV", "LEACE"):
            with self.subTest(method=method):
                sub = fit_subspace(self.X, self.y, method, rank=5)
                self.assertEqual(sub.requested_rank, 5)
                expected_max = 5 if method in {"RAND", "PCA"} else 2
                self.assertLessEqual(sub.effective_rank, expected_max)
                np.testing.assert_allclose(sub.basis.T @ sub.basis, np.eye(sub.effective_rank), atol=1e-10)
                projected, erased = sub.project(self.X), sub.erase(self.X)
                np.testing.assert_allclose(projected @ sub.basis.T + erased, self.X - sub.center, atol=1e-10)
                np.testing.assert_allclose(erased @ sub.input_factor, 0, atol=1e-10)
                P = sub.projection_operator
                np.testing.assert_allclose(P @ P, P, atol=1e-10)
                np.testing.assert_allclose(sub.formal_erase(self.X), erased, atol=1e-10)

    def test_constant_and_missing_labels_never_pad_supervised_rank(self):
        for method in ("MLR", "LDA", "CPCA", "COV", "LEACE"):
            sub = fit_subspace(self.X, np.zeros(len(self.X)), method, rank=7)
            self.assertEqual(sub.effective_rank, 0)
            self.assertEqual(sub.project(self.X).shape, (len(self.X), 0))
        y = self.y.astype(float)
        y[::3] = np.nan
        y[1::3] = -1
        sub = fit_subspace(self.X, y, "COV", rank=7)
        self.assertTrue(np.isfinite(sub.basis).all())
        self.assertLessEqual(sub.effective_rank, 2)

    def test_pca_drops_null_directions_and_rand_is_seeded(self):
        t = np.arange(10.)
        X = np.column_stack([t, 2 * t, np.ones(10)])
        self.assertEqual(fit_subspace(X, None, "PCA", 3).effective_rank, 1)
        a = fit_subspace(self.X, self.y, "RAND", 3, seed=9)
        b = fit_subspace(self.X, self.y, "RAND", 3, seed=9)
        np.testing.assert_array_equal(a.basis, b.basis)

    def test_lda_handles_zero_within_class_variance(self):
        X = np.repeat([[0., 0.], [1., 0.], [0., 1.]], 3, axis=0)
        sub = fit_subspace(X, np.repeat([0, 1, 2], 3), "LDA", rank=2)
        self.assertEqual(sub.effective_rank, 2)

    def test_oblique_leace_removes_training_label_covariance_without_ridge(self):
        sub = fit_subspace(self.X, self.y, "LEACE", 2, ridge=0)
        Y = np.eye(3)[self.y]
        residual_cov = sub.formal_erase(self.X).T @ (Y - Y.mean(0)) / len(Y)
        np.testing.assert_allclose(residual_cov, 0, atol=1e-10)
        np.testing.assert_allclose(sub.erase_operator @ sub.erase_operator, sub.erase_operator, atol=1e-10)
        # Orthogonal image diagnostics are separate from the primary LEACE map.
        self.assertGreater(np.linalg.norm(sub.erase(self.X) - sub.orthogonal_erase(self.X)), .1)
        np.testing.assert_array_equal(sub.erase(self.X), sub.formal_erase(self.X))

    @staticmethod
    def reference_leace_operator(X, y, rank):
        """Reference column-factor formula, converted to row-vector convention."""
        Xc = X - X.mean(0)
        _, encoded = np.unique(y, return_inverse=True)
        Y = np.eye(encoded.max() + 1)[encoded]
        covariance = Xc.T @ Xc / len(X)
        eigen, V = np.linalg.eigh(covariance)
        support = eigen > eigen[-1] * len(eigen) * np.finfo(float).eps
        W = (V[:, support] / np.sqrt(eigen[support])) @ V[:, support].T
        Winv = (V[:, support] * np.sqrt(eigen[support])) @ V[:, support].T
        U = np.linalg.svd(W @ (Xc.T @ (Y - Y.mean(0)) / len(Y)), full_matrices=False)[0][:, :rank]
        return W @ U @ U.T @ Winv

    def test_leace_compact_coordinates_match_reference_oblique_metric(self):
        for rank in (1, 2):
            with self.subTest(rank=rank):
                sub = fit_estimator(self.X, self.y, "LEACE", max_rank=2).at_rank(rank)
                ref = self.reference_leace_operator(self.X, self.y, rank)
                np.testing.assert_allclose(sub.projection_operator, ref, atol=1e-10)
                compact = sub.project(self.X[:30])
                ambient = (self.X[:30] - sub.center) @ ref
                np.testing.assert_allclose(compact @ sub.basis.T, ambient, atol=1e-10)
                np.testing.assert_allclose(compact @ compact.T, ambient @ ambient.T, atol=1e-10)
                a = LogisticRegression(C=.7, max_iter=10000, tol=1e-10).fit(sub.project(self.X), self.y)
                b = LogisticRegression(C=.7, max_iter=10000, tol=1e-10).fit((self.X-sub.center) @ ref, self.y)
                np.testing.assert_allclose(a.predict_proba(compact), b.predict_proba(ambient), atol=2e-7)

    def test_leace_binary_counterexample_does_not_collapse_to_cov_retention(self):
        rng = np.random.default_rng(20261019)
        def examples(n):
            y = np.tile([0, 1], n // 2)
            z = np.repeat(rng.normal(size=n // 2), 2)
            X = np.column_stack([2*y-1+10*z, 10*z]) + rng.normal(0, .05, (n, 2))
            return X, y
        space, ys = examples(1200)
        probe, yp = examples(800)
        test, yt = examples(800)
        sub = fit_subspace(space, ys, "LEACE", 1)
        cov = fit_subspace(space, ys, "COV", 1)
        # Their IMAGE spans coincide, but the oblique input map contains
        # information that an orthogonal projection onto that image loses.
        np.testing.assert_allclose(sub.basis @ sub.basis.T, cov.basis @ cov.basis.T, atol=1e-10)
        primary = fit_probe(sub.project(probe), yp)
        orthogonal = fit_probe(sub.orthogonal_project(probe), yp)
        self.assertGreater((primary.predict(sub.project(test)) == yt).mean(), .98)
        self.assertLess((orthogonal.predict(sub.orthogonal_project(test)) == yt).mean(), .65)

    def test_leace_moore_penrose_whitening_handles_singular_features(self):
        X = np.column_stack([self.X, self.X[:, 0] * 2, np.ones(len(self.X))])
        sub = fit_subspace(X, self.y, "LEACE", 2)
        reference = self.reference_leace_operator(X, self.y, 2)
        np.testing.assert_allclose(sub.projection_operator, reference, atol=1e-9)
        Y = np.eye(3)[self.y]
        np.testing.assert_allclose(sub.erase(X).T @ (Y-Y.mean(0)), 0, atol=1e-8)

    def test_cpca_matches_equal_centroid_pca_with_imbalanced_classes(self):
        rng = np.random.default_rng(19)
        y = np.repeat([0, 1, 2], [10000, 1000, 10])
        X = np.array([[0., 0.], [1., 0.], [0., 5.]])[y] + rng.normal(0, .001, (len(y), 2))
        means = np.stack([X[y == k].mean(0) for k in range(3)])
        expected = PCA(n_components=1).fit(means).components_.T
        actual = fit_subspace(X, y, "CPCA", 1).basis
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-10)

    def test_fisher_prefixes_match_sklearn_without_reordering(self):
        fitted = fit_estimator(self.X, self.y, "LDA", max_rank=2)
        reference = LinearDiscriminantAnalysis().fit(self.X, self.y).scalings_
        for rank in (1, 2):
            expected = np.linalg.qr(reference[:, :rank], mode="reduced")[0]
            actual = fitted.at_rank(rank).basis
            np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-9)

    def test_rank_sweep_fits_mlr_once_and_reuses_each_spectral_fit(self):
        calls = []
        original = LogisticRegression.fit
        def recording_fit(model, *args, **kwargs):
            calls.append(1)
            return original(model, *args, **kwargs)
        with patch.object(LogisticRegression, "fit", new=recording_fit):
            fitted = fit_estimator(self.X, self.y, "MLR", max_rank=7)
            for rank in (0, 1, 2, 4, 7):
                fitted.at_rank(rank)
        self.assertEqual(len(calls), 1)
        from idiom_subspace import geometry
        for method in ("CPCA", "COV", "LDA", "LEACE"):
            with self.subTest(method=method), patch.object(geometry, "_svd_basis", wraps=geometry._svd_basis) as svd:
                fitted = fit_estimator(self.X, self.y, method, max_rank=7)
                for rank in (0, 1, 2, 4, 7):
                    fitted.at_rank(rank)
                self.assertEqual(svd.call_count, 1)

    def test_primary_complement_coordinates_are_isometric(self):
        for method in ("MLR", "LEACE"):
            with self.subTest(method=method):
                sub = fit_subspace(self.X, self.y, method, 2)
                compact = sub.erase_coordinates(self.X[:20])
                ambient = sub.erase(self.X[:20])
                np.testing.assert_allclose(compact @ sub.complement_basis.T, ambient, atol=1e-10)
                np.testing.assert_allclose(compact @ compact.T, ambient @ ambient.T, atol=1e-10)

    def test_preprocessor_is_fit_once_on_space_only(self):
        a = fitted_preprocessor(self.X[:100], pca_dim=4, seed=0)
        b = fitted_preprocessor(self.X[:100], pca_dim=4, seed=99)
        np.testing.assert_allclose(a.transform(self.X), b.transform(self.X), atol=1e-12)
        np.testing.assert_allclose(a.transform(self.X[:100]).mean(0), 0, atol=1e-12)
        self.assertAlmostEqual(np.mean(a.transform(self.X[:100]) ** 2), 1)
        center = a.mean_.copy()
        a.transform(self.X + 1000)
        np.testing.assert_array_equal(a.mean_, center)

    def test_degenerate_preprocessing_and_probes(self):
        p = fitted_preprocessor(np.ones((5, 3)), pca_dim=3)
        self.assertEqual(p.transform(np.ones((2, 3))).shape, (2, 0))
        constant = fit_probe(self.X, np.zeros(len(self.X)))
        np.testing.assert_array_equal(constant.predict(self.X[:2]), [0, 0])
        nofeatures = fit_probe(np.empty((len(self.y), 0)), self.y)
        self.assertEqual(nofeatures.predict_proba(np.empty((2, 0))).shape, (2, 3))
        incomplete = fit_probe(np.array([[0.], [1.], [2.]]), [0, None, 1])
        np.testing.assert_array_equal(incomplete.classes_, [0, 1])

    def test_tuning_does_not_refit_on_validation(self):
        X = np.arange(6.).reshape(-1, 1)
        model, c = tune_probe(X[:3], [0, 0, 0], X[3:], [1, 1, 1], Cs=[10, .1, 1])
        self.assertEqual(c, .1)  # deterministic tie rule
        np.testing.assert_array_equal(model.classes_, [0])


class StatisticsTests(unittest.TestCase):
    def test_indices_and_undefined_denominators(self):
        self.assertAlmostEqual(shared_subspace_index(.4, .5, .1), .75)
        self.assertAlmostEqual(null_adjusted_index(.4, .5, .3), .5)
        self.assertTrue(np.isnan(shared_subspace_index(.4, .11, .1)))
        self.assertTrue(np.isnan(shared_subspace_index(np.nan, .5, .1)))
        self.assertGreater(shared_subspace_index(.7, .5, .1), 1)

    def test_monte_carlo_nan_guard_and_positive_floor(self):
        self.assertAlmostEqual(monte_carlo_pvalue(10, [1, 2, np.nan, 3]), .25)
        self.assertTrue(np.isnan(monte_carlo_pvalue(np.nan, [1, 2])))
        self.assertTrue(np.isnan(monte_carlo_pvalue(1, [np.nan])))
        self.assertAlmostEqual(monte_carlo_pvalue(-3, [-2, 0, 2], tail="two-sided"), .25)

    def test_holm_matches_known_stepdown_example(self):
        np.testing.assert_allclose(holm([.01, .04, .03, np.nan]), [.03, .06, .06, np.nan], equal_nan=True)

    def test_holm_preserves_planned_family_when_results_are_missing(self):
        observed = np.array([.001, .01, np.nan])
        planned = holm(observed, family_size=72)
        np.testing.assert_allclose(planned, [.072, .71, np.nan], equal_nan=True)
        # Equivalent to retaining each unavailable hypothesis as a nonrejecting
        # p=1, while preserving the original missing-value reporting.
        padded = holm(np.concatenate([observed, np.ones(70)]))
        np.testing.assert_allclose(planned[:2], padded[:2])
        self.assertTrue(np.isnan(planned[2]))
        np.testing.assert_array_equal(holm([np.nan], family_size=72), [np.nan])
        for invalid in (1, -1, 1.5, True):
            with self.subTest(family_size=invalid), self.assertRaises(ValueError):
                holm(observed, family_size=invalid)
        with self.assertRaises(ValueError):
            holm([.1, np.inf], family_size=72)

    def test_cluster_bootstrap_keeps_entire_pairs(self):
        # Each group has identical paired outcomes; a resample must retain both
        # rows of every selected group, even when a group is drawn twice.
        groups = np.repeat(np.arange(4), 2)
        values = {"x": np.repeat([0., 1., 2., 3.], 2), "y": np.repeat([10., 11., 12., 13.], 2)}
        def stat(v):
            self.assertEqual(len(v["x"]), 8)
            np.testing.assert_array_equal(v["y"] - v["x"], np.full(8, 10.))
            np.testing.assert_array_equal(v["x"][::2], v["x"][1::2])
            return v["x"].mean()
        result = paired_cluster_bootstrap(values, groups, stat, n_boot=100, seed=2)
        self.assertEqual(result["n_valid"], 100)
        self.assertEqual(result["estimate"], 1.5)
        self.assertLess(result["ci_lo"], result["estimate"])
        self.assertGreater(result["ci_hi"], result["estimate"])

    def test_undefined_point_estimate_has_no_interval(self):
        values = np.array([0., 1.])
        result = paired_cluster_bootstrap(values, [0, 1], lambda a: np.nan if a.mean() == .5 else a.mean(), n_boot=100)
        self.assertTrue(np.isnan(result["estimate"]))
        self.assertTrue(np.isnan(result["ci_lo"]))
        self.assertTrue(np.isnan(result["ci_hi"]))
        self.assertGreater(result["n_valid"], 0)


if __name__ == "__main__":
    unittest.main()
