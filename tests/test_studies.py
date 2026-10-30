"""Offline integration checks for matched-condition studies and null inference."""

from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from idiom_subspace.studies import LayerStudy, moderator_analysis


def synthetic_study(*, methods=("MLR", "LEACE", "RAND", "PCA"), ranks=(1, 2), checkpoint_dir=None, progress=None):
    """Distinct split IDs, shared condition records, and known semantic signal."""
    rng = np.random.default_rng(42)
    rows, features = [], []
    for split in ("space", "probe", "validation", "test"):
        for j in range(18):
            label = j % 3
            # Match the same source record across conditions, but use noisy
            # independently represented realizations in the two conditions.
            for condition, lang in (("bn-idiom", "bn"), ("en-idiom", "en")):
                rows.append(dict(condition=condition, lang=lang, split=split,
                                 F=label, S=j % 2, L=(label + 1) % 3,
                                 idiom_id=f"{split}_{j}", item_id=f"{split}_{j}_{condition}",
                                 idiom_group=f"type_{split}_{j}", source_record_id="",
                                 component_id=f"component_{split}_{j // 2}",
                                 anchored=j % 2 == 0, rare=j % 3 == 0,
                                 cultural=j % 4 == 0, religious=False,
                                 historical=False, rural=j % 5 == 0))
                x = rng.normal(0, .08, size=6)
                x[:3] += np.eye(3)[label] * 3
                x[3] += (j % 2) * 2
                x[4] += .5 * (lang == "en")
                features.append(x)
    cfg = SimpleNamespace(
        pairs=[("en-idiom", "bn-idiom"), ("bn-idiom", "en-idiom")],
        concepts=["F", "S"], methods=list(methods), ranks=list(ranks),
        probe_Cs=[.1, 1.], min_group=3, inference_method="MLR", null_draws=3,
        bootstrap=25, min_den=.02, open_class_reps=1, open_class_N=[6, 12], split_mode="strict",
        compute_leakage=True, inference_groups=["all"], primary_test_groups=["all"],
        primary_test_pairs=[("en-idiom", "bn-idiom"), ("bn-idiom", "en-idiom")],
        primary_test_concepts=["F", "S"], permutation_unit="component",
        min_permutation_movable_fraction=.5, diagnostic_null_draws=2,
    )
    return LayerStudy(np.asarray(features), pd.DataFrame(rows), cfg,
                      dict(seed=7, label_seed=0, model="synthetic", layer=0),
                      checkpoint_dir=checkpoint_dir, progress=progress)


class StudyIntegrationTests(unittest.TestCase):
    def test_condition_split_indices_and_test_pairing(self):
        study = synthetic_study(methods=("MLR",), ranks=(1, 2))
        for condition in ("bn-idiom", "en-idiom"):
            indices = {split: study.idx(condition, split, "F")
                       for split in ("space", "probe", "validation", "test")}
            for split, ix in indices.items():
                frame = study.items.iloc[ix]
                self.assertTrue(frame.condition.eq(condition).all())
                self.assertTrue(frame.split.eq(split).all())
                self.assertEqual(len(ix), 18)
            self.assertEqual(len(set(np.concatenate(list(indices.values())))), 72)
        grid = study.run_grid()
        self.assertFalse(grid.empty)
        cross = study.evidence[("F", "MLR", 2, "en-idiom", "bn-idiom")]
        own = study.evidence[("F", "MLR", 2, "bn-idiom", "bn-idiom")]
        np.testing.assert_array_equal(cross["test_idx"], own["test_idx"])
        self.assertEqual(len(cross["correct"]), 18)

    def test_binary_mlr_does_not_pad_second_direction(self):
        study = synthetic_study()
        for method in ("MLR", "LEACE", "LDA", "COV", "CPCA"):
            self.assertEqual(study.sub("bn-idiom", "S", method, 1).effective_rank, 1)
            self.assertIsNone(study.sub("bn-idiom", "S", method, 2))
        self.assertEqual(study.sub("bn-idiom", "S", "RAND", 2).effective_rank, 2)

    def test_subspace_and_tuning_never_depend_on_test_labels(self):
        original = synthetic_study(methods=("MLR",), ranks=(1,))
        changed = synthetic_study(methods=("MLR",), ranks=(1,))
        changed.items.loc[changed.items.split.eq("test"), "F"] = 9
        a = original.sub("en-idiom", "F", "MLR", 1)
        b = changed.sub("en-idiom", "F", "MLR", 1)
        np.testing.assert_allclose(a.basis @ a.basis.T, b.basis @ b.basis.T, atol=1e-12)
        ea = original.evaluate(a.project(original.X), "bn-idiom", "F")
        eb = changed.evaluate(b.project(changed.X), "bn-idiom", "F")
        self.assertEqual(ea["C"], eb["C"])
        np.testing.assert_array_equal(ea["validation_correct"], eb["validation_correct"])
        self.assertEqual(eb["unseen_test_classes"], 18)

    def test_label_permutation_moves_whole_counterpart_component_bundles(self):
        study = synthetic_study(methods=("COV",), ranks=(1,))
        ix = study.idx("bn-idiom", "space", "F")
        study.items.loc[ix, "idiom_group"] = [f"different_type_{j}" for j in range(18)]
        original = study.items.F.to_numpy().copy()
        captured = []
        from idiom_subspace.geometry import fit_estimator as actual_fit
        def record(X, y, *args, **kwargs):
            captured.append(np.asarray(y).copy())
            return actual_fit(X, y, *args, **kwargs)
        with patch("idiom_subspace.studies.fit_estimator", side_effect=record):
            study.sub("bn-idiom", "F", "COV", 1, permutation=True, draw=2)
        before = sorted(map(tuple, original[ix].reshape(-1, 2)))
        after = sorted(map(tuple, captured[0].reshape(-1, 2)))
        self.assertEqual(before, after)
        np.testing.assert_array_equal(study.items.F, original)

    def test_leace_grid_fits_one_exact_residual_and_keeps_compatibility_alias(self):
        study = synthetic_study(methods=("LEACE",), ranks=(1,))
        grid = study.run_grid()
        leace = grid[grid.method.eq("LEACE")]
        self.assertGreater(len(leace), 0)
        self.assertTrue(np.isfinite(leace[["L", "L_formal"]]).all().all())
        np.testing.assert_array_equal(leace.L, leace.L_formal)
        self.assertTrue(leace.leakage_convention.eq("exact_oblique").all())
        self.assertTrue(leace.effective_rank.eq(1).all())
        sub = study.sub("bn-idiom", "F", "LEACE", 1)
        self.assertEqual(sub.project(study.X).shape, (len(study.items), 1))
        self.assertEqual(sub.formal_erase(study.X).shape, study.X.shape)

    def test_ranks_share_one_estimator_and_repeated_grid_uses_cached_readouts(self):
        study = synthetic_study(methods=("MLR",), ranks=(1, 2))
        from idiom_subspace.geometry import fit_estimator as actual_fit, tune_probe as actual_tune
        with patch("idiom_subspace.studies.fit_estimator", wraps=actual_fit) as estimators:
            a = study.sub("bn-idiom", "F", "MLR", 1)
            b = study.sub("bn-idiom", "F", "MLR", 2)
            self.assertEqual(estimators.call_count, 1)
            np.testing.assert_allclose(a.basis, b.basis[:, :1])
        with patch("idiom_subspace.studies.tune_probe", wraps=actual_tune) as probes:
            first = study.run_grid()
            initial_calls = probes.call_count
            second = study.run_grid()
            self.assertEqual(probes.call_count, initial_calls)
        pd.testing.assert_frame_equal(first, second)

    def test_leakage_can_be_disabled_and_leace_has_no_duplicate_formal_fit(self):
        from idiom_subspace.geometry import tune_probe as actual_tune
        for enabled, expected in [(True, 10), (False, 6)]:
            study = synthetic_study(methods=("LEACE",), ranks=(1,))
            study.cfg.concepts = ["F"]
            study.cfg.compute_leakage = enabled
            with patch("idiom_subspace.studies.tune_probe", wraps=actual_tune) as probes:
                grid = study.run_grid()
            self.assertEqual(probes.call_count, expected)
            estimates = grid[grid.method.eq("LEACE")]
            self.assertEqual(estimates.L.notna().all(), enabled)

    def test_completed_grid_and_null_draws_resume_without_fits_and_can_extend(self):
        with TemporaryDirectory() as checkpoint:
            first = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            first_grid = first.run_grid()
            first_rows, first_predictions = first.inference()
            resumed = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            with patch("idiom_subspace.studies.fit_estimator", side_effect=AssertionError("unexpected estimator refit")), \
                 patch("idiom_subspace.studies.tune_probe", side_effect=AssertionError("unexpected probe refit")):
                resumed_grid = resumed.run_grid()
                resumed_rows, resumed_predictions = resumed.inference()
            pd.testing.assert_frame_equal(first_grid, resumed_grid)
            pd.testing.assert_frame_equal(first_rows, resumed_rows)
            pd.testing.assert_frame_equal(first_predictions, resumed_predictions)
            # Changing the null budget must not invalidate already fitted draws.
            from idiom_subspace.geometry import tune_probe as actual_tune
            resumed.cfg.null_draws = 5
            with patch("idiom_subspace.studies.tune_probe", wraps=actual_tune) as probes:
                extended, _ = resumed.inference()
            self.assertEqual(probes.call_count, 16)  # 4 concept/directions × 2 added draws × RAND/PERM
            self.assertTrue(extended.RAND_draws.eq(5).all())

    def test_interrupted_grid_reuses_completed_and_partial_task_evidence(self):
        from idiom_subspace.geometry import tune_probe as actual_tune
        with TemporaryDirectory() as checkpoint:
            interrupted = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            attempts = []
            def fail_fourth(*args, **kwargs):
                attempts.append(1)
                if len(attempts) == 4: raise RuntimeError("simulated interruption")
                return actual_tune(*args, **kwargs)
            with patch("idiom_subspace.studies.tune_probe", side_effect=fail_fourth):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    interrupted.run_grid()
            resumed = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            with patch("idiom_subspace.studies.tune_probe", wraps=actual_tune) as probes:
                completed = resumed.run_grid()
            self.assertEqual(probes.call_count, 17)  # Three completed readouts survive the interrupted fourth.
            fresh = synthetic_study(methods=("MLR",), ranks=(1,)).run_grid()
            pd.testing.assert_frame_equal(completed, fresh)

    def test_resume_false_bypasses_grid_and_readout_checkpoint_reads(self):
        from idiom_subspace.geometry import tune_probe as actual_tune
        with TemporaryDirectory() as checkpoint:
            first = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            first_grid = first.run_grid()
            first.inference()
            forced = synthetic_study(methods=("MLR",), ranks=(1,), checkpoint_dir=checkpoint)
            forced.cfg.resume = False
            with patch("idiom_subspace.studies.tune_probe", wraps=actual_tune) as probes:
                forced_grid = forced.run_grid()
                grid_fits = probes.call_count
                forced.inference()
                null_fits = probes.call_count - grid_fits
            self.assertEqual(grid_fits, 20)
            self.assertEqual(null_fits, 28)  # Four tasks × (3 RAND + 1 PCA + 3 PERM).
            pd.testing.assert_frame_equal(first_grid, forced_grid)

    def test_low_movable_component_mass_blocks_permutation_significance_and_fits(self):
        study = synthetic_study(methods=("MLR",), ranks=(1,))
        for condition in study.items.condition.unique():
            ix = study.idx(condition, "space", "F")
            study.items.loc[ix, "component_id"] = ["giant"] * 16 + ["small_a", "small_b"]
        study.run_grid()
        rows, _ = study.inference()
        self.assertTrue(rows.PERM_draws.eq(0).all())
        self.assertTrue(rows.PERM_p.isna().all())
        self.assertTrue(rows.PERM_status.eq("insufficient_movable_component_mass").all())
        np.testing.assert_allclose(rows.permutation_movable_fraction, 2 / 18)
        self.assertTrue(rows.permutation_frozen_units.eq(1).all())

    def test_null_p_values_are_restricted_to_prespecified_primary_family(self):
        study = synthetic_study(methods=("MLR",), ranks=(1,))
        study.cfg.primary_test_pairs = [("en-idiom", "bn-idiom")]
        study.cfg.primary_test_concepts = ["F"]
        study.run_grid()
        rows, _ = study.inference()
        diagnostic = rows[~rows.primary_test]
        self.assertTrue(diagnostic.RAND_p.isna().all())
        self.assertTrue(diagnostic.PERM_p.isna().all())
        self.assertTrue(diagnostic.RAND_draws.eq(2).all())
        self.assertTrue(rows[rows.primary_test].RAND_draws.eq(3).all())

    def test_planned_primary_family_is_retained_when_ssi_is_undefined(self):
        study = synthetic_study(methods=("MLR",), ranks=(1,))
        study.cfg.null_draws = 3
        study.run_grid()
        # One observed primary statistic is deliberately made undefined; the
        # declared four concept/direction hypotheses still define multiplicity.
        own = study.evidence[("F", "MLR", 1, "bn-idiom", "bn-idiom")]
        own["correct"] = own["floor"].copy()
        rows, _ = study.inference()
        self.assertTrue(rows.RAND_formal_family_size_this_refit.eq(4).all())
        self.assertTrue(rows.RAND_observed_formal_tests_this_refit.lt(4).all())
        self.assertTrue(rows.RAND_planned_minimum_p_primary.eq(.25).all())
        self.assertTrue(rows.RAND_best_possible_holm_p_this_refit.eq(1.).all())

    def test_inference_has_matched_arrays_and_finite_null_resolution(self):
        study = synthetic_study(methods=("MLR",), ranks=(1, 2))
        study.run_grid()
        rows, predictions = study.inference()
        self.assertFalse(rows.empty)
        overall = rows[rows.group.eq("all")]
        self.assertTrue(overall.n.eq(18).all())
        self.assertTrue(overall.n_components.eq(9).all())
        for concept, rank in (("F", 2), ("S", 1)):
            self.assertTrue(overall.loc[overall.concept.eq(concept), "M"].eq(rank).all())
        self.assertTrue(overall.PCA_p.isna().all())
        for kind in ("RAND", "PERM"):
            valid = overall[overall[kind + "_p"].notna()]
            self.assertTrue((valid[kind + "_p"] >= 1 / (valid[kind + "_draws"] + 1)).all())
            self.assertTrue((valid[kind + "_p"] <= 1).all())
        expected = ["cross_correct", "within_correct", "floor_correct", "cross_margin", "within_margin"]
        self.assertFalse(predictions[expected].isna().any().any())
        for _, frame in predictions.groupby(["concept", "source_condition", "condition"]):
            self.assertEqual(len(frame), 18)
            self.assertTrue(frame.idiom_id.is_unique)
            self.assertTrue(frame.component_id.str.startswith("component_test_").all())

    def test_retrieval_pairs_on_record_identity_without_optional_source_ids(self):
        study = synthetic_study(methods=("MLR",), ranks=(1,))
        rows = study.baselines()
        self.assertFalse(rows.empty)
        self.assertTrue(rows.n.eq(18).all())
        self.assertTrue(np.isfinite(rows.CKA).all())
        self.assertTrue(rows.space.eq("original_hidden").all())
        study.cfg.pca_dim = 4
        self.assertTrue(study.baselines().space.eq("PCA_4").all())
        study.meta["ambient_space"] = "explicit_space_label"
        self.assertTrue(study.baselines().space.eq("explicit_space_label").all())


class ModeratorDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def observations():
        rng = np.random.default_rng(3)
        anchored = rng.integers(0, 2, 120)
        correct = rng.binomial(1, .3 + .3 * anchored)
        return pd.DataFrame(dict(model="tiny", anchored=anchored, cross_correct=correct,
                                 component_id=np.repeat(np.arange(60), 2)))

    def test_nonseparated_association_has_finite_estimates(self):
        with TemporaryDirectory() as output:
            result = moderator_analysis(self.observations(), output)
        self.assertTrue(result.status.eq("ok").all())
        self.assertTrue(np.isfinite(result[["beta_standardized", "se", "p"]]).all().all())

    def test_perfect_separation_does_not_export_an_association(self):
        frame = self.observations()
        frame["cross_correct"] = frame.anchored
        with TemporaryDirectory() as output:
            result = moderator_analysis(frame, output)
        self.assertTrue(result.status.ne("ok").all())
        self.assertNotIn("beta_standardized", result)
        diagnostic = " ".join(result.reason.astype(str)) + " " + " ".join(result.get("warnings", pd.Series(dtype=str)).astype(str))
        self.assertIn("separation", diagnostic.lower())

    def test_rank_deficient_design_is_rejected_before_glm_fit(self):
        frame = self.observations()
        frame["rare"] = frame.anchored
        with TemporaryDirectory() as output, patch("statsmodels.api.GLM", side_effect=AssertionError("unidentifiable fit")):
            result = moderator_analysis(frame, output)
        self.assertTrue(result.status.eq("rank_deficient_design").all())
        self.assertNotIn("beta_standardized", result)

    def test_nonconverged_nonfinite_and_convergence_warning_results_are_rejected(self):
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        for mode in ("nonconverged", "nonfinite", "warning"):
            with self.subTest(mode=mode):
                fitted = SimpleNamespace(converged=mode != "nonconverged",
                    params=pd.Series([0., np.nan if mode == "nonfinite" else .2], index=["const", "anchored"]),
                    bse=pd.Series([.1, .1]), pvalues=pd.Series([1., .05]), tvalues=pd.Series([0., 2.]))
                def fake_fit(**kwargs):
                    if mode == "warning": warnings.warn("optimizer warning", ConvergenceWarning)
                    return fitted
                with TemporaryDirectory() as output, patch("statsmodels.api.GLM", return_value=SimpleNamespace(fit=fake_fit)):
                    result = moderator_analysis(self.observations(), output)
                self.assertTrue(result.status.ne("ok").all())
                self.assertNotIn("beta_standardized", result)
                self.assertTrue(result.reason.str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()
