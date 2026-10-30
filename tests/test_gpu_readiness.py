"""Structural prefetch regressions; no optimizer fits or existing smoke reruns."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from idiom_subspace.studies import LayerStudy


CORE_EVIDENCE_KEYS = {"correct", "validation_correct", "floor", "margin", "test_idx", "val_idx",
                      "C", "unseen_test_classes"}


def _study(checkpoint_dir=None, resume=True):
    rows = [dict(condition="bn-fig", lang="bn", split=split, F=i % 2,
                 idiom_id=f"{split}_{i}", component_id=f"{split}_{i}")
            for split in ["space", "probe", "validation", "test"] for i in range(4)]
    cfg = SimpleNamespace(concepts=["F"], methods=["RAND", "MLR"], ranks=[1, 2], probe_Cs=[1.],
                          probe_backend="torch", probe_device="cpu", probe_max_iter=3,
                          mlr_variant="centered", grid_rand_draws=1, resume=resume, estimator_batch=2,
                          probe_batch_compact=8, probe_batch_ambient=8, permutation_unit="component")
    features = np.random.default_rng(2).normal(size=(len(rows), 3))
    return LayerStudy(features, pd.DataFrame(rows), cfg, dict(seed=0, model="structural", layer=1),
                      checkpoint_dir=checkpoint_dir)


def _spec(method="RAND", draw=1, rank=1, permutation=False):
    return ("bn-fig", "F", "bn-fig", "F", method, rank, "project", permutation, draw, None, False)


def _consume(study, spec):
    return study._evaluate_sub(spec[0], spec[1], spec[2], spec[3], spec[4], spec[5],
                               representation=spec[6], permutation=spec[7], draw=spec[8],
                               subset=spec[9], need_margin=spec[10])


def _solver_stub(features, y, indices, Cs, **kwargs):
    _, validation, test = indices
    return [dict(correct=np.zeros(len(test), dtype=bool), validation_correct=np.zeros(len(validation), dtype=bool),
                 floor=np.zeros(len(test), dtype=bool), margin=np.empty(0), test_idx=test, val_idx=validation,
                 C=1., unseen_test_classes=0) for _ in range(features.train.shape[0])]


@pytest.mark.parametrize("mode", ["no_checkpoint", "no_resume"])
def test_nonresumable_prefetch_falls_back_without_duplicate_fits(tmp_path, mode):
    spec = _spec()
    if mode == "no_resume":
        # A forced recomputation must ignore an older successful checkpoint.
        old = _study(tmp_path, resume=True)
        with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=_solver_stub):
            old._prefetch([spec])
    study = _study(None if mode == "no_checkpoint" else tmp_path, resume=mode != "no_resume")
    with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=_solver_stub) as solver:
        study._prefetch([spec])
        assert solver.call_count == 0
        result = _consume(study, spec)
        assert solver.call_count == 1
    assert set(result) == CORE_EVIDENCE_KEYS


def test_resumable_prefetch_preserves_evidence_keys_and_warm_resume(tmp_path):
    spec = _spec()
    study = _study(tmp_path)
    with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=_solver_stub) as solver:
        study._prefetch([spec, spec])  # Duplicate metadata must not duplicate a fit.
        prefetched = _consume(study, spec)
        assert solver.call_count == 1
    assert set(prefetched) == CORE_EVIDENCE_KEYS
    resumed = _study(tmp_path)
    with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=AssertionError("unexpected readout refit")), \
         patch("idiom_subspace.studies.fit_estimator", side_effect=AssertionError("unexpected estimator refit")):
        resumed._prefetch([spec])
        restored = _consume(resumed, spec)
    assert set(restored) == CORE_EVIDENCE_KEYS
    for key in CORE_EVIDENCE_KEYS:
        np.testing.assert_array_equal(restored[key], prefetched[key])


def test_permutation_states_are_materialized_per_chunk_and_skips_are_cleaned(tmp_path):
    study = _study(tmp_path)
    specs = [_spec("MLR", draw, permutation=True) for draw in range(1, 8)]
    # Binary labels cannot support rank two. Its prefitted estimator must also
    # be released, and its unavailable evidence must be resumable.
    skipped = _spec("MLR", 8, rank=2, permutation=True)
    specs.append(skipped)
    observed = []

    def coefficient_stub(features, labels, *args, **kwargs):
        return [np.array([[1., 0., 0.]]) for _ in labels]

    def bounded_solver(*args, **kwargs):
        observed.append((len(study._estimators), len(study.subspaces), args[0].train.shape[0]))
        assert len(study._estimators) <= study.cfg.estimator_batch
        assert len(study.subspaces) <= study.cfg.estimator_batch
        return _solver_stub(*args, **kwargs)

    with patch("idiom_subspace.studies.mlr_coefficients", side_effect=coefficient_stub), \
         patch("idiom_subspace.fastprobe.tune_readouts", side_effect=bounded_solver):
        study._prefetch(specs)
    assert [batch for _, _, batch in observed] == [2, 2, 2, 1]
    assert not study._estimators
    assert not study.subspaces
    with patch("idiom_subspace.fastprobe.tune_readouts", side_effect=AssertionError("unexpected refit")), \
         patch("idiom_subspace.studies.fit_estimator", side_effect=AssertionError("unexpected estimator refit")):
        for spec in specs[:-1]:
            assert CORE_EVIDENCE_KEYS <= set(_consume(study, spec))
        assert _consume(study, skipped) is None
