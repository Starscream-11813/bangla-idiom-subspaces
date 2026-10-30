"""Orthogonal maps between paired conditions, fitted on a chosen set of pairs (X10; reused by X9).

X7 fitted its Procrustes maps on paired SPACE records, the records whose labels
also define both languages' class means and hence the subspaces being rotated.
Here the same maps (X7's ``procrustes`` and ``reduced_procrustes``, on rows
centred by the condition's unlabelled SPACE mean and normalized) are fitted on
any union of splits. PROBE and VALIDATION pairs are disjoint from SPACE (no
class mean, estimator or subspace ever sees them) and from TEST, and no label
enters a map. ``shuffle`` pairs each source row with a random target row of the
same splits: a control map with the right marginals but no correspondence.
"""
from __future__ import annotations

import numpy as np

from idiom_subspace.geometry import Subspace

from experiments.x7_alignment import (REDUCED_K, condition_mean, paired_rows, procrustes,  # noqa: F401
                                      reduced_procrustes, unit)

FIT_SETS = {"space": ("space",), "probe": ("probe",), "validation": ("validation",),
            "probe_validation": ("probe", "validation")}
PRIMARY_FIT = "probe_validation"
REDUCED = f"procrustes_pca{REDUCED_K}"
MAPS = ("procrustes", REDUCED)


def pair_rows(study, source, target, splits):
    """Row indices of one-to-one (source, target) pairs of the same records in ``splits``."""
    found = [paired_rows(study, source, target, split) for split in splits]
    return (np.concatenate([a for a, _ in found]).astype(int), np.concatenate([b for _, b in found]).astype(int))


def fit_maps(study, source, target, fit_on, shuffle=None):
    """Full and top-128-plane Procrustes maps W (rows: a W ~ b) fitted on the pairs of ``fit_on``."""
    a, b = pair_rows(study, source, target, FIT_SETS[fit_on])
    A = unit(study.X[a] - condition_mean(study, source))
    B = unit(study.X[b] - condition_mean(study, target))
    if shuffle is not None:
        B = B[np.random.default_rng(int(shuffle)).permutation(len(B))]
    maps = {"procrustes": procrustes(A, B), REDUCED: reduced_procrustes(A, B)}
    return maps, dict(fit_on=fit_on, fit_pairs=int(len(a)), shuffle_seed=shuffle)


def pair_cosine(study, source, target, splits, W=None):
    """Mean cosine of paired rows (centred, normalized) after mapping the source rows by W."""
    a, b = pair_rows(study, source, target, splits)
    A = unit(study.X[a] - condition_mean(study, source))
    B = unit(study.X[b] - condition_mean(study, target))
    if W is not None:
        A = unit(A @ W)
    return float(np.mean(np.sum(A * B, 1))), int(len(a))


def rotate(sub, W, center):
    """A source subspace carried into target coordinates: basis W^T Q and input map W^T F (as in X7)."""
    factor = None if sub.projection_factor is None else W.T @ sub.projection_factor
    return Subspace(center, W.T @ sub.basis, sub.effective_rank, sub.method, sub.requested_rank, factor)


def shared_dims(reference, basis):
    """Sum of squared cosines of the principal angles between two orthonormal bases."""
    return float(np.linalg.norm(np.asarray(reference).T @ np.asarray(basis)) ** 2)


def overlap(reference, basis):
    """Share of the reference subspace that lies in span(basis): tr(P_S P_U) / dim U (the read/erase bound's omega)."""
    reference = np.asarray(reference)
    return shared_dims(reference, basis) / max(reference.shape[1], 1)
