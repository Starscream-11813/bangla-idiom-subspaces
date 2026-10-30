"""Unit tests for the covariance-matched null and the item-level contrast (CPU, seconds)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.matched import matched_basis, matched_subspaces  # noqa: E402
from experiments.x17_human_items import auc, contrast, sample_id  # noqa: E402


def _anisotropic(n=3000, d=60, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.exp(-np.arange(d) / 4.)          # a few dominant directions, as in encoder spaces
    return rng.standard_normal((n, d)) * scales


def test_matched_subspaces_are_orthonormal_and_reproducible():
    X = _anisotropic()
    a = matched_subspaces(X, 5, [1, 2], 11)
    b = matched_subspaces(X, 5, [1, 2], 11)
    for s, t in zip(a, b):
        assert np.allclose(s.basis.T @ s.basis, np.eye(5), atol=1e-10)
        assert np.array_equal(s.basis, t.basis)
    assert not np.allclose(a[0].basis, a[1].basis)


def test_matched_subspaces_follow_the_variance_profile():
    X = _anisotropic()
    Xc = X - X.mean(0)
    total = (Xc ** 2).sum()
    rng = np.random.default_rng(3)
    matched = np.mean([((Xc @ matched_basis(Xc, 5, s)) ** 2).sum() / total for s in range(50)])
    haar = np.mean([((Xc @ np.linalg.qr(rng.standard_normal((X.shape[1], 5)))[0]) ** 2).sum() / total
                    for _ in range(50)])
    top = np.linalg.svd(Xc, full_matrices=False)[2][:5].T
    best = ((Xc @ top) ** 2).sum() / total
    # random Haar directions capture about rank/d of the variance; matched ones most of it; PCA is the ceiling
    assert haar < .2 < matched <= best + 1e-12


def test_contrast_recovers_a_planted_within_class_effect():
    rng = np.random.default_rng(0)
    rows = []
    for item in range(240):
        cls = item % 12
        high = float(rng.random() < .5)
        native = float(rng.random() < .5)
        for model in range(3):
            p = .15 + .25 * high + .2 * native + .01 * cls
            rows.append(dict(sample_id=f"s{item}", F=cls, model=f"m{model}", flag=high, native_cov=native,
                             cross_cov=float(rng.random() < p)))
    d = pd.DataFrame(rows)
    result = contrast(d, "flag", "cross_cov", n_perm=199, n_boot=199, seed=1)
    assert 10 < result["estimate_pp"] < 40 and result["p_perm"] < .05 and result["ci_lo_pp"] > 0
    # the same items with flags shuffled across items: no effect left
    items = d.drop_duplicates("sample_id")[["sample_id", "flag"]]
    shuffled = items.assign(flag=rng.permutation(items.flag.to_numpy()))
    null = contrast(d.drop(columns="flag").merge(shuffled, on="sample_id"), "flag", "cross_cov",
                    n_perm=199, n_boot=99, seed=2)
    assert null["p_perm"] > .01 and abs(null["estimate_pp"]) < result["estimate_pp"]


def test_sample_id_and_auc():
    assert sample_id("1234").startswith("validation_") and len(sample_id("1234")) == len("validation_") + 12
    assert auc([3, 4, 5, 1, 2], [True, True, True, False, False]) == 1.0
    assert auc([1, 2, 3, 4], [True, False, False, True]) == .5
