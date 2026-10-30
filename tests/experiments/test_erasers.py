"""Unit tests of the INLP and R-LACE erasers and the Fantope projection on Gaussian classes (CPU, < 1 minute).

    python -m pytest tests/experiments/test_erasers.py -q -p no:cacheprovider
"""
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import erasers as er  # noqa: E402


def gaussian(D=24, K=4, n=1500, signal=1.5, seed=0):
    """K classes whose means span a (K-1)-dimensional subspace U, with isotropic unit noise."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.normal(size=(D, K - 1)))[0]
    means = signal * rng.normal(size=(K, K - 1)) @ U.T

    def draw(m):
        y = rng.integers(0, K, m)
        return means[y] + rng.normal(size=(m, D)), y

    X, y = draw(n)
    Xv, yv = draw(n // 2)
    return X, y, Xv, yv, U


def test_fantope_projection_has_the_right_spectrum_and_fixes_projections():
    import torch
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.normal(size=(20, 20)))
    A = (A + A.T) / 2
    P, _, lam = er.fantope_project(A, 3)
    assert abs(float(torch.trace(P)) - 3) < 1e-6
    spectrum = torch.linalg.eigvalsh(P)
    assert float(spectrum.min()) > -1e-8 and float(spectrum.max()) < 1 + 1e-8
    Q = torch.linalg.qr(torch.tensor(rng.normal(size=(20, 3))))[0]
    projected, _, _ = er.fantope_project(Q @ Q.T, 3)
    assert torch.allclose(projected, Q @ Q.T, atol=1e-6)


def test_inlp_guards_its_language_and_removes_the_class_mean_span():
    X, y, Xv, yv, U = gaussian()
    out = er.inlp(X, y, Xv, yv, max_rank=12, device="cpu")
    assert out["guarded"] and out["orthonormal_error"] < 1e-10
    assert out["path"][0]["val_acc"] > out["majority"] + .2
    assert out["ranks"][1] == 3                        # K - 1 directions per step
    B = out["basis"][:, :out["ranks"][out["stop"]]]
    assert np.linalg.norm(U.T @ B) ** 2 / U.shape[1] > .9


def test_rlace_guards_a_low_dimensional_concept():
    X, y, Xv, yv, U = gaussian(D=16, K=3, signal=2.0)
    out = er.rlace(X, y, Xv, yv, rank=2, steps=3000, eval_every=100, device="cpu", seed=0)
    assert out["orthonormal_error"] < 1e-10
    assert abs(out["best_val_acc"] - out["majority"]) < .05
    assert np.linalg.norm(U.T @ out["basis"]) ** 2 / 2 > .6
    assert out["lr_used"] in er.RLACE_LRS and len(out["pilot"]) == len(er.RLACE_LRS)


def test_rlace_reports_divergence_instead_of_returning_garbage():
    import pytest
    X, y, Xv, yv, _ = gaussian(D=8, K=3)
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        er.rlace(X, y, Xv, yv, rank=1, steps=200, pilot_steps=50, eval_every=25, device="cpu")


def test_guard_of_the_full_space_and_of_the_true_span():
    X, y, Xv, yv, U = gaussian()
    full = er.guard(X, y, Xv, yv)
    removed = er.guard(X, y, Xv, yv, U, U)
    assert full["guard_gap"] > .2
    assert abs(removed["guard_gap"]) < .03
