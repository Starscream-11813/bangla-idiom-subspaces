"""End-to-end smoke test of X1--X8 on a miniature synthetic campaign (CPU, ~5 minutes).

    python -m pytest tests/experiments/test_smoke_x1_x8.py -q

It builds a campaign with the real on-disk layout from two tiny local BERT
encoders (no downloads), runs X1--X8 with tiny null/bootstrap budgets, and
runs the verifier. It checks code paths and file formats, not results.
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

STEPS = [("X5", []), ("X1", []), ("X3", []), ("X7", []), ("X8", []),
         ("X2", ["--fbn-backend", "tfidf", "--dom-k", "6", "--fbn-k", "6"]),
         ("X6", ["--models", "weights_0"]), ("X4", ["--device", "cpu"])]


def test_every_experiment_on_a_synthetic_campaign(tmp_path):
    from synthetic_campaign import build
    import run_experiments as run
    from experiments.verify import verify
    campaign = build(tmp_path)
    common = ["--campaign", str(campaign), "--out", str(tmp_path / "out"), "--results", str(tmp_path / "results"),
              "--allow-cpu", "--smoke", "--threads", "2"]
    for code, extra in STEPS:
        run.main([code] + common + extra)
    results = tmp_path / "results"
    assert verify(results) == 0
    tests = pd.read_csv(results / "X5_balanced_metrics" / "tests.csv")
    assert {"SSI_acc", "SSI_bal", "SSI_f1", "RAND_p_bal", "PERM_p_f1"} <= set(tests)
    assert tests.SSI_acc.notna().any()
    # Re-running restores finished cells instead of recomputing them.
    run.main(["X5"] + common)
    again = pd.read_csv(results / "X5_balanced_metrics" / "tests.csv")
    pd.testing.assert_frame_equal(tests.drop(columns=[c for c in tests if c.startswith("diff_")], errors="ignore"),
                                  again.drop(columns=[c for c in again if c.startswith("diff_")], errors="ignore"))
