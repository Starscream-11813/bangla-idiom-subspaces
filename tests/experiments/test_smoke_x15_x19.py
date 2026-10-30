"""End-to-end smoke test of X15--X19 on a miniature synthetic campaign (CPU, a few minutes).

    python -m pytest tests/experiments/test_smoke_x15_x19.py -q -p no:cacheprovider

It builds the synthetic campaign (two tiny local BERT encoders, no downloads), writes a volunteers' sheet with
random answers for up to 25 discovery TEST records per F class, runs X15, X16, X17, X19 and X18 with tiny budgets,
and then the verifier. It checks code paths, file formats and the gates, not results. The released reference tables
do not describe the synthetic encoders, so their gates report "no common cells" as warnings here.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _judgments(campaign, path):
    """A volunteers' sheet for the synthetic campaign (random answers; no corpus text)."""
    from experiments.x17_human_items import sample_id
    corpus = pd.read_json(next((campaign / "primary").glob("*/split_0/labels_0/corpus_prepared.json")))
    test = corpus[corpus.split.eq("test") & corpus.F.ge(0)].groupby("F").head(25)
    rng = np.random.default_rng(0)
    rows = []
    for r in test.itertuples():
        row = dict(sample_id=sample_id(r.idiom_id), assigned_F=int(r.F))
        for a in "AB":
            row[f"f_plausibility_{a}"] = rng.choice(["good", "partial", "poor"])
            row[f"function_{a}"] = rng.choice(["referential", "evaluative", "both"])
            row[f"counterpart_type_{a}"] = rng.choice(["idiom", "literal_paraphrase", "proverb"])
            row[f"counterpart_equivalence_{a}"] = rng.choice(["equivalent", "partly_equivalent", "not_equivalent"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_x15_to_x19_on_a_synthetic_campaign(tmp_path):
    from synthetic_campaign import build
    import run_experiments as run
    from experiments.verify3 import verify
    campaign = build(tmp_path)
    human = _judgments(campaign, tmp_path / "judgments.csv")
    common = ["--campaign", str(campaign), "--allow-cpu", "--smoke", "--threads", "2", "--out", str(tmp_path / "out3"),
              "--results", str(tmp_path / "results3")]
    for code in ["X15", "X16", "X17", "X19", "X18"]:
        run.main([code] + common + (["--human", str(human)] if code == "X17" else []))
    results = tmp_path / "results3"
    assert verify(results) == 0

    tests = pd.read_csv(results / "X15_matched_nulls" / "tests.csv")
    assert {"bn-fig", "bn-idiom"} <= set(tests.target) and {"MLR", "COV"} <= set(tests.method)
    assert tests.RANDCOV_p_acc.notna().all() and tests.RANDCOV_p_bal_holm.notna().all()
    assert tests.minus_PCA_acc_pp.notna().all() and tests.PCA_SSI_acc.notna().any()

    prospective = pd.read_csv(results / "X16_idiom_prospective" / "tests.csv")
    assert set(prospective.split) == {1} and set(prospective.target) == {"bn-idiom"}
    assert {"RAND_p_bal_holm", "PERM_p_bal_holm", "RANDCOV_p_bal_holm", "primary_pass"} <= set(prospective)
    assert prospective[prospective.method.eq("COV")].primary.all()
    assert len(pd.read_csv(results / "X16_idiom_prospective" / "gate.csv"))
    assert pd.read_csv(results / "X16_idiom_prospective" / "zero_shot.csv").RAND_zs_p_bal.notna().any()

    contrasts = pd.read_csv(results / "X17_human_items" / "contrasts.csv")
    assert {"bn-idiom", "bn-fig"} <= set(contrasts.target)
    assert {"cross_cov", "zs_cov", "native_cov"} <= set(contrasts.outcome) and contrasts.p_perm.between(0, 1).all()
    coverage = pd.read_csv(results / "X17_human_items" / "coverage.csv")
    assert coverage.F_matches_packet.all() and coverage["items"].gt(0).all()

    summary = pd.read_csv(results / "X18_refit_bootstrap" / "summary.csv")
    assert summary.agreement_cross.ge(.98).all() and summary.refit_ci_lo_acc.notna().any()

    subsets = pd.read_csv(results / "X19_counterpart_type" / "subsets.csv")
    assert {"all", "opaque", "transparent", "random"} <= set(subsets.subset)
    assert pd.read_csv(results / "X19_counterpart_type" / "summary.csv").opaque_minus_random_acc.notna().any()

    # Re-running restores finished cells instead of recomputing them.
    run.main(["X15"] + common)
    pd.testing.assert_frame_equal(tests, pd.read_csv(results / "X15_matched_nulls" / "tests.csv"))
