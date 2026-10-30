"""End-to-end smoke test of X9--X14 on a miniature synthetic campaign (CPU, a few minutes).

    python -m pytest tests/experiments/test_smoke_x9_x14.py -q -p no:cacheprovider

It builds the synthetic campaign (two tiny local BERT encoders, no downloads),
runs X7 to obtain the reproduction reference, adds a decoder tree (a copy of
one encoder's tree, as the campaign's runner lays it out), then runs X10, X12,
X13, X9, X11 and X14 with tiny budgets and the verifier, whose reproduction
gates must pass. It checks code paths, file formats and the gates, not results.
"""
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _decoder_tree(campaign):
    """A decoder tree with the runner's layout: a copy of the first encoder's primary tree."""
    from idiom_subspace.config import model_slug
    config = json.loads((campaign / "configs" / "full_3090.json").read_text(encoding="utf-8"))
    name = config["models"][0]["name"]
    shutil.copytree(campaign / "primary" / model_slug(name), campaign / "decoder" / model_slug(name))
    (campaign / "configs" / "decoder_3090.json").write_text(
        json.dumps(dict(config, models=[{"name": name, "revision": "main", "kind": "decoder"}])), encoding="utf-8")
    return name


def test_x9_to_x14_on_a_synthetic_campaign(tmp_path):
    from synthetic_campaign import build
    import run_experiments as run
    from experiments.verify2 import verify
    campaign = build(tmp_path)
    decoder = _decoder_tree(campaign)
    flags = ["--campaign", str(campaign), "--allow-cpu", "--smoke", "--threads", "2"]
    run.main(["X7", "--out", str(tmp_path / "out1"), "--results", str(tmp_path / "results1")] + flags)
    x7 = tmp_path / "results1" / "X7_alignment"
    common = ["--out", str(tmp_path / "out2"), "--results", str(tmp_path / "results2")] + flags
    for code in ["X10", "X12", "X13", "X9", "X11", "X14"]:
        run.main([code] + common)
    results = tmp_path / "results2"
    assert verify(results, x7=x7) == 0

    aligned = pd.read_csv(results / "X10_heldout_alignment" / "aligned.csv")
    assert {"space", "probe", "validation", "probe_validation", "probe_validation_shuffled"} <= set(aligned.fit_on)
    assert {"omega_aligned", "read_ok_aligned", "erase_share_aligned"} <= set(aligned)
    retrieval = pd.read_csv(results / "X10_heldout_alignment" / "retrieval.csv")
    assert {"none", "space", "probe_validation"} <= set(retrieval.fit_on)

    linear = pd.read_csv(results / "X9_erasers_nonlinear" / "linear.csv")
    assert {"COV", "LEACE", "MLR", "INLP", "RLACE", "RAND"} <= set(linear.eraser)
    assert {"raw", "procrustes", "procrustes_pca128"} <= set(linear.condition)
    assert linear[linear.target.eq("bn-fig") & linear.source.eq("en-idiom")].erase_share.notna().any()
    mlp = pd.read_csv(results / "X9_erasers_nonlinear" / "mlp.csv")
    assert {"COV", "LEACE", "INLP", "RLACE"} <= set(mlp.eraser)
    summary = pd.read_csv(results / "X9_erasers_nonlinear" / "summary.csv")
    assert len(summary) and summary.omega.notna().all()

    taxonomy = pd.read_csv(results / "X11_taxonomy_robustness" / "tests.csv")
    assert {"F1", "F2", "F3", "TAG"} <= set(taxonomy.concept)
    report = json.loads((results / "X11_taxonomy_robustness" / "label_report_split0.json").read_text(encoding="utf-8"))
    assert report["reproduction"]["agreement"] == 1.0

    path = pd.read_csv(results / "X12_penalty_path" / "path.csv")
    assert {"MLR", "LEACE", "COV"} <= set(path.method) and set(path.split) == {0, 1}
    assert path.SSI_zs_acc.notna().any() and path.RAND_zs_p_acc.notna().any()

    isolated = pd.read_csv(results / "X13_idiom_isolated" / "tests.csv")
    assert set(isolated.target) == {"bn-idiom"} and isolated.PERM_p_acc_holm.notna().any()
    absolute = pd.read_csv(results / "X13_idiom_isolated" / "absolute.csv")
    assert {"bn-idiom", "bn-fig"} <= set(absolute.target) and absolute.SSI_ambient_acc.notna().any()

    decoder_tests = pd.read_csv(results / "X14_decoder_powered" / "tests.csv")
    assert set(decoder_tests.model) == {decoder} and set(decoder_tests.M) == {4, 2}

    # Re-running restores finished cells instead of recomputing them.
    run.main(["X9"] + common)
    again = pd.read_csv(results / "X9_erasers_nonlinear" / "summary.csv")
    pd.testing.assert_frame_equal(summary, again)
