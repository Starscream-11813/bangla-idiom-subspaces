"""v4 additions: transparency moderator, geometric sharing, language subspace,
machine literal glosses, and dashboard queue/log/snapshot views."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from idiom_subspace.data import load_corpus, split_corpus
from idiom_subspace.labels import build_labels
from test_studies import synthetic_study


def _records(n=24):
    rows = []
    glosses = ["anger and rage", "fear and panic", "money and wealth"]
    for i in range(n):
        rows.append(dict(id=str(i), idiom=f"বাগধারা {i}", figurative_meaning_en=f"{glosses[i % 3]} {i}",
                         figurative_meaning_bn=f"অর্থ {i}", literal_meaning=f"আক্ষরিক {i}",
                         example_sentences_in_bangla=[f"সে বাগধারা {i} বলে।"],
                         example_sentences_in_english=[f"He says expression {i}."],
                         similar_in_english=[glosses[i % 3] if i % 2 else f"spill the beans {i}"],
                         tags=["x"], usage_domain=["y"], sentiment="negative" if i % 2 else "positive",
                         cultural_significance="1", religious_significance="0", historical_significance="0",
                         frequency="common"))
    return rows


def _corpus(tmp_path, rows):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    return load_corpus(path)[0], path


def test_transparency_is_a_space_fitted_covariate(tmp_path):
    df, _ = _corpus(tmp_path, _records())
    df = split_corpus(df, seed=1)
    labeled, report, _ = build_labels(df, k=3)
    assert "cp_transparency" in labeled and labeled.cp_transparency.notna().all()
    paraphrase = labeled.cp_transparency[labeled.sims.map(lambda s: not s[0].startswith("spill"))]
    idiom_like = labeled.cp_transparency[labeled.sims.map(lambda s: s[0].startswith("spill"))]
    assert paraphrase.mean() > idiom_like.mean()
    info = report["counterpart_transparency"]
    assert info["fitted_on"] == "space" and info["lower_tertile"] <= info["upper_tertile"]
    assert not (labeled.transparent & labeled.opaque).any()


def test_subspace_overlap_identity_and_random_expectation():
    study = synthetic_study(methods=("MLR", "COV", "RAND"), ranks=(1, 2))
    rows = study.subspace_overlap()
    assert not rows.empty and not rows.method.eq("RAND").any()
    assert rows.mean_cos2.between(0, 1).all()
    assert np.allclose(rows.random_expected_mean_cos2, rows.M / study.X.shape[1])
    # Same concept signal in both conditions of the fixture: well above random overlap.
    f = rows[(rows.concept == "F") & (rows.method == "COV") & (rows.M == 2)]
    assert (f.mean_cos2 > f.random_expected_mean_cos2).all()


def test_language_difference_subspace_reports_angles_and_both_sides():
    study = synthetic_study(methods=("MLR", "LEACE"), ranks=(1, 2))
    rows = study.language_subspace()
    assert not rows.empty
    assert {"concept_error_inside_language", "concept_error_outside_language", "floor_error",
            "MLR_mean_cos2_with_language", "language_variance_share"} <= set(rows)
    # The fixture's language offset lives in its own coordinate: F is not inside it.
    one = rows[(rows.concept == "F") & (rows.M == 1)].iloc[0]
    assert one.concept_error_inside_language > one.concept_error_outside_language
    assert one.MLR_mean_cos2_with_language < .5


def test_machine_literal_glosses_feed_L_with_provenance(tmp_path, monkeypatch):
    from idiom_subspace import literal
    from idiom_subspace.runner import apply_literal_reviews
    rows = _records()
    df, _ = _corpus(tmp_path, rows)

    class FakeTokenizer:
        def __init__(self): self.calls = 0
        def __call__(self, batch, **kwargs):
            self.batch = list(batch)
            return types.SimpleNamespace(to=lambda device: {"batch": self.batch})
        def convert_tokens_to_ids(self, token): return 7
        def batch_decode(self, output, skip_special_tokens=True):
            return [("anger and rage" if int(t.split()[-1]) % 3 == 0 else "fear and panic") for t in output]

    class FakeModel:
        config = types.SimpleNamespace(_commit_hash="abc123")
        def to(self, device): return self
        def eval(self): return self
        def generate(self, batch, **kwargs): return batch

    fake_tokenizer = FakeTokenizer()
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: fake_tokenizer),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    table = literal.translate_literals(df, model="fake/nllb-model", device="cpu", batch_size=5)
    assert len(table) == len(df) and table.approved.eq("machine").all()
    assert table.source.str.endswith("@abc123").all()
    table.loc[0, ["literal_en", "approved"]] = ["fear and panic", "true"]     # one human-reviewed row
    path = tmp_path / "literal.csv"
    table.to_csv(path, index=False)
    reviewed = apply_literal_reviews(df, path)
    origins = reviewed.lit_en_source.str.split(":").str[0]
    assert origins.iloc[0] == "human_review" and origins.iloc[1:].eq("machine_translation").all()
    labeled, report, _ = build_labels(split_corpus(reviewed, seed=1), k=3)
    assert labeled.L.ge(0).all() and report["literal_english_available"] == len(df)


def test_second_exact_context_becomes_bn_fig2(tmp_path):
    from idiom_subspace.data import build_items
    rows = _records(4)
    rows[0]["example_sentences_in_bangla"] = ["সে বাগধারা 0 বলে।", "আবার বাগধারা 0 শুনি।"]
    rows[0]["example_sentences_in_english"] = ["He says it.", "Again I hear it."]
    df, _ = _corpus(tmp_path, rows)
    items = build_items(df)
    second = items[items.condition.eq("bn-fig2")]
    assert len(second) == 1 and second.iloc[0].text == "আবার বাগধারা 0 শুনি।" and second.iloc[0].span_found
    assert items.attrs["item_audit"]["skipped"]["bn-fig2:no_second_exact_example"] == 3


def _identity_study():
    from types import SimpleNamespace
    from idiom_subspace.studies import LayerStudy
    rng = np.random.default_rng(3)
    rows, feats, dims = [], [], 24
    for split, count in (("space", 40), ("probe", 12), ("validation", 12), ("test", 30)):
        for j in range(count):
            identity = np.r_[rng.normal(size=8) * 1.2, np.zeros(dims - 8)]   # identity lives in 8 dimensions
            label = j % 3
            for condition in ("bn-fig", "bn-fig2"):
                x = identity + rng.normal(0, .3, size=dims)
                x[8:11] += np.eye(3)[label]
                feats.append(x)
                rows.append(dict(condition=condition, lang="bn", split=split, F=label, S=j % 2,
                                 idiom_id=f"{split}_{j}", item_id=f"{split}_{j}_{condition}",
                                 idiom_group=f"g_{split}_{j}", component_id=f"c_{split}_{j}"))
    cfg = SimpleNamespace(pairs=[], concepts=["F"], methods=["LDA", "LEACE", "CPCA", "COV", "PCA", "RAND"], ranks=[1, 2],
                          probe_Cs=[1.], min_group=3, inference_method="MLR", null_draws=3, bootstrap=10, min_den=.02,
                          open_class_reps=1, open_class_N=[10], split_mode="input_disjoint", compute_leakage=True,
                          inference_groups=["all"], primary_test_groups=["all"], primary_test_pairs=[],
                          primary_test_concepts=["F"], permutation_unit="component", min_permutation_movable_fraction=.5,
                          diagnostic_null_draws=2, identity_N=[10, 20], identity_ranks=[4, 8], identity_reps=1,
                          identity_methods=["LDA", "LEACE", "CPCA", "PCA", "RAND"])
    return LayerStudy(np.asarray(feats), pd.DataFrame(rows), cfg, dict(seed=0, label_seed=0, model="synthetic", layer=0))


def test_identity_open_class_matches_unseen_idioms():
    rows = _identity_study().identity_open_class()
    assert set(rows.method) == {"LDA", "LEACE", "CPCA", "PCA", "RAND"} and set(rows.N) == {10, 20, 40}
    full = rows[rows.N.eq(40) & rows.M.eq(8)].set_index("method")
    assert (full.ambient_R > 10 * full.chance).all()
    assert full.loc["CPCA", "R"] > full.loc["RAND", "R"] and full.loc["LDA", "R"] > full.loc["RAND", "R"]
    assert full.loc["CPCA", "L"] < full.loc["CPCA", "ambient_R"]          # identity leaves the complement
    assert full.purity_err.notna().all() and rows[rows.N.lt(40)].get("purity_err", pd.Series(dtype=float)).isna().all()
    assert set(rows.loc[rows.method.isin(["LDA", "LEACE"]), "estimator_ridge"]) == {"ledoit_wolf"}


def test_open_class_reports_purity_against_the_other_concept():
    rows = synthetic_study(methods=("MLR", "RAND"), ranks=(1,)).open_class()
    f = rows[rows.concept.eq("F")]
    assert f.purity_concept.eq("S").all() and f.purity_err.between(0, 1).all()
    assert rows[rows.concept.eq("S")].purity_concept.eq("F").all()


def test_open_class_shrinks_small_estimation_sets_only():
    from unittest.mock import patch
    from idiom_subspace.geometry import fit_estimator as actual
    study = synthetic_study(methods=("LEACE", "MLR"), ranks=(1,))
    seen = []
    def record(X, y, method, *args, **kwargs):
        seen.append((method, kwargs.get("ridge")))
        return actual(X, y, method, *args, **kwargs)
    with patch("idiom_subspace.studies.fit_estimator", side_effect=record):
        rows = study.open_class()
        assert getattr(study, "_ridge_override", None) is None
        study.sub("bn-idiom", "F", "LEACE", 1)          # main-grid estimator
    assert ("LEACE", "ledoit_wolf") in seen
    assert seen[-1] == ("LEACE", 0.)
    assert set(rows.loc[rows.method.eq("LEACE"), "estimator_ridge"]) == {"ledoit_wolf"}
    assert set(rows.loc[rows.method.eq("MLR"), "estimator_ridge"]) == {"none"}


def test_capped_probe_fits_are_recorded_not_printed():
    import warnings
    from idiom_subspace.geometry import fit_probe
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped = fit_probe(X, y, max_iter=1)
        converged = fit_probe(X, y, max_iter=5000)
    assert capped.converged_ is False and converged.converged_ is True
    assert not [w for w in caught if "converge" in str(w.message).lower()]
    study = synthetic_study(methods=("MLR",), ranks=(1,))
    grid = study.run_grid()
    assert grid.probe_converged.dtype == bool and "leak_probe_converged" in grid


def test_dashboard_reports_queue_logs_and_writes_offline_snapshot(tmp_path):
    from datetime import datetime, timezone
    from idiom_subspace.dashboard_server import DashboardData, export_snapshot
    root = tmp_path / "jobs"
    job = root / "xlm-roberta-large" / "split_0" / "labels_0" / "xlm-roberta-large"
    job.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    (job / "progress.json").write_text(json.dumps(dict(status="running", stage="grid", completed=5, total=10,
        updated_at=now, context={"model": "xlm-roberta-large", "seed": 0, "label_seed": 0},
        milestones={"layer": {"done": 2, "total": 9, "eta_seconds": 700., "durations": [300., 320.]}})), encoding="utf8")
    (root / "launcher.json").write_text(json.dumps(dict(status="running", updated_at=now, gpus=["0"],
        planned=["xlm-roberta-large", "bert-base-multilingual-cased"], queued=["bert-base-multilingual-cased"],
        running=[{"model": "xlm-roberta-large", "gpu": "0"}], finished=[])), encoding="utf8")
    (root / "xlm-roberta-large" / "job.log").write_text(
        "line one\nprogress 10%\rprogress 55%\n\x1b[31mred warning\x1b[0m\n", encoding="utf8")
    data = DashboardData(root)
    overview = data.overview()
    assert overview["summary"]["planned_jobs"] == 2 and overview["summary"]["queued"] == 1
    assert overview["launchers"][0]["planned"][0] == "xlm-roberta-large"
    detail = data.detail(overview["jobs"][0]["id"])
    assert detail["log_tail"] == ["line one", "progress 55%", "red warning"]
    assert DashboardData(root, logs=False).detail(overview["jobs"][0]["id"])["log_tail"] == []
    page = export_snapshot(root, tmp_path / "snapshot.html")
    html = page.read_text(encoding="utf8")
    assert "window.__OBSERVATORY_SNAPSHOT__" in html and "<style>" in html and 'src="/app.js"' not in html
    # Inline scripts must run after the body is parsed (they cannot be deferred).
    assert html.index("window.__OBSERVATORY_SNAPSHOT__") > html.index('id="log-tail"')
    payload = html.split("window.__OBSERVATORY_SNAPSHOT__ = ", 1)[1].split(";</script>", 1)[0]
    snapshot = json.loads(payload.replace("<\\/", "</"))
    assert snapshot["overview"]["total"] == 1 and len(snapshot["details"]) == 1
