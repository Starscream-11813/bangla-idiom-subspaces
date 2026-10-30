"""Offline runner checks using real corpus/taxonomy code and synthetic features."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from idiom_subspace.config import Config
from idiom_subspace import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = []
        for j in range(80):
            topic = "wealth savings income fortune" if j % 2 else "anger fury wrath frustration"
            rows.append(dict(id=str(j), idiom=f"বাগধারা {j}",
                             figurative_meaning_en=f"{topic} meaning {j}",
                             figurative_meaning_bn=f"অর্থ {j}", literal_meaning=f"আক্ষরিক {j}",
                             example_sentences_in_bangla=[f"সে বাগধারা {j} বলে।"],
                             example_sentences_in_english=[f"He uses expression {j}."],
                             similar_in_english=[f"expression {j}"], sentiment="positive" if j % 2 else "negative",
                             usage_domain=["daily life"], tags=["emotion"], frequency="common",
                             cultural_significance=False, religious_significance=False,
                             historical_significance=False))
        self.corpus = self.root / "fixture.json"
        self.corpus.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def config(self, out_name):
        return Config(data=str(self.corpus), out=str(self.root / out_name),
                      split_fractions=[.4, .2, .2, .2], seeds=[3], label_seeds=[0],
                      label_backend="tfidf", classes=2,
                      models=[{"name": "synthetic", "revision": "offline"}],
                      conditions=["bn-fig", "en-idiom"],
                      pairs=[["en-idiom", "bn-fig"], ["bn-fig", "en-idiom"]],
                      concepts=["F"], methods=["MLR"], ranks=[1],
                      pca_dim=4, layers=[0, 1], bootstrap=10, null_draws=2,
                      probe_Cs=[.1, 1.], min_group=100, studies=["S1", "S3"], robustness=False)

    @staticmethod
    def fake_features(items, spec, cfg, cache_dir, progress=None):
        # The raw semantic signal is derived from source text, independently of
        # assigned taxonomy labels. Changing test labels cannot change features.
        signal = items.fig_en.str.contains("wealth").to_numpy(float) * 2 - 1
        rng = np.random.default_rng(123)
        features = rng.normal(0, 1, size=(len(items), 2, 6))
        features[:, 0, 0] = 5 * signal + rng.normal(0, .01, len(items))
        pooling = pd.DataFrame(dict(item_id=items.item_id, valid=True, status="ok", token_count=2))
        return features, pooling, dict(layers=[0, 1], model_layer_count=2,
                                       resolved_revision="offline-fixed", fingerprint="synthetic")

    def run_offline(self, cfg, **patches):
        with patch("idiom_subspace.extraction.extract_features", side_effect=self.fake_features), \
             patch("idiom_subspace.runner.environment", return_value={"test": "offline"}), \
             redirect_stdout(io.StringIO()):
            runner.run(cfg, **patches)
        return Path(cfg.out) / "split_3" / "labels_0" / "synthetic"

    def test_audit_and_templates_work_offline(self):
        cfg = self.config("audit")
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(cfg.as_dict()), encoding="utf-8")
        # If audit ever starts building labels, it would violate its read-only,
        # model-free role; templates intentionally fit the selected TF-IDF backend.
        with patch("idiom_subspace.labels.build_labels", side_effect=AssertionError("audit must not fit labels")), \
             redirect_stdout(io.StringIO()):
            runner.main(["audit", "--config", str(config_path)])
        audit = json.loads((Path(cfg.out) / "data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["records"], 80)
        with redirect_stdout(io.StringIO()):
            runner.main(["templates", "--config", str(config_path), "--n", "7"])
        folder = Path(cfg.out) / "annotations"
        controls = pd.read_csv(folder / "controls.csv", keep_default_na=False)
        self.assertEqual(len(controls), 49)
        self.assertTrue(controls.text.eq("").all())
        self.assertFalse(controls.approved.any())
        scores = pd.read_csv(folder / "behavior_scores.csv", keep_default_na=False)
        self.assertEqual(len(scores), 7)
        self.assertTrue(scores.score.eq("").all())
        prompts = json.loads((folder / "intervention_prompts.json").read_text(encoding="utf-8"))
        for prompt in prompts:
            start, end = prompt["span"]
            self.assertTrue(prompt["prompt"][start:end].startswith("বাগধারা"))

    def test_full_mocked_run_persists_preprocessing_and_paired_inference(self):
        cfg = self.config("run")
        dest = self.run_offline(cfg)
        status = json.loads((dest / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        selected = json.loads((dest / "layer_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["selected_using"], "validation_only")
        self.assertEqual(selected["layer"], 0)
        transform = np.load(dest / "preprocess_layer_0.npz")
        self.assertTrue(np.isfinite(transform["mean"]).all())
        self.assertGreater(float(transform["scale"]), 0)
        self.assertEqual(transform["components"].shape, (4, 6))
        infer = pd.read_csv(dest / "S3_inference.csv")
        self.assertEqual(len(infer), 2)
        self.assertTrue(infer.n_valid.gt(0).all())
        pred = pd.read_csv(dest / "predictions.csv")
        self.assertFalse(pred[["cross_correct", "within_correct", "floor_correct"]].isna().any().any())
        for _, pair in pred.groupby(["source_condition", "condition"]):
            self.assertTrue(pair.idiom_id.is_unique)
        self.assertTrue((Path(cfg.out) / "inference_all.csv").exists())
        self.assertTrue((Path(cfg.out) / "refit_summary.csv").exists())

    def test_selected_layer_is_invariant_to_changed_test_labels(self):
        original_cfg = self.config("original")
        changed_cfg = self.config("changed")
        original = self.run_offline(original_cfg)
        from idiom_subspace.labels import build_labels as actual_labels
        def altered_labels(*args, **kwargs):
            frame, report, artifacts = actual_labels(*args, **kwargs)
            mask = frame.split.eq("test")
            frame.loc[mask, "F"] = 1 - frame.loc[mask, "F"]
            return frame, report, artifacts
        with patch("idiom_subspace.labels.build_labels", side_effect=altered_labels):
            changed = self.run_offline(changed_cfg)
        a = json.loads((original / "layer_selection.json").read_text(encoding="utf-8"))
        b = json.loads((changed / "layer_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(a, b)
        original_grid = pd.read_csv(original / "grid_layer_0.csv")
        changed_grid = pd.read_csv(changed / "grid_layer_0.csv")
        # Confirm that the intervention mattered to test scores while leaving
        # both the selected layer and all validation criteria unchanged.
        self.assertFalse(np.allclose(original_grid.R, changed_grid.R))
        np.testing.assert_allclose(original_grid.validation_R, changed_grid.validation_R)

    def test_completed_runner_resumes_without_estimator_or_probe_fits(self):
        cfg = self.config("resume")
        dest = self.run_offline(cfg)
        before = pd.read_csv(dest / "S1_S3_grid.csv")
        with patch("idiom_subspace.studies.fit_estimator", side_effect=AssertionError("refitted estimator")), \
             patch("idiom_subspace.studies.tune_probe", side_effect=AssertionError("refitted probe")):
            self.run_offline(cfg)
        pd.testing.assert_frame_equal(before, pd.read_csv(dest / "S1_S3_grid.csv"))
        state = json.loads((dest / "progress.json").read_text(encoding="utf8"))
        self.assertEqual(state["status"], "completed")

    def test_legacy_analysis_directory_is_preserved_and_rejected(self):
        cfg = self.config("legacy")
        root = Path(cfg.out); root.mkdir()
        original = '{"historical":true}'
        (root / "run_config.json").write_text(original)
        with self.assertRaisesRegex(ValueError, "fresh --out"):
            self.run_offline(cfg)
        self.assertEqual((root / "run_config.json").read_text(), original)
        self.assertFalse((root / "pipeline_manifest.json").exists())

    def test_extract_only_skips_labels_and_all_analysis(self):
        cfg = self.config("extract_only"); cfg.extract_only = True
        with patch("idiom_subspace.labels.build_labels", side_effect=AssertionError("fit labels")), \
             patch("idiom_subspace.studies.lexical_baselines", side_effect=AssertionError("fit lexical probes")), \
             patch("idiom_subspace.studies.LayerStudy", side_effect=AssertionError("started analysis")):
            dest = self.run_offline(cfg)
        self.assertTrue(json.loads((dest / "status.json").read_text())["analysis_pending"])
        self.assertFalse((dest / "S1_S3_grid.csv").exists())

    def test_decoder_exports_eligible_nonfinal_layer_separately(self):
        cfg = self.config("decoder")
        cfg.models[0]["kind"] = "decoder"
        dest = self.run_offline(cfg)
        selected = json.loads((dest / "intervention_layer_selection.json").read_text())
        self.assertEqual(selected["layer"], 1)
        with np.load(dest / "basis_F_bn-fig.npz", allow_pickle=False) as basis:
            self.assertEqual(int(basis["layer_index"]), 0)
            self.assertEqual(int(basis["hidden_state_index"]), 1)
            self.assertEqual(basis["basis"].shape, (6, 1))

    def test_generation_cli_preserves_recorded_precision(self):
        import torch
        cfg=self.config("generate")
        config=self.root/"generation_config.json";config.write_text(json.dumps(cfg.as_dict()))
        basis=self.root/"F.npz"
        np.savez(basis,basis=np.eye(4)[:,:1],center=np.zeros(4),model_id="synthetic",layer_index=0,
                 space="decoder_block_output",resolved_revision="fixed-commit",quantization="null",precision="torch.bfloat16")
        records=self.root/"records.json";records.write_text("[]")
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as model_loader, \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("idiom_subspace.interventions.run_interventions",return_value=[]), \
             patch("torch.cuda.is_available",return_value=False):
            runner.main(["intervene","--config",str(config),"--model","synthetic","--basis",str(basis),"--records",str(records)])
        self.assertEqual(model_loader.call_args.kwargs["torch_dtype"],torch.bfloat16)
        self.assertEqual(model_loader.call_args.kwargs["revision"],"fixed-commit")

    def test_generation_cli_rejects_mixed_precision_bases_before_loading_weights(self):
        cfg=self.config("bad_generate")
        config=self.root/"generation_config.json";config.write_text(json.dumps(cfg.as_dict()))
        values=dict(basis=np.eye(4)[:,:1],center=np.zeros(4),model_id="synthetic",layer_index=0,
                    space="decoder_block_output",resolved_revision="fixed-commit",quantization="null")
        a,b=self.root/"F.npz",self.root/"L.npz"
        np.savez(a,**values,precision="torch.float16");np.savez(b,**values,precision="torch.bfloat16")
        records=self.root/"records.json";records.write_text("[]")
        with patch("transformers.AutoModelForCausalLM.from_pretrained",side_effect=AssertionError("loaded weights")), \
             self.assertRaisesRegex(ValueError,"same precision"):
            runner.main(["intervene","--config",str(config),"--model","synthetic","--basis",str(a),
                         "--literal-basis",str(b),"--records",str(records)])


if __name__ == "__main__":
    unittest.main()
