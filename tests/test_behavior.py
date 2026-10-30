"""Behavior joins and causal hooks checked without downloaded models/data."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from idiom_subspace.behavior import (geometry_behavior_analysis,
                                    paired_intervention_effects,
                                    validate_behavior_scores,
                                    validate_rubric_concordance,
                                    write_score_template)
from idiom_subspace.interventions import (InterventionBasis, run_interventions,
                                         transform_activations)


def test_score_validation_preserves_raters_and_rejects_pseudoreplication():
    ratings = pd.DataFrame({"idiom_id": ["a", "a"], "model": ["m", "m"],
                            "score": [2, 3], "rater": ["one", "two"]})
    assert len(validate_behavior_scores(ratings)) == 2
    with pytest.raises(ValueError, match="unique"):
        validate_behavior_scores(ratings.drop(columns="rater"))
    with pytest.raises(ValueError, match="between 0 and 5"):
        validate_behavior_scores(ratings.assign(score=[2, 6]))
    with pytest.raises(ValueError, match="Aggregate"):
        validate_behavior_scores(ratings.assign(idiom_id="overall"))


def test_analysis_matches_exact_model_ids_and_clusters_idioms(tmp_path):
    geometry = pd.DataFrame([{"idiom_id": str(i), "model": "decoder-a", "cross_correct": i % 2,
                              "cross_margin": i / 10, "token_count": i + 1} for i in range(8)])
    scores = pd.DataFrame([{"idiom_id": str(i), "model": "decoder-a", "score": i * 0.5} for i in range(8)] +
                          [{"idiom_id": "0", "model": "other-model", "score": 4}])
    report = geometry_behavior_analysis(geometry, scores, tmp_path, bootstrap_iterations=30)
    assert report["unmatched_score_rows"] == 1
    assert report["matched_analysis_rows"] == 8
    assert all(model["status"] == "not_fit" for model in report["mixed_models"])
    correlations = pd.read_csv(tmp_path / "behavior_correlations.csv")
    assert set(correlations.bootstrap_unit) == {"idiom_id"}
    assert correlations.query("predictor == 'cross_margin'").spearman_rho.iloc[0] == pytest.approx(1)
    assert "n_ratings" in pd.read_csv(tmp_path / "behavior_matched_rows.csv")
    with pytest.raises(ValueError, match="SSI"):
        geometry_behavior_analysis(geometry.assign(per_idiom_ssi=1), scores, tmp_path)


def test_rater_means_do_not_duplicate_behavior_rows(tmp_path):
    geometry = pd.DataFrame({"idiom_id": ["a", "b", "c"], "model": "m", "cross_correct": [0, 1, 1]})
    scores = pd.DataFrame({"idiom_id": ["a", "a", "b", "b", "c", "c"], "model": "m",
                            "rater": ["x", "y"] * 3, "score": [0, 2, 2, 4, 3, 5]})
    geometry_behavior_analysis(geometry, scores, tmp_path, bootstrap_iterations=0)
    rows = pd.read_csv(tmp_path / "behavior_matched_rows.csv")
    assert len(rows) == 3
    assert rows.score.tolist() == [1, 3, 4]
    assert rows.n_ratings.tolist() == [2, 2, 2]


def test_geometry_representation_and_generation_conditions_are_distinct(tmp_path):
    geometry = pd.DataFrame({"idiom_id": ["a", "b", "c"] * 2, "model": "m",
                              "condition": ["bn-fig"] * 3 + ["en-idiom"] * 3,
                              "cross_correct": [0, 1, 1, 1, 1, 0]})
    scores = pd.DataFrame({"idiom_id": ["a", "b", "c"], "model": "m",
                            "condition": "baseline", "score": [1, 3, 4]})
    report = geometry_behavior_analysis(geometry, scores, tmp_path, bootstrap_iterations=0)
    assert report["matched_analysis_rows"] == 6
    assert "geometry_condition" in report["strata"]
    rows = pd.read_csv(tmp_path / "behavior_matched_rows.csv")
    assert rows.condition.eq("baseline").all()
    with pytest.raises(ValueError, match="No exact"):
        geometry_behavior_analysis(geometry, scores, tmp_path, condition_matches_geometry=True)


def test_crossed_random_effect_model_uses_idiom_and_model_variance_components(tmp_path):
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(9)
    geometry, scores = [], []
    for idiom in range(30):
        idiom_effect = rng.normal(0, 0.3)
        for model in range(3):
            margin = rng.normal()
            geometry.append({"idiom_id": str(idiom), "model": str(model), "cross_margin": margin,
                             "layer": model * 3 + 1, "layer_role": "validation_selected"})
            scores.append({"idiom_id": str(idiom), "model": str(model),
                           "score": np.clip(2.5 + 0.3 * margin + idiom_effect + 0.15 * model + rng.normal(0, 0.2), 0, 5)})
    report = geometry_behavior_analysis(pd.DataFrame(geometry), pd.DataFrame(scores), tmp_path, bootstrap_iterations=0)
    fitted = report["mixed_models"][0]
    assert fitted["status"] == "converged"
    assert set(fitted["variance_components"]) == {"idiom_id", "model"}
    assert fitted["predictor_estimate"] > 0
    assert "layer" not in report["strata"]
    assert "layer_role" in report["strata"]
    matched = pd.read_csv(tmp_path / "behavior_matched_rows.csv")
    assert set(matched.layer) == {1, 4, 7}
    geometry[0]["layer"] = 99
    with pytest.raises(ValueError, match="only one actual layer"):
        geometry_behavior_analysis(pd.DataFrame(geometry), pd.DataFrame(scores), tmp_path, bootstrap_iterations=0)


def test_bootstrap_preserves_counterpart_components(tmp_path):
    geometry = pd.DataFrame({"idiom_id": list("abcdef"), "model": "m",
                              "component_id": ["p", "p", "q", "q", "r", "r"],
                              "cross_margin": [0., 0.2, 0.3, 0.4, 0.6, 0.8]})
    scores = pd.DataFrame({"idiom_id": list("abcdef"), "model": "m", "score": [0, 1, 2, 2, 4, 5]})
    report = geometry_behavior_analysis(geometry, scores, tmp_path, bootstrap_iterations=30)
    assert report["n_components"] == 3
    assert report["n_idioms"] == 6
    correlations = pd.read_csv(tmp_path / "behavior_correlations.csv")
    assert correlations.bootstrap_unit.eq("component_id").all()
    assert correlations.n_bootstrap_clusters.eq(3).all()
    assert correlations.n_idioms.eq(6).all()
    assert correlations.n_components.eq(3).all()


def test_paired_intervention_effects_average_raters_and_cluster_components(tmp_path):
    scores = []
    metadata = []
    for idiom in range(12):
        for condition, values in (("baseline", (2, 4)), ("erase_F", (1, 3)), ("sham", (2, 4))):
            for rater, value in zip(("a", "b"), values):
                scores.append({"idiom_id": str(idiom), "model": "decoder", "condition": condition,
                               "rater": rater, "score": value})
        for layer in (1, 4):  # repeated prediction strata must deduplicate safely
            metadata.append({"idiom_id": str(idiom), "model": "decoder", "component_id": str(idiom // 2),
                             "anchored": idiom < 6, "layer": layer})
    scores.append({"idiom_id": "no-baseline", "model": "decoder", "condition": "erase_F", "rater": "a", "score": 5})
    report = paired_intervention_effects(pd.DataFrame(scores), tmp_path, pd.DataFrame(metadata), bootstrap_iterations=30)
    assert report["bootstrap_unit"] == "component_id"
    assert report["paired_rows"] == 24
    assert report["unpaired_intervention_rows"] == 1
    effects = pd.read_csv(tmp_path / "intervention_effects.csv")
    erasure = effects.query("condition == 'erase_F' and group == 'all'").iloc[0]
    assert erasure.n_pairs == 12 and erasure.n_components == 6
    assert erasure.mean_difference == -1
    assert erasure.ci_95_low == -1 and erasure.ci_95_high == -1
    assert set(effects.group) == {"all", "anchored", "nonanchored"}
    assert effects.query("condition == 'sham'").mean_difference.eq(0).all()
    paired = pd.read_csv(tmp_path / "intervention_paired_differences.csv")
    assert paired.baseline_score.eq(3).all()
    conflicting = pd.DataFrame(metadata)
    conflicting.loc[0, "component_id"] = "inconsistent"
    with pytest.raises(ValueError, match="inconsistent"):
        paired_intervention_effects(pd.DataFrame(scores), tmp_path, conflicting, bootstrap_iterations=0)


def test_concordance_pairs_rubric_scores_and_exports_blank_template(tmp_path):
    ratings = pd.DataFrame({"idiom_id": ["a", "b", "c", "d"], "model": "m", "score": [0, 2, 4, 5]})
    concordance = validate_rubric_concordance(ratings, ratings.assign(rater="human"), tmp_path)
    assert concordance["quadratic_weighted_kappa"] == pytest.approx(1)
    assert concordance["spearman_rho"] == pytest.approx(1)
    template = write_score_template(ratings, tmp_path / "template.csv")
    assert template.score.eq("").all()
    with pytest.raises(ValueError):
        validate_behavior_scores(tmp_path / "template.csv")
    generations = pd.DataFrame({"idiom_id": ["a"] * 3, "model": "m",
                                "condition": ["baseline", "erase_F", "patch_F"],
                                "generation": ["before", "after", None],
                                "status": ["generated", "generated", "skipped"]})
    template = write_score_template(generations, tmp_path / "generated-template.csv")
    assert template.generation.tolist() == ["before", "after"]
    assert template.condition.tolist() == ["baseline", "erase_F"]


def test_erasure_and_patch_change_only_target_tokens_and_basis_coordinates():
    hidden = torch.tensor([[[3., 7., 9.], [5., 8., 10.], [9., 2., 3.]]])
    basis = InterventionBasis(torch.tensor([[1.], [0.], [0.]]), torch.tensor([1., 0., 0.]))
    erased = transform_activations(hidden, [1], basis, operation="erase")
    assert erased[0, 1].tolist() == [1, 8, 10]
    assert torch.equal(erased[0, [0, 2]], hidden[0, [0, 2]])
    assert hidden[0, 1, 0] == 5  # no in-place edit
    patched = transform_activations(hidden, [1], basis, operation="patch", donor=torch.tensor([42., -1., -1.]))
    assert patched[0, 1].tolist() == [42, 8, 10]


class CharacterTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[ord(char) % 50 + 1 for char in text]]),
                "attention_mask": torch.ones((1, len(text)), dtype=torch.long),
                "offset_mapping": torch.tensor([[(i, i + 1) for i in range(len(text))]])}

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens)


class ToyBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1, "untouched-cache")


class ToyDecoder(torch.nn.Module):
    """Greedy generation calls its block at prefill and two cached tokens."""

    def __init__(self, fail_generation=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3, _name_or_path="toy-decoder", is_encoder_decoder=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyBlock(), torch.nn.Identity()])
        self.embeddings = torch.nn.Embedding(100, 3)
        self.fail_generation = fail_generation
        self.block_outputs = []

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embeddings(input_ids)
        for block in self.model.layers:
            output = block(hidden)
            if isinstance(output, tuple):
                assert output[1] == "untouched-cache"
                hidden = output[0]
            else:
                hidden = output
        self.block_outputs.append(hidden.detach().clone())
        return hidden

    def generate(self, input_ids, **kwargs):
        hidden = self.forward(input_ids)
        if self.fail_generation:
            raise RuntimeError("simulated generation failure")
        token = int(torch.round(hidden.sum()).item()) % 50
        for _ in range(2):
            self.forward(torch.tensor([[token]]))
        return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([[token]])], dim=1))


def test_decoder_hooks_prefill_only_and_restore_training_state():
    model = ToyDecoder()
    model.train()
    tokenizer = CharacterTokenizer()
    basis = InterventionBasis(np.eye(3)[:, :1], np.zeros(3), "toy-decoder", 0)
    rows = run_interventions(model, tokenizer,
                             [{"idiom_id": "a", "prompt": "abcd", "span": [1, 3],
                               "donor_prompt": "wxyz", "donor_span": [0, 2]}], 0, basis, basis)
    assert {row["condition"] for row in rows} == {"baseline", "erase_F", "erase_L", "random_rank_matched", "patch_F", "sham"}
    assert all(row["status"] == "generated" for row in rows)
    assert all(row["prefill_interventions"] == 1 for row in rows if row["condition"] != "baseline")
    assert model.training
    assert not model.model.layers[0]._forward_hooks
    # One donor forward, then three block calls per generation.
    baseline, erased = model.block_outputs[1], model.block_outputs[4]
    assert torch.equal(baseline[:, [0, 3]], erased[:, [0, 3]])
    assert torch.equal(erased[:, [1, 2], 0], torch.zeros((1, 2)))
    assert torch.equal(baseline[:, :, 1:], erased[:, :, 1:])
    assert rows[0]["generation"] == rows[-1]["generation"]


def test_hooks_removed_on_generation_failure_and_metadata_checked():
    model = ToyDecoder(fail_generation=True)
    model.eval()
    basis = InterventionBasis(np.eye(3)[:, :1], np.zeros(3), "toy-decoder", 0)
    # Baseline failure exercises outer cleanup. The direct edit helper is tested
    # above; a second-call failure below exercises an installed erase hook.
    calls = {"n": 0}
    original_generate = model.generate

    def second_call_failure(*args, **kwargs):
        calls["n"] += 1
        model.fail_generation = calls["n"] == 2
        return original_generate(*args, **kwargs)

    model.generate = second_call_failure
    with pytest.raises(RuntimeError, match="simulated"):
        run_interventions(model, CharacterTokenizer(), [{"idiom_id": "a", "prompt": "abc", "span": [1, 2]}], 0, basis)
    assert not model.model.layers[0]._forward_hooks
    assert not model.training
    with pytest.raises(ValueError, match="model_id"):
        run_interventions(model, CharacterTokenizer(), [], 0,
                          InterventionBasis(np.eye(3)[:, :1], np.zeros(3), "other", 0))
    with pytest.raises(ValueError, match="orthonormal"):
        run_interventions(model, CharacterTokenizer(), [], 0,
                          InterventionBasis(np.ones((3, 1)), np.zeros(3)))


def test_explicit_span_required_and_missing_optional_inputs_reported():
    model = ToyDecoder()
    basis = InterventionBasis(np.eye(3)[:, :1], np.zeros(3))
    with pytest.raises(ValueError, match="requires idiom_id, prompt and character span"):
        run_interventions(model, CharacterTokenizer(), [{"idiom_id": "a", "prompt": "abc"}], 0, basis)
    rows = run_interventions(model, CharacterTokenizer(), [{"idiom_id": "a", "prompt": "abc", "span": [1, 2],
                                                          "donor_prompt": "", "donor_span": None}], 0, basis)
    assert {row["condition"] for row in rows if row["status"] == "skipped"} == {"erase_L", "patch_F"}
    with pytest.raises(ValueError, match="both donor_prompt and donor_span"):
        run_interventions(model, CharacterTokenizer(), [{"idiom_id": "a", "prompt": "abc", "span": [1, 2],
                                                        "donor_prompt": "abc", "donor_span": None}], 0, basis)


def test_figurative_and_literal_controls_require_the_same_rank():
    model = ToyDecoder().train()
    figurative = InterventionBasis(np.eye(3)[:, :1], np.zeros(3), "toy-decoder", 0)
    literal = InterventionBasis(np.eye(3)[:, :2], np.zeros(3), "toy-decoder", 0)
    with pytest.raises(ValueError, match="matched rank"):
        run_interventions(model, CharacterTokenizer(), [], 0, figurative, literal)
    assert model.training
    assert not any(layer._forward_hooks for layer in model.model.layers)


def test_real_two_block_decoder_propagates_span_edits_and_rejects_final_block():
    """A controlled GPT-2 tests actual causal attention/KV behavior, offline.

    Block zero is an identity residual block. Block one's uniform causal
    attention mixes the first token into the final prompt position. The fixed
    output readout makes this change visible in greedy generation, including
    its next cached decoding step. No pretrained weights or network are used.
    """
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(vocab_size=8, n_positions=16, n_embd=4, n_layer=2, n_head=1, n_inner=4,
                        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., bos_token_id=None,
                        eos_token_id=None, pad_token_id=0, tie_word_embeddings=False)
    config._name_or_path = "tiny-real-decoder"
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.wte.weight[1] = torch.tensor([2., 0., -2., 0.])
        model.transformer.wte.weight[2] = torch.tensor([0., 2., 0., -2.])
        for block in model.transformer.h:
            block.ln_1.weight.fill_(1.)
            block.ln_2.weight.fill_(1.)
        model.transformer.ln_f.weight.fill_(1.)
        # Zero Q/K gives uniform causal attention; V and output are fixed.
        model.transformer.h[1].attn.c_attn.weight[:, 8:12] = torch.eye(4)
        model.transformer.h[1].attn.c_proj.weight.copy_(5 * torch.eye(4))
        model.lm_head.weight[3] = torch.tensor([2., -1., -2., 1.])
        model.lm_head.weight[4] = -model.lm_head.weight[3]

    class TinyTokenizer(CharacterTokenizer):
        eos_token_id = None

        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            encoded["input_ids"] = torch.tensor([[{"a": 1, "b": 2}[char] for char in text]])
            return encoded

    tokenizer = TinyTokenizer()
    basis = InterventionBasis(np.eye(4)[:, :1], np.array([-100., 0., 0., 0.]), "tiny-real-decoder", 0)
    tokens = torch.tensor([[1, 2]])
    with torch.inference_mode():
        baseline_logits = model(tokens).logits[0, -1].clone()

    def edit_first_span(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        replacement = transform_activations(hidden, [0], basis, operation="erase")
        return (replacement,) + output[1:] if isinstance(output, tuple) else replacement

    handle = model.transformer.h[0].register_forward_hook(edit_first_span)
    try:
        with torch.inference_mode():
            edited_logits = model(tokens).logits[0, -1].clone()
    finally:
        handle.remove()
    assert baseline_logits.argmax().item() == 3
    assert edited_logits.argmax().item() == 4
    assert not torch.allclose(baseline_logits, edited_logits)
    model.train()
    records = [{"idiom_id": "a", "prompt": "ab", "span": [0, 1]}]
    first = run_interventions(model, tokenizer, records, 0, basis, max_new_tokens=2)
    repeated = run_interventions(model, tokenizer, records, 0, basis, max_new_tokens=2)
    assert first == repeated
    generated = {row["condition"]: row for row in first}
    assert generated["baseline"]["generated_token_ids"] == [3, 3]
    assert generated["erase_F"]["generated_token_ids"] == [4, 4]
    assert generated["sham"]["generated_token_ids"] == [3, 3]
    assert all(row["causal_generation_eligible"] for row in first)
    assert model.training
    assert not any(block._forward_hooks for block in model.transformer.h)

    final_basis = InterventionBasis(basis.basis, basis.center, "tiny-real-decoder", 1)
    with pytest.raises(ValueError, match="final decoder block is ineligible"):
        run_interventions(model, tokenizer, records, 1, final_basis, max_new_tokens=2)
    assert model.training
    assert not any(block._forward_hooks for block in model.transformer.h)
    diagnostic = run_interventions(model, tokenizer, records, 1, final_basis, max_new_tokens=2,
                                   allow_final_block_diagnostic=True)
    diagnostic_generated = {row["condition"]: row for row in diagnostic}
    assert diagnostic_generated["baseline"]["generated_token_ids"] == diagnostic_generated["erase_F"]["generated_token_ids"]
    assert not any(row["causal_generation_eligible"] for row in diagnostic)
    assert {row["intervention_scope"] for row in diagnostic} == {"final_block_diagnostic_only"}
    assert model.training
    assert not any(block._forward_hooks for block in model.transformer.h)
