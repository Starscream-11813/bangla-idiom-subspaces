"""Actual local tiny-transformer extraction: no network or pretrained weights."""
from pathlib import Path
import hashlib
import json
import sys
import types
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from idiom_subspace.config import Config
from idiom_subspace.extraction import FeatureCacheError, extract_features, fingerprint, pooling_mask, normalize_inputs


def test_normalization_keeps_raw_span_provenance_and_normalizes_both_sides():
    items = pd.DataFrame(dict(text=["one  two", "three"], span=["one  two", "THREE"]))
    result = normalize_inputs(items, lambda s: " ".join(s.lower().split()))
    assert result.text.tolist() == ["one two", "three"]
    assert result.span.tolist() == ["one two", "three"]
    assert result.raw_span_found.tolist() == [True, False]
    assert result.normalization_changed.all()


def test_pooling_never_falls_back_or_accepts_truncated_spans():
    offsets = np.array([[0, 0], [0, 3], [4, 7], [8, 10], [0, 0]])
    attention = np.ones(5)
    mask, status = pooling_mask("one two xx", "two", offsets, attention)
    assert status == "ok" and np.array_equal(mask, [False, False, True, False, False])
    for text, span, expected in [("one two xx", "absent", "span_absent"),
                                 ("one two xx", "xx tail", "span_absent"),
                                 ("one two xx tail", "xx tail", "span_truncated"),
                                 ("one two two", "two", "span_ambiguous")]:
        mask, status = pooling_mask(text, span, offsets, attention)
        assert status == expected and not mask.any()
    mask, status = pooling_mask("one two xx tail", "", offsets, attention, "sentence")
    assert status == "sentence_truncated" and not mask.any()
    mask, status = pooling_mask("one two xx", "two", offsets, attention, "context")
    assert status == "ok" and mask.sum() == 2 and not mask[2]


def test_cache_includes_text_order_pooling_and_quantization():
    items = pd.DataFrame(dict(item_id=["a", "b"], text=["one", "two"], span=["one", "two"],
                              condition=["bn-fig"]*2, pool_target=["span"]*2))
    a = fingerprint(items, {"revision": "one"})
    assert a != fingerprint(items.iloc[::-1], {"revision": "one"})
    assert a != fingerprint(items.assign(text=["new", "two"]), {"revision": "one"})
    assert a != fingerprint(items, {"revision": "two"})


def _cache_request(tmp_path, monkeypatch):
    """A real on-disk legacy-format cache with no model/tokenizer loading."""
    import torch
    import transformers
    items = pd.DataFrame(dict(item_id=["a", "b", "c"], condition=["bn-fig"]*3,
                              text=["one two", "one two", "one three"], span=["two", "absent", "three"],
                              pool_target=["span"]*3))
    spec = dict(name="example/model", revision="main", kind="encoder")
    cfg = Config(device="cpu", batch_size=2, max_length=8, layers=[0, 1, 2])
    cfg.feature_cache_dirs, cfg.cache_only = [str(tmp_path / "old-cache")], True
    settings = dict(name=spec["name"], resolved_revision="old-commit", max_length=8, layers=[0, 1, 2],
                    dtype="torch.float32", normalization="none", normalizer_code_hash=None,
                    framework=torch.__version__, model_spec=spec.copy(), representation_space="hidden_states")
    key = fingerprint(items, settings)
    cache = tmp_path / "old-cache" / key
    cache.mkdir(parents=True)
    values = np.ones((3, 3, 8), dtype=np.float32)
    values[1] = np.nan
    np.save(cache / "features.npy", values)
    pd.DataFrame(dict(item_id=items.item_id, valid=[True, False, True], status=["ok", "span_absent", "ok"],
                      token_count=[1, 0, 1])).to_csv(cache / "pooling.csv", index=False)
    meta = dict(settings, fingerprint=key, shape=list(values.shape), tokenizer_class="test",
                item_order_hash=fingerprint(items, {}))
    (cache / "manifest.json").write_text(json.dumps(meta), encoding="utf8")
    state = {"current_revision": "old-commit"}
    def config(name, revision="main", **kwargs):
        return SimpleNamespace(num_hidden_layers=2, hidden_size=8,
                               _commit_hash=state["current_revision"] if revision == "main" else revision)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    def no_weights(*args, **kwargs):
        raise AssertionError("Cache requests must not load model/tokenizer weights")
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", no_weights)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_weights)
    return items, spec, cfg, cache, tmp_path / "new-cache", state


def test_external_legacy_cache_reused_read_only_with_provenance_and_progress(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in cache.iterdir()}
    updates = []
    reporter = SimpleNamespace(update=lambda **value: updates.append(value))
    values, pooling, meta = extract_features(items, spec, cfg, destination, progress=reporter)
    assert not values.flags.writeable and pooling.item_id.tolist() == items.item_id.tolist()
    assert meta["cache_match"] == "exact_fingerprint"
    assert Path(meta["cache_reused_from"]) == cache.resolve()
    assert updates[-1]["stage"] == "extract" and updates[-1]["completed"] == len(items)
    assert before == {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in cache.iterdir()}
    assert not destination.exists()


def test_pinned_revision_can_find_compatible_legacy_main_key(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    spec["revision"] = "old-commit"
    _, _, meta = extract_features(items, spec, cfg, destination)
    assert meta["cache_match"] == "verified_settings_and_inputs"
    assert meta["resolved_revision"] == "old-commit"
    assert not meta["cache_revision_override"]


def test_cache_only_main_drift_is_explicit_and_pinned_mismatch_is_rejected(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, state = _cache_request(tmp_path, monkeypatch)
    state["current_revision"] = "new-main-commit"
    _, _, meta = extract_features(items, spec, cfg, destination)
    assert meta["resolved_revision"] == "old-commit"
    assert meta["requested_resolved_revision"] == "new-main-commit"
    assert meta["cache_revision_override"]
    spec["revision"] = "new-main-commit"
    with pytest.raises(FileNotFoundError, match="cache-only"):
        extract_features(items, spec, cfg, destination)


@pytest.mark.parametrize("corruption", ["row_order", "nonfinite", "shape"])
def test_cache_rejects_corrupt_row_identity_shape_or_valid_features(tmp_path, monkeypatch, corruption):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    if corruption == "row_order":
        info = pd.read_csv(cache / "pooling.csv")
        info.iloc[::-1].to_csv(cache / "pooling.csv", index=False)
    else:
        values = np.load(cache / "features.npy")
        if corruption == "nonfinite": values[0, 0, 0] = np.inf
        else: values = values[:, :, :7]
        np.save(cache / "features.npy", values)
    with pytest.raises(FeatureCacheError):
        extract_features(items, spec, cfg, destination)
    assert not destination.exists()


def test_cache_only_miss_never_loads_weights_and_analysis_only_is_alias(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    cfg.cache_only, cfg.analysis_only = False, True
    changed = items.copy()
    changed.loc[0, "text"] = "different context two"
    with pytest.raises(FileNotFoundError, match="will not load model weights"):
        extract_features(changed, spec, cfg, destination)
    assert not destination.exists()


def test_explicit_precision_must_match_cached_model_precision(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    cfg.precision = "float16"
    with pytest.raises(FileNotFoundError, match="cache-only"):
        extract_features(items, spec, cfg, destination)


def test_exact_key_checks_effective_inputs_after_normalizer_dependency_change(tmp_path, monkeypatch):
    items, spec, cfg, cache, destination, _ = _cache_request(tmp_path, monkeypatch)
    package = tmp_path / "normalizer"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("# Unchanged normalizer source; external dependency behavior changed.\n", encoding="utf8")
    fake = types.ModuleType("normalizer")
    fake.__file__ = str(source)
    fake.normalize = lambda text: text.replace("two", "TWO")
    monkeypatch.setitem(sys.modules, "normalizer", fake)
    spec["normalization"] = "csebuetnlp"
    meta = json.loads((cache / "manifest.json").read_text(encoding="utf8"))
    fields = ["name", "resolved_revision", "max_length", "layers", "dtype", "normalization",
              "normalizer_code_hash", "framework", "model_spec", "representation_space"]
    settings = {key: meta[key] for key in fields}
    settings.update(normalization="csebuetnlp", normalizer_code_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                    model_spec=spec.copy())
    exact_key = fingerprint(items, settings)
    meta.update(settings, fingerprint=exact_key)
    # The cached effective texts used the previous dependency's identity output;
    # raw input rows, source-code hash and therefore the extraction key agree.
    relocated = cache.parent / exact_key
    cache.rename(relocated)
    (relocated / "manifest.json").write_text(json.dumps(meta), encoding="utf8")
    with pytest.raises(FeatureCacheError, match="effective input hash"):
        extract_features(items, spec, cfg, destination)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_real_tiny_local_model_and_cache(tmp_path, kind):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, BertConfig, BertModel, GPT2Config, GPT2Model
    model_path = tmp_path / "weights"
    model_path.mkdir()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "one": 2, "two": 3, "three": 4, "four": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tok.save_pretrained(model_path)
    torch.manual_seed(0)
    if kind == "encoder":
        model = BertModel(BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=2,
                                    num_attention_heads=2, intermediate_size=12, max_position_embeddings=32))
    else:
        model = GPT2Model(GPT2Config(vocab_size=6, n_embd=8, n_layer=2, n_head=2, n_positions=32))
    model.save_pretrained(model_path)
    items = pd.DataFrame(dict(item_id=["a", "b", "c"], condition=["bn-fig"]*3,
                              text=["one two three", "one two four", "one two three"],
                              span=["two", "missing", "three"], pool_target=["span"]*3))
    cfg = Config(device="cpu", batch_size=2, max_length=8, layers=[0, 1, 2])
    spec = dict(name=str(model_path), revision="main", kind=kind)
    features, pooling, manifest = extract_features(items, spec, cfg, tmp_path / "cache")
    assert features.shape == (3, 3, 8)
    assert pooling.valid.tolist() == [True, False, True]
    assert np.isnan(features[1]).all() and np.isfinite(features[0]).all()
    again, info, metadata = extract_features(items, spec, cfg, tmp_path / "cache")
    np.testing.assert_allclose(features, again, equal_nan=True)
    assert manifest["fingerprint"] == metadata["fingerprint"]
    assert info.item_id.tolist() == items.item_id.tolist()
