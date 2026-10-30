"""Exact-span hidden states, content-addressed caches, bounded host/GPU memory.

Only requested layers are pooled, one at a time; no B x L x T x D stack.
Truncated, absent, or ambiguous spans never turn into sentence features.
"""
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXTRACTOR_VERSION = "v2.1"


class FeatureCacheError(ValueError):
    """A cache exists but cannot safely supply the requested feature rows."""


def _update(progress, stage, completed, total, message, **context):
    if progress is not None:
        progress.update(stage=stage, completed=completed, total=total, message=message, **context)


def _cache_roots(cache_dir, cfg):
    roots = [Path(cache_dir)]
    additional = getattr(cfg, "feature_cache_dirs", None) or []
    if isinstance(additional, (str, Path)):
        additional = [additional]
    for value in additional:
        path = Path(value).expanduser()
        if path.resolve() not in {p.resolve() for p in roots}:
            roots.append(path)
    return roots


def _load_cache(path, items, expected_layers, expected_width=None, *, progress=None):
    """Read and validate a cache without altering any existing file."""
    try:
        meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        arr = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
        info = pd.read_csv(path / "pooling.csv", keep_default_na=False, dtype={"item_id": str})
    except (OSError, ValueError, KeyError) as exc:
        raise FeatureCacheError(f"Cannot read feature cache {path}: {exc}") from exc
    if "item_id" not in info or "valid" not in info:
        raise FeatureCacheError(f"Feature cache {path} lacks item_id/valid pooling columns")
    if meta.get("item_order_hash") != fingerprint(items, {}):
        raise FeatureCacheError(f"Feature cache {path} effective input hash does not match current text/span/pooling")
    if info.item_id.tolist() != items.item_id.astype(str).tolist():
        raise FeatureCacheError(f"Feature cache {path} item_id order does not match the requested inputs")
    if arr.ndim != 3 or arr.shape[:2] != (len(items), len(expected_layers)):
        raise FeatureCacheError(f"Feature cache {path} has incompatible shape {arr.shape}")
    if expected_width is not None and arr.shape[-1] != expected_width:
        raise FeatureCacheError(f"Feature cache {path} has hidden width {arr.shape[-1]}, expected {expected_width}")
    if meta.get("shape") != list(arr.shape) or meta.get("layers") != list(expected_layers):
        raise FeatureCacheError(f"Feature cache {path} manifest shape/layers disagree with its feature array")
    flags = info.valid.astype(str).str.casefold()
    if not flags.isin(["true", "false"]).all():
        raise FeatureCacheError(f"Feature cache {path} has invalid pooling validity flags")
    info["valid"] = flags.eq("true")
    if "status" in info and not np.array_equal(info.valid.to_numpy(), info.status.eq("ok").to_numpy()):
        raise FeatureCacheError(f"Feature cache {path} validity/status columns disagree")
    if "raw_span_found" in items and "status" in info:
        raw_missing = items.pool_target.eq("span") & ~items.raw_span_found.astype(bool)
        if (info.loc[raw_missing.to_numpy(), "valid"].any()
                or (items.pool_target.eq("span").to_numpy() & ~raw_missing.to_numpy()
                    & info.status.eq("raw_span_absent").to_numpy()).any()):
            raise FeatureCacheError(f"Feature cache {path} has incompatible raw-span validity")
    valid = info.valid.to_numpy()
    # Bound the temporary array while checking even multi-gigabyte caches.
    for start in range(0, len(items), 256):
        stop = min(start + 256, len(items))
        if not np.isfinite(arr[start:stop][valid[start:stop]]).all():
            raise FeatureCacheError(f"Feature cache {path} contains nonfinite values in valid rows {start}:{stop}")
        _update(progress, "cache_validation", stop, len(items), "Validating cached feature rows", model=meta.get("name"))
    return arr, info, meta


def _compatible_manifest(meta, settings, item_hashes, *, cache_only, automatic_precision, requested_revision):
    """Prove input/pooling and inference settings equal before legacy-key reuse."""
    required = ("name", "max_length", "layers", "normalization", "normalizer_code_hash", "representation_space")
    if any(meta.get(key) != settings.get(key) for key in required):
        return False
    if meta.get("item_order_hash") not in item_hashes:
        return False
    old_spec, new_spec = meta.get("model_spec", {}), settings.get("model_spec", {})
    if old_spec.get("kind", "encoder") != new_spec.get("kind", "encoder"):
        return False
    if old_spec.get("quantization") != new_spec.get("quantization"):
        return False
    if not (cache_only and automatic_precision) and meta.get("dtype") != settings.get("dtype"):
        return False
    if not cache_only and meta.get("framework") != settings.get("framework"):
        return False
    same_revision = meta.get("resolved_revision") == settings.get("resolved_revision")
    # Explicitly cache-only 'main' means use recorded weights, even if the Hub's
    # current main changed. A pinned requested commit is never silently replaced.
    return bool(meta.get("resolved_revision")) and (same_revision or (cache_only and requested_revision == "main"))


def normalize_inputs(items, normalize):
    """Normalize complete input and target together, retaining raw-span validity."""
    result = items.copy()
    result["raw_span_found"] = [bool(span and str(span) in text) for text, span in zip(items.text, items.span)]
    result["text"] = items.text.map(normalize)
    result["span"] = items.span.map(lambda s: normalize(s) if s else "")
    result["normalization_changed"] = (result.text != items.text) | (result.span != items.span)
    return result


def fingerprint(items, settings):
    cols = [c for c in ["item_id", "condition", "text", "span", "pool_target"] if c in items]
    payload = {"version": EXTRACTOR_VERSION, "settings": settings,
               "items": items[cols].fillna("").to_dict("records")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def pooling_mask(text, span, offsets, attention, pool_target="span"):
    """Return mask and a diagnostic; require complete coverage of the intended span."""
    offsets = np.asarray(offsets)
    real = (offsets[:, 1] > offsets[:, 0]) & np.asarray(attention, dtype=bool)
    if pool_target == "sentence":
        end = max((b for a, b in offsets[real]), default=0)
        # Any non-whitespace content after the last represented character means truncation.
        if text[end:].strip():
            return np.zeros_like(real), "sentence_truncated"
        return real, "ok" if real.any() else "empty"
    start = text.find(str(span)) if span else -1
    if start < 0:
        return np.zeros_like(real), "span_absent"
    if text.find(str(span), start + len(str(span))) >= 0:
        return np.zeros_like(real), "span_ambiguous"
    stop = start + len(str(span))
    hit = real & (offsets[:, 0] < stop) & (offsets[:, 1] > start)
    if pool_target == "context":
        end = max((b for a, b in offsets[real]), default=0)
        if text[end:].strip():
            return np.zeros_like(real), "context_truncated"
        mask = real & ~hit
        return mask, "ok" if mask.any() else "empty_context"
    if not hit.any() or offsets[hit, 1].max() < stop or offsets[hit, 0].min() > start:
        return np.zeros_like(real), "span_truncated"
    return hit, "ok"


def extract_features(items, spec, cfg, cache_dir, *, progress=None):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    name, revision = spec["name"], spec.get("revision", "main")
    model_cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=False)
    resolved = getattr(model_cfg, "_commit_hash", None) or revision
    if Path(name).exists():
        # Local weights have no Hub commit; hash files so edits cannot reuse stale features.
        digest = hashlib.sha256()
        for f in sorted(Path(name).rglob("*")):
            if f.is_file():
                digest.update(str(f.relative_to(name)).encode())
                with f.open("rb") as src:
                    for chunk in iter(lambda: src.read(1 << 20), b""): digest.update(chunk)
        resolved = digest.hexdigest()
    n_layers = getattr(model_cfg, "num_hidden_layers", getattr(model_cfg, "n_layer", None))
    if n_layers is None:
        raise ValueError(f"Cannot resolve hidden layer count for {name}")
    layers = cfg.layers if cfg.layers is not None else sorted(set(range(0, n_layers + 1, cfg.layer_step)) | {n_layers})
    if any(l < 0 or l > n_layers for l in layers):
        raise ValueError(f"layers must be between 0 and {n_layers}")
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    if (cfg.precision == "auto" and spec.get("kind") == "decoder" and device != "cpu"
            and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        # Decoder LMs (e.g. Qwen3) are trained in bf16 and can overflow in fp16.
        # Encoders keep fp16 so that existing feature caches stay valid.
        dtype = torch.bfloat16
    if cfg.precision != "auto": dtype = getattr(torch, cfg.precision)
    normalization = spec.get("normalization", "none")
    norm_hash = None
    original_items = items
    if normalization == "csebuetnlp":
        import normalizer
        from normalizer import normalize
        package = Path(normalizer.__file__).parent
        digest = hashlib.sha256()
        for file in sorted(package.rglob("*.py")): digest.update(file.read_bytes())
        norm_hash = digest.hexdigest()
        items = normalize_inputs(items, normalize)
    elif normalization != "none":
        raise ValueError(f"Unsupported normalization: {normalization}")
    settings = dict(name=name, resolved_revision=resolved, max_length=cfg.max_length,
                    layers=layers, dtype=str(dtype), normalization=normalization,
                    normalizer_code_hash=norm_hash, framework=torch.__version__,
                    model_spec=spec, representation_space=("decoder_block_output" if spec.get("kind") == "decoder"
                                                            else "hidden_states"))
    key = fingerprint(original_items, settings)
    dest = Path(cache_dir) / key
    manifest = dest / "manifest.json"
    cache_only = bool(getattr(cfg, "cache_only", False) or getattr(cfg, "analysis_only", False))
    roots = _cache_roots(cache_dir, cfg)
    width = getattr(model_cfg, "hidden_size", getattr(model_cfg, "n_embd", None))
    direct = []
    for root in roots:
        for candidate in ([root] if (root / "manifest.json").is_file() else [root / key]):
            if (candidate / "manifest.json").is_file() and candidate.resolve() not in {p.resolve() for p in direct}:
                direct.append(candidate)
    for candidate in direct:
        metadata = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != key:
            # A direct leaf from an external root may need the settings proof below.
            if candidate.name == key:
                raise FeatureCacheError(f"Feature cache manifest fingerprint mismatch: {candidate}")
            continue
        arr, info, meta = _load_cache(candidate, items, layers, width, progress=progress)
        meta = dict(meta, model_layer_count=n_layers, cache_reused_from=str(candidate.resolve()), cache_match="exact_fingerprint", cache_read_only=True)
        _update(progress, "extract", len(items), len(items), "Reused exact feature cache", model=name,
                cache_path=str(candidate.resolve()))
        return arr, info, meta
    # This also permits explicitly pinning a legacy cache's resolved commit:
    # model_spec.revision changes the key, but not the actual recorded weights.
    candidates = []
    for root in roots:
        manifests = [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
        for file in manifests:
            if file.parent.resolve() not in {p.resolve() for p in candidates}:
                candidates.append(file.parent)
    # Legacy manifests hash the *effective* model inputs: normalized for the
    # Bangla normalizer, original otherwise. Raw text equality alone cannot
    # justify reuse if a normalizer's effective output changed.
    item_hashes = {fingerprint(items, {})}
    matches = []
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _compatible_manifest(metadata, settings, item_hashes, cache_only=cache_only,
                                automatic_precision=cfg.precision == "auto", requested_revision=revision):
            matches.append((candidate, metadata))
    revisions = {meta["resolved_revision"] for _, meta in matches}
    if len(revisions) > 1:
        raise FeatureCacheError(f"Multiple compatible cached revisions for {name}: {sorted(revisions)}; pin a model revision explicitly")
    if matches:
        candidate, metadata = matches[0]
        arr, info, meta = _load_cache(candidate, items, layers, width, progress=progress)
        meta = dict(meta, model_layer_count=n_layers, cache_reused_from=str(candidate.resolve()), cache_match="verified_settings_and_inputs",
                    cache_read_only=True, requested_resolved_revision=resolved, requested_dtype=str(dtype),
                    runtime_framework=torch.__version__,
                    cache_revision_override=meta["resolved_revision"] != resolved)
        _update(progress, "extract", len(items), len(items), "Reused compatible recorded feature cache", model=name,
                cache_path=str(candidate.resolve()), resolved_revision=meta["resolved_revision"])
        return arr, info, meta
    if cache_only:
        raise FileNotFoundError(f"No compatible feature cache for {name}; cache-only mode will not load model weights or extract. "
                                f"Searched: {[str(root) for root in roots]}")
    dest.mkdir(parents=True, exist_ok=True)
    pinned_revision = resolved if not Path(name).exists() else revision
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision, use_fast=True, trust_remote_code=False)
    if not tok.is_fast: raise ValueError("Fast tokenizer required for character span alignment")
    if tok.pad_token_id is None:
        if tok.eos_token_id is None: raise ValueError("Tokenizer has neither pad nor EOS token")
        tok.pad_token = tok.eos_token
    kwargs = dict(revision=pinned_revision, torch_dtype=dtype, trust_remote_code=False)
    if spec.get("quantization"):
        from transformers import BitsAndBytesConfig
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
    model = AutoModel.from_pretrained(name, **kwargs)
    if not spec.get("quantization"): model.to(device)
    model.eval()
    rows, arr = [], None
    try:
        try:
            from .progress import progress_iter
        except ImportError:
            def progress_iter(iterable, **kwargs):
                return iterable
        batches = range(0, len(items), cfg.batch_size)
        # The job reporter already echoes a status line; avoid a second, interleaved bar.
        show_bar = progress is None and getattr(cfg, "progress", True)
        for start in progress_iter(batches, desc=f"Extract {name}", total=len(batches), unit="batch", leave=False,
                                   enabled=show_bar):
            batch = items.iloc[start:start + cfg.batch_size]
            enc = tok(batch.text.tolist(), padding=True, truncation=True, max_length=cfg.max_length,
                      return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").numpy()
            attention = enc["attention_mask"].numpy()
            masks = []
            for j, (_, row) in enumerate(batch.iterrows()):
                mask, status = pooling_mask(row.text, row.span, offsets[j], attention[j], row.pool_target)
                if row.pool_target == "span" and not row.get("raw_span_found", True):
                    mask, status = np.zeros_like(mask), "raw_span_absent"
                masks.append(mask)
                ids = enc["input_ids"][j].numpy()
                rows.append(dict(item_id=row.item_id, valid=status == "ok", status=status,
                                 token_count=int(mask.sum()), full_token_count=int(attention[j].sum()),
                                 normalization_changed=bool(row.get("normalization_changed", False)),
                                 unk_count=int(((ids == tok.unk_token_id) & attention[j].astype(bool)).sum())
                                 if tok.unk_token_id is not None else 0))
            enc = {k: v.to(model.device) for k, v in enc.items()}
            def save_pooled(h, layer):
                    nonlocal arr
                    if arr is None:
                        arr = np.lib.format.open_memmap(dest / "features.npy", mode="w+", dtype=np.float32,
                                                       shape=(len(items), len(layers), h.shape[-1]))
                    mask = torch.as_tensor(np.array(masks), device=h.device, dtype=torch.float32)
                    pooled = (h.float() * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
                    result = pooled.cpu().numpy()
                    result[mask.sum(1).cpu().numpy() == 0] = np.nan
                    arr[start:start + len(batch), layers.index(layer)] = result
            with torch.inference_mode():
                if spec.get("kind") == "decoder":
                    from .interventions import resolve_decoder_layers
                    blocks, _ = resolve_decoder_layers(model)
                    handles = []
                    try:
                        if 0 in layers:
                            handles.append(blocks[0].register_forward_pre_hook(lambda mod, args: save_pooled(args[0], 0)))
                        for layer in layers:
                            if layer == 0: continue
                            def hook(mod, args, output, index=layer):
                                h = output[0] if isinstance(output, (list, tuple)) else output
                                save_pooled(h, index)
                            handles.append(blocks[layer-1].register_forward_hook(hook))
                        out = model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)
                    finally:
                        for handle in handles: handle.remove()
                else:
                    out = model(**enc, output_hidden_states=True, return_dict=True)
                    hs = out.hidden_states
                    if hs is None or len(hs) <= max(layers): raise ValueError("Model did not return requested hidden states")
                    for layer in layers: save_pooled(hs[layer], layer)
                    del hs
            del out
            _update(progress, "extract", start + len(batch), len(items), "Extracting hidden states", model=name)
        if arr is None: raise ValueError("No items selected for extraction")
        arr.flush()
        info = pd.DataFrame(rows)
        info.to_csv(dest / "pooling.csv", index=False)
        meta = dict(settings, fingerprint=key, shape=list(arr.shape), model_layer_count=n_layers,
                    tokenizer_class=type(tok).__name__, item_order_hash=fingerprint(items, {}))
        manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.load(dest / "features.npy", mmap_mode="r"), info, meta
