"""Optional causal generation probes with decoder-block output interventions.

Nothing is downloaded or loaded on import. Supply an already loaded causal
language model, a fast tokenizer, character spans, and bases fitted on training
data in *this exact decoder block output space*. Encoder/PCA coordinates cannot
be passed directly as a decoder activation basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class InterventionBasis:
    """Column-orthonormal ambient basis [hidden_dimension, rank] and center.

    Metadata is optional for direct programmatic use; when supplied, model_id
    and layer_index are checked against the model and selected block.
    """

    basis: Any
    center: Any
    model_id: str | None = None
    layer_index: int | None = None
    space: str = "decoder_block_output"


def _basis(value: InterventionBasis | dict | np.ndarray | torch.Tensor,
           hidden_size: int, layer_index: int, model_id: str | None) -> InterventionBasis:
    if isinstance(value, InterventionBasis):
        candidate = value
    elif isinstance(value, dict):
        if "basis" not in value or "center" not in value:
            raise ValueError("Basis dictionaries require basis[D, rank] and center[D].")
        candidate = InterventionBasis(**{k: value[k] for k in
                                         ("basis", "center", "model_id", "layer_index", "space") if k in value})
    else:
        # An uncentered array is explicit zero-centering, never a guessed PCA
        # mean. Persisted analysis artifacts should always include their center.
        candidate = InterventionBasis(value, np.zeros(hidden_size))
    matrix = torch.as_tensor(candidate.basis).detach().to(device="cpu", dtype=torch.float64)
    center = torch.as_tensor(candidate.center).detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != hidden_size or not 0 < matrix.shape[1] <= hidden_size:
        raise ValueError(f"Basis must have shape [{hidden_size}, rank] with 1 <= rank <= {hidden_size}.")
    if center.shape != (hidden_size,):
        raise ValueError(f"Basis center must have shape [{hidden_size}].")
    if not torch.isfinite(matrix).all() or not torch.isfinite(center).all():
        raise ValueError("Basis and center must contain finite numbers.")
    gram = matrix.T @ matrix
    if not torch.allclose(gram, torch.eye(matrix.shape[1], dtype=gram.dtype), atol=1e-4, rtol=1e-4):
        raise ValueError("Basis columns must be orthonormal; rows are hidden dimensions.")
    if candidate.space != "decoder_block_output":
        raise ValueError("Intervention basis must be in decoder_block_output ambient coordinates.")
    if candidate.layer_index is not None and candidate.layer_index != layer_index:
        raise ValueError("Basis layer_index does not match the selected decoder block.")
    if candidate.model_id is not None and model_id is not None and candidate.model_id != model_id:
        raise ValueError("Basis model_id does not match the supplied causal model.")
    return InterventionBasis(matrix, center, candidate.model_id, layer_index, candidate.space)


def resolve_decoder_layers(model: Any) -> tuple[Sequence[Any], str]:
    """Resolve common Hugging Face block layouts, returning the exact path."""
    paths = ("model.layers", "transformer.h", "model.decoder.layers", "decoder.layers",
             "gpt_neox.layers", "base_model.model.layers", "base_model.transformer.h",
             "base_model.layers", "base_model.model.decoder.layers", "layers", "h")
    for path in paths:
        value = model
        for component in path.split("."):
            value = getattr(value, component, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__") and len(value):
            if all(hasattr(layer, "register_forward_hook") for layer in value):
                return value, path
    raise ValueError("Unsupported decoder layout. Expected model.layers, transformer.h, or a supported base_model wrapper.")


def _hidden(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError("Decoder block must return hidden states [batch, tokens, hidden] or a tuple starting with them.")
    return value


def _replace_hidden(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement,) + output[1:]
    if isinstance(output, list):
        return [replacement] + output[1:]
    return replacement


def transform_activations(hidden: torch.Tensor, token_indices: Sequence[int],
                          basis: InterventionBasis, *, operation: str,
                          donor: torch.Tensor | None = None) -> torch.Tensor:
    """Return a copy with only specified prefill tokens erased or patched.

    Row-vector convention: erase x -= ((x-center) @ B) @ B.T;
    patch x += ((donor-x) @ B) @ B.T. A donor vector is the mean over its
    explicitly supplied idiom span, allowing spans of different token lengths.
    """
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Interventions require a single sequence [1, tokens, hidden].")
    if not token_indices or len(set(token_indices)) != len(token_indices):
        raise ValueError("Provide distinct target token positions.")
    if min(token_indices) < 0 or max(token_indices) >= hidden.shape[1]:
        raise ValueError("Target token index is outside the prefill sequence.")
    matrix = torch.as_tensor(basis.basis, device=hidden.device, dtype=hidden.dtype)
    center = torch.as_tensor(basis.center, device=hidden.device, dtype=hidden.dtype)
    selected = hidden[:, list(token_indices), :]
    if operation == "erase":
        changed = selected - ((selected - center) @ matrix) @ matrix.T
    elif operation == "patch":
        if donor is None:
            raise ValueError("Patching requires donor span activations.")
        donor_value = donor.to(device=hidden.device, dtype=hidden.dtype)
        if donor_value.shape != (hidden.shape[-1],):
            raise ValueError("Donor mean must have one value per hidden dimension.")
        changed = selected + ((donor_value - selected) @ matrix) @ matrix.T
    elif operation == "sham":
        changed = selected
    else:
        raise ValueError(f"Unknown intervention operation: {operation}")
    result = hidden.clone()
    result[:, list(token_indices), :] = changed
    return result


def _device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _tokenize(tokenizer: Any, prompt: str, span: Sequence[int], device: torch.device) -> tuple[dict, list[int]]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Each intervention record requires a nonempty prompt.")
    if not isinstance(span, (list, tuple)) or len(span) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in span):
        raise ValueError("Span must be [start, end] character offsets; no guessed-span fallback is used.")
    start, end = int(span[0]), int(span[1])
    if start < 0 or end > len(prompt) or start >= end or not prompt[start:end].strip():
        raise ValueError("Span must select non-whitespace characters inside the prompt.")
    if getattr(tokenizer, "is_fast", True) is False:
        raise ValueError("A fast tokenizer with character offset mappings is required.")
    try:
        encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    except (NotImplementedError, TypeError) as exc:
        raise ValueError("Tokenizer must provide actual character offset mappings.") from exc
    if "offset_mapping" not in encoded or "input_ids" not in encoded:
        raise ValueError("Tokenizer returned no offset mapping or input IDs.")
    offsets = encoded["offset_mapping"]
    offsets = offsets.tolist() if hasattr(offsets, "tolist") else offsets
    if len(offsets) != 1:
        raise ValueError("Intervention tokenization must contain exactly one sequence.")
    offsets = offsets[0]
    positions = [i for i, (left, right) in enumerate(offsets) if right > left and right > start and left < end]
    if not positions:
        raise ValueError("The supplied idiom span overlaps no actual tokens.")
    uncovered = [i for i in range(start, end) if not prompt[i].isspace() and
                 not any(int(left) <= i < int(right) for left, right in (offsets[j] for j in positions))]
    if uncovered:
        raise ValueError("Tokenizer offsets do not cover all characters in the idiom span.")
    inputs = {name: value.to(device) for name, value in encoded.items()
              if name not in ("offset_mapping", "special_tokens_mask") and isinstance(value, torch.Tensor)}
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Input IDs must have shape [1, tokens].")
    if inputs["input_ids"].shape[1] != len(offsets):
        raise ValueError("Offset mapping length does not match token sequence length.")
    return inputs, positions


def _capture_donor(model: Any, layer: Any, inputs: dict, positions: Sequence[int]) -> torch.Tensor:
    captured = []
    sequence_length = inputs["input_ids"].shape[1]

    def capture(_module, _args, output):
        hidden = _hidden(output)
        if hidden.shape[:2] != (1, sequence_length):
            raise ValueError("Donor decoder output does not match its prefill token sequence.")
        captured.append(hidden[0, list(positions), :].detach().float().mean(dim=0).cpu())

    handle = layer.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f"Expected exactly one donor prefill block call; observed {len(captured)}.")
    return captured[0]


def _generate(model: Any, tokenizer: Any, layer: Any, inputs: dict, positions: Sequence[int],
              basis: InterventionBasis | None, operation: str, donor: torch.Tensor | None,
              max_new_tokens: int) -> dict:
    sequence_length = inputs["input_ids"].shape[1]
    state = {"calls": 0, "edited_prefills": 0}

    def intervene(_module, _args, output):
        hidden = _hidden(output)
        state["calls"] += 1
        # Only the first block evaluation is eligible. During cached decoding a
        # block sees one new token; with no cache it sees growing full prefixes.
        if state["calls"] != 1:
            return output
        if hidden.shape[:2] != (1, sequence_length):
            raise ValueError("The first generation call was not the expected full prefill.")
        state["edited_prefills"] += 1
        replacement = transform_activations(hidden, positions, basis, operation=operation, donor=donor)
        return _replace_hidden(output, replacement)

    handle = layer.register_forward_hook(intervene) if basis is not None else None
    kwargs = dict(inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                  num_beams=1, return_dict_in_generate=True)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    try:
        with torch.inference_mode():
            result = model.generate(**kwargs)
    finally:
        if handle is not None:
            handle.remove()
    if basis is not None and state["edited_prefills"] != 1:
        raise ValueError("No prefill intervention was applied; generation results are invalid.")
    sequences = result.sequences if hasattr(result, "sequences") else result
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise ValueError("Generation must return a single token sequence.")
    completion = sequences[0, sequence_length:].detach().cpu().tolist()
    return {"generation": tokenizer.decode(completion, skip_special_tokens=True),
            "generated_token_ids": completion, "prefill_interventions": state["edited_prefills"],
            "decoder_block_calls": state["calls"]}


def run_interventions(model: Any, tokenizer: Any, records: Sequence[dict], layer_index: int,
                      figurative_basis: InterventionBasis | dict | np.ndarray | torch.Tensor,
                      literal_basis: InterventionBasis | dict | np.ndarray | torch.Tensor | None = None,
                      seed: int = 0, max_new_tokens: int = 64, *,
                      allow_final_block_diagnostic: bool = False) -> list[dict]:
    """Generate baseline, erase-F/L, random-rank, patch-F and sham conditions.

    Every record needs idiom_id, prompt, and span=[character_start,character_end].
    A donor_prompt and donor_span are required for patch-F. Without them patch-F
    is explicitly recorded as skipped. Raw generations require independent
    scoring; this function never invents an idiom-understanding score.
    Layer indices are zero-based raw decoder block outputs. Final-normalized
    HF hidden_states can differ; fit bases with the matching raw-output hooks.
    The final decoder block is excluded: edits before the last prompt token
    cannot influence generation through later attention or cached keys/values.
    An explicit final-block diagnostic override labels its outputs ineligible
    for the ordinary causal generation comparison. F/L erasures require equal
    rank; choose the same validation-selected rank before fitting both bases.
    """
    if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
        raise ValueError("layer_index must be a non-negative zero-based block index.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        raise ValueError("This workflow supports decoder-only causal language models.")
    layers, layer_path = resolve_decoder_layers(model)
    if layer_index >= len(layers):
        raise ValueError(f"layer_index {layer_index} is outside {layer_path} ({len(layers)} blocks).")
    if not isinstance(allow_final_block_diagnostic, bool):
        raise ValueError("allow_final_block_diagnostic must be an explicit boolean.")
    final_block = layer_index == len(layers) - 1
    if final_block and not allow_final_block_diagnostic:
        raise ValueError("The final decoder block is ineligible for prefill-span generation interventions: "
                         "its output edits cannot propagate through later attention or cached keys/values. "
                         "Supply a basis fitted at a nonfinal decoder block, or explicitly request a "
                         "final-block diagnostic with allow_final_block_diagnostic=True.")
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    if hidden_size is None:
        raise ValueError("Model config must provide hidden_size or n_embd to validate basis coordinates.")
    model_id = getattr(config, "_name_or_path", None) or None
    figurative = _basis(figurative_basis, hidden_size, layer_index, model_id)
    literal = _basis(literal_basis, hidden_size, layer_index, model_id) if literal_basis is not None else None
    if literal is not None and literal.basis.shape[1] != figurative.basis.shape[1]:
        raise ValueError("Figurative and literal erasure bases must have matched rank for a comparable intervention.")
    generator = np.random.default_rng(seed)
    random_matrix, _ = np.linalg.qr(generator.standard_normal((hidden_size, figurative.basis.shape[1])))
    random_basis = InterventionBasis(random_matrix, figurative.center, model_id, layer_index)
    device = _device(model)
    layer = layers[layer_index]
    training = getattr(model, "training", False)
    results = []
    model.eval()
    try:
        seen = set()
        for record in records:
            if "idiom_id" not in record or "prompt" not in record or "span" not in record:
                raise ValueError("Every intervention record requires idiom_id, prompt and character span.")
            record_id = str(record.get("record_id", record["idiom_id"]))
            if record_id in seen:
                raise ValueError("Intervention record_id (or idiom_id) must be unique.")
            seen.add(record_id)
            inputs, positions = _tokenize(tokenizer, record["prompt"], record["span"], device)
            if "span_text" in record and record["prompt"][record["span"][0]:record["span"][1]] != record["span_text"]:
                raise ValueError(f"span_text does not match actual prompt offsets for {record_id}.")
            donor = None
            donor_prompt = record.get("donor_prompt")
            donor_span = record.get("donor_span")
            prompt_present = not (donor_prompt is None or isinstance(donor_prompt, str) and not donor_prompt.strip())
            span_present = not (donor_span is None or isinstance(donor_span, (list, tuple, str)) and len(donor_span) == 0)
            has_donor = prompt_present or span_present
            donor_positions = None
            if has_donor:
                if not prompt_present or not span_present:
                    raise ValueError("Patching requires both donor_prompt and donor_span.")
                donor_inputs, donor_positions = _tokenize(tokenizer, record["donor_prompt"], record["donor_span"], device)
                donor = _capture_donor(model, layer, donor_inputs, donor_positions)
            conditions = [("baseline", None, "sham", None), ("erase_F", figurative, "erase", None),
                          ("erase_L", literal, "erase", None),
                          ("random_rank_matched", random_basis, "erase", None),
                          ("patch_F", figurative, "patch", donor), ("sham", figurative, "sham", None)]
            metadata = {"record_id": record_id, "idiom_id": str(record["idiom_id"]), "model": model_id,
                        "prompt": record["prompt"], "span": list(record["span"]), "target_token_indices": positions,
                        "layer_index": layer_index, "layer_path": f"{layer_path}.{layer_index}",
                        "activation_space": "decoder_block_output", "basis_rank_F": figurative.basis.shape[1],
                        "basis_rank_L": literal.basis.shape[1] if literal is not None else None,
                        "seed": seed, "max_new_tokens": max_new_tokens,
                        "causal_generation_eligible": not final_block,
                        "intervention_scope": "final_block_diagnostic_only" if final_block else "nonfinal_block_prefill",
                        "generation_policy": "greedy; num_beams=1; model.eval; prefill idiom span only"}
            for condition, basis, operation, donor_value in conditions:
                row = dict(metadata, condition=condition)
                if condition == "erase_L" and literal is None:
                    results.append(dict(row, status="skipped", reason="No literal basis supplied."))
                elif condition == "patch_F" and donor is None:
                    results.append(dict(row, status="skipped", reason="No donor prompt/span supplied."))
                else:
                    generated = _generate(model, tokenizer, layer, inputs, positions, basis, operation, donor_value, max_new_tokens)
                    if condition == "patch_F":
                        row.update(donor_prompt=record["donor_prompt"], donor_span=list(record["donor_span"]),
                                   donor_token_indices=donor_positions, donor_pooling="mean of explicit donor idiom span")
                    results.append(dict(row, status="generated", **generated))
    finally:
        model.train(training)
    return results
