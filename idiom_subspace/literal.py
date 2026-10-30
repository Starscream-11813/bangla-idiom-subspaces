"""Machine translation of Bangla literal glosses into English (optional S4a input).

FLS needs a literal-sense label L in the *same* taxonomy as F, which is fitted
on English figurative glosses. The corpus field ``literal_meaning`` is Bangla,
so L is unavailable unless English literal glosses are supplied. This module
writes a CSV in the reviewed-gloss format (``idiom_id, literal_en, approved``)
with ``approved=machine`` and the translation model as provenance; human
reviewers can correct rows and set ``approved=true``. Nothing is downloaded on
import, and nothing here is used unless the CSV is passed to ``run``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def clean_literal_bn(text: str) -> str:
    """Normalize whitespace and separators; keep parenthetical English hints."""
    text = re.sub(r"\s*\+\s*", " + ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip(" ;,")


def translate_literals(df: pd.DataFrame, model: str = DEFAULT_MODEL, revision: str = "main", *,
                       batch_size: int = 32, device: str = "auto", max_new_tokens: int = 96,
                       num_beams: int = 4, progress=None) -> pd.DataFrame:
    """Translate ``lit_bn`` for every record; returns a reviewable gloss table."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    source_texts = [clean_literal_bn(v) for v in df.lit_bn.fillna("")]
    unique = sorted({t for t in source_texts if t})
    is_nllb = "nllb" in model.lower()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, **({"src_lang": "ben_Beng"} if is_nllb else {}))
    translator = AutoModelForSeq2SeqLM.from_pretrained(model, revision=revision).to(device).eval()
    resolved = getattr(translator.config, "_commit_hash", None) or revision
    generate = dict(max_new_tokens=max_new_tokens, num_beams=num_beams)
    if is_nllb:
        generate["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("eng_Latn")
    translated = {}
    order = np.argsort([len(t) for t in unique])
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = [unique[i] for i in order[start:start + batch_size]]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            output = translator.generate(**encoded, **generate)
            for text, english in zip(batch, tokenizer.batch_decode(output, skip_special_tokens=True)):
                translated[text] = english.strip()
            if progress is not None:
                progress.update(stage="translate_literals", completed=min(start + batch_size, len(unique)),
                                total=len(unique), unit="glosses", message="Machine-translating literal glosses")
    rows = []
    for record, text in zip(df.to_dict("records"), source_texts):
        rows.append(dict(idiom_id=record["idiom_id"], source_id=record.get("source_id", ""), idiom=record.get("idiom", ""),
                         lit_bn=record.get("lit_bn", ""), literal_en=translated.get(text, ""),
                         approved="machine" if translated.get(text) else "false",
                         source=f"{model}@{resolved}", reviewer="", notes=""))
    return pd.DataFrame(rows)
