"""Build a miniature campaign with the exact on-disk layout of the real discovery campaign.

Two tiny random BERT encoders (saved locally, no downloads) encode a synthetic
corpus through the real pipeline functions (load_corpus, split_corpus,
build_labels with TF-IDF, build_items, extract_features), so every file the
experiments read -- items.json, corpus_prepared.json, caches, model_revision.json,
layer_selection.json, preprocess npz, the follow-up prepared corpus -- is real.
Translation-equivalent Bangla/English words share embedding rows, so English
directions carry some Bangla class information and the tests are not degenerate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
CLASSES = [
    (["আকাশ", "মেঘ", "বৃষ্টি", "রোদ"], ["sky", "cloud", "rain", "sun"], "weather"),
    (["নদী", "জল", "নৌকা", "মাছ"], ["river", "water", "boat", "fish"], "politics"),
    (["ঘর", "মা", "বাবা", "ভাই"], ["home", "mother", "father", "brother"], "family"),
    (["টাকা", "হাট", "দাম", "ব্যবসা"], ["money", "market", "price", "trade"], "finance"),
    (["মন্দির", "পূজা", "ঈশ্বর", "প্রার্থনা"], ["temple", "worship", "god", "prayer"], "religion"),
    (["রাগ", "ভয়", "আনন্দ", "দুঃখ"], ["anger", "fear", "joy", "sorrow"], "emotions"),
]
SENTIMENT = [("ভালো", "good", "positive"), ("খারাপ", "bad", "negative"), ("সাধারণ", "plain", "neutral")]
FILLER_BN = ["সে", "তার", "এই", "কথা", "দিন", "লোক", "কাজ", "পথ"]
FILLER_EN = ["he", "his", "this", "word", "day", "people", "work", "path"]
IDIOM_BN = ["গাছ", "কাঁঠাল", "গোঁফ", "তেল", "বাঘ", "ঘোড়া", "চাঁদ", "সাপ", "ব্যাঙ", "কলা"]


def corpus(n=600, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        c = int(rng.integers(len(CLASSES)))
        s = int(rng.choice(3, p=[.2, .5, .3] if c % 2 else [.5, .2, .3]))
        bn_words, en_words, domain = CLASSES[c]
        j = rng.choice(4, size=2, replace=False)
        tag = str(i).translate(BN_DIGITS)
        # The idiom carries a class word and a sentiment word whose embeddings are tied
        # to the English counterpart's, so English directions can read Bangla classes.
        idiom = f"{IDIOM_BN[i % 10]} {bn_words[j[1]]} {SENTIMENT[s][0]} {tag}"
        context_bn = " ".join(rng.choice(FILLER_BN, 2)) + f" {bn_words[j[0]]}"
        context_en = " ".join(rng.choice(FILLER_EN, 2)) + f" {en_words[j[0]]}"
        bn_ex = f"{context_bn} {idiom} হলো।"
        en_ex = f"{context_en} like {en_words[j[1]]} {SENTIMENT[s][1]} number{i}."
        records.append({
            "id": str(i + 1), "idiom": idiom,
            "example_sentences_in_bangla": [bn_ex], "example_sentences_in_english": [en_ex],
            "figurative_meaning_bn": f"{bn_words[j[0]]} {bn_words[(j[0] + 1) % 4]} {SENTIMENT[s][0]} {tag}",
            "figurative_meaning_en": f"{en_words[j[0]]} {en_words[(j[0] + 1) % 4]} {SENTIMENT[s][1]} gloss{i}",
            "literal_meaning": f"{IDIOM_BN[i % 10]} {tag}",
            "similar_in_english": [f"{en_words[j[1]]} {SENTIMENT[s][1]} saying{i}"] if rng.random() < .95 else [],
            "sentiment": SENTIMENT[s][2],
            "usage_domain": [domain.capitalize(), "Everyday conversation"] if rng.random() < .8 else ["Colloquial"],
            "tags": [domain], "frequency": "common" if rng.random() < .8 else "rare",
            "cultural_significance": "0", "religious_significance": "0", "historical_significance": "0",
            "scape": "country", "alternative_idioms": []})
    return records


def tiny_encoder(path, texts, seed):
    """A random 6-layer BERT with a whitespace word-level tokenizer covering ``texts``."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    pre = Whitespace()
    words = sorted({w for t in texts for w, _ in pre.pre_tokenize_str(t)})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                  mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]")
    path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(path)
    torch.manual_seed(seed)
    model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
                                 intermediate_size=64, max_position_embeddings=128))
    with torch.no_grad():
        table = model.embeddings.word_embeddings.weight
        for bn_words, en_words, _ in CLASSES:
            for b, e in zip(bn_words, en_words):
                if b in vocab and e in vocab:
                    table[vocab[b]] = table[vocab[e]]
        for b, e, _ in SENTIMENT:
            if b in vocab and e in vocab:
                table[vocab[b]] = table[vocab[e]]
    model.save_pretrained(path)
    return str(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def build(root, n=600, seed=0, records=None, models=None, label_backend="tfidf", classes=6, ranks=(1, 2, 3, 4),
          max_length=64):
    """Create CAMPAIGN under ``root``; returns its path.

    ``records`` replaces the synthetic corpus (e.g. a subset of the real export) and
    ``models`` the tiny encoders (e.g. a cached Hugging Face model id), so the same
    layout can be exercised with real tokenizers and dimensions.
    """
    from idiom_subspace.config import Config, model_slug
    from idiom_subspace.data import build_items, load_corpus, split_corpus
    from idiom_subspace.extraction import extract_features
    from idiom_subspace.geometry import fitted_preprocessor
    from idiom_subspace.labels import build_labels
    root = Path(root)
    campaign = root / "campaign"
    data = campaign / "inputs" / "bagdhara_export_done_2026-03-01_010356.json"
    write_json(data, records if records is not None else corpus(n, seed))
    raw, _ = load_corpus(data)
    split0 = split_corpus(raw, seed=0)
    labelled0, report0, _ = build_labels(split0, seed=0, k=classes, backend=label_backend)
    split1 = split_corpus(raw, seed=1)
    labelled1, report1, _ = build_labels(split1, seed=0, k=classes, backend=label_backend)
    conditions = ["bn-fig", "bn-sentence", "bn-idiom", "bn-context", "bn-para", "en-idiom", "en-trans", "en-para"]
    items = build_items(labelled0)
    items = items[items.condition.isin(conditions)].reset_index(drop=True)
    texts = items.text.tolist() + ["[MASK]"]
    if models is None:
        models = [tiny_encoder(root / f"weights_{k}", texts, seed + k) for k in range(2)]
    base = json.loads((REPO / "configs" / "full.json").read_text(encoding="utf-8"))
    base.update(models=[{"name": m, "revision": "main"} for m in models], label_backend=label_backend,
                classes=classes, ranks=list(ranks), null_draws=9, bootstrap=50, device="cpu", batch_size=16,
                max_length=max_length, probe_backend="torch", diagnostic_null_draws=3)
    write_json(campaign / "configs" / "full_3090.json", base)
    cfg = Config.load(campaign / "configs" / "full_3090.json")
    followup = campaign / "paper_followup_seed1_20261021"
    (followup / "prepared").mkdir(parents=True, exist_ok=True)
    labelled1.to_json(followup / "prepared" / "corpus_prepared.json", orient="records", force_ascii=False)
    write_json(followup / "prepared" / "label_audit.json", report1)
    for name in models:
        slug = model_slug(name)
        prepared = campaign / "primary" / slug / "split_0" / "labels_0"
        prepared.mkdir(parents=True, exist_ok=True)
        labelled0.to_json(prepared / "corpus_prepared.json", orient="records", force_ascii=False)
        items.to_json(prepared / "items.json", orient="records", force_ascii=False)
        write_json(prepared / "label_audit.json", report0)
        features, pooling, manifest = extract_features(items, {"name": name, "revision": "main"}, cfg,
                                                       campaign / "primary" / slug / "cache")
        dest = prepared / slug
        write_json(dest / "model_revision.json", manifest)
        pooling.to_csv(dest / "pooling.csv", index=False)
        layer = manifest["layers"][1]
        write_json(dest / "layer_selection.json", dict(layer=layer, selected_using="validation_only"))
        valid = pooling.valid.to_numpy(bool)
        li = manifest["layers"].index(layer)
        raw_features = np.asarray(features[valid, li, :], dtype=float)
        pre = fitted_preprocessor(raw_features[items.loc[valid, "split"].eq("space").to_numpy()])
        np.savez(dest / f"preprocess_layer_{layer}.npz", mean=pre.mean_, scale=pre.scale_,
                 components=np.eye(raw_features.shape[1]))
        write_json(followup / slug / "layer_selection.json", dict(layer=manifest["layers"][-1],
                                                                  selected_using="validation_only", method="COV"))
    return campaign


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
