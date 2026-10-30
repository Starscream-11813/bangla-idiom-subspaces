"""Two concepts that are not constructed from English glosses (X2).

FBN  -- a figurative taxonomy rebuilt from the *Bangla* figurative glosses:
        k-means (k=12, as for F) over gloss embeddings fitted on SPACE only.
        The default embedder is the encoder of NLLB-200 (distilled 600M), which
        is never probed and was already used by the campaign (literal glosses);
        ``tfidf`` is an encoder-free alternative (character n-gram TF-IDF + LSA),
        whose clusters are partly morphological (verbal vs nominal glosses).
DOM  -- the corpus's usage-domain metadata as one coarse label per record:
        register descriptors that apply to most idioms (conversation, colloquial,
        everyday, daily, informal, speech, ...) are dropped, the K most frequent
        remaining tags in SPACE are kept, and each record takes its most frequent
        kept tag (ties alphabetical); records without a kept tag are unlabelled.

Both vocabularies are fitted on SPACE rows of the given split only, and both are
reported with class sizes, stability and agreement with F and S.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

from idiom_subspace.data import _list, _text, normalize_phrase

GENERIC_DOMAIN = re.compile(
    r"\b(conversation|conversations|conversational|colloquial|informal|everyday|daily|casual|speech|"
    r"language|communication|descriptive|description|general|social interactions?|social)\b")
BENGALI = re.compile("[ঀ-৿]")
NLLB = "facebook/nllb-200-distilled-600M"


def _agreement(frame, column, others=("F", "S")):
    from sklearn.metrics import adjusted_mutual_info_score
    out = {}
    space = frame.split.eq("space")
    for other in others:
        if other not in frame:
            continue
        ok = space & frame[column].ge(0) & pd.to_numeric(frame[other], errors="coerce").ge(0)
        out[f"AMI({column},{other})_space"] = (float(adjusted_mutual_info_score(frame.loc[ok, column], frame.loc[ok, other]))
                                              if ok.sum() > 1 else None)
    return out


def _class_report(frame, column):
    out = {}
    for split, group in frame.groupby("split", sort=True):
        values = group.loc[group[column].ge(0), column]
        counts = values.value_counts().sort_index()
        out[str(split)] = dict(records=int(len(group)), labelled=int(len(values)), classes=int(values.nunique()),
                               counts={int(k): int(v) for k, v in counts.items()},
                               majority_share=float(counts.max() / counts.sum()) if len(counts) else None)
    return out


def usage_domain_labels(corpus, k=12):
    """DOM: one coarse usage-domain label per record, vocabulary from SPACE only."""
    space = corpus.split.eq("space").to_numpy()
    tags = [sorted({normalize_phrase(t) for t in _list(v)} - {""}) for v in corpus.domain_raw]
    counts = Counter(t for ts, s in zip(tags, space) if s for t in ts)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dropped = [t for t, _ in ordered if GENERIC_DOMAIN.search(t)]
    retained = [t for t, _ in ordered if not GENERIC_DOMAIN.search(t)][:k]
    rank = {t: i for i, t in enumerate(retained)}
    assigned, multiple = [], 0
    for ts in tags:
        hits = sorted(rank[t] for t in ts if t in rank)
        multiple += len(hits) > 1
        assigned.append(hits[0] if hits else -1)
    out = corpus[["idiom_id", "split"] + [c for c in ["F", "S"] if c in corpus]].copy()
    out["DOM"] = np.asarray(assigned, dtype=int)
    report = dict(concept="DOM", rule="most frequent retained SPACE usage-domain tag; register descriptors removed",
                  generic_pattern=GENERIC_DOMAIN.pattern, k=k,
                  classes=[dict(label=i, tag=t, space_records=int(counts[t])) for i, t in enumerate(retained)],
                  most_frequent_dropped_tags=[dict(tag=t, space_records=int(counts[t])) for t in dropped[:25]],
                  labelled_records=int((out.DOM >= 0).sum()), records=int(len(out)),
                  records_with_several_retained_tags=int(multiple),
                  by_split=_class_report(out, "DOM"), agreement=_agreement(out, "DOM"))
    return out, report


def _nllb_embeddings(texts, model=NLLB, revision="main", device="auto", batch_size=64):
    import torch
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    # The explicit class avoids an AutoTokenizer lookup that fails for m2m_100 in transformers 5.x.
    tokenizer = NllbTokenizerFast.from_pretrained(model, revision=revision, src_lang="ben_Beng")
    network = AutoModelForSeq2SeqLM.from_pretrained(model, revision=revision)
    resolved = getattr(network.config, "_commit_hash", None) or revision
    encoder = network.get_encoder().to(device).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), network.config.d_model), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chosen = order[start:start + batch_size]
            encoded = tokenizer([texts[i] for i in chosen], return_tensors="pt", padding=True, truncation=True,
                                max_length=128, return_special_tokens_mask=True)
            special = encoded.pop("special_tokens_mask")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state.float()
            mask = (encoded["attention_mask"].bool() & ~special.to(device).bool()).float()[..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            out[chosen] = pooled.cpu().numpy()
    del network, encoder
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12), dict(model=model, resolved_revision=resolved)


def _tfidf_embeddings(texts, train, seed):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_df=.5, sublinear_tf=True)
    fitted = vectorizer.fit_transform([texts[i] for i in train])
    svd = TruncatedSVD(n_components=min(100, fitted.shape[1] - 1), random_state=seed).fit(fitted)
    return normalize(svd.transform(vectorizer.transform(texts))), dict(model="char_wb(2,4) TF-IDF + LSA(100)",
                                                                       resolved_revision=None)


def bangla_gloss_labels(corpus, backend="nllb", seed=0, k=12, model=NLLB, revision="main", device="auto"):
    """FBN: k-means over Bangla figurative-gloss embeddings, fitted on SPACE only."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score
    texts = [_text(v) for v in corpus.fig_bn]
    valid = np.asarray([bool(BENGALI.search(t)) for t in texts])
    space = corpus.split.eq("space").to_numpy()
    train = np.flatnonzero(valid & space)
    unique = sorted({texts[i] for i in np.flatnonzero(valid)})
    position = {t: j for j, t in enumerate(unique)}
    if backend == "nllb":
        embedded, provenance = _nllb_embeddings(unique, model=model, revision=revision, device=device)
    elif backend == "tfidf":
        train_unique = sorted({position[texts[i]] for i in train})
        embedded, provenance = _tfidf_embeddings(unique, train_unique, seed)
    else:
        raise ValueError("backend must be nllb or tfidf")
    rows = np.asarray([position[texts[i]] for i in np.flatnonzero(valid)])
    E = np.zeros((len(texts), embedded.shape[1]))
    E[valid] = embedded[rows]
    taxonomy = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(E[train])
    out = corpus[["idiom_id", "split"] + [c for c in ["F", "S"] if c in corpus]].copy()
    out["FBN"] = -1
    out.loc[valid, "FBN"] = taxonomy.predict(E[valid])
    out["FBN"] = out.FBN.astype(int)
    stability = [float(adjusted_rand_score(taxonomy.labels_, KMeans(n_clusters=k, n_init=5, random_state=s)
                                           .fit_predict(E[train]))) for s in (seed + 1, seed + 2, seed + 3)]
    words = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)[^\s;,।.!?()\-'\"]+", min_df=2, max_df=.3)
    weights = words.fit_transform([texts[i] for i in train])
    terms = words.get_feature_names_out()
    keywords = []
    for c in range(k):
        chosen = taxonomy.labels_ == c
        mean = np.asarray(weights[chosen].mean(axis=0)).ravel() if chosen.any() else np.zeros(len(terms))
        keywords.append(dict(label=c, space_records=int(chosen.sum()),
                             keywords=terms[np.argsort(mean)[::-1][:8]].tolist()))
    report = dict(concept="FBN", backend=backend, embedder=provenance, k=k, seed=seed,
                  training_records=int(len(train)), unique_glosses=len(unique),
                  missing_or_non_bengali_glosses=int((~valid).sum()), reseed_ARI_space=stability,
                  cluster_keywords_bn=keywords, by_split=_class_report(out, "FBN"), agreement=_agreement(out, "FBN"))
    return out, report
