"""Train-only concept taxonomies with shared figurative/literal coordinates."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from .data import _list, _text, normalize_phrase


_SENTIMENT = {
    "positive": "positive", "pos": "positive", "+": "positive", "ইতিবাচক": "positive",
    "negative": "negative", "neg": "negative", "-": "negative", "নেতিবাচক": "negative",
    "neutral": "neutral", "neu": "neutral", "নিরপেক্ষ": "neutral",
    "mixed": "mixed", "ambivalent": "mixed", "positive and negative": "mixed", "মিশ্র": "mixed",
}
# Conservative aliases only. Other supplied tags/domains retain their names.
_CATEGORY_ALIASES = {"everyday life": "daily life", "day to day life": "daily life", "educational": "education"}


def _english_only(text: str) -> bool:
    """Accept English-field prose with Latin letters, reject mislabeled scripts."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    return bool(letters) and all("LATIN" in unicodedata.name(c, "") for c in letters)


def _category_set(value) -> tuple[str, ...]:
    names = (normalize_phrase(v) for v in _list(value))
    return tuple(sorted({_CATEGORY_ALIASES.get(v, v) for v in names if v}))


def _counts_by_split(df: pd.DataFrame, name: str) -> dict:
    return {str(split): {"records": len(group), "available": int(group[name].ge(0).sum()),
                         "classes": int(group.loc[group[name].ge(0), name].nunique())}
            for split, group in df.groupby("split", sort=True)}


def build_labels(df: pd.DataFrame, seed: int = 0, k: int = 12, backend: str = "tfidf",
                 encoder: str = "sentence-transformers/all-MiniLM-L6-v2", revision: str = "main") -> tuple[pd.DataFrame, dict, dict]:
    """Fit F on SPACE English glosses and assign L in exactly that taxonomy.

    ``L=-1`` when an English literal gloss is unavailable or out of vocabulary.
    S is a canonical four-way human sentiment label. D/Ft classify complete
    supplied human category sets using SPACE-only vocabularies: a composite set
    is explicitly one class, not an arbitrarily chosen first tag. These are
    human metadata targets, not independent semantic-cluster validations.

    The optional sentence encoder is loaded only when explicitly requested. It
    is a label constructor and must never also be a probed representation model.
    """
    if "split" not in df or not df["split"].eq("space").any():
        raise ValueError("Call split_corpus first; F requires a nonempty SPACE split.")
    if backend not in {"tfidf", "sentence-transformers"}:
        raise ValueError("backend must be 'tfidf' or 'sentence-transformers'.")
    if k < 2:
        raise ValueError("F taxonomy requires k >= 2.")
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    out = df.copy()
    figurative = out.fig_en.map(_text).tolist()
    literal = out.lit_en.map(_text).tolist() if "lit_en" in out else [""] * len(out)
    fig_valid = np.asarray([_english_only(v) for v in figurative])
    lit_valid = np.asarray([_english_only(v) for v in literal])
    train = out.split.eq("space").to_numpy() & fig_valid
    train_ix = np.flatnonzero(train)
    n_unique = len({figurative[i].casefold().strip() for i in train_ix})
    effective_k = min(k, n_unique, len(train_ix))
    if effective_k < 2:
        raise ValueError("SPACE needs at least two distinct English figurative glosses to fit F.")
    artifacts = {"backend": backend, "encoder_name": encoder if backend == "sentence-transformers" else None,
                 "train_indices": train_ix, "train_idiom_ids": out.iloc[train_ix].idiom_id.tolist()}
    notes = []
    if backend == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000,
                                     token_pattern=r"(?u)\b\w+\b", dtype=np.float64)
        try:
            vectorizer.fit([figurative[i] for i in train_ix])
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            vectorizer.set_params(stop_words=None)
            vectorizer.fit([figurative[i] for i in train_ix])
            notes.append("SPACE glosses yielded no non-stopword tokens; TF-IDF retains stopwords.")
        F_features = vectorizer.transform([v if ok else "" for v, ok in zip(figurative, fig_valid)])
        L_features = vectorizer.transform([v if ok else "" for v, ok in zip(literal, lit_valid)])
        fig_valid &= np.asarray(F_features.getnnz(axis=1) > 0)
        lit_valid &= np.asarray(L_features.getnnz(axis=1) > 0)
        artifacts["vectorizer"] = vectorizer
        feature_model = vectorizer
    else:
        from sentence_transformers import SentenceTransformer
        feature_model = SentenceTransformer(encoder, revision=revision, trust_remote_code=False)
        texts = list(dict.fromkeys([v for v, ok in zip(figurative, fig_valid) if ok] +
                                  [v for v, ok in zip(literal, lit_valid) if ok]))
        embedded = feature_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        cache = dict(zip(texts, embedded))
        width = embedded.shape[1]
        F_features = np.stack([cache[v] if ok else np.zeros(width) for v, ok in zip(figurative, fig_valid)])
        L_features = np.stack([cache[v] if ok else np.zeros(width) for v, ok in zip(literal, lit_valid)])
        artifacts["label_encoder"] = feature_model
    # Recompute eligibility after TF-IDF empty-vector detection.
    train = out.split.eq("space").to_numpy() & fig_valid
    train_ix = np.flatnonzero(train)
    effective_k = min(effective_k, len(train_ix))
    if effective_k < 2:
        raise ValueError("SPACE has fewer than two nonempty English gloss feature vectors.")
    taxonomy = KMeans(n_clusters=effective_k, n_init=10, random_state=seed)
    taxonomy.fit(F_features[train_ix])
    out["F"], out["L"] = -1, -1
    if fig_valid.any():
        out.loc[fig_valid, "F"] = taxonomy.predict(F_features[fig_valid])
    if lit_valid.any():
        out.loc[lit_valid, "L"] = taxonomy.predict(L_features[lit_valid])
    out["F_valid_english"], out["L_valid_english"] = fig_valid, lit_valid
    transparency = counterpart_transparency(out, feature_model, backend, F_features, fig_valid)
    out["F_label_source"] = f"space_only_{backend}_figurative_taxonomy"
    out["L_label_source"] = f"assignment_to_same_F_taxonomy_{backend}"
    out["S_canonical"] = out.sentiment_raw.map(lambda x: _SENTIMENT.get(normalize_phrase(x), ""))
    out["D_labels"] = out.domain_raw.map(_category_set)
    out["Ft_labels"] = out.tags.map(_category_set)
    vocabularies = {}
    category_report = {}
    for name, source in (("S", "S_canonical"), ("D", "D_labels"), ("Ft", "Ft_labels")):
        values = out[source].tolist()
        training_values = [v for v, is_train in zip(values, out.split.eq("space")) if is_train and v]
        vocab = {value: i for i, value in enumerate(sorted(set(training_values)))}
        out[name] = [vocab.get(value, -1) if value else -1 for value in values]
        vocabularies[name] = [{"label": i, "value": list(value) if isinstance(value, tuple) else value}
                              for value, i in vocab.items()]
        counts = Counter(training_values)
        category_report[name] = {"semantics": "canonical_sentiment" if name == "S" else "exact_canonical_human_category_set",
                                 "training_classes": len(vocab), "singleton_training_classes": sum(v == 1 for v in counts.values()),
                                 "multilabel_records": sum(isinstance(v, tuple) and len(v) > 1 for v in values),
                                 "unseen_or_missing_records": int(out[name].eq(-1).sum())}
    out["S_label_source"] = "supplied_human_sentiment_train_vocabulary"
    out["D_label_source"] = "supplied_human_domain_exact_set_train_vocabulary"
    out["Ft_label_source"] = "supplied_human_tags_exact_set_train_vocabulary"
    # Agreement/stability are diagnostics, never substitutes for human validation.
    agreement = {}
    for first, second in (("F", "L"), ("F", "S"), ("F", "D"), ("F", "Ft")):
        valid = out[first].ge(0) & out[second].ge(0)
        agreement[f"AMI({first},{second})"] = {
            "n": int(valid.sum()),
            "value": float(adjusted_mutual_info_score(out.loc[valid, first], out.loc[valid, second])) if valid.sum() >= 2 else None,
        }
    stability = []
    for reseed in (seed + 1, seed + 2, seed + 3):
        repeat = KMeans(n_clusters=effective_k, n_init=5, random_state=reseed).fit_predict(F_features[train_ix])
        stability.append(float(adjusted_rand_score(out.iloc[train_ix].F, repeat)))
    # Keywords use only SPACE corpus text, including with the optional neural constructor.
    if backend == "tfidf":
        keyword_vectorizer, keyword_features = feature_model, F_features[train_ix]
    else:
        keyword_vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b", max_features=10000)
        try:
            keyword_features = keyword_vectorizer.fit_transform([figurative[i] for i in train_ix])
        except ValueError:
            keyword_vectorizer.set_params(stop_words=None)
            keyword_features = keyword_vectorizer.fit_transform([figurative[i] for i in train_ix])
    terms = keyword_vectorizer.get_feature_names_out()
    keywords = []
    train_labels = out.iloc[train_ix].F.to_numpy()
    for c in range(effective_k):
        selected = train_labels == c
        if selected.any():
            weights = np.asarray(keyword_features[selected].mean(axis=0)).ravel()
            names = terms[np.argsort(weights)[::-1][:10]].tolist()
        else:
            names = []
        keywords.append({"F": c, "space_n": int(selected.sum()), "keywords": names})
    report = {
        "backend": backend, "label_encoder": encoder if backend == "sentence-transformers" else "SPACE-fitted English TF-IDF",
        "label_encoder_requested_revision": revision if backend == "sentence-transformers" else None,
        "label_encoder_resolved_revision": (getattr(getattr(getattr(feature_model[0], "auto_model", None), "config", None), "_commit_hash", None)
                                             if backend == "sentence-transformers" else None),
        "label_encoder_is_probed": False, "fit_split": "space", "requested_k": k, "effective_k": effective_k,
        "training_records": len(train_ix), "same_taxonomy_for_F_and_L": True,
        "literal_english_available": int(lit_valid.sum()), "literal_english_missing_or_invalid": int((~lit_valid).sum()),
        "figurative_non_english_or_missing": int(sum(not _english_only(v) for v in figurative)),
        "literal_non_english_rejected": int(sum(bool(v) and not _english_only(v) for v in literal)),
        "label_availability": {name: _counts_by_split(out, name) for name in ("F", "L", "S", "D", "Ft")},
        "category_semantics": category_report, "vocabularies": vocabularies,
        "agreement_diagnostics": agreement, "F_SPACE_reseed_ARI": stability,
        "cluster_keywords": keywords,
        "counterpart_transparency": transparency,
        "notes": notes + ["All fitted vocabularies and F clusters use SPACE rows only.",
                          "L is predicted by the fitted F taxonomy; no independent literal clusters are compared.",
                          "No Bangla literal text is sent to the English label constructor.",
                          "D/Ft preserve full human category sets as composite classes; inspect rare-class support before probing.",
                          "AMI and reseed stability do not establish semantic validity; independent annotations are required."],
    }
    artifacts.update(kmeans=taxonomy, F_features=F_features, L_features=L_features, train_indices=train_ix,
                     train_idiom_ids=out.iloc[train_ix].idiom_id.tolist(), vocabularies=vocabularies)
    out.attrs["label_report"] = report
    return out, report, artifacts


def counterpart_transparency(out: pd.DataFrame, feature_model, backend: str, F_features, fig_valid) -> dict:
    """Cosine between the probed English counterpart and the English figurative gloss.

    Uses the same never-probed taxonomy constructor as F. A high score means
    the counterpart restates the meaning (a paraphrase-like counterpart); a low
    score means it shares little wording or meaning with the gloss (often a
    conventional English idiom). Tertile cut-offs are fitted on SPACE only.
    This is a covariate, not a label: no probe or subspace is fitted on it.
    Adds ``cp_transparency`` (NaN without a counterpart) and boolean
    ``transparent`` / ``opaque`` groups to ``out`` in place.
    """
    first = [(_list(v) or [""])[0] for v in out.sims] if "sims" in out else [""] * len(out)
    valid = np.asarray([bool(c) and _english_only(c) for c in first], dtype=bool) & np.asarray(fig_valid, dtype=bool)
    score = np.full(len(out), np.nan)
    if valid.any():
        texts = [c for c, ok in zip(first, valid) if ok]
        if backend == "tfidf":
            C = feature_model.transform(texts)
            Fv = F_features[np.flatnonzero(valid)]
            num = np.asarray(C.multiply(Fv).sum(axis=1)).ravel()
            den = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel() * np.asarray(Fv.multiply(Fv).sum(axis=1)).ravel())
            score[valid] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        else:
            C = feature_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            score[valid] = np.einsum("ij,ij->i", np.asarray(C), np.asarray(F_features)[valid])
    out["cp_transparency"] = score
    space = out.split.eq("space").to_numpy() & np.isfinite(score)
    report = {"available": int(np.isfinite(score).sum()), "missing_counterpart_or_gloss": int((~np.isfinite(score)).sum())}
    if space.sum() >= 3:
        lo, hi = np.quantile(score[space], [1 / 3, 2 / 3])
        out["transparent"] = np.isfinite(score) & (score >= hi)
        out["opaque"] = np.isfinite(score) & (score <= lo)
        report.update(lower_tertile=float(lo), upper_tertile=float(hi), fitted_on="space",
                      transparent=int(out.transparent.sum()), opaque=int(out.opaque.sum()))
    else:
        out["transparent"], out["opaque"] = False, False
    report["interpretation"] = ("covariate for English anchoring: high = paraphrase-like counterpart, "
                                "low = lexically/semantically distant counterpart (often an English idiom)")
    return report


def annotation_sample(df: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Round-robin available F/anchoring/frequency strata, preserving rare groups.

    This deliberately oversamples sparse strata, so annotation prevalence must
    not be interpreted as corpus prevalence.
    """
    target = min(max(int(n), 0), len(df))
    if target == 0: return df.iloc[:0].copy()
    columns = [c for c in ["F", "anchored", "rare"] if c in df]
    if not columns: return df.sample(n=target, random_state=seed)
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(ix).tolist() for ix in df.groupby(columns, dropna=False, sort=True).indices.values()]
    chosen = []
    while len(chosen) < target:
        for j in rng.permutation(len(buckets)):
            if buckets[j]: chosen.append(buckets[j].pop())
            if len(chosen) == target: break
    return df.iloc[chosen].copy()


def export_annotation_sample(df: pd.DataFrame, out_path: str | Path, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Export paired blank annotation rows; supplied machine labels are only proposals."""
    if not df.idiom_id.is_unique:
        raise ValueError("Annotation sampling requires unique idiom_id values.")
    sample = annotation_sample(df, n, seed)
    rows = []
    context = ("idiom_id", "source_id", "idiom", "fig_en", "fig_bn", "lit_en", "lit_bn", "split")
    for item in sample.to_dict("records"):
        for slot in (1, 2):
            row = {name: item.get(name, "") for name in context}
            row.update(annotation_slot=slot, annotator_id="")
            for name in ("F", "L", "S", "D", "Ft"):
                row[f"proposed_{name}"] = item.get(name, -1)
                row[f"human_{name}"] = ""
            row["literal_gloss_en_reviewed"] = ""
            row["notes"] = ""
            rows.append(row)
    result = pd.DataFrame(rows)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def validate_annotations(path: str | Path) -> dict:
    """Validate independent annotator joins and report paired Cohen's kappa.

    Blank templates produce an explicit incomplete report, never synthetic
    agreement. More than two annotators, duplicate ratings and self-pairs are
    rejected rather than silently picking two rows by position.
    """
    from sklearn.metrics import cohen_kappa_score
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {"idiom_id", "annotator_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Annotation file is missing {sorted(required - set(frame.columns))}")
    if frame.idiom_id.str.strip().eq("").any():
        raise ValueError("Every annotation row requires an idiom_id.")
    complete = frame[frame.annotator_id.str.strip().ne("")].copy()
    complete["annotator_id"] = complete.annotator_id.str.strip()
    if complete.duplicated(["idiom_id", "annotator_id"]).any():
        raise ValueError("Duplicate idiom_id/annotator_id rating; annotators must be independent.")
    n_raters = complete.groupby("idiom_id").annotator_id.nunique()
    if n_raters.gt(2).any():
        raise ValueError("Expected two independent annotators per idiom; found more than two.")
    annotators = sorted(complete.annotator_id.unique())
    if len(annotators) > 2:
        raise ValueError("Two-annotator Cohen's kappa requires the same two named annotators across items.")
    result = {"records": len(frame), "idioms": int(frame.idiom_id.nunique()), "annotators": annotators,
              "blank_annotator_rows": int(frame.annotator_id.str.strip().eq("").sum()),
              "paired_idioms": int(n_raters.eq(2).sum()), "fields": {}, "status": "incomplete"}
    for field in ("F", "L", "S", "D", "Ft"):
        col = f"human_{field}"
        if col not in frame or len(annotators) != 2:
            result["fields"][field] = {"paired_ratings": 0, "kappa": None, "agreement": None}
            continue
        pivot = complete.pivot(index="idiom_id", columns="annotator_id", values=col).reindex(columns=annotators)
        paired = pivot.dropna()
        paired = paired[(paired != "").all(axis=1) & (paired != "-1").all(axis=1)]
        a, b = paired.iloc[:, 0], paired.iloc[:, 1]
        kappa = float(cohen_kappa_score(a, b)) if len(paired) >= 2 and len(set(a) | set(b)) > 1 else None
        result["fields"][field] = {"paired_ratings": len(paired), "kappa": kappa,
                                    "agreement": float((a == b).mean()) if len(paired) else None}
    if (len(annotators) == 2 and result["blank_annotator_rows"] == 0 and n_raters.eq(2).all()
            and all(v["paired_ratings"] == result["idioms"] for v in result["fields"].values())):
        result["status"] = "complete"
    return result
