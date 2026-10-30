"""X11: does the planned result survive re-seeded taxonomies, and one built from neither gloss field?

Motivation: (i) F is an unstable partition (re-seed ARI 0.40--0.43), so the
planned result might belong to one k-means seed; (ii) RQ2 is circular: F clusters
the English glosses and FBN the Bangla glosses, which are near-translations of
each other, so both favour gloss inputs by construction; only DOM is clean.

concepts  F1, F2, F3: F rebuilt by the pipeline's own ``build_labels`` with k-means
          seeds 1, 2 and 3 (same SPACE glosses, label encoder, k and n_init). Seed 0
          must reproduce the campaign's F (the gate).
          TAG: F's recipe applied to the corpus's tags instead of a gloss field. Each
          record's tags, minus form, register and polarity descriptors (figurative,
          idiom, proverbial, colloquial, slang, onomatopoeia, negative, positive, ...)
          and non-Latin tags, are joined and embedded with F's label encoder (its
          TF-IDF backend when the campaign used TF-IDF); k-means (k = 12, seed 0) is
          fitted on SPACE. Records without a retained tag are unlabelled.
tests     X2's design for every concept: the planned test (en-idiom -> bn-fig; MLR and
          COV; 999 RAND + 999 PERM draws; accuracy, balanced accuracy and macro-F1;
          Holm within concept x estimator over the six encoders), the input ladder
          (bootstrap intervals, no nulls) and the gloss-minus-counterpart contrast on
          the Bangla gloss (component wild-sign p; Holm within concept x estimator).
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.data import _list, normalize_phrase
from idiom_subspace.geometry import holm

from experiments.common import Log, concat, load_cell, make_study, rank_for, split_corpus_frame, write_csv, write_json
from experiments.concepts import _agreement, _class_report
from experiments.metrics import METRICS
from experiments.nulls import null_test
from experiments.x2_concepts import CONDITIONS, ladder_rows

from .common2 import cell_tables, cells_root, finish

RESEEDS = (1, 2, 3)
CONCEPTS = ["F1", "F2", "F3", "TAG"]
# Tags that describe an idiom's form, register or polarity rather than what it means.
NON_SEMANTIC_TAG = re.compile(
    r"\b(figurative|idioms?|idiomatic|proverbs?|proverbial|sayings?|expressions?|phrases?|metaphors?|metaphorical|"
    r"similes?|colloquial\w*|informal|slang|speech|conversation\w*|onomatopoe\w*|reduplicat\w*|rhym\w*|alliteration|"
    r"poetic|literary|literature|traditional|negative|positive|neutral|connotations?|usage|common|everyday|daily)\b")
LATIN = re.compile(r"[A-Za-z]")


def reseeded_taxonomies(cfg, corpus, seeds):
    """F under each k-means seed, from the pipeline's own build_labels on the split's prepared corpus."""
    from idiom_subspace.labels import build_labels
    out = {}
    for seed in seeds:
        labelled, _, _ = build_labels(corpus, seed=seed, k=cfg.classes, backend=cfg.label_backend,
                                      encoder=cfg.label_encoder, revision=cfg.label_revision)
        out[seed] = labelled.F.to_numpy().astype(int)
    return out


def tag_texts(corpus):
    """Each record's semantic tags (form, register, polarity and non-Latin tags removed), joined."""
    texts, removed = [], Counter()
    for value in corpus.tags:
        tags = sorted({normalize_phrase(t) for t in _list(value)} - {""})
        kept = []
        for t in tags:
            if LATIN.search(t) and not NON_SEMANTIC_TAG.search(t):
                kept.append(t)
            else:
                removed[t] += 1
        texts.append(", ".join(kept))
    return texts, removed


def tag_labels(cfg, corpus, k=12, seed=0):
    """TAG: k-means over embeddings of the semantic tags, fitted on SPACE (F's recipe on another field)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    texts, removed = tag_texts(corpus)
    valid = np.asarray([bool(t) for t in texts])
    train = np.flatnonzero(valid & corpus.split.eq("space").to_numpy())
    if cfg.label_backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(cfg.label_encoder, revision=cfg.label_revision, trust_remote_code=False)
        unique = list(dict.fromkeys(t for t in texts if t))
        embedded = encoder.encode(unique, normalize_embeddings=True, show_progress_bar=False)
        lookup = dict(zip(unique, embedded))
        E = np.stack([lookup[t] if t else np.zeros(embedded.shape[1]) for t in texts])
        embedder = dict(backend="sentence-transformers", encoder=cfg.label_encoder, revision=cfg.label_revision)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), dtype=np.float64)
        vectorizer.fit([texts[i] for i in train])
        E = vectorizer.transform(texts)
        embedder = dict(backend="tfidf")
    effective_k = min(k, len({texts[i] for i in train}), len(train))
    taxonomy = KMeans(n_clusters=effective_k, n_init=10, random_state=seed).fit(E[train])
    assigned = np.full(len(texts), -1)
    assigned[valid] = taxonomy.predict(E[valid])
    stability = [float(adjusted_rand_score(taxonomy.labels_, KMeans(n_clusters=effective_k, n_init=5, random_state=s)
                                           .fit_predict(E[train]))) for s in (seed + 1, seed + 2, seed + 3)]
    keywords = []
    for c in range(effective_k):
        members = [texts[i] for i, label in zip(train, taxonomy.labels_) if label == c]
        counts = Counter(t for text in members for t in text.split(", "))
        keywords.append(dict(label=c, space_records=len(members), top_tags=[t for t, _ in counts.most_common(8)]))
    report = dict(concept="TAG", embedder=embedder, k=effective_k, seed=seed, training_records=int(len(train)),
                  labelled_records=int(valid.sum()), records=int(len(texts)), reseed_ARI_space=stability,
                  non_semantic_pattern=NON_SEMANTIC_TAG.pattern,
                  most_frequent_removed_tags=[dict(tag=t, records=int(n)) for t, n in removed.most_common(30)],
                  cluster_tags=keywords)
    return assigned, report


def build_concepts(ctx, split, log):
    """Concept labels for one split (shared by all models), cached with the run's finished cells."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from experiments.concepts import usage_domain_labels
    folder = cells_root(ctx) / "labels" / f"split{split}"
    path = folder / "labels.csv"
    if path.exists():
        return pd.read_csv(path)
    corpus = split_corpus_frame(ctx, ctx.models[0], split)
    seeds = (0,) + tuple(int(s) for s in str(ctx.option("reseeds", ",".join(map(str, RESEEDS)))).split(",") if s)
    log(f"rebuilding F for split {split} with k-means seeds {seeds}")
    fits = reseeded_taxonomies(ctx.cfg, corpus, seeds)
    original = pd.to_numeric(corpus.F, errors="coerce").fillna(-1).astype(int).to_numpy()
    either = (original >= 0) | (fits[0] >= 0)
    table = corpus[["idiom_id", "split"]].copy()
    report = dict(split=split, reproduction=dict(
        records=int(either.sum()), agreement=float(np.mean(original[either] == fits[0][either])),
        ARI=float(adjusted_rand_score(original[either], fits[0][either]))), reseeded={})
    space = corpus.split.eq("space").to_numpy()
    for seed in seeds[1:]:
        name = f"F{seed}"
        table[name] = fits[seed]
        ok = space & (original >= 0) & (fits[seed] >= 0)
        report["reseeded"][name] = dict(seed=seed, ARI_with_F_space=float(adjusted_rand_score(original[ok],
                                                                                              fits[seed][ok])),
                                        by_split=_class_report(table.assign(F=original), name))
    log(f"building TAG for split {split}")
    table["TAG"], report["TAG"] = tag_labels(ctx.cfg, corpus, k=int(ctx.option("tag_k", ctx.cfg.classes)))
    dom, _ = usage_domain_labels(corpus)
    frame = table.assign(F=original, S=pd.to_numeric(corpus.S, errors="coerce").fillna(-1).astype(int).to_numpy(),
                         DOM=dom.DOM.to_numpy())
    report["TAG"]["by_split"] = _class_report(frame, "TAG")
    report["TAG"]["agreement"] = _agreement(frame, "TAG", others=("F", "S", "DOM"))
    ok = space & (frame.TAG >= 0) & (frame.F >= 0)
    report["TAG"]["ARI_with_F_space"] = float(adjusted_rand_score(frame.F[ok], frame.TAG[ok])) if ok.any() else None
    report["TAG"]["AMI_with_F_space"] = (float(adjusted_mutual_info_score(frame.F[ok], frame.TAG[ok]))
                                         if ok.any() else None)
    write_csv(path, table)
    write_json(folder / "label_report.json", report)
    write_json(ctx.portable / f"label_report_split{split}.json", report)
    write_csv(ctx.portable / f"labels_split{split}.csv.gz", table.drop(columns=["split"]))
    return table


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    rand = int(ctx.option("null_draws", 999))
    perm = int(ctx.option("perm_draws", 999))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    methods = ctx.option("methods", "MLR,COV").split(",")
    wanted = ctx.option("concepts", ",".join(CONCEPTS)).split(",")
    tests, ladders, contrasts = [], [], []
    for split in ctx.splits:
        table = build_concepts(ctx, split, log)
        concepts = [c for c in wanted if c in table]
        lookup = table.set_index("idiom_id")
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=CONDITIONS)
                for concept in concepts:
                    cell.frame[concept] = cell.frame.idiom_id.map(lookup[concept]).fillna(-1).astype(int)
                study = make_study(ctx, cell, "taxonomy", extra_columns=concepts)
                rows, ladder, contrast = [], [], []
                for concept in concepts:
                    rank = rank_for(study, concept)
                    if rank is None:
                        continue
                    for method in methods:
                        log(f"{name} split{split} {concept} {method} M={rank}: {rand} RAND + {perm} PERM",
                            model=name, concept=concept, method=method)
                        row, _, _ = null_test(study, concept, method, "en-idiom", "bn-fig", rank, rand, perm, n_boot,
                                              cell.meta["seed"], label=f"X11 {concept} {method}")
                        if row is not None:
                            row["selected_layer"] = cell.selected_layer
                            rows.append(row)
                        part, c = ladder_rows(study, concept, method, rank, n_boot, cell.meta["seed"])
                        ladder += part
                        contrast += c
                saved = cell_tables(ctx, key, {"tests.csv": pd.DataFrame(rows), "ladder.csv": pd.DataFrame(ladder),
                                               "contrasts.csv": pd.DataFrame(contrast)})
                del study, cell
            else:
                log(f"restored {key}")
            tests.append(saved["tests.csv"])
            ladders.append(saved["ladder.csv"])
            contrasts.append(saved["contrasts.csv"])
    tests = concat(tests)
    for _, index in tests.groupby(["split", "concept", "method"]).groups.items():
        for null in ["RAND", "PERM"]:
            for m in METRICS:
                column = f"{null}_p_{m}"
                tests.loc[index, column + "_holm"] = holm(tests.loc[index, column].to_numpy(), family_size=len(index))
    contrasts = concat(contrasts)
    if not contrasts.empty:
        for _, index in contrasts.groupby(["split", "concept", "method"]).groups.items():
            contrasts.loc[index, "p_holm"] = holm(contrasts.loc[index, "p"].to_numpy(), family_size=len(index))
    finish(ctx, {"tests.csv": tests, "ladder.csv": concat(ladders), "contrasts.csv": contrasts},
           dict(null_draws=rand, perm_draws=perm, bootstrap=n_boot, methods=methods, concepts=wanted,
                reseeds=list(RESEEDS), non_semantic_tag_pattern=NON_SEMANTIC_TAG.pattern))
    log("finished")
