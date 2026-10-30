"""Corpus normalization and leakage-aware inputs for the revised experiments.

Importing this module never downloads data. Missing counterparts and duplicate
senses are retained; availability is a property of an experimental condition.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


_MISSING = {"", "none", "null", "nan", "n/a", "na", "[]", "{}", "-"}
_FALSE = _MISSING | {"no", "false", "0", "not applicable", "absent"}
_TRUE = {"yes", "true", "1", "present"}
_FIELDS = {
    "idiom": ("idiom", "idiom_bn", "phrase", "bagdhara"),
    "bn_ex": ("bn_ex", "example_sentences_in_bangla", "example_sentences_in_bengali", "example_bn"),
    "en_ex": ("en_ex", "example_sentences_in_english", "example_en"),
    "fig_en": ("fig_en", "figurative_meaning_en", "figurative_meaning_english"),
    "fig_bn": ("fig_bn", "figurative_meaning_bn", "figurative_meaning_bangla"),
    "lit_en": ("lit_en", "literal_meaning_en", "literal_meaning_english"),
    "lit_bn": ("lit_bn", "literal_meaning_bn", "literal_meaning_bangla", "literal_meaning"),
    "sims": ("sims", "similar_in_english", "english_counterparts", "sim_en"),
    "tags": ("tags", "tags_l"),
    "domain_raw": ("domain_raw", "usage_domain", "domain", "dom_l"),
    "sentiment_raw": ("sentiment_raw", "sentiment", "sent"),
    "freq_raw": ("freq_raw", "frequency", "freq"),
    "cultural_raw": ("cultural_raw", "cultural_significance", "cult"),
    "religious_raw": ("religious_raw", "religious_significance", "rel"),
    "historical_raw": ("historical_raw", "historical_significance", "hist"),
    "scape_raw": ("scape_raw", "scape"),
}
_LIST_FIELDS = {"sims", "tags", "domain_raw"}
SPLITS = ("space", "probe", "validation", "test")


def normalize_phrase(value: Any) -> str:
    """Normalize spelling/spacing for grouping without removing internal words."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip(
        " .,!?:;\"'“”‘’\u0964"
    )


_JOINERS = re.compile(r"[\s\-­‐-―−​-‍⁠﻿]+")


def normalize_idiom_type(value: Any) -> str:
    """Group spelling-equivalent forms: whitespace, hyphens and (zero-width) joiners are ignored.

    This deliberately does not merge supplied semantic alternatives. Record and
    item identities continue to use the legacy content digest unchanged.
    """
    return _JOINERS.sub("", normalize_phrase(value))


def is_form_variant(a: Any, b: Any, threshold: float = .7) -> bool:
    """True for spelling/form variants of one expression (e.g. তেঁদড়/ত্যাঁদড়).

    Most supplied ``alternative_idioms`` are *different* idioms with a similar
    meaning (e.g. কচু / অষ্টরম্ভা); those are not variants and stay unlinked.
    """
    import difflib
    x, y = normalize_idiom_type(a), normalize_idiom_type(b)
    if not x or not y:
        return False
    return x == y or difflib.SequenceMatcher(None, x, y).ratio() >= threshold


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return next((t for v in value if (t := _text(v))), "")
    if isinstance(value, dict):
        return " ; ".join(t for v in value.values() if (t := _text(v)))
    value = str(value).strip()
    return "" if value.casefold() in _MISSING else value


def _list(value: Any) -> list[str]:
    if isinstance(value, str) and value.lstrip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    values = value if isinstance(value, (list, tuple)) else [value]
    return list(dict.fromkeys(t for v in values if (t := _text(v))))


def _flag(value: Any) -> bool:
    # Source significance fields can be booleans, 0/1, or descriptive prose.
    # Keep the original field next to this explicitly derived presence flag.
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and value != 0)
    return _text(value).casefold() not in (_FALSE | {"0.0"})


def _hash(value: Any, n: int = 20) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False).to_dict("records")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path).to_dict("records")
    with path.open(encoding="utf-8-sig") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            obj = [json.loads(line) for line in handle if line.strip()]
    if isinstance(obj, dict):
        for key in ("data", "records", "idioms", "items"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            if any(key in obj for key in _FIELDS["idiom"]):
                obj = [obj]
            elif all(isinstance(v, dict) for v in obj.values()):
                obj = list(obj.values())
            else:
                candidates = [v for v in obj.values() if isinstance(v, list)]
                if len(candidates) != 1:
                    raise ValueError("Cannot identify the records list in JSON; use a records/data wrapper.")
                obj = candidates[0]
    if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
        raise ValueError("Corpus must contain an array of record objects.")
    return obj


def load_corpus(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a local corpus without filtering missing conditions or duplicate senses.

    ``idiom_id`` identifies a sense/record by source ID and canonical content;
    ``idiom_group`` identifies the normalized Bangla expression. Exact duplicate
    records receive deterministic occurrence suffixes and remain separate rows.
    Unqualified ``literal_meaning`` is Bangla, never an English label input.
    """
    path = Path(path)
    records = _read_records(path)
    if not records:
        raise ValueError(f"Corpus has no records: {path}")
    columns = {str(c).casefold(): c for r in records for c in r}
    mapping = {name: next((columns[a.casefold()] for a in aliases if a.casefold() in columns), None)
               for name, aliases in _FIELDS.items()}
    if mapping["idiom"] is None:
        raise ValueError(f"No idiom field found. Columns: {list(columns.values())}")
    rows, occurrences = [], Counter()
    for index, source in enumerate(records):
        row = {name: (_list(source.get(col)) if name in _LIST_FIELDS else _text(source.get(col)))
               for name, col in mapping.items()}
        examples = {}
        for lang in ("bn", "en"):
            raw_examples = source.get(mapping[f"{lang}_ex"])
            raw_examples = raw_examples if isinstance(raw_examples, (list, tuple)) else [raw_examples]
            examples[lang] = [_text(v) for v in raw_examples]
        exact = [j for j, sentence in enumerate(examples["bn"]) if row["idiom"] and row["idiom"] in sentence]
        first_available = next((j for j, sentence in enumerate(examples["bn"]) if sentence), 0)
        selected = exact[0] if exact else first_available
        row["bn_ex"] = examples["bn"][selected] if selected < len(examples["bn"]) else ""
        row["en_ex"] = examples["en"][selected] if selected < len(examples["en"]) else ""
        row["bn_examples"], row["en_examples"] = examples["bn"], examples["en"]
        row["example_index"] = selected
        row["example_selection"] = "first_exact_idiom_span" if exact else "first_available_no_exact_span"
        source_id = _text(source.get("id", source.get("source_record_id", "")))
        digest = _hash({"source_id": source_id, **row})
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        row.update(idiom_id=f"idiom_{digest}" + (f"_{occurrence}" if occurrence else ""),
                   source_record_id=source_id, source_id=source_id, source_row=index,
                   idiom_group=f"bn_{_hash(normalize_idiom_type(row['idiom']))}")
        # These sidecars MUST follow the legacy digest. Adding an alias to the
        # source is grouping metadata, not a new record/input identity; changing
        # the digest here would unnecessarily invalidate existing GPU features.
        row["alternative_idioms"] = _list(source.get("alternative_idioms"))
        row["idiom_spelling_key"] = normalize_idiom_type(row["idiom"])
        row["anchored"] = bool(row["sims"])
        for name in ("cultural", "religious", "historical"):
            row[name] = _flag(row[f"{name}_raw"])
        row["religious_or_historical"] = row["religious"] or row["historical"]
        row["rural"] = normalize_phrase(row["scape_raw"]) in {"deep rural", "rural"}
        row["rare"] = bool(re.search(r"\brare\b", row["freq_raw"], flags=re.I))
        row["idiom_len"] = len(row["idiom"].split())
        row["lit_en_source"] = mapping["lit_en"] or "unavailable"
        row["lit_bn_source"] = mapping["lit_bn"] or "unavailable"
        row["lex"] = any(normalize_phrase(cp) in normalize_phrase(row["en_ex"]) for cp in row["sims"])
        rows.append(row)
    df = pd.DataFrame(rows)
    # Missing idiom strings remain visible in the audit, but must never group
    # otherwise unrelated missing records together.
    missing = df.idiom.eq("")
    df.loc[missing, "idiom_group"] = df.loc[missing, "idiom_id"].map(lambda v: f"missing_{v}")
    audit = {
        "path": str(path.resolve()), "records": len(df), "retained_records": len(df),
        "column_mapping": mapping, "unique_idioms": int(df.loc[~missing, "idiom_group"].nunique()),
        "duplicate_idiom_rows": int((~missing).sum() - df.loc[~missing, "idiom_group"].nunique()),
        "exact_duplicate_rows": sum(n - 1 for n in occurrences.values()),
        "absent_counterparts": int((~df.anchored).sum()),
        "idiom_grouping": "NFKC/casefold/punctuation-trim followed by whitespace removal; semantic alternatives are not merged",
        "example_selection": df.example_selection.value_counts().to_dict(),
        "later_matching_examples_selected": int(df.example_index.gt(0).sum()),
        "first_example_exact_span": int(sum(bool(row['idiom']) and bool(row['bn_examples']) and
                                              row['idiom'] in row['bn_examples'][0] for row in rows)),
        "any_example_exact_span": int(df.example_selection.eq("first_exact_idiom_span").sum()),
        "missing": {name: int(df[name].eq("").sum()) for name in ("idiom", "bn_ex", "en_ex", "fig_en", "fig_bn", "lit_en", "lit_bn")},
        "flag_counts": {name: {"true": int(df[name].sum()), "false": int((~df[name]).sum())}
                        for name in ("cultural", "religious", "historical", "rural", "rare")},
        "sentiment_counts": df.sentiment_raw.value_counts(dropna=False).to_dict(),
        "notes": ["No rows removed for unavailable English counterparts or repeated Bangla idioms.",
                  "Anchored means at least one supplied English counterpart; it is not a transparency score.",
                  "Unqualified literal_meaning is retained as lit_bn, not passed to an English encoder.",
                  "Significance flags are source-value presence/0-1 flags; inspect their prevalence before moderator analysis."],
    }
    df.attrs["corpus_audit"] = audit
    return df, audit


class SplitFeasibilityError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


_INPUT_FIELDS = ("bn_ex", "en_ex", "fig_en", "fig_bn")


def _input_keys(row: dict) -> list:
    """Normalized texts that can become model inputs (all examples, glosses, probed counterpart)."""
    keys = [(name, normalize_phrase(row.get(name, ""))) for name in _INPUT_FIELDS]
    for lang in ("bn", "en"):
        examples = row.get(f"{lang}_examples")
        if isinstance(examples, (list, tuple)):
            keys += [(f"{lang}_ex", normalize_phrase(v)) for v in examples if isinstance(v, str)]
    sims = _list(row.get("sims"))
    if sims:
        keys.append(("en_cp", normalize_phrase(sims[0])))
    return [k for k in keys if k[1]]


def split_corpus(df: pd.DataFrame, seed: int = 0, fractions=(.40, .25, .10, .25), *,
                 mode: str = "input_disjoint", max_deviation: float = .05) -> pd.DataFrame:
    """Split connected components of records that must never be separated.

    ``input_disjoint`` (default) links records that share a Bangla idiom type
    (whitespace/hyphen/joiner-insensitive), a listed *form variant* that is
    itself a corpus entry, or any identical model input: the selected example
    sentences, either figurative gloss, or the probed (first) English
    counterpart. No input string can therefore occur in two splits.
    ``counterpart_closure`` (alias ``strict``) additionally links *every*
    listed English counterpart transitively; on Bagdhara this forms a giant
    component (~64% of records), which forces it into SPACE and restricts the
    test population. ``idiom_only`` links Bangla types only (exploratory).
    A giant component makes the requested split infeasible rather than
    justifying leakage or dropping records.
    """
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 4 or not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("fractions must contain four positive values summing to one.")
    if mode == "counterpart_closure":
        mode = "strict"
    if mode not in {"input_disjoint", "strict", "idiom_only"}:
        raise ValueError("split mode must be 'input_disjoint', 'counterpart_closure'/'strict' or 'idiom_only'.")
    if not len(df) or not df.idiom_id.is_unique:
        raise ValueError("Splitting requires nonempty data with unique idiom_id values.")
    n, parent = len(df), list(range(len(df)))
    corpus_types = {normalize_idiom_type(v) for v in df.idiom if normalize_idiom_type(v)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    seen = {}
    linked_variants = 0
    for i, row in enumerate(df.to_dict("records")):
        idiom = normalize_idiom_type(row["idiom"])
        keys = [("bn", idiom)] if idiom else [("record", row["idiom_id"])]
        if mode in {"input_disjoint", "strict"}:
            for alias in _list(row.get("alternative_idioms")):
                key = normalize_idiom_type(alias)
                if key and key != idiom and key in corpus_types and is_form_variant(alias, row["idiom"]):
                    keys.append(("bn", key))
                    linked_variants += 1
            keys += _input_keys(row)
        if mode == "strict":
            keys += [("en", normalize_phrase(cp)) for cp in _list(row.get("sims")) if normalize_phrase(cp)]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    components = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    # Hash sorted member IDs, not unstable union-find roots or source ordering.
    groups = [(f"component_{_hash(sorted(df.iloc[ix].idiom_id.astype(str)))}", ix) for ix in components.values()]
    sizes = sorted((len(ix) for _, ix in groups), reverse=True)
    report = {"mode": mode, "records": n, "components": len(groups), "largest_component": sizes[0],
              "largest_fraction": sizes[0] / n, "largest_component_sizes": sizes[:20],
              "requested_fractions": dict(zip(SPLITS, fractions.tolist())), "seed": seed,
              "counterpart_disjoint": mode == "strict", "input_disjoint": mode in {"input_disjoint", "strict"},
              "form_variant_links": linked_variants,
              "linking_keys": {"input_disjoint": "Bangla type (whitespace/hyphen/joiner-insensitive) + listed form variants "
                                                 "+ identical example/gloss/probed-counterpart inputs",
                               "strict": "input_disjoint keys + every listed English counterpart (transitive closure)",
                               "idiom_only": "Bangla type only"}[mode]}
    rounding = 1 / n
    if len(groups) < 4 or sizes[0] / n > fractions.max() + max(max_deviation, rounding):
        raise SplitFeasibilityError(
            f"Requested {mode} split is infeasible: {len(groups)} components; largest has {sizes[0]}/{n} "
            f"rows ({sizes[0]/n:.1%}), requested maximum {fractions.max():.1%}. "
            "No records were dropped. Use an explicitly documented alternative split policy.", report)
    rng = np.random.default_rng(seed)
    # Stable input ordering + seeded tie breaks gives reproducible assignments.
    groups.sort(key=lambda g: g[0])
    jitter = {key: rng.random() for key, _ in groups}
    groups.sort(key=lambda g: (-len(g[1]), jitter[g[0]]))
    targets, counts = fractions * n, np.zeros(4)
    assignment, comp_ids = np.empty(n, dtype=object), np.empty(n, dtype=object)
    component_split_counts = Counter()
    for number, (key, ix) in enumerate(groups):
        candidates = []
        for j in range(4):
            trial = counts.copy()
            trial[j] += len(ix)
            candidates.append(float(np.sum((trial - targets) ** 2 / targets)))
        j = 0 if number == 0 else int(np.argmin(candidates))
        counts[j] += len(ix)
        assignment[ix], comp_ids[ix] = SPLITS[j], key
        component_split_counts[SPLITS[j]] += 1
    result = df.copy()
    result["split"], result["component_id"] = assignment, comp_ids
    counterpart_splits = {}
    for row in result.to_dict("records"):
        for cp in _list(row.get("sims")):
            key = normalize_phrase(cp)
            if key:
                counterpart_splits.setdefault(key, set()).add(row["split"])
    leaks = {key: sorted(value) for key, value in counterpart_splits.items() if len(value) > 1}
    input_splits = {}
    for row in result.to_dict("records"):
        for key in _input_keys(row):
            input_splits.setdefault(key, set()).add(row["split"])
    shared_inputs = {f"{k[0]}:{k[1]}": sorted(v) for k, v in input_splits.items() if len(v) > 1}
    if mode in {"input_disjoint", "strict"} and shared_inputs:
        raise AssertionError(f"Identical model inputs cross split boundaries: {list(shared_inputs)[:5]}")
    report.update(cross_split_identical_inputs=len(shared_inputs),
                  identical_input_examples=dict(list(sorted(shared_inputs.items()))[:10]))
    report.update(counts={split: int(count) for split, count in zip(SPLITS, counts)},
                  achieved_fractions={split: float(count / n) for split, count in zip(SPLITS, counts)},
                  max_fraction_deviation=float(np.max(np.abs(counts / n - fractions))),
                  overlapping_counterparts=len(leaks), counterpart_overlap_examples=dict(list(sorted(leaks.items()))[:20]),
                  split_component_counts=dict(component_split_counts), giant_component_id=groups[0][0],
                  giant_component_split="space", coverage_warning=(
                      "The largest connected component is assigned to SPACE. Test estimates apply to other components, "
                      "not to the full connected semantic network; report this coverage bias."
                      if sizes[0] / n > .2 else ""))
    spelling_groups = {}
    aliases = Counter()
    skipped_alias_examples = []
    split_of_type = {}
    for row in result.to_dict("records"):
        key = normalize_idiom_type(row["idiom"])
        if key:
            group = spelling_groups.setdefault(key, {"forms": set(), "splits": set(), "rows": 0})
            group["forms"].add(normalize_phrase(row["idiom"]))
            group["splits"].add(row["split"])
            group["rows"] += 1
            split_of_type.setdefault(key, set()).add(row["split"])
    variant_pairs_crossing = 0
    for row in result.to_dict("records"):
        key = normalize_idiom_type(row["idiom"])
        for alias in _list(row.get("alternative_idioms")):
            aliases["supplied_entries"] += 1
            alias_key = normalize_idiom_type(alias)
            if key and alias_key == key:
                aliases["spelling_equivalent_entries"] += 1
            elif alias_key in corpus_types and is_form_variant(alias, row["idiom"]):
                aliases["form_variant_entries"] += 1
                if mode != "idiom_only" and split_of_type.get(alias_key, set()) - {row["split"]}:
                    variant_pairs_crossing += 1
            else:
                aliases["semantic_alternatives_not_linked"] += 1
                if len(skipped_alias_examples) < 20:
                    skipped_alias_examples.append({"source_id": row.get("source_id", ""), "alias": alias,
                                                   "reason": "different_expression_or_not_a_corpus_entry"})
    variants = [group for group in spelling_groups.values() if len(group["forms"]) > 1]
    cross_split_variants = [group for group in variants if len(group["splits"]) > 1]
    report.update(idiom_grouping="whitespace/hyphen/joiner-insensitive normalized primary spelling"
                                 + (" plus listed form variants" if mode != "idiom_only" else ""),
                  spelling_variant_groups=len(variants), spelling_variant_rows=sum(g["rows"] for g in variants),
                  cross_split_spelling_variant_groups=len(cross_split_variants),
                  cross_split_form_variant_pairs=variant_pairs_crossing,
                  alias_audit={**dict(aliases), "policy": "spelling-equivalent forms and listed form variants (similarity >= 0.7) "
                                                          "are linked; listed alternatives that are different idioms are not merged",
                               "unlinked_examples": skipped_alias_examples})
    if cross_split_variants or variant_pairs_crossing:
        raise AssertionError("Spelling-equivalent idiom types or listed form variants cross split boundaries.")
    result.attrs["split_report"] = report
    return result


_CONTROL_LANGS = {"bn-lit": "bn", "bn-ctrl": "bn", "bn-rom": "bn-Latn", "natural-en": "en",
                  "natural-hi": "hi", "natural-ur": "ur", "natural-kok": "kok"}
_CONTROL_ALIASES = {"natural-hindi": "natural-hi", "natural-urdu": "natural-ur", "natural-konkani": "natural-kok"}


def build_items(df: pd.DataFrame, controls_path: str | Path | None = None) -> pd.DataFrame:
    """Build explicit input conditions and auditable span/pooling metadata.

    External controls require approval and an exact span in their supplied text.
    Literal readings, nonidiomatic controls, romanization and third-language
    examples are never invented from glosses by this function.
    """
    if not df.idiom_id.is_unique:
        raise ValueError("idiom_id must be unique before building conditions.")
    rows, skipped = [], Counter()

    def add(meta, condition, lang, text, span, pool_target="span", **extra):
        text, span = _text(text), _text(span)
        if not text:
            skipped[f"{condition}:missing_text"] += 1
            return
        start = text.find(span) if span else -1
        found = start >= 0
        row = dict(meta)
        row.update(condition=condition, lang=lang, text=text, span=span, pool_target=pool_target,
                   span_found=found, span_start=start, span_end=start + len(span) if found else -1,
                   span_count=text.count(span) if span else 0,
                   span_status=("ambiguous_multiple" if found and text.count(span) > 1 else "exact") if found else "missing",
                   item_id=f"item_{_hash([meta['idiom_id'], condition, lang, text, span, pool_target])}", **extra)
        rows.append(row)

    for meta in df.to_dict("records"):
        idiom = _text(meta.get("idiom"))
        bn, en = _text(meta.get("bn_ex")), _text(meta.get("en_ex"))
        add(meta, "bn-fig", "bn", bn, idiom)
        add(meta, "bn-sentence", "bn", bn, bn, "sentence")
        add(meta, "en-trans", "en", en, en, "sentence")
        if idiom:
            add(meta, "bn-idiom", "bn", f'এখানে “{idiom}” অভিব্যক্তিটি ব্যবহার করা হয়েছে।', idiom)
        counterparts = _list(meta.get("sims"))
        if counterparts:
            cp = counterparts[0]
            add(meta, "en-idiom", "en", f'The expression “{cp}” is used here.', cp, counterpart_selected=cp)
        else:
            skipped["en-idiom:missing_counterpart"] += 1
        if idiom and idiom in bn:
            add(meta, "bn-context", "bn", bn.replace(idiom, "[MASK]"), "[MASK]", "context")
        else:
            skipped["bn-context:missing_exact_idiom_span"] += 1
        # A second, different exact-span context of the same idiom (identity study).
        others = [s for j, s in enumerate(meta.get("bn_examples") or [])
                  if j != meta.get("example_index") and isinstance(s, str) and idiom and idiom in s and s != bn]
        if others:
            add(meta, "bn-fig2", "bn", others[0], idiom)
            # Same second context with the idiom masked: identity from context alone.
            add(meta, "bn-context2", "bn", others[0].replace(idiom, "[MASK]"), "[MASK]", "context")
        else:
            skipped["bn-fig2:no_second_exact_example"] += 1
        for lang in ("bn", "en"):
            gloss = _text(meta.get(f"fig_{lang}"))
            add(meta, f"{lang}-para", lang, gloss, gloss, "sentence")
    control_audit = {"provided": controls_path is not None, "approved_rows": 0, "unapproved_rows": 0}
    if controls_path is not None:
        controls = pd.read_csv(controls_path, keep_default_na=False)
        required = {"idiom_id", "condition", "lang", "text", "span", "approved"}
        if not required.issubset(controls.columns):
            raise ValueError(f"Control CSV is missing columns: {sorted(required - set(controls.columns))}")
        lookup = {str(r["idiom_id"]): r for r in df.to_dict("records")}
        approved_keys = set()
        for line, row in enumerate(controls.to_dict("records"), start=2):
            if str(row["approved"]).strip().casefold() not in _TRUE:
                control_audit["unapproved_rows"] += 1
                continue
            condition = _CONTROL_ALIASES.get(row["condition"], row["condition"])
            if condition not in _CONTROL_LANGS:
                raise ValueError(f"Control line {line}: unsupported condition {condition!r}.")
            if str(row["idiom_id"]) not in lookup:
                raise ValueError(f"Control line {line}: unknown idiom_id {row['idiom_id']!r}.")
            expected_lang = _CONTROL_LANGS[condition]
            if row["lang"] != expected_lang:
                raise ValueError(f"Control line {line}: {condition} requires lang={expected_lang!r}.")
            if not _text(row["text"]) or not _text(row["span"]) or row["span"] not in row["text"]:
                raise ValueError(f"Control line {line}: approved control needs nonempty text and an exact span.")
            key = (str(row["idiom_id"]), condition)
            if key in approved_keys:
                raise ValueError(f"Control line {line}: duplicate approved idiom_id/condition {key}.")
            approved_keys.add(key)
            add(lookup[key[0]], condition, expected_lang, row["text"], row["span"],
                control_approved=True, control_source=_text(row.get("source")))
            control_audit["approved_rows"] += 1
    items = pd.DataFrame(rows)
    if items.empty:
        raise ValueError("No input conditions could be constructed from the supplied corpus.")
    if not items.item_id.is_unique:
        raise ValueError("Duplicate item_id generated; verify unique corpus records and controls.")
    items.attrs["item_audit"] = {
        "items": len(items), "conditions": items.condition.value_counts().to_dict(),
        "missing_spans": items.loc[~items.span_found, "condition"].value_counts().to_dict(),
        "ambiguous_spans": items.loc[items.span_count.gt(1), "condition"].value_counts().to_dict(),
        "skipped": dict(skipped), "controls": control_audit,
        "notes": ["Sentence conditions pool the explicitly supplied full sentence/gloss.",
                  "Context conditions exclude [MASK] placeholder tokens from pooling.",
                  "Missing exact idiom spans remain invalid; no sentence fallback is permitted."],
    }
    return items
