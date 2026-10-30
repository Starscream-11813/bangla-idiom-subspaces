# Bagdhara data audit

Audited 17–20 October 2026 (spelling groups, then the input-disjoint split policy), using the export `bagdhara_export_done_2026-03-01_010356.json`. Split tables below use seed 0. These are corpus, tokenizer and offline TF-IDF label diagnostics, not neural experiment results. The raw corpus is not included in this repository; see [data/README.md](../data/README.md).

Source SHA-256: `9f370e6705185f0fdd349cb332571d3c8622954b1b8a1419d6bd482ec4c2a640`.

## Record retention

| Property | Count |
|---|---:|
| Source records, all retained | 10,359 |
| Distinct original Bangla idiom strings | 8,877 |
| Distinct whitespace-insensitive normalized Bangla idiom groups | 8,801 |
| Rows beyond one row per normalized idiom group | 1,558 |
| Records with at least one supplied English counterpart | 10,238 |
| Records without supplied English counterparts | 121 |
| Records with supplied Bangla/English examples and figurative glosses | 10,359 |
| Records with an explicit English literal-gloss field | 0 |

The corpus paper cites 10,361 entries; this export contains 10,359. Analyses must identify the actual export and checksum rather than assume the published count. Repeated idioms can represent different senses and remain separate records. The loader preserves original `source_id`/`source_record_id`, creates a stable content-based record `idiom_id`, and keeps an `idiom_group` for whitespace-insensitive normalized spelling. Exact duplicate records, if present, receive occurrence suffixes instead of being discarded.

The initial normalization retained internal word boundaries and produced 8,875 groups. A later audit found 29 candidate joined/spaced spelling groups crossing those old splits, affecting 62 records, including 20 test records. The correction groups primary spellings with the same Unicode-normalized character sequence after whitespace removal. Across the complete corpus, 74 spelling-variant groups contain 169 records; none crosses a corrected split.

The source's 4,321 `alternative_idioms` entries are preserved as sidecars **after the legacy record digest is computed**. 10 are spelling-equivalent to their primary form (whitespace, hyphens and zero-width joiners ignored) and 262 are form variants that are themselves corpus entries; both kinds are linked by the split. The remaining 4,049 are different (often synonymous) idioms and are not merged.

`anchored` means only that the source supplies a nonempty English counterpart list. It is not an embedding-similarity tertile or a measure of semantic transparency. Missing counterparts are retained for Bangla conditions and omitted only from conditions that require an English counterpart.

## Span availability and condition coverage

| Check | Count |
|---|---:|
| First supplied Bangla example contains the exact idiom | 8,286 |
| A later supplied example recovers an exact match | 540 |
| Selected Bangla example contains an exact idiom | 8,826 |
| No exact idiom occurrence across supplied examples | 1,533 |
| Selected examples containing multiple exact occurrences | 4 |

For the recovered 540 rows the selected English example uses the same source example index. This selects an existing aligned example; it does not synthesize a new sentence or substitute a paraphrase for a missing span.

With no external controls, item construction creates 81,218 rows:

| Condition | Constructed rows | Data-level limitation |
|---|---:|---|
| `bn-fig` | 10,359 | 1,533 absent exact spans; 4 repeated spans |
| `bn-sentence` | 10,359 | Explicit sentence pooling, including examples without exact idiom spans |
| `en-trans` | 10,359 | Entire translated sentence pooled |
| `bn-idiom` | 10,359 | Neutral Bangla frame; one target also repeats in its surrounding frame |
| `en-idiom` | 10,238 | 121 records lack a counterpart |
| `bn-context` | 8,826 | Requires an exact source idiom occurrence; repeated placeholders are separately diagnosed |
| `bn-para` | 10,359 | Supplied Bangla figurative gloss |
| `en-para` | 10,359 | Supplied English figurative gloss |

These are item-construction counts. Model-specific truncation, repeated spans and tokenizer coverage can further reduce usable feature counts. The main `bn-fig` condition never converts an invalid span into a sentence representation. `bn-sentence` is the separately labeled sentence sensitivity condition. Compare conditions on explicitly matched records when interpreting differences.

An actual tokenizer-only pass through all 81,218 inputs using mBERT, XLM-R, MuRIL and BanglaBERT fast tokenizers at maximum length 128 accepted 8,822 `bn-fig` items for each: 1,533 lacked an exact span and four had repeated spans. BanglaBERT used its supplied normalizer. This normalizer introduced no missing spans among raw-valid inputs and no new cross-split primary-type collision. This checks input alignment, not model activations or comprehension. Other conditions have small tokenizer-dependent exclusions, so their emitted pooling diagnostics remain authoritative.

## Leakage graph and split policy

A split must keep apart (i) every Bangla idiom type, including spelling/form variants, and (ii) every *identical model input*. The default `input_disjoint` policy therefore links records that share a whitespace/hyphen/joiner-insensitive Bangla type, a listed alternative that is a form variant of the idiom (character similarity ≥ 0.7, e.g. তেঁদড়/ত্যাঁদড়, চীনে/ছিনে জোঁক) and is itself a corpus entry, or an identical example sentence (any supplied Bangla or English example), English gloss, Bangla gloss or **probed (first) English counterpart**. Most listed `alternative_idioms` are *different* idioms with a similar meaning (e.g. কচু/অষ্টরম্ভা); linking those would split by meaning neighbourhood rather than by idiom, so they are not merged (4,049 of 4,321 entries). Secondary counterparts are never model inputs, so sharing one does not put the same input in two splits and does not link records. Without the input links (`idiom_only`), 1,544 identical inputs (mostly duplicated glosses) cross splits.

| Policy | Components | Largest component | Identical inputs across splits | Shared secondary counterparts |
|---|---:|---:|---:|---:|
| `input_disjoint` (default, 40/25/10/25) | 4,816 | 901 (8.7%) | 0 | 1,792 |
| `counterpart_closure` (earlier "strict" policy, 65/15/5/15) | 2,546 | 6,836 (66.0%) | 0 | 0 |
| `idiom_only` (exploratory) | 8,782 | 7 | 1,544 | 2,790 |

Seed 0, `input_disjoint`:

| Split | Rows | Components | Religious | Historical | Rare | Deep rural | Exact Bangla span |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPACE | 4,143 | 1,914 | 12.5% | 5.4% | 17.0% | 10.1% | 3,521 |
| PROBE | 2,590 | 1,209 | 13.9% | 6.2% | 17.8% | 10.3% | 2,227 |
| VALIDATION | 1,036 | 484 | 14.2% | 4.3% | 18.3% | 11.9% | 910 |
| TEST | 2,590 | 1,209 | 12.2% | 5.8% | 17.8% | 9.9% | 2,168 |
| Corpus | 10,359 | — | 12.9% | 5.6% | 17.5% | 10.3% | 8,826 |

Seed 0, `counterpart_closure` (the earlier default, kept as a sensitivity profile):

| Split | Rows | Components | Religious | Historical | Rare | Deep rural | Exact Bangla span |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPACE | 6,836 | 1 | 10.5% | 4.2% | 15.0% | 10.5% | 5,816 |
| PROBE | 1,510 | 1,090 | 17.7% | 8.7% | 22.3% | 9.1% | 1,277 |
| VALIDATION | 503 | 364 | 18.9% | 7.8% | 24.9% | 12.3% | 442 |
| TEST | 1,510 | 1,091 | 17.1% | 8.0% | 21.8% | 9.6% | 1,291 |

Transitively linking every listed counterpart forms one component holding two thirds of the corpus. It must sit in SPACE, so SPACE is a single permutation bundle (the component-level label-permutation null cannot move any labels) and the test set over-represents religious, historical and rare idioms. The input-disjoint split avoids both problems: every split mirrors the corpus composition and SPACE has many exchangeable components. Use `configs/counterpart_closure.json` only as a sensitivity analysis.

7,830 records have a second, different supplied Bangla example containing the exact idiom; they form the optional `bn-fig2` condition used only by the identity open-class profile (`configs/identity_openclass.json`). The default conditions still produce exactly 81,218 items, so earlier encoder caches remain valid.

## Moderator census

| Source attribute / derived flag | Count |
|---|---:|
| Cultural significance true | 10,337 |
| Cultural significance false | 22 |
| Religious significance true | 1,339 |
| Historical significance true | 577 |
| Religious OR historical significance true | 1,702 |
| Scape: country / deep rural / city | 8,961 / 1,063 / 335 |
| Frequency: common / very common | 7,245 / 1,297 |
| Frequency: rare / very rare | 1,790 / 27 |
| Derived rare flag (rare OR very rare) | 1,817 |

The cultural flag is 99.79% true and is unsuitable as a binary cultural contrast. Religious, historical and rural variables must retain their own names; substituting religious-or-historical for cultural would silently change the claim. The source contains four frequency levels. The supplied `rare` indicator is a transparent two-level collapse; analyses needing finer frequency control should use the retained `freq_raw` values.

A normalized substring diagnostic finds a supplied English counterpart in 1,805 selected English examples. This is a surface-string heuristic, not a validated count of natural idiomatic English usage or of faithful translations. The translated English examples must remain distinct from natural-English controls.

## Labels: usable targets and gaps

| Supplied sentiment | Records |
|---|---:|
| Negative | 4,952 |
| Neutral | 3,531 |
| Positive | 1,724 |
| Mixed | 152 |

All four sentiment categories occur in SPACE and remain available in all splits. They provide a supplied human-label target, while F is derived from English glosses. The offline TF-IDF audit fits 12 F classes on 6,719 eligible SPACE records. Two figurative glosses contain non-Latin/missing English-field content; additional otherwise valid glosses can become zero/out-of-vocabulary vectors under a SPACE-only TF-IDF vocabulary. Label availability therefore depends on the explicitly selected constructor.

| Offline TF-IDF target availability | SPACE | PROBE | VALIDATION | TEST |
|---|---:|---:|---:|---:|
| F | 6,719 | 1,524 | 506 | 1,510 |
| S | 6,733 | 1,554 | 518 | 1,554 |
| L | 0 | 0 | 0 | 0 |
| D, exact category-set labels | 6,733 | 345 | 114 | 364 |
| Ft, exact category-set labels | 6,733 | 3 | 2 | 4 |

D has 5,018 SPACE category-set classes, including 4,388 singleton classes. Ft has 6,629, including 6,540 singleton classes. Every source row supplies multiple domain/tag labels; selecting only the first would impose an undocumented arbitrary target. Preserving full sets reveals why these raw metadata are not a usable small human taxonomy. Defaults use F/S; a reviewed domain/tag codebook is required before making a human-domain replication claim. The TF-IDF validation split contains 11 of the fitted 12 F classes, so class coverage should also accompany evaluation counts.

The unqualified literal field contains Bangla, often with an English parenthetical. Automatically treating the entire field as English, or constructing an unrelated literal cluster index, would invalidate F/L comparison. The default does neither: L stays unavailable until English literal glosses are supplied. The `run_research.py translate-literals` command machine-translates every Bangla literal gloss (NLLB-200 by default) into the reviewed-gloss CSV format with `approved=machine`; human-corrected rows (`approved=true`) take precedence, and each record's provenance is kept in `lit_en_source` and the label audit.

## Feature compatibility and scope of verification

Before/after comparison against the unmodified loader confirmed that all 10,359 `idiom_id` and source-ID values, source order, and all 81,218 item IDs/texts/spans/pooling targets remain identical. The extraction fingerprint is unchanged. Aliases and the corrected type key are sidecar/grouping metadata, not new model inputs. Existing completed hidden-state caches can be reused read-only; old split labels, preprocessing, probes, LEACE metrics and inferential outputs must be rebuilt after a split change.

The audit ran the real local corpus through normalization, corrected strict splitting, all available built-in item construction and offline train-only labels. Lightweight tests cover stable legacy IDs, spelling equivalence, nonmerging of semantic alternatives, all-counterpart transitivity, split feasibility, paired example selection, no span fallback, control approval and independent annotation joins. Cache tests additionally reject mismatched order, shapes, effective normalized inputs and nonfinite valid rows, verify read-only external reuse and ensure cache-only misses do not load representation-model weights.

This local audit did not run the revised full neural cohort, collect literal/control annotations, validate a human F/L taxonomy, acquire natural-English or additional-language corpora, collect per-item behavioral scores, or establish causal intervention effects. Re-run the emitted audits whenever source data, normalization, split seed, label constructor or external annotations change; the current counts above describe this checksum and corrected seed-0 protocol.
