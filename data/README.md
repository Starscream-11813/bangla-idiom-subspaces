# Data

## The Bagdhara corpus (not redistributed)

All experiments use the Bangla (Bengali) idiom corpus Bagdhara ([Sakhawat et al., 2026](https://aclanthology.org/2026.lrec-1.546/)), distributed on Kaggle as [`sakhadib/bangla-bagdhara`](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara) under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This repository does not include the corpus. Download it from Kaggle, or with `python run_research.py download-data`, and pass its path to `--data`.

The study used the export `bagdhara_export_done_2026-03-01_010356.json`:

| Property | Value |
|:---|:---|
| SHA-256 | `9f370e6705185f0fdd349cb332571d3c8622954b1b8a1419d6bd482ec4c2a640` |
| Sense records | 10,359 (the corpus paper describes 10,361) |
| Distinct idiom strings | 8,877 |
| Sentiment field | four values, including 152 `mixed` records |

The export is used unchanged. [`docs/DATA_AUDIT.md`](../docs/DATA_AUDIT.md) describes its records, spans, splits and labels, and [`results/replication/input_provenance.json`](../results/replication/input_provenance.json) records the checksum with the feature-cache fingerprints and model revisions.

## Human validation sheets

[`human_validation/`](human_validation/) holds the only new data of the study: the judgments of two volunteers, native Bangla speakers fluent in English, on 300 discovery TEST items (25 per figurative class F, one item per input-disjoint component). Each volunteer worked independently in a spreadsheet from brief written instructions. Disagreements were not adjudicated.

| File | Contents |
|:---|:---|
| `annotator_A.csv`, `annotator_B.csv` | The two filled sheets, 300 rows each |
| `annotator_A.original.csv`, `annotator_B.original.csv` | The same sheets as handed out, blank, in each volunteer's item order |

| Column | Meaning |
|:---|:---|
| `sample_id` | Stable item ID (a hash of the corpus record ID) |
| `idiom`, `bangla_example`, `english_example`, `bangla_figurative_gloss`, `english_figurative_gloss`, `english_counterpart` | Source text from Bagdhara |
| `assigned_F`, `assigned_F_name` | The item's class in the machine-built figurative taxonomy F |
| `f_plausibility` | Does the idiom fit its F class? `good`, `partial`, `poor` or `uncertain` |
| `function` | `referential`, `evaluative`, `both` or `uncertain` |
| `counterpart_type` | The English counterpart is an `idiom`, `proverb`, `literal_paraphrase`, `other` or `uncertain` |
| `counterpart_equivalence` | `equivalent`, `partly_equivalent`, `not_equivalent` or `uncertain` |
| `rationale` | Optional free-text note |
| `flagged`, `saved_at` | Unused; always empty |
| `collection_method`, `annotation_timing_status`, `imported_at` | `spreadsheet`; `unavailable` (no session times were recorded); when the sheets were imported, not when they were filled |

[`analysis/human_validation.py`](../analysis/human_validation.py) computes agreement (Cohen's κ with a 2,000-draw item bootstrap, and Gwet's AC1) and the class-level summaries in [`results/human_validation/`](../results/human_validation/). [`analysis/human_validation_packet.py`](../analysis/human_validation_packet.py) shows how the 300 items were drawn.

### License

The source-text columns derive from Bagdhara, so these sheets are distributed under the same [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, with attribution to the corpus authors:

```bibtex
@inproceedings{sakhawat-etal-2026-words,
  title     = {When Words Don{'}t Mean What They Say: Figurative Understanding in {B}engali Idioms},
  author    = {Sakhawat, Adib and Parveen, Shamim Ara and Amin, Md Ruhul and
               Khatun, Tahera and Mahmud, Shamim Al and Islam, Md Saiful},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  pages     = {6870--6879},
  publisher = {ELRA Language Resource Association},
  year      = {2026},
  doi       = {10.63317/546w2cys6m6t},
  url       = {https://aclanthology.org/2026.lrec-1.546/}
}
```
