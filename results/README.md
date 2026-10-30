# Results

Every number the paper reports is computed from the aggregates in this folder. They hold no corpus records beyond short illustrative cases, no model weights and no item-level feature caches; checkpoints and item-level predictions stayed on the GPU machine.

| Folder | Contents | Written by |
|:---|:---|:---|
| [`discovery/`](discovery/) | Derived tables of the discovery campaign (split seed 0): the planned confirmatory family (`primary_confirmatory.csv`), all selected-layer inference, layer profiles, estimator and baseline comparisons, retrieval, subspace overlap, input controls, identity, decoder, moderators, audits and phase durations | [`analysis/analyze.py`](../analysis/analyze.py) and the other campaign scripts in `analysis/` |
| [`discovery/synthesis/`](discovery/synthesis/) | Descriptive re-analyses of those tables: the input ladder, estimator families, erasure transfer, class transfer, the F taxonomy's class aggregates | [`analysis/synthesize.py`](../analysis/synthesize.py), [`analysis/rebuild_taxonomy.py`](../analysis/rebuild_taxonomy.py) |
| [`replication/`](replication/) | The prospective internal replication on split seed 1: the frozen-plan record (`PLAN_FROZEN.json`), 48 Holm-corrected hypotheses, readout scores, mechanisms, layer selections, split and label audits, the split overlap with the discovery study, and the corpus checksum (`input_provenance.json`) | [`run_replication.py`](../run_replication.py), [`analysis/validation_selected_mlr.py`](../analysis/validation_selected_mlr.py) |
| [`experiments/`](experiments/) | One folder per follow-up experiment X1–X19, each with a `MANIFEST.json`, and the verification reports | [`run_experiments.py`](../run_experiments.py) |
| [`human_validation/`](human_validation/) | The volunteers' judgments without corpus text, agreement statistics and per-class summaries | [`analysis/human_validation.py`](../analysis/human_validation.py) |
| [`provenance/`](provenance/) | Pinned Hub revisions, licenses, parameter counts and layer counts of the six encoders and the decoder, and the auxiliary models (MiniLM taxonomy encoder, NLLB) | |

Machine-specific paths in the recorded JSON are replaced by `CAMPAIGN`, `PYTHON` and `BACKUP`. The planned confirmatory family is the twelve MLR tests in `discovery/primary_confirmatory.csv`; the replication's 48 hypotheses form their own Holm family, and X16 tests its own frozen plan. Everything else is descriptive or exploratory, as each file's producing script states.
