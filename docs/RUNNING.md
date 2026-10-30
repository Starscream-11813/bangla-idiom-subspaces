# Running the experiments

This runbook reproduces the study from the raw corpus: the discovery campaign (studies S1–S4), its analysis, the prospective internal replication on a second split (E1–E4), and the follow-up experiments X1–X19. The released `results/` folder already holds every aggregate the paper reports, so none of this is needed to inspect them.

The study ran on one 24 GB NVIDIA RTX 3090. Measured durations of the discovery campaign are in [`results/discovery/phase_durations.csv`](../results/discovery/phase_durations.csv): about 6 hours for the six-encoder main phase, 50 minutes for the identity profile and 3 hours for the decoder profile. The follow-up experiments were budgeted at roughly 11–17 hours for X1–X8, 12–19 hours for X9–X14 and 6–9 hours for X15–X19 on the same GPU (estimates, not measurements). X1–X8 need about 17 GB of free disk below `OUT`, the other groups a few GB, beyond the model and feature caches.

Placeholders used below: `CAMPAIGN` is the campaign directory, `OUT` a work directory for checkpoints, and `/path/to/bagdhara.json` the corpus export.

## 1. Environment

Use Python 3.10 or later (the study used 3.11) and a CUDA build of PyTorch that matches the machine's driver, installed first from the [official selector](https://pytorch.org/get-started/locally/). Then:

```bash
python -m pip install -r requirements.txt        # includes the pinned BanglaBERT normalizer from GitHub
python -m pip install -r requirements-decoder.txt   # only for 4-bit decoder loading on GPUs under 20 GB
python -m pytest -q                               # CPU only; no pretrained weights or downloads needed
```

The test suite includes three end-to-end smoke tests that build a miniature campaign from two tiny random encoders and run every follow-up experiment on it (a few minutes each). `python smoke_offline.py --out out_offline_smoke --probe-backend torch` exercises extraction, all studies and checkpoint resumption the same way.

## 2. The corpus

Download the Bagdhara export (CC BY-SA 4.0) from Kaggle, either manually or with `python run_research.py download-data`, and check its checksum. The study used `bagdhara_export_done_2026-03-01_010356.json`, whose SHA-256 is recorded in [`results/replication/input_provenance.json`](../results/replication/input_provenance.json):

```bash
sha256sum /path/to/bagdhara.json   # 9f370e6705185f0fdd349cb332571d3c8622954b1b8a1419d6bd482ec4c2a640
python run_research.py audit --config configs/full.json --data /path/to/bagdhara.json --out CAMPAIGN/audit
```

The audit is offline and writes the split, item and label audits that [DATA_AUDIT.md](DATA_AUDIT.md) summarizes. The follow-up scripts expect a copy of the export inside the campaign, at `CAMPAIGN/inputs/bagdhara_export_done_2026-03-01_010356.json`.

## 3. Campaign configurations

The profiles in `configs/` are device-neutral. For a CUDA campaign, derive copies that enable the GPU paths without changing the scientific design; the follow-up scripts read `CAMPAIGN/configs/full_3090.json`:

```bash
mkdir -p CAMPAIGN/configs CAMPAIGN/inputs
cp /path/to/bagdhara.json CAMPAIGN/inputs/bagdhara_export_done_2026-03-01_010356.json
python - <<'PY'
import json
from pathlib import Path
from idiom_subspace.config import Config
src, dst = Path("configs"), Path("CAMPAIGN/configs")
for name in ["quick", "full", "identity_openclass", "decoder_3090"]:
    data = json.loads((src / f"{name}.json").read_text(encoding="utf-8"))
    data.update(device="cuda", probe_backend="torch", probe_device="cuda")
    target = dst / (f"{name}.json" if name.endswith("_3090") else f"{name}_3090.json")
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    Config.load(target)
PY
```

Pin `models[].revision` and `label_revision` in these copies to the Hub commits listed in [`results/provenance/model_artifacts.json`](../results/provenance/model_artifacts.json) so that a restart cannot pick up a newer upstream revision.

| Profile | Purpose |
|:---|:---|
| `configs/quick.json` | mBERT, three layers, ranks 1/2/3, 9 null draws: an execution check |
| `configs/full.json` | Six encoders, seven estimators, full hidden dimensions, all studies, 1,999 planned null draws: the main run |
| `configs/identity_openclass.json` | Idiom identity as an open-class concept, with second contexts |
| `configs/decoder_3090.json` | Qwen3-8B in bf16; nine null draws, so its inference is exploratory |
| `configs/screen_3090.json` | XLM-R-large, S1 and S3 only: a fast screen |
| `configs/counterpart_closure.json`, `idiom_only.json`, `pca128_sensitivity.json`, `taxonomy16.json`, `tfidf_labels.json` | Sensitivity profiles (split policy, reduced space, taxonomy) |
| `configs/annotations.json` | Annotation templates (`templates` command) |

Every profile runs one split seed and one taxonomy seed. Replicate refits are opt-in (`--seeds 0,1,2 --label-seeds 0,1` with a fresh output folder) and enlarge the planned Holm family, which `inference_design.json` records.

## 4. Discovery campaign (studies S1–S4)

Run the phases in order, one GPU job at a time:

```bash
python run_research.py run --config CAMPAIGN/configs/quick_3090.json --data /path/to/bagdhara.json \
    --out CAMPAIGN/gpu_smoke --probe-backend torch                 # hardware check

python run_research.py translate-literals --config CAMPAIGN/configs/full_3090.json \
    --data /path/to/bagdhara.json --out CAMPAIGN/literals          # English literal glosses (NLLB), approved=machine

python launch_gpu_jobs.py --config CAMPAIGN/configs/full_3090.json --data /path/to/bagdhara.json \
    --out CAMPAIGN/primary --gpus 0 --probe-backend torch \
    --literal-reviews CAMPAIGN/literals/literal_glosses_mt.csv    # six encoders, sequentially

python launch_gpu_jobs.py --config CAMPAIGN/configs/identity_openclass_3090.json \
    --data /path/to/bagdhara.json --out CAMPAIGN/identity --gpus 0 --probe-backend torch

python run_research.py run --config CAMPAIGN/configs/decoder_3090.json --data /path/to/bagdhara.json \
    --out CAMPAIGN/decoder --probe-backend torch --literal-reviews CAMPAIGN/literals/literal_glosses_mt.csv
```

`launch_gpu_jobs.py` writes one job log per model, a live queue (`launcher.json`) and the cross-job Holm-corrected `inference_all_jobs.csv`, whose `*_p_holm_all_jobs` columns are authoritative for the planned family. Interrupted runs resume from completed readouts; a change to the analysis code needs a fresh `--out`. Compatible feature caches from an earlier run can be reused read-only with `--cache-dir`.

Watch a run with the read-only dashboard, which binds to localhost:

```bash
python dashboard.py --root CAMPAIGN/primary --port 8787            # http://127.0.0.1:8787
python dashboard.py --root CAMPAIGN/primary --snapshot status.html # one self-contained file
```

The analysis scripts read the identity phase's per-model `S4_identity.csv` files concatenated into `CAMPAIGN/exports/identity_tables/S4_identity.csv`, with a `model` column.

## 5. Analysis of the discovery campaign

These scripts read the campaign read-only and write the tables in `results/discovery/`:

```bash
export IDIO_CAMPAIGN=CAMPAIGN
python analysis/analyze.py                       # planned family, layer profiles, estimators, baselines
python analysis/audit_inputs.py                  # coverage, tokenization and population diagnostics
python analysis/audit_identity_ties.py           # tie-aware identity retrieval
python analysis/additional_diagnostics.py        # post hoc balanced accuracy, class recall, complete-case GLMs
python analysis/rebuild_taxonomy.py /path/to/bagdhara.json   # class-level F taxonomy aggregates (no corpus text)
python analysis/synthesize.py                    # results/discovery/synthesis/ and analysis/figures/
python analysis/plot_results.py                  # optional exploratory figures in analysis/figures/
```

`synthesize.py` needs only the released tables, so it runs without the campaign; rerunning it reproduces every file in `results/discovery/synthesis/`.

## 6. Prospective internal replication (split seed 1)

The replication refits everything on a second input-disjoint split under the plan frozen in [`configs/plans/FOLLOWUP_PLAN.md`](../configs/plans/FOLLOWUP_PLAN.md): E1 is one Holm family of 48 tests (hypotheses H1–H4), and E2–E4 are exploratory mechanism diagnostics. The frozen plan names its runner by the development path `scripts/run_followup.py`; it is `run_replication.py` here. The replication reuses the discovery feature caches read-only, so it needs no new forward passes:

```bash
IDIO_PYTHON=python bash scripts/run_replication.sh CAMPAIGN      # writes CAMPAIGN/paper_followup_seed1_20261021
python analysis/validation_selected_mlr.py --followup CAMPAIGN/paper_followup_seed1_20261021
```

The runner records the plan's SHA-256 and the analysis sources in `PLAN_FROZEN.json` and refuses to resume if they changed. Its portable aggregates go to `results/replication/`. The second split re-partitions the same corpus: 932 of its 2,590 TEST records were discovery TEST records and the other 1,658 were discovery training records ([`split_overlap.json`](../results/replication/split_overlap.json)). It therefore repeats the analysis on a new partition, not on new data.

## 7. Follow-up experiments X1–X19

`run_experiments.py` runs the follow-up experiments against the finished campaign and its replication split. Each group has a reproduction gate that must pass before the group's other results are trusted: X5 reproduces the planned discovery tests and their nulls, X10 reproduces X7's alignment, and X15 reproduces the planned test and the estimator table.

```bash
python run_experiments.py list
IDIO_PYTHON=python bash scripts/run_experiments.sh CAMPAIGN OUT   # everything, in order, with the gates
```

The driver runs, for each group, `preflight`, the gate, `verify --check <gate>`, the remaining experiments and a full `verify`. Single commands:

```bash
python run_experiments.py preflight --group X1-X8 --campaign CAMPAIGN --out OUT
python run_experiments.py X5 --campaign CAMPAIGN --out OUT
python run_experiments.py verify --check X5
python run_experiments.py verify --group X1-X8
```

Every command resumes when re-run with the same `OUT`. Finished model × split cells are restored from `OUT/<folder>/cells/`; their store is keyed by the options and the experiment code, so changed budgets start fresh cells while the expensive readout checkpoints are still reused. `--smoke --allow-cpu` runs tiny budgets on CPU for a pipeline check. X16 refuses to start if [`configs/plans/PLAN_X16.json`](../configs/plans/PLAN_X16.json) differs from the SHA-256 pinned in `experiments/x16_idiom_prospective.py`. X14 (the decoder with a powered planned test) is optional and was not run for the paper.

Three inputs are optional, and only one experiment fails without each:

| Missing input | Fix |
|:---|:---|
| NLLB model (X2's Bangla-gloss taxonomy) | Download once: `HF_HUB_OFFLINE=0 python -c "from transformers import AutoModelForSeq2SeqLM as M, NllbTokenizerFast as T; T.from_pretrained('facebook/nllb-200-distilled-600M'); M.from_pretrained('facebook/nllb-200-distilled-600M')"`, or run X2 with `--fbn-backend tfidf` |
| Pinned encoder revision (X4's new forward passes) | Run X4 once with `HF_HUB_OFFLINE=0`; it downloads only that revision |
| BanglaBERT normalizer (X4 for BanglaBERT and BanglishBERT) | Install `requirements.txt`, which pins it |

Aggregates go to `results/experiments/<folder>/` with a `MANIFEST.json`; checkpoints, fitted erasers and item-level predictions stay below `OUT`.

## 8. Human validation

Two volunteers judged 300 discovery TEST items, 25 per F class. [`analysis/human_validation_packet.py`](../analysis/human_validation_packet.py) shows how the packet was drawn (one record per input-disjoint component, round-robin over classes) and wrote the blank sheets in `data/human_validation/annotator_{A,B}.original.csv`. The filled sheets are `annotator_{A,B}.csv`. To recompute agreement and the class-level summaries:

```bash
python analysis/human_validation.py              # writes results/human_validation/
```

X17 links these judgments to item-level readouts through `results/human_validation/judgments.csv`.
