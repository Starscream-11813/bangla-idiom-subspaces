# Aggregates of the follow-up experiments X1–X19

Written by [`run_experiments.py`](../../run_experiments.py) on one GPU, after each group's reproduction gate passed. Each folder holds small CSV aggregates and a `MANIFEST.json` with the options, row counts, environment and the SHA-256 of the code that ran. X16's manifest also records the SHA-256 of its frozen plan, [`configs/plans/PLAN_X16.json`](../../configs/plans/PLAN_X16.json). Checkpoints, fitted erasers and item-level predictions stayed below the run's `OUT` directory.

| Folder | Question | Main files |
|:---|:---|:---|
| `X1_zero_shot` | Does an English-trained classifier read Bangla through English directions with no Bangla labels? | `summary.csv` |
| `X2_concepts` | Does the planned test hold for concepts not built from English glosses (a Bangla-gloss taxonomy FBN, usage domains DOM)? | `tests.csv`, `ladder.csv`, `contrasts.csv`, `labels_split0.csv.gz` |
| `X3_leace_ablation` | Is LEACE's cross-lingual deficit due to whitening error, sample size or the probe penalty? | `summary.csv` |
| `X4_pooling_matched` | Does the input or the pooled tokens move accuracy (idiom present or masked × span or context pooling)? | `cells.csv`, `contrasts.csv`, `extraction_check.csv`, `tokens.csv` |
| `X5_balanced_metrics` | Do the planned tests and their nulls agree under balanced accuracy and macro-F1? Gate for X1–X8 | `tests.csv`, `agreement.csv`, `null_metrics.csv.gz` |
| `X6_nonlinear_mdl` | Do MLP readouts or online-code MDL change the picture? | `mlp.csv`, `mdl.csv`, `mdl_summary.csv` |
| `X7_alignment` | Do CSLS retrieval and an orthogonal Procrustes map change retrieval, SSI or erasure? | `aligned.csv`, `retrieval.csv`, `procrustes_fit.csv` |
| `X8_layerwise` | Do COV versus MLR and the null comparisons hold at every cached layer? | `layers.csv`, `contrasts.csv`, `summary.csv` |
| `X9_erasers_nonlinear` | Do INLP, R-LACE or rotated erasers guard the concept, against linear and MLP readouts? | `linear.csv`, `mlp.csv`, `erasers.csv`, `path.csv`, `summary.csv` |
| `X10_heldout_alignment` | Do maps fitted on held-out pairs change the alignment results? Gate for X9–X14 | `aligned.csv`, `retrieval.csv`, `procrustes_fit.csv`, `summary.csv` |
| `X11_taxonomy_robustness` | Do re-seeded F taxonomies and a tag-built taxonomy (TAG) reproduce the planned test? | `tests.csv`, `ladder.csv`, `contrasts.csv`, `labels_split0.csv.gz` |
| `X12_penalty_path` | Does transfer survive along MLR's penalty path and LEACE's ridge path? | `path.csv`, `selected.csv`, `summary.csv` |
| `X13_idiom_isolated` | Is the idiom read without its context (`bn-idiom`), with absolute effects? | `tests.csv`, `absolute.csv`, `paired.csv`, `zero_shot.csv` |
| `X15_matched_nulls` | Does COV beat a covariance-matched random null and label-free PCA? Gate for X15–X19 | `tests.csv` |
| `X16_idiom_prospective` | The idiom without context, tested prospectively on split seed 1 under a frozen plan | `tests.csv`, `gate.csv`, `zero_shot.csv` |
| `X17_human_items` | Do the volunteers' item-level judgments predict cross-lingual readouts? | `contrasts.csv`, `accuracy.csv`, `agreement.csv`, `coverage.csv` |
| `X18_refit_bootstrap` | How wide are intervals that refit the subspaces and probes in every draw? | `summary.csv`, `draws.csv` |
| `X19_counterpart_type` | Do idiom-like or paraphrase-like English counterparts carry the shared signal? | `subsets.csv`, `summary.csv` |

X14, the decoder with a powered planned test, is optional and was not run.

The verification reports are `VERIFY_X1-X8.json`, `VERIFY_X9-X14.json` and `VERIFY_X15-X19.json`, with the gate checks in `VERIFY_X5.json`, `VERIFY_X10.json` and `VERIFY_X15.json`. Every gate and every experiment that ran passed. The reports were written under their original names (`VERIFY.json` per group) and renamed for this release.

The manifests record code hashes under the development layout, before the code was reorganized for release (`new_experiments*/` for the experiment modules, `code_and_results/` for the library). The library entries still match [`idiom_subspace/`](../../idiom_subspace/) byte for byte. The experiment modules here differ from the recorded ones in paths (imports, reference tables, output folders and report names), in which files they hash, and in docstrings; no computation changed.
