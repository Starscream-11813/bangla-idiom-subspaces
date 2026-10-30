# Experiment protocol

This maps the study design to the `idiom_subspace` package as of 20 October 2026, the version that ran the discovery campaign. It describes the implemented analyses and what each needs; implementation and tests do not establish any hypothesis. The prospective replication and the follow-up experiments X1–X19 reuse this package unchanged; [RUNNING.md](RUNNING.md) shows how to run everything.

## What the corpus can establish

The corpus supports experiments on figurative-gloss classes F, supplied human sentiment S, Bangla examples, translated English examples, neutral idiom frames, context-only inputs and supplied figurative paraphrases. With `translate-literals`, a literal-sense label L in the F taxonomy becomes available (machine translation, provenance-tagged). Natural English idiom usage, additional languages and behavioural scores require external data, which the control/score interfaces accept.

## Splits

SPACE fits preprocessing, taxonomies and subspaces; PROBE fits probes; VALIDATION selects regularization and the layer; TEST only supplies final scores. Default fractions are 40/25/10/25 over connected components of the `input_disjoint` graph: records are linked when they share a Bangla idiom type (whitespace/hyphen/joiner-insensitive), a listed form variant that is itself a corpus entry, or any identical model input (selected Bangla/English examples, either gloss, probed English counterpart). No input string occurs in two splits, and no idiom type does either. Listed alternatives that are different idioms with similar meanings are not merged. A class-level probe is meant to generalize across such synonyms.

An earlier policy, which links every listed counterpart transitively, forms a 66% component that must be placed in SPACE. That makes the component-level permutation null impossible and biases the test population (see [DATA_AUDIT.md](DATA_AUDIT.md)); it is kept as `configs/counterpart_closure.json`. `idiom_only` (Bangla types only) leaks identical inputs and is exploratory.

## Labels

- **F**: k-means (k = 12) of English figurative glosses fitted on SPACE with `all-MiniLM-L6-v2`, which is never a probed model. Cluster keywords, reseed ARI and AMI with S/D/Ft are diagnostics, not validation; two volunteers' judgments of 300 TEST items are in `data/human_validation/`.
- **L**: the *same* fitted F taxonomy applied to English literal glosses. Glosses come from `translate-literals` (`approved=machine`) and/or human review (`approved=true`, which takes precedence); the label audit records counts by source. FLS uses items with F ≠ L.
- **S**: supplied four-way sentiment (including `mixed`), SPACE vocabulary.
- **D, Ft**: complete human category sets; they have thousands of singleton classes and are skipped by default.
- **`cp_transparency`**: cosine between the probed English counterpart and the English gloss under the taxonomy encoder; tertile cut-offs fitted on SPACE define `transparent` and `opaque`. It is a covariate for English anchoring, because the binary `anchored` flag is true for 98.8% of records.

## Input conditions and pooling

| Condition | Text | Pooling | Role |
|---|---|---|---|
| `bn-fig` | Bangla example with an exact idiom occurrence | idiom span | primary contextual representation |
| `bn-sentence` | same sentence | all tokens | sentence-level comparison |
| `bn-context` | same sentence, idiom replaced by `[MASK]` | context tokens only | what the context alone carries |
| `bn-idiom` | idiom in a neutral Bangla frame | idiom span | form, without context |
| `bn-para` | Bangla figurative gloss | all tokens | plain meaning, non-idiomatic control |
| `en-idiom` | first English counterpart in a neutral frame | counterpart span | English anchor |
| `en-trans` | aligned English translation of the example | all tokens | parallel-sentence positive control |
| `en-para` | English figurative gloss | all tokens | plain meaning; label source of F |
| `bn-lit`, `bn-ctrl`, `bn-rom`, `natural-*` | approved external controls | supplied span | literal reading, compositional control, script, natural English/other languages |

`bn-fig2` (a second, different supplied Bangla example with the exact idiom; 7,830 records) and `bn-context2` (the same sentence with the idiom masked) are used only by the identity profile. Absent, repeated or truncated spans are invalid; no sentence fallback is used.

## Studies and outputs

| Study | Analyses (files) | Interpretation limits |
|---|---|---|
| S1 containment | Layer-wise retention and leakage, ambient and majority bounds, seven estimators, matched ranks (`grid_layer_*.csv`, `S1_S3_grid.csv`); `spectral_gap` per rank | High leakage after removing MLR/LDA readout directions does not imply redundant encoding: readout directions differ from class-mean directions (COV/LEACE remove the latter). |
| S2 language | Purity/interference w.r.t. language for pooled concept subspaces and the 1-D language direction (`S2_language.csv`); **multi-dimensional language subspace** from paired translation differences: concept error inside/outside it and principal angles to concept subspaces (`S2_language_subspace.csv`) | Neutral frames and translations differ in format; distinguish span and sentence conditions. |
| S3 transfer | SSI both directions, random/PCA/label-permutation nulls, null-adjusted SSI, asymmetry, CKA and paired retrieval with raw and per-condition centering (`S3_inference.csv`, `S3_baselines.csv`, `S3_asymmetry.csv`); **principal angles between separately estimated subspaces** (`S3_subspace_overlap.csv`); moderator contrasts including `transparent − opaque` (`S3_contrasts.csv`); per-item predictions and an associational logit (`predictions.csv`, `moderators.csv`) | SSI is a ratio that can be negative, exceed 1 or be undefined; report numerator and denominator. Moderator results are associations. |
| S4a literal separation | FLS = error of an L probe inside the F subspace (and the reverse, plus interference), on F ≠ L items (`S4_FLS.csv`); figurative-vs-control cosine margins when approved `bn-lit`/`bn-ctrl` controls exist (`S4_geometry.csv`) | Requires L glosses; machine-translated L must be labelled as such until reviewed. |
| S4b open class | (i) Subspaces for F/S from N idiom types, evaluated on held-out types with fixed class inventories, with RAND/PCA references and purity (the other concept's probe error inside the subspace) for the containment/purity trade-off (`S4_openclass.csv`). (ii) **Idiom identity I** as an open-class concept (`configs/identity_openclass.json`, `S4_identity.csv`): identity subspaces from N SPACE idioms seen in two contexts, scored by 1-nearest-neighbour matching of the two contexts of *unseen* TEST idioms (top-1 and MRR, inside and outside the subspace), ranks up to 256, with ambient/random references and F purity. Two representations are reported: idiom spans (`bn-fig`/`bn-fig2`), where matching is largely lexical because both contexts contain the same string, and idiom-masked contexts (`bn-context`/`bn-context2`), which test semantic identity and are the meaningful open-class case | LEACE/LDA use Ledoit–Wolf shrinkage in both sweeps because small estimation sets have fewer rows than dimensions. (i) tests unseen idiom *types* with known classes; (ii) is the unseen-class analogue of the reference paper's speaker experiment. |
| S5a behaviour | Joins per-idiom, per-model scores to geometry; cluster bootstrap and mixed models (`behavior` command) | Needs actual generation scores for the same identities. |
| S5b causal | Decoder hooks: erase F/L, rank-matched random, cross-lingual patching, sham (`intervene`, `intervention-effects`) | Bases must be fitted in the exact decoder block space; final block excluded; needs scoring. |

## Estimators

RAND (random orthonormal directions; grid rows average five draws, as in the reference), PCA, MLR (reference recipe: no intercept, unit-norm class weight rows, SVD row space), LDA (Fisher directions), CPCA (PCA of equally weighted centroids), COV (SVD of the feature/one-hot cross-covariance) and LEACE (oblique). Supervised estimators have rank at most K − 1; unsupported ranks are skipped, not padded. Every subspace records the spectral gap after its rank; a gap near zero means the truncated subspace is not well determined. The [reference audit](REFERENCE_AUDIT.md) documents the equivalence checks against the reference code.

## Probe engine

Probes are L2 logistic regressions with C = 1 by default (a `probe_Cs` grid is tuned on VALIDATION). Every readout records whether its probe converged (`probe_converged`, `leak_probe_converged` in the grid); capped scikit-learn fits are counted rather than printed. `probe_backend: "auto"` solves them with a batched PyTorch L-BFGS on CUDA (independent per-problem curvature and line search; erased and ambient features computed as low-rank corrections of one matrix) and with scikit-learn otherwise. Both minimize the same objective to a gradient tolerance; with shared estimators they give identical study outputs in the test suite. Permutation-null MLR estimators are fitted in GPU batches.

## Confirmatory family and null budgets

The formal family is MLR, F and S, `en-idiom → bn-fig`, group `all`, at the validation-selected layer and the highest supported rank. With the default single split and taxonomy, six encoders give 12 planned tests per null family; 1,999 random-subspace draws give a minimum p of 0.0005 and a minimum Holm-adjusted p of 0.006. Label-permutation draws use `perm_null_draws` (default: `null_draws`) and shuffle whole component bundles among equal-size components; the input-disjoint split leaves most SPACE mass movable. Replicate refits (`--seeds`, `--label-seeds`) are optional and enlarge the planned family, which is recorded in `inference_design.json`. Everything else is descriptive.

## Runtime and resumption

Estimator fits are shared across ranks, readouts are checkpointed individually (JSON + NumPy, no pickles), and grids/null draws/open-class readouts are batched on the GPU. Interrupted runs resume without refitting completed readouts; added null draws compute only the new suffix. Terminal status lines and `progress.json` report stage throughput, a stage ETA and a layers ETA from measured layer durations; these are estimates.

## Minimal evidence for a claim

Keep the source checksum, split/item/label audits, effective ranks and spectral gaps, feature-validity diagnostics, model revisions, configuration, validation selections and complete test results. Replicate the main direction on S as well as F and across estimators; include ambient, majority, random, PCA, permutation, context, neutral-frame and parallel/paraphrase comparisons, and state unavailable controls explicitly.
