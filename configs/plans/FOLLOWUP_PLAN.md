# E1–E4: prospective follow-up analysis plan

This plan is frozen and SHA-256 stamped by `scripts/run_followup.py` before
seed-1 outcomes are calculated. It is a local prospective analysis plan, not a
public preregistration. The original seed-0 campaign is the discovery study.
Seed 1 reuses the same corpus: this is an internal split-sensitivity replication,
not an independent dataset or an untouched holdout. Both successful and failed
hypotheses will be reported. No test-dependent tuning or stopping is allowed.

## Common design

- Original corpus and pinned model revisions from campaign `20261020T114718Z`.
- Six original encoders; reuse original cached hidden states, without changing
  text, pooling, model weights or cache files. GPU logistic probes and GPU label
  embeddings; CPU decompositions and bookkeeping where appropriate.
- Rebuild input-disjoint connected-component SPACE/PROBE/VALIDATION/TEST split
  with split seed 1 and the original fractions. Refit F on new SPACE only with
  the original label encoder/revision, 12 clusters, taxonomy seed 0. Preserve S.
- Select one layer per encoder using mean native Bangla F COV validation
  accuracy across supported ranks {1,2,3,4,8,11}; ties select the earlier layer.
  All estimators and controls use that same layer. No TEST selection.
- F rank 11 and S rank 3; original global SPACE-only centering/scaling;
  estimator C=1, readout C=1, original reference MLR normalization.
- Save item-level labels, predictions, correctness, component IDs and majority
  correctness for all seven estimators on the primary pair and native reference.
- Paired component percentile intervals: 5,000 draws. SSI intervals: 1,000
  component draws, native denominator must exceed 2 percentage points. All
  intervals condition on the fitted split/taxonomy/subspaces/readouts and are
  unadjusted 95% intervals, not simultaneous confidence statements.

## One new 48-test Holm family, separate from seed 0

H1 contributes 12 tests; H2, H3 and H4 contribute 12 tests each. Undefined
hypotheses occupy their planned positions as p=1. Apply Holm once across all
48; do not select a smaller family after seeing results. Alpha=.05.

1. **H1, COV readability:** for BanglaBERT, MuRIL and LaBSE and each F/S,
   English-counterpart -> contextual Bangla SSI exceeds (a) 1,999 matched-rank
   Gaussian-orthogonal random subspaces and (b) 1,999 COV fits to label bundles
   permuted among equal-size SPACE components. Require at least 50% movable
   SPACE mass. Plus-one Monte Carlo p-values; test-specific same native
   reference and majority floor. This is a distribution of source-estimator
   controls, not evidence that COV outperforms an optimally tuned classifier.
2. **H2, estimator comparison:** COV cross accuracy exceeds MLR cross accuracy,
   on identical Bangla-span TEST items, for all six encoders and F/S.
3. **H3, fixed-target gloss comparison:** COV directions from English glosses
   exceed directions from English counterparts, on identical Bangla-gloss TEST
   items, for all six encoders and F/S. This contrasts both source content and
   presentation; it does not isolate compositionality causally.
4. **H4, bounded erasure transfer:** the effect of an English-fitted LEACE
   eraser is less than 25% of the native Bangla-fitted effect. With per-item
   correctness a0 (ambient), aE (English erasure), aB (Bangla erasure), test
   mean[.25(a0-aB)-(a0-aE)] > 0. Require a positive observed native effect;
   report the ratio and its conditional bootstrap interval as well, marking
   intervals undefined when the native effect is nonpositive. All six F/S.

For H2–H4 use 19,999 component wild-sign draws of paired score sums and the
plus-one one-sided p-value at the zero boundary. Component sign calibration
assumes independent components and sign symmetry at the boundary; it is not
an exact randomization guarantee for these observational data. Report paired
bootstrap intervals and raw differences so conclusions do not rest on p alone.
H4 is a directional non-inferiority-style bound on the erasure-effect fraction;
failure to reject is not equivalence, and passing H4 does not imply zero effect.

## E2–E4: exploratory mechanism diagnostics

No additional confirmatory claims or p-value families. Use the same seed-1
chosen layers and fixed probe C=1. Report every configured setting, including
outcomes contrary to the discovery explanation.

- **E2:** English MLR estimator C in {0.001,0.01,0.1,1}; hold all other choices
  fixed. Report cross accuracy and overlap with English COV, F and S, all six
  models. Row normalization/no-intercept means C->0 is not guaranteed to
  recover the exact class-mean pattern under imbalanced labels.
- **E3:** hold the English full-rank class-covariance image fixed. Construct
  LEACE operators using English covariance, equal-weight pooled within-language
  covariance, or Bangla covariance, always estimated on SPACE. Report retained
  accuracy, erasure accuracy, image agreement and idempotence. Compare the
  English-covariance construction against the original LEACE implementation.
  Target/pooled covariance is a diagnostic adaptation with target-language
  features, not a strict source-only transfer setting. Outcomes may also reflect
  the readout's changed metric and regularization, not just covariance mismatch.
- **E4:** XLM-R baseline, per-dimension SPACE standardization, and removal of
  top {1,3,10} PCs fitted on pooled English-counterpart/Bangla-span SPACE
  features. Refit COV/MLR and their readouts. Report cross/native accuracy,
  leading variance fraction and participation-ratio effective dimension.
  Improvement alone is not proof of isotropy or of a unique causal mechanism.

## Human validation (E5)

Actual human judgments cannot be supplied by an automated agent. Prepare
300 component-distinct, F-stratified held-out records, two independent blinded
annotation sheets, definitions, consent/compensation guidance and an analysis
script. Do not fill human fields with model judgments. Until real independent
annotations are available, keep class names descriptive, treat aboutness/stance
as an unvalidated hypothesis, and make no claim of observed agreement.
