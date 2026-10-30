# Post-hoc validation-selected MLR diagnostic

This is an additional exploratory analysis, outside the frozen 48-test family.
It uses the four E2 settings already fitted on SPACE and read out on PROBE,
at the common COV-validation-selected layer. For each model and concept, choose
estimator C using the saved primary-pair **VALIDATION accuracy only**; ties
select the smaller C. Do not choose the best TEST setting. Compare the chosen
MLR readout with the existing E1 COV readout on the same TEST item IDs.

Report all four validation scores, the chosen setting, both test accuracies,
their paired difference, and an unadjusted 5,000-component-bootstrap interval.
Do not add p-values to the completed E1 family or call this confirmatory.
The TEST results for the regularization path have already been inspected, so
this is a transparent diagnostic of a stronger baseline, not a new holdout.
The fixed common layer means neither estimator is separately optimized over
all possible hyperparameters. Freeze this file before reading validation scores.
