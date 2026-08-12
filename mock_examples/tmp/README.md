# PhIPML synthetic classification examples

These datasets are reproducible mock data for testing and demonstrating the
PhIPML command-line workflow. They contain no real participants or biological
measurements and must not be interpreted as biological benchmarks.

## Included examples

### `perfect_signal/`

- 30 samples: 15 `Control` and 15 `Case`
- 50 peptide features
- `agilent_signal` equals the encoded target
- `twist_signal` is the inverse target
- 48 independent binary noise peptides
- two binary clinical variables: `Sex`, `Smoking`
- two continuous clinical variables: `Age`, `BMI`
- expected nested-CV ROC-AUC and PR-AUC: approximately `1.0`

### `noisy_signal/`

- 60 samples: 30 `Control` and 30 `Case`
- the same 50-peptide and four-clinical-feature structure
- both signal peptides are flipped for exactly 20% of samples, balanced across
  the two target classes
- expected nested-CV ROC-AUC and PR-AUC: usually around `0.7-0.8`

A reference three-fold run produced mean ROC-AUC `0.743` and mean PR-AUC
`0.730`; the perfect example produced `1.000` for both metrics.

Exact non-perfect metrics can vary slightly with scikit-learn versions and
estimator behavior. The noisy example is intended to demonstrate realistic
non-perfect performance, not require one exact AUC.

### `tuned_noisy_signal_v2/`

- self-contained copy of the 60-sample noisy cohort
- uses eight Bayesian-search iterations in every inner fold
- jointly tunes elastic-net `l1_ratio` and `C`
- also tunes Random Forest tree count, depth, split size, leaf size, and
  `max_features`
- repeats the inner search on the complete cohort and saves the fitted pipeline
  for later reuse

### `external_validation_noisy_v3/`

- 60-sample noisy training cohort and an independently generated 40-sample
  noisy external cohort
- both cohorts are balanced and have 20% signal flips
- metadata column `cohort` selects `training` versus `external`
- runs tuned nested CV only on the training cohort
- tunes one full training-cohort model and evaluates it once on the untouched
  external cohort
- expected external ROC-AUC/PR-AUC are useful but non-perfect, approximately
  in the `0.7-0.9` range

A fixed-parameter reference run gave external ROC-AUC `0.808` and PR-AUC
`0.714`. The runnable YAML performs Bayesian tuning, so exact values may differ.

## File structure

Each example contains:

- `data.csv`: peptide-by-sample matrix. The YAML therefore uses
  `transposed: true`.
- `metadata.csv`: one row per sample with target and clinical variables.
- `config.yaml`: complete runnable configuration using relative paths.

`phipml_mock_datasets_overview.xlsx` provides a human-readable
inspection copy of both datasets and the tuned search space. PhIPML itself
reads the CSV files, not this workbook.

## Run the examples

From the root of the PhIPML repository, after installing it in editable mode:

```bash
python -m pip install -e .

phipml -c mock_examples/perfect_signal/config.yaml
phipml -c mock_examples/noisy_signal/config.yaml
phipml -c mock_examples/tuned_noisy_signal/config_tuned_random_forest.yaml
phipml -c mock_examples/external_validation_noisy/config_external_validation.yaml
```

Each command runs three-fold outer nested CV without hyperparameter tuning,
then fits and saves a full-cohort Random Forest model. Results are written to
the corresponding example's `results/` directory.

Expected files include:

```text
perfect_signal/results/nested_random-forest_perfect_demo_420.joblib
perfect_signal/results/training_random-forest_perfect_demo_420.joblib

noisy_signal/results/nested_random-forest_noisy_demo_420.joblib
noisy_signal/results/training_random-forest_noisy_demo_420.joblib

tuned_noisy_signal/results_tuned/nested_random-forest_tuned_noisy_demo_420.joblib
tuned_noisy_signal/results_tuned/training_random-forest_tuned_noisy_demo_420.joblib

external_validation_noisy/results_external/nested_random-forest_noisy_train_420.joblib
external_validation_noisy_v3/results_external/validation_random-forest_noisy_external_420.joblib
```

The external-validation joblib contains the fitted full-cohort estimator,
external SHAP values, prediction scores, ROC/PR metrics, selected features, and
the feature-alignment report.

To reuse the tuned full-cohort model later, point `classification.input_dir`
to `results_tuned`, set `classification.input_name` to
`tuned_noisy_demo`, and enable `classification.use_pretrained: true` in a
validation configuration.

## Load the nested-CV results

```python
import joblib

result = joblib.load(
    "mock_examples/"
    "noisy_signal/results/nested_random-forest_noisy_demo_420.joblib"
)

print(result["metrics_train"]["roc"]["auc"])
print(result["metrics_train"]["pr"]["ap"])
```
