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

### `tuned_noisy_signal/`

- self-contained copy of the 60-sample noisy cohort
- uses eight Bayesian-search iterations in every inner fold
- jointly tunes elastic-net `l1_ratio` and `C`
- also tunes Random Forest tree count, depth, split size, leaf size, and
  `max_features`
- repeats the inner search on the complete cohort and saves the fitted pipeline
  for later reuse

### `external_validation_noisy/`

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

### `external_validation_perfect/`

- 30-sample perfect training cohort and independent 20-sample perfect external
  cohort
- fitted elastic-net feature selection with fixed defaults
- fixed/default Random Forest parameters
- no Bayesian hyperparameter tuning
- expected external ROC-AUC and PR-AUC: approximately `1.0`

### `external_validation_noisy/`

- 60-sample noisy training cohort and independent 40-sample noisy external
  cohort
- one shared data/metadata pair and two directly comparable configs:
  - `config_noisy_external_no_tuning.yaml` fits feature selection and the
    forest using fixed settings
  - `config_noisy_external_with_tuning.yaml` tunes the selector and forest
- useful but non-perfect external performance is expected

A fixed-parameter reference run on the noisy CSVs produced ROC-AUC `0.782`
and PR-AUC `0.782`; the final tuned result may differ.

## Scenario matrix

| Scenario | Feature selection fitted? | Hyperparameter tuning? |
|---|---:|---:|
| Perfect nested CV | Yes | No |
| Noisy nested CV | Yes | No |
| Noisy nested CV tuned | Yes | Yes |
| Perfect external validation | Yes | No |
| Noisy external validation untuned | Yes | No |
| Noisy external validation tuned | Yes | Yes |

## File structure

Each example contains a peptide-by-sample CSV matrix, sample metadata, and one
or more YAML configurations. The YAML files use `transposed: true`. Files with
`training` and `external` cohorts use the metadata column `cohort` for
selection.

`phipml_mock_datasets_overview.xlsx` provides a human-readable
inspection copy of the datasets and tuning space. PhIPML reads the CSV files,
not this workbook.

## Run the examples

From the root of the PhIPML repository, after installing it in editable mode:

```bash
python -m pip install -e .

phipml -c mock_examples/perfect_signal/config.yaml
phipml -c mock_examples/noisy_signal/config.yaml
phipml -c mock_examples/tuned_noisy_signal/config_tuned_random_forest.yaml
phipml -c mock_examples/external_validation_noisy/config_external_validation_noisy.yaml
phipml -c mock_examples/external_validation_perfect/config_perfect_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_with_tuning.yaml
```

The first three commands demonstrate nested CV. The final four validation
commands isolate full-cohort fitting followed by evaluation on an independent
external cohort. The two noisy commands use identical data, so their only
material difference is whether hyperparameters are tuned.

Expected files include:

```text
perfect_signal/results/nested_random-forest_perfect_demo_420.joblib
perfect_signal/results/training_random-forest_perfect_demo_420.joblib

noisy_signal/results/nested_random-forest_noisy_demo_420.joblib
noisy_signal/results/training_random-forest_noisy_demo_420.joblib

tuned_noisy_signal/results_tuned/nested_random-forest_tuned_noisy_demo_420.joblib
tuned_noisy_signal/results_tuned/training_random-forest_tuned_noisy_demo_420.joblib

external_validation_noisy/results_external/nested_random-forest_noisy_train_420.joblib
external_validation_noisy/results_external/validation_random-forest_noisy_external_420.joblib

external_validation_perfect/results_perfect_external/validation_random-forest_perfect_external_420.joblib

external_validation_noisy/results_noisy_external_untuned/validation_random-forest_noisy_external_untuned_420.joblib
external_validation_noisy/results_noisy_external_tuned/validation_random-forest_noisy_external_tuned_420.joblib
```

The external-validation joblib contains the fitted full-cohort estimator,
external SHAP values, prediction scores, ROC/PR metrics, selected features, and
the feature-alignment report.

To reuse the tuned full-cohort model later, point `classification.input_dir`
to `results_tuned_v2`, set `classification.input_name` to
`tuned_noisy_demo_v2`, and enable `classification.use_pretrained: true` in a
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
