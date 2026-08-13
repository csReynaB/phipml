# phipml

`phipml` is a Python package for reproducible binary classification of
PhIP-seq and other high-dimensional biological data. It provides a YAML-driven
workflow for loading data, defining cohorts, fitting Random Forest or XGBoost
models, evaluating performance, interpreting models with SHAP, and validating
a final model in independent cohorts.

## Main features

- Combined CSV, TSV/TXT, XLSX, or XLS sample-by-feature matrices.
- Directories or manifests of per-sample enrichment files, converted into a
  binary presence/absence matrix.
- CSV, TSV/TXT, Excel, or pickle peptide-library metadata.
- Absolute paths or YAML-relative paths.
- Metadata-defined training, split, and external-validation cohorts.
- Boolean, numeric, and string-valued peptide-library filters.
- Training-only peptide prevalence filtering.
- Peptide features plus arbitrary numeric clinical variables.
- Optional fold-safe imputation of continuous clinical variables.
- Random Forest and XGBoost classifiers.
- Nested stratified cross-validation with optional Bayesian tuning.
- Out-of-fold scores, classification metrics, selected features, and SHAP
  values.
- Fitted-pipeline-aware alignment of external validation features.
- Bootstrap confidence intervals for external-validation performance.
- ROC, precision-recall, confusion-matrix, classification-metric, SHAP, and
  feature-summary plots.
- YAML defaults with explicit command-line overrides.

## Modelling safeguards

The modelling workflow is designed to reduce information leakage:

- Peptide prevalence is learned from the final training cohort only.
- Variance filtering, feature selection, and optional imputation are fitted
  inside the scikit-learn pipeline and therefore separately within CV folds.
- Every nested-CV score is produced by a model that did not train on that
  sample.
- External data are aligned to the raw features expected by the fitted
  pipeline. Missing peptides may be filled with zero; missing clinical
  variables raise an error.
- The classification threshold is specified before evaluation and is not
  selected from external-validation outcomes.

## Installation

### Micromamba or Conda

The supplied `ML_env.yml` contains the tested scientific, plotting, Excel, and
Jupyter dependencies:

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml

micromamba create --yes --name phipml --file ML_env.yml
micromamba activate phipml

python -m pip install --no-build-isolation --no-deps -e .
```

Use a non-editable installation for a fixed deployment:

```bash
python -m pip install --no-build-isolation --no-deps .
```

### Pip

Inside an existing Python 3.10+ environment:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

Confirm the installation:

```bash
phipml --version
phipml -h
```

## Docker

Build the image from the repository root:

```bash
docker build -t phipml:latest .
```

Show the CLI help:

```bash
docker run --rm phipml:latest
```

Run a configuration stored in the current directory:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/workspace" \
  -w /workspace \
  phipml:latest \
  phipml -c configs/config.yaml
```

Relative paths in the YAML should point to files available under the mounted
workspace. If a configuration contains absolute host paths outside the project,
mount those directories at the same container paths.

Start JupyterLab with `phipml` installed in its kernel:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  -w /workspace \
  phipml:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the tokenized URL printed by Jupyter in a local browser.

## Configuration

Start from [`configs/config_all_options.yaml`](configs/config_all_options.yaml),
which documents all supported settings. The top-level section controls data
loading, target encoding, metadata filtering, and library metadata. Execution
settings live under `classification:`.

Configuration precedence is:

```text
internal default < YAML value < explicit CLI option
```

Relative paths are resolved from the directory containing the YAML file, not
from the shell's current working directory.

### Minimal example

```yaml
metadata_input: ../data/metadata.xlsx
data_input: ../data/exist.csv
data_input_mode: matrix
lib_metadata_input: ../data/library_metadata.pkl

group_tests: [Controls, HCC]
col_sample_name: SampleName
col_target: group_test
extra_features_to_include: [Sex, Age]
peptide_prefixes: [agilent, corona2, twist]
transposed: true

classification:
  model_type: xgboost
  param_grid_name: xgboost
  seed: 420

  run_nested_cv: true
  only_train_model: false

  subgroup: all
  with_oligos: true
  with_additional_features: true
  prevalence_threshold_min: 5
  prevalence_threshold_max: 95

  outer_cv_splits: 5
  inner_cv_splits: 3
  n_iter: 30
  n_jobs_outer: 1
  n_jobs_inner: -1

  classification_threshold: 0.5
  bootstrap_validation: true
  bootstrap_n_resamples: 1000
  bootstrap_confidence_level: 0.95

  train_filters:
    cohort: training

  validation_sets:
    - name: external
      filters:
        cohort: external

  output_dir: ../results
  output_name: Controls_vs_HCC

param_grid:
  xgboost:
    estimator__max_depth:
      type: integer
      low: 3
      high: 10
```

The order of `group_tests` defines target encoding: the first group is class
`0` and the second is class `1`. Existing numeric `0/1` targets are also
accepted.

If `param_grid` is empty, the pipeline still performs variance filtering and
model-based peptide feature selection, but uses the configured default feature
selector and classifier parameters without hyperparameter search.

The prevalence thresholds are inclusive. With minimum `5` and maximum `95`, a
peptide is retained when its training prevalence is between 5% and 95%,
including both boundaries.

### Individual sample files

Instead of one combined matrix, `data_input` may point to a directory containing
one enrichment table per sample:

```yaml
data_input: ../data/ABS
data_input_mode: auto
sample_file_patterns:
  - "R52P01_*.csv"
sample_file_peptide_column: ID
sample_name_regex: null
```

Each file contains the enriched peptide IDs for one sample. By default, its
filename stem becomes the sample ID. `phipml` builds the union of all peptide
IDs and returns a sample-by-peptide `uint8` presence/absence matrix.

Multiple and recursive glob patterns are supported, for example
`"**/*_significant.csv"`. `transposed` applies only to combined matrices and is
ignored in sample-file mode.

A tab-delimited manifest can also be used:

```yaml
data_input: ../data/samplefiles.txt
data_input_mode: sample-files
sample_file_peptide_column: ID
```

The manifest may contain one path per line:

```text
ABS/Sample_01.csv
ABS/Sample_02.csv
```

or explicit sample names and paths:

```text
sample_name	path
Sample-A	ABS/raw_a.csv
Sample-B	ABS/raw_b.csv
```

Relative manifest paths are resolved from the manifest directory. A filename
prefix or suffix can be removed with a named regular-expression group:

```yaml
sample_name_regex: 'prefix_(?P<sample>.+)_enriched'
```

## Cohort selection

`train_filters` selects the principal training cohort. Every entry under
`validation_sets` independently selects an external validation cohort from the
metadata:

```yaml
classification:
  train_filters:
    treatment: ICI
    center: MUW

  validation_sets:
    - name: HCC_TKI
      filters:
        treatment: TKI
    - name: HCC_external
      filters:
        treatment: ICI
        center: Graz
```

Multiple filter columns are combined with AND. A list of values within one
column means OR for that column.

### Optional split cohort

`split_filters` selects an additional cohort that is divided stratifiably into
training and held-out portions:

```yaml
classification:
  train_filters:
    disease: HCC
  split_filters:
    disease: Controls
  train_size: 0.7
  split_only: false
```

With `split_only: false`, the split training portion is appended to the main
training cohort and its held-out portion is appended to every validation set.
With `split_only: true`, only the selected split cohort is used for training and
hold-out evaluation.

Keep training, split, and validation filters disjoint to avoid duplicated
samples and data leakage.

## Running

Run the settings from a YAML file:

```bash
phipml -c configs/config.yaml
```

Override selected YAML values from the command line:

```bash
phipml -c configs/config.yaml \
  --model-type random-forest \
  --outer-cv-splits 5 \
  --inner-cv-splits 3 \
  --classification-threshold 0.5 \
  --train '{"cohort":"training"}' \
  --validate '{"cohort":"external"}' external
```

Boolean settings support positive and negative forms, for example:

```bash
phipml -c configs/config.yaml --run-nested-cv
phipml -c configs/config.yaml --no-run-nested-cv
phipml -c configs/config.yaml --bootstrap-validation
phipml -c configs/config.yaml --no-bootstrap-validation
```

### Train only

Fit and save a full-cohort model without nested CV or external validation:

```yaml
classification:
  run_nested_cv: false
  only_train_model: true
```

Equivalent CLI override:

```bash
phipml -c configs/config.yaml \
  --no-run-nested-cv \
  --only-train-model
```

## Evaluation and uncertainty

Nested CV and external validation deliberately report uncertainty differently:

- **Nested CV:** ROC-AUC and average precision are reported as the mean ± sample
  SD across outer folds. Classification metrics are calculated from pooled
  out-of-fold predictions, while their fold-level means and SDs are also saved.
  Fold SD describes split-to-split variability and is not labelled as a formal
  confidence interval.
- **External validation:** the final model is fitted on the complete training
  cohort and evaluated in the independent cohort. A class-stratified paired
  bootstrap provides confidence intervals for ROC-AUC, average precision,
  accuracy, balanced accuracy, precision, recall/sensitivity, specificity,
  negative predictive value, F1, and MCC. The model remains fixed during this
  bootstrap.

Threshold-dependent metrics use `classification_threshold`, which defaults to
`0.5`. Choose a different threshold using prior knowledge or training data, not
the external validation outcomes.

## Output files

Depending on the requested workflow, `phipml` writes:

```text
nested_<model>_<name>_<seed>.joblib
training_<model>_<name>_<seed>.joblib
validation_<model>_<validation-name>_<seed>.joblib
```

The main saved objects are:

- **Nested CV:** fold models, out-of-fold scores, ROC/PR and classification
  metrics, SHAP values, validation indices, and selected features.
- **Training:** the fitted full-cohort scikit-learn pipeline.
- **Validation:** the fitted pipeline, external scores, bootstrapped performance
  metrics, SHAP values, selected features, and an external-alignment report.

Saved pipelines include preprocessing and feature selection. Always predict
through the complete pipeline rather than extracting only its final estimator.

## Plotting saved results

The plotting API can consume one result file or aggregate repeated runs:

```python
from phipml.plots.auc_shap_summary import plot_result_files

plots = plot_result_files(
    ["results/nested_random-forest_Controls_vs_HCC_420.joblib"],
    split="train",
    class_labels=("Controls", "HCC"),
    title="Controls vs HCC",
    output_dir="results/plots",
    output_prefix="controls_vs_hcc",
    max_display=20,
)
```

This produces a performance summary and, when SHAP values are present, global
SHAP importance and heatmap figures. Supplying the corresponding feature matrix
enables a SHAP beeswarm; supplying both features and targets also enables the
feature-statistics table. Continuous clinical variables are treated separately
from binary prevalence features in that table.

Individual plotting functions are also available from
`phipml.plots.helpers`, including:

- `plot_roc_metrics`
- `plot_precision_recall_metrics`
- `plot_confusion_matrix_metrics`
- `plot_classification_metric_bars`
- `plot_performance_summary`
- `plot_shap_importance_bar`
- `plot_shap_heatmap`
- `plot_shap_values`
- `plot_feature_importance_table`

## Development checks

```bash
python -m isort --check-only --diff src tests
python -m black --check src tests
python -m pytest -q
```

To apply formatting:

```bash
python -m isort src tests
python -m black src tests
```

## License

See [`LICENSE`](LICENSE).