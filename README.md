# phipml

phipml is a Python package for reproducible binary classification of PhIP-Seq
and other high-dimensional biological data. 

## Features

- combined CSV, TSV/TXT, XLSX, and XLS sample/feature tables
- directories or manifests of per-sample enrichment tables, converted to 0/1
- CSV, TSV/TXT, Excel, or pickle library metadata
- relative or absolute input paths
- metadata-defined training, split, and external-validation cohorts
- Boolean, numeric, and string-valued peptide-library filters
- train-only peptide prevalence filtering
- arbitrary numeric clinical features, with optional continuous imputation
- XGBoost and RandomForest classifiers
- nested stratified cross-validation and Bayesian hyperparameter search
- out-of-fold scores and SHAP values
- fitted-pipeline-aware external feature alignment
- YAML settings with explicit CLI overrides

## Installation

Create or activate an environment containing the scientific dependencies, then
install phipml:

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Start from [configs/config_standard.yaml](configs/config_standard.yaml). The
top-level keys describe data loading and target encoding. Runtime options live
under `classification:`.

Paths may be absolute or relative. Relative paths are resolved from the YAML
file, not from the shell's current directory.

A minimal configuration looks like this:

```yaml
metadata_input: ../data/metadata.xlsx
data_input: ../data/exist.csv
lib_metadata_input: ../data/library_metadata.pkl

group_tests: [Controls, HCC]
col_sample_name: SampleName
col_target: group_test
extra_features_to_include: [Sex, Age]
peptide_prefixes: [agilent, corona2, twist]
transposed: true

classification:
  model_type: xgboost
  run_nested_cv: true
  subgroup: all
  with_oligos: true
  with_additional_features: true
  prevalence_threshold_min: 5
  prevalence_threshold_max: 95
  outer_cv_splits: 5
  inner_cv_splits: 3
  n_iter: 30
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

The positional target encoding follows `group_tests`: the first group is 0,
the second is 1. Existing numeric 0/1 targets are also accepted.

### Per-sample enrichment files

Instead of supplying one combined matrix, `data_input` can point to a directory
of one-sample files. Each file contains the enriched peptide IDs for one sample;
its filename stem becomes the sample ID. phipml builds the union of peptide IDs
and returns a sample-by-peptide `uint8` presence/absence matrix.

```yaml
data_input: ../data/ABS
data_input_mode: auto       # an existing directory is detected automatically
sample_file_patterns:
  - "R52P01_*.csv"          # glob handles prefixes, suffixes, or subdirectories
sample_file_peptide_column: ID
sample_name_regex: null      # by default: SampleName = filename stem
```

Multiple patterns can be supplied, and recursive discovery is available with a
pattern such as `"**/*_significant.csv"`. `transposed` applies only to an
existing combined matrix and is ignored for per-sample input.

Alternatively, point `data_input` to a tab-delimited manifest and explicitly
select `sample-files` mode:

```yaml
data_input: ../data/samplefiles.txt
data_input_mode: sample-files
sample_file_peptide_column: ID
```

The manifest may contain one relative or absolute path per line:

```text
ABS/R52P01_30_EC86_J25_F12_P3_A_T_C2.csv
ABS/R52P01_31_EC86_J25_G12_P3_A_T_C2.csv
```

or an explicit sample name and path, separated by a tab:

```text
sample_name	path
Sample-A	ABS/raw_file_a.csv
Sample-B	ABS/raw_file_b.csv
```

Relative manifest entries are resolved from the manifest's directory. To strip
prefixes or suffixes from filename-derived sample names, use a regular
expression with a named `sample` group, for example
`sample_name_regex: 'prefix_(?P<sample>.+)_enriched'`.

## Running

Use the YAML as-is:

```bash
phipml -c configs/config_standard.yaml
```

CLI values override only the specified YAML settings:

```bash
phipml -c configs/config_standard.yaml \
  --model-type random-forest \
  --outer-cv-splits 5 \
  --inner-cv-splits 3 \
  --train '{"cohort":"training"}' \
  --validate '{"cohort":"external"}' external
```

Existing argument files remain supported:

```bash
phipml @configs/args.txt
```

The legacy `trainTest` executable is retained as an alias.

### Optional split cohort

Some classification designs split one cohort first, then append its training
portion to another training cohort and its held-out portion to every validation
cohort:

```yaml
classification:
  train_filters:
    disease: HCC
  split_filters:
    disease: Controls
  train_size: 0.7
  split_only: false
```

Set `split_only: true` to train and evaluate using only that stratified
train/hold-out split.

### Train only

To fit and save the full-cohort tuned model without external validation:

```yaml
classification:
  run_nested_cv: false
  only_train_model: true
```

or:

```bash
phipml -c configs/config_standard.yaml \
  --no-run-nested-cv \
  --only-train-model
```

## Outputs

Depending on the run settings, phipml writes:

- `nested_<model>_<name>_<seed>.joblib`
- `training_<model>_<name>_<seed>.joblib`
- `validation_<model>_<name>_<seed>.joblib`

Validation artifacts include the fitted pipeline, scores, ROC/PR metrics, SHAP
values, selected features, and an external-alignment report. Missing peptide
features may be zero-filled; missing clinical features raise an error.

## Development checks

```bash
python -m isort --check-only --diff src tests
python -m black --check src tests
python -m pytest -q
```
