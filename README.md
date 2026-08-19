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

`ML_env.yml` pins the exact versions used for testing. The compatible version
ranges in `pyproject.toml` use those tested releases as their lower bounds, so
ordinary pip installations remain flexible without crossing major-version
compatibility boundaries.

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
  metrics, SHAP values, validation indices, fold-specific selected features,
  encoded targets, and compact resolved input provenance.
- **Training:** the fitted full-cohort scikit-learn pipeline, training targets,
  raw feature names, and compact resolved input provenance.
- **Validation:** the fitted pipeline, external scores, bootstrapped performance
  metrics, SHAP values, selected features, encoded targets, resolved input
  provenance, and an external-alignment report.

Saved pipelines include preprocessing and feature selection. Always predict
through the complete pipeline rather than extracting only its final estimator.

## Plotting saved results

The plotting API can consume one result file or aggregate repeated runs:

```python
from phipml.plots.result_summary import plot_result_files

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
SHAP importance, heatmap, and beeswarm figures. Standalone ROC,
precision-recall, confusion-matrix, and classification-metric panels are also
saved by default. Every selected figure is written as PDF, SVG, and PNG unless
`output_formats`/`--formats` selects a smaller set. New result artifacts save
the encoded target and a compact
resolved configuration snapshot. The plotter uses the SHAP index and columns
as the authoritative samples/features and reloads only their original values
from the configured data and metadata. It does not rerun splitting, prevalence
filtering, model fitting, or feature selection.

The peptide-library metadata is loaded only for feature-table annotation, such
as taxonomic, protein, or description columns. Continuous clinical variables
are summarized by group mean rather than colored as peptide prevalence. Nested
CV feature tables also report how often a feature was selected across outer
folds; repeated-run tables combine selection opportunities across runs/folds.

For repeated result files, feature display uses top-K SHAP frequency by
default: each run ranks features by its mean absolute SHAP, and the plotter
counts how often every feature occurs among that run's top K. Features are
ordered by occurrence frequency, mean rank while present, and finally mean
absolute SHAP. This prevents one unusually large run-specific SHAP value from
dominating consistently important features. A single result retains the
standard mean-absolute-SHAP ranking.

For one result file, the plots use the uncertainty already saved during model
evaluation: outer-fold SD for nested CV or bootstrap confidence intervals for
external validation. For repeated files, run-level point estimates and SHAP
values are aggregated and empirical between-run intervals are shown. These are
labelled as repeated-run variability, not formal confidence intervals.

The equivalent CLI accepts explicit paths or quoted glob patterns:

```bash
phipml-plot 'results/nested_random-forest_demo_*.joblib' \
  --split train \
  --class-labels Controls HCC \
  --title 'Controls vs HCC' \
  --feature-ranking top-k-frequency \
  --ranking-top-k 30 \
  --min-top-k-frequency 50 \
  --max-display 20 \
  --output-dir results/plots \
  --output-prefix controls_vs_hcc
```

Running `phipml-plot` without arguments prints its help and exits successfully.
Use `phipml-plot --version` to display the installed package version.

Select only the required plots and formats when preparing a specific panel:

```bash
phipml-plot results/validation_random-forest_external_420.joblib \
  --split test \
  --plots roc pr confusion \
  --formats pdf svg \
  --roc-color '#264653' \
  --roc-band-color '#A8DADC' \
  --pr-color '#8A5A44' \
  --pr-band-color '#DDBEA9' \
  --output-dir results/plots
```

Available plot names are `performance`, `roc`, `pr`, `confusion`,
`classification`, `shap-beeswarm`, `shap-importance`, `shap-heatmap`, and
`feature-table`; `all` is the default.

For reproducible figure settings, start from
[`configs/config_plotting.yaml`](configs/config_plotting.yaml). A plotting YAML
can contain result paths, plot selection, output formats, ranking settings,
colors, and feature-table options. Relative paths are resolved from that YAML,
and explicit CLI arguments override its values:

```bash
phipml-plot --plot-config configs/config_plotting.yaml
```

SHAP beeswarm dots use the muted `phipml_blue_gray_red` feature-value scale by
default (`#6699CC` through a light gray/lilac midpoint to `#CC6677`). When all
displayed features are binary, the continuous colorbar is replaced
automatically by a discrete 0/1 legend. If at least one displayed feature is
continuous, the continuous colorbar is retained. `shap_cmap`/`--shap-cmap`
overrides the scale; `class_colors`/`--class-colors` affects only the two
prediction-direction labels above the beeswarm. Signed SHAP heatmaps use a
separate `phipml_purple_gray_orange` diverging scale: negative contributions
are purple, zero is near-white, and positive contributions are orange. This
keeps SHAP contribution direction visually distinct from feature value. When
targets are available, heatmaps also draw labelled class-range brackets and a
strong separator between classes.

Confusion matrices display the positive class (1) first on both axes, followed
by the negative class (0). Figure annotations use two decimal places by
default. Feature-table prevalence cells use the `phipml_prevalence` scale
(muted red at 0%, pale yellow at 50%, olive green at 100%); continuous clinical
summaries remain uncolored.

`--ranking-top-k` determines how many features count as top-ranked in each
run; `--max-display` independently controls how many of the frequency-ranked
features are drawn. `--min-top-k-frequency` optionally requires occurrence in
a minimum percentage of runs. Use `--feature-ranking mean-abs-shap` to restore
the conventional across-run mean-importance ordering, or leave it as `auto`
to use frequency ranking only when multiple files are supplied.

For new artifacts, no extra input argument is needed while the saved absolute
data paths remain valid. If the project or data moved, provide the updated YAML;
its data, metadata, and optional library-metadata paths replace the embedded
ones:

```bash
phipml-plot results/validation_random-forest_external_420.joblib \
  --split test \
  --config configs/config_standard.yaml \
  --class-labels Controls HCC \
  --table-annotation-columns Description Species \
  --output-dir results/plots
```

Explicit tables remain available for old artifacts or custom plotting. Here
the target may be part of the feature table:

```bash
phipml-plot results/validation_random-forest_external_420.joblib \
  --split test \
  --features-table data/external_features.csv \
  --sample-column SampleName \
  --target-column group_test \
  --class-labels Controls HCC \
  --output-dir results/plots
```

Alternatively, use separate `--features-table` and `--target-table` files.
Explicit tables take precedence over reconstructed values, and
`--library-metadata` overrides the library table from the configuration.

The generated feature-importance CSV can also be curated and rendered again
without recalculating its values. Keep `Feature`, `Feature type`, `Statistic`,
`Mean |SHAP|`, and the two class-statistic columns unchanged; annotation text
and row order may be edited:

```bash
phipml-plot results/nested_random-forest_demo_420.joblib \
  --plots feature-table \
  --feature-importance-table results/curated_feature_importance.csv \
  --table-annotation-columns 'Short description' 'Short taxon' \
  --max-display 15 \
  --output-dir results/plots
```

The compact feature-table figure shows `Feature`, requested annotation
columns, the two class summaries, and `Mean |SHAP|` by default. Audit columns
remain in the generated CSV. Add them to the figure only when needed:

```bash
phipml-plot results/nested_random-forest_demo_420.joblib \
  --plots feature-table \
  --table-extra-columns \
    'Feature type' \
    'Statistic' \
    'Top-k SHAP frequency (%)' \
    'Mean rank when in top K' \
    'Selection frequency (%)'
```

`--output-dir` takes precedence over all defaults. If omitted, the plotter uses
`classification.plot_output_dir` from an explicitly supplied YAML when present;
otherwise it writes to a `plots/` directory beside the first result file.

### Cohort metric heatmaps

`phipml-heatmap` uses a tidy CSV/TSV manifest rather than inferring cohorts from
filenames. Required columns are `training`, `validation`, and `path`; `split`
is optional. The default palette is `inferno`, with annotation text selected
for contrast against each cell; override it with `--palette` when needed:

```text
training,validation,path,split
Dutch,Dutch,results/nested_random-forest_dutch_420.joblib,train
Dutch,German,results/validation_random-forest_german_420.joblib,test
Dutch,Norwegian,results/validation_random-forest_norwegian_420.joblib,test
```

Plot ROC-AUC, AP, or a threshold-dependent classification metric:

```bash
phipml-heatmap --manifest results/manifest.csv \
  --metric roc.auc \
  --output results/roc_auc_heatmap

phipml-heatmap --manifest results/manifest.csv \
  --metric pr.ap \
  --output results/ap_heatmap

phipml-heatmap --manifest results/manifest.csv \
  --metric classification.balanced_accuracy \
  --vmin 0 --vmax 1 \
  --output results/balanced_accuracy_heatmap
```

By default, each heatmap is saved as PDF, SVG, and PNG. Use, for example,
`--formats pdf svg` to request a smaller set. A supported suffix on `--output`
is treated as an optional hint, so `--output results/roc_auc_heatmap.pdf` also
creates all three default formats with the `roc_auc_heatmap` stem.

Cells with repeated files display the mean, SD, and number of runs. A cell
represented by one external-validation file displays its saved confidence
interval when available, while one nested-CV file displays its saved
outer-fold SD. By default, rows and columns are derived independently from the
observed validation and training labels, producing a compact rectangular
matrix without empty combinations. `--order` deliberately requests the legacy
square layout; use `--training-order` and `--validation-order` to control the
two axes separately.

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

The preferred high-level Python APIs are:

```python
from phipml.plots.result_summary import plot_result_files
from phipml.plots.metric_heatmap import (
    build_metric_matrix,
    plot_metric_heatmap,
)
```

`result_summary` is not limited to AUC: it normalizes and plots ROC-AUC,
precision-recall/AP, confusion counts, threshold-dependent classification
metrics, SHAP summaries, and feature tables. `metric_heatmap` accepts any saved
scalar metric using a dotted name such as `roc.auc`, `pr.ap`,
`classification.f1`, or `classification.balanced_accuracy`.

The older modules `phipml.plots.auc_shap_summary` and
`phipml.plots.auc_heatmap` are retained for compatibility with existing
notebooks and legacy CLI commands. New code should use `result_summary` and
`metric_heatmap` respectively.

A runnable 50-peptide plotting example, including simulated sample metadata
and peptide-library annotations, is available under
[`mock_examples/plotting`](mock_examples/plotting/README.md).

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
