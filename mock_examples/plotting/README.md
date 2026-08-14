# Reproducible plotting demonstration

This demonstration reuses the 50-peptide synthetic datasets in the parent
`mock_examples` directory. The inputs contain:

- a peptide-by-sample presence/absence matrix;
- sample metadata with the binary target and four clinical variables;
- `peptide_library_metadata.csv`, containing synthetic description, species,
  protein, library, and feature-role annotations for every peptide.

No precomputed joblib files are committed. Generate them with the installed
phipml version so that fitted estimators remain compatible with the local
scikit-learn, XGBoost, and SHAP versions.

## Run the complete example

From the repository root:

```bash
bash mock_examples/plotting/run_plotting_demo.sh
```

The script performs two nested-CV runs, one external validation, and then
creates single-run, repeated-run, external-validation, and metric-heatmap
plots.

## Run individual steps

Create one noisy nested-CV result and its full-cohort model:

```bash
phipml -c mock_examples/noisy_signal/config.yaml
```

Plot that one nested-CV artifact. The embedded resolved configuration reloads
the original data, target, clinical variables, and peptide annotations, so no
separate feature or target table is required:

```bash
phipml-plot \
  mock_examples/noisy_signal/results/nested_random-forest_noisy_demo_420.joblib \
  --split train \
  --feature-ranking mean-abs-shap \
  --table-annotation-columns Description Species Protein \
  --output-dir mock_examples/plotting/output/nested_single \
  --output-prefix noisy_nested
```

Create a second run and aggregate both runs empirically:

```bash
phipml -c mock_examples/noisy_signal/config.yaml --seed 421

phipml-plot --plot-config \
  mock_examples/plotting/config_repeated_plots.yaml
```

The plotting YAML demonstrates result globs, all-plot generation, PDF/SVG/PNG
output, colors, repeated top-K frequency ranking, and compact-table styling.
Any explicit CLI option overrides the corresponding YAML value.

Create and plot an independent external-validation result:

```bash
phipml -c \
  mock_examples/external_validation_noisy/config_noisy_external_no_tuning.yaml

phipml-plot \
  mock_examples/external_validation_noisy/results_noisy_external_untuned/validation_random-forest_noisy_external_untuned_420.joblib \
  --split test \
  --table-annotation-columns Description Species Protein \
  --output-dir mock_examples/plotting/output/external_validation \
  --output-prefix noisy_external
```

The external artifact already contains bootstrap intervals because external
bootstrapping is enabled by default. The nested-CV result instead contains
outer-fold variability.

After the standard nested and external results exist, create a metric heatmap:

```bash
phipml-heatmap \
  --manifest mock_examples/plotting/metric_manifest.csv \
  --metric roc.auc \
  --output mock_examples/plotting/output/roc_auc_heatmap.pdf
```

Use `--metric pr.ap`, `classification.balanced_accuracy`, or another saved
scalar metric to summarize a different quantity.

## Expected plotting outputs

Each `phipml-plot` invocation can produce:

- a combined ROC, precision-recall, confusion-matrix, and classification panel;
- standalone ROC and precision-recall curves;
- standalone confusion-matrix and classification-metric plots;
- mean absolute SHAP importance;
- a sample-by-feature SHAP heatmap;
- a SHAP beeswarm using the reconstructed feature values;
- a feature-importance CSV and compact annotated table.

All available plots and PDF/SVG/PNG output are enabled by default. Use
`--plots` and `--formats` (or the equivalent plotting-YAML keys) to request a
smaller subset.

For repeated joblib inputs, performance and SHAP values are aggregated across
runs. The repeated example ranks features by how often they occur among each
run's top 15 and displays the leading 10 features that occur in at least 50%
of runs. The feature table also reports mean top-K rank, SHAP stability, and
model feature-selection frequency. The synthetic annotations exist only to
demonstrate the interface and have no biological meaning.
