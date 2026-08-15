#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/phipml-matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

# One standard nested-CV run plus a second seed for repeated-run aggregation.
phipml -c "$EXAMPLES_DIR/noisy_signal/config.yaml"
phipml -c "$EXAMPLES_DIR/noisy_signal/config.yaml" --seed 421

# One full-cohort model evaluated in an independent external cohort.
phipml -c \
  "$EXAMPLES_DIR/external_validation_noisy/config_noisy_external_no_tuning.yaml"

phipml-plot \
  "$EXAMPLES_DIR/noisy_signal/results/nested_random-forest_noisy_demo_420.joblib" \
  --split train \
  --feature-ranking mean-abs-shap \
  --library-metadata "$EXAMPLES_DIR/peptide_library_metadata.csv" \
  --library-id-column peptide_id \
  --table-annotation-columns Description Species Protein \
  --output-dir "$OUTPUT_DIR/nested_single" \
  --output-prefix noisy_nested

phipml-plot --plot-config "$SCRIPT_DIR/config_repeated_plots.yaml"

phipml-plot \
  "$EXAMPLES_DIR/external_validation_noisy/results_noisy_external_untuned/validation_random-forest_noisy_external_untuned_420.joblib" \
  --split test \
  --library-metadata "$EXAMPLES_DIR/peptide_library_metadata.csv" \
  --library-id-column peptide_id \
  --table-annotation-columns Description Species Protein \
  --output-dir "$OUTPUT_DIR/external_validation" \
  --output-prefix noisy_external

phipml-heatmap \
  --manifest "$SCRIPT_DIR/metric_manifest.csv" \
  --metric roc.auc \
  --output "$OUTPUT_DIR/roc_auc_heatmap.pdf"

printf 'Plotting demonstration completed: %s\n' "$OUTPUT_DIR"
