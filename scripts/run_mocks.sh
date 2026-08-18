#!/bin/bash

# nested-cv
phipml -c mock_examples/perfect_signal/config.yaml 
phipml -c mock_examples/noisy_signal/config.yaml 
phipml -c mock_examples/tuned_noisy_signal/config_tuned_random_forest.yaml 

# full training and validation
phipml -c mock_examples/external_validation_perfect/config_perfect_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_with_tuning.yaml


# plots
phipml-plot \
  "mock_examples/perfect_signal/results/nested_random-forest_perfect_demo_420.joblib" \
  --split train \
  --feature-ranking mean-abs-shap \
  --library-metadata "mock_examples/peptide_library_metadata.csv" \
  --library-id-column peptide_id \
  --table-annotation-columns Description Species Protein \
  --output-dir "mock_examples/plotting/output/nested_single_perfect" \
  --output-prefix perfect_nested

