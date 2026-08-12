#!/bin/bash

# nested-cv
phipml -c mock_examples/perfect_signal/config.yaml 
phipml -c mock_examples/noisy_signal/config.yaml 
phipml -c mock_examples/tuned_noisy_signal/config_tuned_random_forest.yaml 

# full training and validation
phipml -c mock_examples/external_validation_perfect/config_perfect_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_no_tuning.yaml
phipml -c mock_examples/external_validation_noisy/config_noisy_external_with_tuning.yaml

