# phipml

**phipml** is a Python package for reproducible machine learning workflows tailored to PhIP-Seq and other high-dimensional biological datasets.

It provides a structured framework to move from feature matrices and metadata to robust, explainable classification models.

---

## 🚀 Key Features

- Nested cross-validation (outer + inner splits)
- XGBoost and Random Forest support
- Bayesian hyperparameter optimization (BayesSearchCV)
- Prevalence-based feature filtering
- SHAP-based model interpretation
- External validation via metadata filtering
- YAML-driven experiment configuration
- CLI-based execution for full reproducibility

---

## 🧠 Intended Use

`phipml` is designed for classification problems where:

- You have a feature matrix (e.g. peptide enrichment scores, antigen scores, omics data)
- You have a metadata file with sample annotations
- You want statistically robust performance estimates
- You require explainable feature importance
- You may validate models on independent cohorts

The framework is especially suited for **PhIP-Seq classification tasks**, but is generalizable to other structured datasets.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/csReynaB/phipml.git
cd phipml
conda env create -f ML_env.yml --prefix /path/to/envs/classification_phipml
conda activate ML_env
python -m pip install . --no-deps
```

---

## 📁 Repository structure

```text
phipml/
│
├── configs/              # YAML configuration templates
├── scripts/              # Helper scripts
├── src/phipml/
│   ├── cli/              # Command-line entry points
│   ├── models/           # Model implementations
│   ├── preprocessing/    # Feature filtering utilities
│   ├── plots/            # Visualization tools
│   └── utils/            # Shared utilities
│
└── pyproject.toml        # Package configuration
``` 

---

## ⚙️  Running a Classification Pipeline
```bash
trainTest \
  --config config.yaml \
  --train '{"group_test": ["Controls", "Disease"]}' \
  --run_nested_cv True \
  --outer_cv_split 5 \
  --inner_cv_split 3 \
  --model_type random-forest \ # or xgboost
  --subgroup all \
  --with_oligos True \ 
  --with_additional_features False \
  --prevalence_threshold_min 2.0 \
  --prevalence_threshold_max 98.0 \
  --out_name resultsDisease \
  --out_dir /path/results 
```
---

## 🧪 Configuration

Experiments are controlled via YAML configuration files.

Typical parameters include:

    Path to feature matrix

    Path to metadata

    Target column

    Feature inclusion/exclusion filters

    Group definitions

    Model hyperparameter search space

---

## 📈 Outputs

Depending on configuration, outputs may include:

    Cross-validation performance metrics

    ROC-AUC / PR-AUC scores

    Trained model objects

    SHAP values

    Feature rankings

    Validation results

    Performance heatmaps and plots

---


## 🧰 Dependencies

Core dependencies include:

- numpy
- pandas
- scikit-learn
- xgboost
- shap
- scikit-optimize
- matplotlib
- seaborn
- joblib
- pyyaml
- tqdm
---


## 🧹 Formatting & Debugging
```bash
isort .
black .
ruff check .
```

