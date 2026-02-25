# ======================
# Standard library
# ======================
import importlib
import os
from pathlib import Path

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import yaml
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from skopt.space import Categorical, Integer, Real

# ======================
# Local / project imports
# ======================
from phipml.utils.peptides_filter import (
    CorrelationFilter,
    EntropyFilter,
    PrevalenceFilter,
)

# ======================
# Global configuration
# ======================
pd.set_option("future.no_silent_downcasting", True)
set_config(transform_output="pandas")


class Config:
    def __init__(
        self,
        # ======================
        # Core
        # ======================
        config_file,
        project=None,
        random_state=None,
        # ======================
        # Paths / directories
        # ======================
        metadata_dir=None,
        data_dir=None,
        # ======================
        # Metadata & input structure
        # ======================
        lib_meta_data=None,
        meta_typefile=None,
        col_sample_name=None,
        col_target=None,
        col_predict=None,
        # ======================
        # Data types & features
        # ======================
        data_types=None,
        extra_features_to_include=None,
        with_oligos_options=None,
        with_additional_features_options=None,
        with_run_plates_options=None,
        # ======================
        # Filtering & preprocessing
        # ======================
        filter_by_entropy=None,
        entropy_threshold=None,
        prevalence_thresholds_min=None,
        prevalence_thresholds_max=None,
        filter_by_correlation=None,
        filters_metadata=None,
        combined_filters_metadata=None,
        fillna=None,
        imputed=None,
        transposed=None,
        # ======================
        # Grouping & stratification
        # ======================
        group_tests=None,
        subgroups_to_include=None,
        subgroups_to_name=None,
        subgroups_order=None,
        subgroups_colors=None,
        # ======================
        # Models & estimators
        # ======================
        estimators_info=None,
        param_grid=None,
        tuning_parameters=None,
        # ======================
        # Cross-validation & tuning
        # ======================
        cv_method=None,
        split_train_test=None,
        train_size=None,
        tuning_n_iter=None,
        tuning_k=None,
        k=None,
        external_set=None,
        libraries_prefixes=None,
        # ======================
        # Outputs & diagnostics
        # ======================
        compute_feature_importance=None,
        return_train=None,
        return_test=None,
    ):

        # ======================
        # Mandatory / core inputs
        # ======================
        self.metadata_dir = (
            Path(metadata_dir) if isinstance(metadata_dir, str) else None
        )
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else None
        self.project = project if isinstance(project, str) else None
        self.lib_meta_data = lib_meta_data if isinstance(lib_meta_data, str) else None
        self.group_tests = group_tests if isinstance(group_tests, str) else None

        # ======================
        # Defaults & schema
        # ======================
        self.meta_typefile = (
            meta_typefile if meta_typefile in {"excel", "csv"} else "excel"
        )  # Default excel, must be set explicitly ('excel' or 'csv')
        self.col_sample_name = (
            col_sample_name if isinstance(col_sample_name, str) else "SampleName"
        )
        self.col_target = col_target if isinstance(col_target, str) else "group_test"
        self.col_predict = (
            col_predict if isinstance(col_predict, str) else "class1_proba"
        )

        # ======================
        # Encoding & reproducibility
        # ======================
        self.sex_encoding = {"F": 0, "M": 1}
        self.random_state = random_state if isinstance(random_state, int) else 420

        # ======================
        # Features & data types
        # ======================
        self.extra_features_to_include = (
            extra_features_to_include
            if (
                isinstance(extra_features_to_include, list)
                and all(isinstance(item, str) for item in extra_features_to_include)
            )
            else ["Sex", "Age"]
        )
        self.data_types = (
            data_types
            if (
                isinstance(data_types, list)
                and all(isinstance(item, str) for item in data_types)
            )
            else ["exist"]
        )

        ###
        self.entropy_threshold = (
            entropy_threshold if isinstance(entropy_threshold, float) else 0.4
        )

        self.prevalence_thresholds_min = (
            prevalence_thresholds_min
            if (
                isinstance(prevalence_thresholds_min, list)
                and all(
                    isinstance(item, (int, float)) for item in prevalence_thresholds_min
                )
            )
            else [5.0, 10.0, 20.0, 50.0]
        )
        self.prevalence_thresholds_max = (
            prevalence_thresholds_max
            if (
                isinstance(prevalence_thresholds_max, list)
                and all(
                    isinstance(item, (int, float)) for item in prevalence_thresholds_max
                )
            )
            else [95.0]
        )
        self.with_oligos_options = (
            with_oligos_options
            if (
                isinstance(with_oligos_options, list)
                and all(isinstance(item, bool) for item in with_oligos_options)
            )
            else [True, False]
        )
        self.with_additional_features_options = (
            with_additional_features_options
            if (
                isinstance(with_additional_features_options, list)
                and all(
                    isinstance(item, bool) for item in with_additional_features_options
                )
            )
            else [True, False]
        )
        self.with_run_plates_options = (
            with_run_plates_options
            if (
                isinstance(with_run_plates_options, list)
                and all(isinstance(item, bool) for item in with_run_plates_options)
            )
            else [False]
        )
        self.filter_by_entropy = (
            filter_by_entropy
            if (
                isinstance(filter_by_entropy, list)
                and all(isinstance(item, bool) for item in filter_by_entropy)
            )
            else [False]
        )
        self.filter_by_correlation = (
            filter_by_correlation
            if (
                isinstance(filter_by_correlation, list)
                and all(isinstance(item, bool) for item in filter_by_correlation)
            )
            else [False]
        )
        self.subgroups_to_name = (
            subgroups_to_name
            if isinstance(subgroups_to_name, dict)
            else {
                "all": "Complete library",
                "bloodtests": "Blood tests",
                "is_ALIGENT": "Aligent library",
                "is_TWIST": "Twist library",
                "is_CORONA": "Corona library",
                "is_PNP": "Metagenomics\nantigens",
                "is_auto": "Human Autoantigens",
                "is_patho": "Pathogenic strains",
                "is_probio": "Probiotic strains",
                "is_IgA": "Antibody-coated\nstrains",
                "is_MPA": "Microbiota\nstrains",
                "is_bac_flagella": "Flagellins",
                "is_infect": "Infectious\npathogens",
                "is_EBV": "Epstein-Barr\nVirus",
                "is_toxin": "Toxin",
                "is_phage": "Phages",
                "is_allergens": "Allergens",
                "is_influenza": "Influenza",
                "is_EM": "Microbiota\ngenes",
                "signalp6_slow": "Secreted proteins",
                "is_topgraph_new_&_old": "Membrane proteins",
                "diamond_mmseqs_intersec_toxin": "Predicted toxins",
                "is_IEDB_or_cntrl": "IEDB/controls",
                "is_pos_cntrl": "Positive control",
                "is_neg_cntrl": "Negative control",
                "is_rand_cntrl": "Random control",
            }
        )
        self.subgroups_order = (
            subgroups_order
            if (
                isinstance(subgroups_order, list)
                and all(isinstance(item, str) for item in subgroups_order)
            )
            else [
                "Complete library",
                "Aligent library",
                "Twist library",  # 'Corona library',
                "Metagenomics\nantigens",
                "Human Autoantigens",
                "Pathogenic strains",
                "Probiotic strains",
                "Antibody-coated\nstrains",
                "Microbiota\nstrains",
                "Flagellins",
                "Infectious\npathogens",
                "Epstein-Barr\nVirus",
                "Toxin",
                "Phages",
                "Allergens",
                "Influenza",
                "Microbiota\ngenes",
                "Secreted proteins",
                "Membrane proteins",
                "Predicted toxins",
                "IEDB/controls",
                "Positive control",
                "Negative control",
                "Random control",
            ]
        )
        self.subgroups_to_include = (
            subgroups_to_include
            if (
                isinstance(subgroups_to_include, list)
                and all(isinstance(item, str) for item in subgroups_to_include)
            )
            else [
                "all",
                "is_ALIGENT",
                "is_TWIST",  # 'is_CORONA',
                "is_PNP",
                "is_auto",
                "is_patho",
                "is_probio",
                "is_IgA",
                "is_MPA",
                "is_bac_flagella",
                "is_infect",
                "is_EBV",
                "is_toxin",
                "is_phage",
                "is_allergens",
                "is_influenza",
                "is_EM",
                "signalp6_slow",
                "is_topgraph_new_&_old",
                "diamond_mmseqs_intersec_toxin",
                "is_IEDB_or_cntrl",
                "is_pos_cntrl",
                "is_neg_cntrl",
                "is_rand_cntrl",
            ]
        )
        self.estimators_info = (
            estimators_info
            if isinstance(estimators_info, dict)
            else {
                "XGBClassifier": {
                    "estimator_class": xgb.XGBClassifier,
                    "estimator_kwargs": {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "random_state": self.random_state,
                        "nthread": 1,
                        "n_jobs": -1,
                        "n_estimators": 150,
                        "learning_rate": 0.1,
                        "max_depth": 6,
                    },
                }
            }
        )
        self.param_grid = (
            param_grid
            if isinstance(param_grid, dict)
            else {
                "XGBClassifier": {
                    "n_estimators": [50, 100, 200, 500, 1000],
                    "learning_rate": [0.01, 0.1, 0.3],  # Learning rate (eta)
                    "max_depth": [4, 6, 8],  # Maximum tree depth for base learners
                    # 'gamma': [0, 0.1, 0.3, 0.5],                              # Minimum loss reduction for a split
                    "subsample": [
                        0.6,
                        0.8,
                        1.0,
                    ],  # Subsample ratio of the training instances
                    "colsample_bytree": [
                        0.6,
                        0.8,
                        1.0,
                    ],  # Subsample ratio of columns when constructing each tree
                    # 'reg_alpha': [0, 0.1, 0.5, 1.0],                          # L1 regularization term
                    "reg_lambda": [1, 1.5, 2, 3],  # L2 regularization term
                    # ,'booster': ['gbtree', 'dart']
                }
            }
        )
        self.subgroups_colors = (
            subgroups_colors
            if isinstance(subgroups_colors, dict)
            else dict(
                zip(
                    self.subgroups_order,
                    sns.color_palette()[0:4]
                    + sns.color_palette("Set2", 12)[5:6]
                    + sns.color_palette("Set3", 12)[9:10]
                    + sns.color_palette()[8:10]
                    + sns.color_palette()[5:6]
                    + [sns.color_palette("Set2", 8)[i] for i in [1, 2, 6]]
                    + sns.color_palette("Accent")[5:6]
                    + sns.color_palette("Set3", 12)[0:1]
                    + sns.color_palette("Set3", 12)[2:7]
                    + sns.color_palette("Set2", 8)[0:1]
                    + sns.color_palette("PRGn")[0:1]
                    + sns.husl_palette(s=0.4),
                )
            )
        )
        self.additional_features_only_color = "black"
        self.run_plates_only_color = "silver"
        self.additional_features_run_plates_only_color = "gray"

        self.filters_metadata = (
            filters_metadata if isinstance(filters_metadata, dict) else None
        )  # Default to an empty dict
        self.combined_filters_metadata = (
            combined_filters_metadata
            if (
                isinstance(combined_filters_metadata, list)
                and all(isinstance(item, dict) for item in combined_filters_metadata)
            )
            else None
        )  # Default to an empty list

        # Oligo Handler
        self.transposed = transposed if isinstance(transposed, bool) else True
        # Meta Handler
        self.imputed = imputed if isinstance(imputed, bool) else False
        # FeatureManager
        self.fillna = fillna if isinstance(fillna, bool) else False

        # Performance validator
        self.libraries_prefixes = (
            libraries_prefixes
            if (
                isinstance(libraries_prefixes, list)
                and all(isinstance(item, str) for item in libraries_prefixes)
            )
            else ["agilent", "corona2", "twist"]
        )
        self.cv_method = cv_method if cv_method in ("loo", "kfold") else "kfold"
        self.split_train_test = (
            split_train_test if isinstance(split_train_test, bool) else True
        )
        self.compute_feature_importance = (
            compute_feature_importance
            if isinstance(compute_feature_importance, bool)
            else False
        )
        self.return_train = return_train if isinstance(return_train, bool) else True
        self.return_test = return_test if isinstance(return_test, bool) else False
        self.external_set = external_set if isinstance(external_set, bool) else False
        self.tuning_parameters = (
            tuning_parameters if isinstance(tuning_parameters, bool) else True
        )
        self.train_size = train_size if isinstance(train_size, (int, float)) else 0.8
        self.k = k if isinstance(k, int) else 10
        self.tuning_n_iter = tuning_n_iter if isinstance(tuning_n_iter, int) else 20
        self.tuning_k = tuning_k if isinstance(tuning_k, int) else 3

        # must be set explicitly
        self.config_file = config_file if isinstance(config_file, str) else None
        # overwrite parameters present in yaml file
        if self.config_file is not None:
            if not (
                self.config_file.endswith(".yaml") or self.config_file.endswith(".yml")
            ):
                raise ValueError(
                    f"Config file '{self.config_file}' does not appear to be a YAML file (expected .yaml or .yml)."
                )
            try:
                self.load_from_file(self.config_file)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"The config file '{self.config_file}' is not valid YAML. Error: {e}"
                )
        else:
            raise ValueError("A YAML file must be provided")

        # Validate that mandatory attributes are set
        self.validate_mandatory_attributes()

        # derived values
        self.label_group_tests = "-".join(self.group_tests)
        self.group_label_encoding = {
            test: idx for idx, test in enumerate(self.group_tests)
        }
        self.predictions_dir = Path(self.data_dir) / "Predictions"
        self.visualization_dir = Path(self.predictions_dir) / "Figures"
        self.figures_dir = (
            Path(self.visualization_dir) / f"figures_{self.label_group_tests}"
        )

        # Ensure the directories exist
        if not os.path.exists(self.predictions_dir):
            os.makedirs(self.predictions_dir)
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

    def validate_mandatory_attributes(self):
        """Validate that mandatory attributes are set and not None."""
        mandatory_attributes = [
            "metadata_dir",
            "data_dir",
            "project",
            "lib_meta_data",
            "group_tests",
        ]
        for attr in mandatory_attributes:
            if getattr(self, attr) is None:
                raise ValueError(
                    f"The mandatory attribute '{attr}' is not set or is None."
                )

    def set_attribute(self, attr_name, value):
        if hasattr(self, attr_name):
            current_value = getattr(self, attr_name, value)
            # Only perform type checking if the attribute already has a non-None value
            if (
                current_value is not None
                and not isinstance(value, type(current_value))
                and not isinstance(current_value, Path)
            ):
                raise TypeError(
                    f"Expected value of type {type(current_value)} for attribute '{attr_name}', but got {type(value)}"
                )
            # setattr(self, attr_name, value)
            setattr(
                self,
                attr_name,
                (
                    Path(value)
                    if attr_name == "metadata_dir" or attr_name == "data_dir"
                    else value
                ),
            )

        else:
            raise AttributeError(f"Config has no attribute named '{attr_name}'")

    def process_estimators_info(self):
        """
        Converts any 'estimator_class' entries from strings to actual classes.
        """

        def get_class_from_string(class_path: str):
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        if hasattr(self, "estimators_info") and isinstance(self.estimators_info, dict):
            for name, info in self.estimators_info.items():
                class_str = info.get("estimator_class")
                if class_str and isinstance(class_str, str):
                    info["estimator_class"] = get_class_from_string(class_str)

    def get_bayesian_param_grid_from_dict_items(self, model_type: str = "xgboost"):
        # Expect structure like param_grid: { xgboost: { ... }, random_forest: { ... } }
        raw_pg = self.param_grid.get(model_type)
        if raw_pg is None:
            raise ValueError(f"No param_grid found for model type '{model_type}'")
        pg = {}
        for name, spec in raw_pg.items():
            t = spec["type"].lower()
            if t == "integer":
                pg[name] = Integer(spec["low"], spec["high"])
            elif t == "real":
                pg[name] = Real(
                    spec["low"], spec["high"], prior=spec.get("prior", "uniform")
                )
            elif t == "categorical":
                pg[name] = Categorical(spec["categories"])
            else:
                raise ValueError(f"Unknown type '{t}' for {name}")
        self.param_grid = pg

    def load_from_file(self, config_file):
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        # Only update attributes that exist in config_data, leaving all other defaults intact.
        for key, value in config_data.items():
            # If the attribute exists in the class, override it.
            # This ensures we only update keys present in the config.
            self.set_attribute(key, value)

        self.process_estimators_info()

    def _set_string_attribute(self, attr_name, value):
        if not isinstance(value, str):
            raise ValueError(f"{attr_name} must be a string.")
        setattr(self, attr_name, value)

        # Special case: if DATA_DIR is updated, automatically set PREDICTIONS_DIR
        if attr_name == "data_dir":
            self.predictions_dir = Path(self.data_dir) / "Predictions"

    def _set_list_string_attribute(self, attr_name, value):
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{attr_name} must be a list of strings.")
        setattr(self, attr_name, value)

    def _set_list_boolean_attribute(self, attr_name, value):
        if not isinstance(value, list) or not all(
            isinstance(item, bool) for item in value
        ):
            raise ValueError(
                f"{attr_name} must be a list of booleans and can only contain [True, False], [True], or [False]."
            )
        if set(value) not in [{True}, {False}, {True, False}]:
            raise ValueError(
                f"{attr_name} must only be [True, False], [True], or [False]."
            )
        setattr(self, attr_name, value)

    @staticmethod
    def _is_str_or_list_of_str(x):
        if isinstance(x, str):
            return True
        if isinstance(x, list) and all(isinstance(item, str) for item in x):
            return True
        return False

    def _set_dict_string_attribute(self, attr_name, value):
        if not isinstance(value, dict):
            raise ValueError(f"The attribute '{attr_name}' must be a dictionary.")

        # Check if all keys and values are strings
        if not all(
            isinstance(k, str) and self._is_str_or_list_of_str(v)
            for k, v in value.items()
        ):
            raise TypeError(
                f"The attribute '{attr_name}' must be a dictionary with string keys and string values."
            )
        setattr(self, attr_name, value)

    def _set_list_of_dict_string_attribute(self, attr_name, value):
        if not isinstance(value, list):
            raise ValueError(
                f"The attribute '{attr_name}' must be a list of dictionaries."
            )

        if not all(isinstance(item, dict) for item in value):
            raise ValueError(
                f"All elements in the attribute '{attr_name}' must be dictionaries."
            )

        # Check if all keys and values in each dictionary are strings
        for dictionary in value:
            if not all(
                isinstance(k, str) and isinstance(v, str) for k, v in dictionary.items()
            ):
                raise ValueError(
                    f"All dictionaries in the attribute '{attr_name}' must have string keys and string values."
                )
        setattr(self, attr_name, value)

    # Using the helper function for each setter
    def set_metadata_dir(self, metadata_dir):
        self._set_string_attribute("metadata_dir", metadata_dir)

    def set_data_dir(self, data_dir):
        self._set_string_attribute("data_dir", data_dir)

    def set_visualization_dir(self, visualization_dir):
        self._set_string_attribute("visualization_dir", visualization_dir)
        self.figures_dir = Path(visualization_dir) / f"figures_{self.label_group_tests}"
        os.makedirs(self.figures_dir, exist_ok=True)

    def set_project(self, project):
        self._set_string_attribute("project", project)

    def set_lib_meta_data(self, lib_meta_data):
        self._set_string_attribute("lib_meta_data", lib_meta_data)

    def set_column_sample_name(self, col_sample_name):
        self._set_string_attribute("col_sample_name", col_sample_name)

    def set_column_target(self, col_target):
        self._set_string_attribute("col_target", col_target)

    def set_column_predict(self, col_predict):
        self._set_string_attribute("col_predict", col_predict)

    def set_group_tests(self, group_tests):
        self._set_list_string_attribute("group_tests", group_tests)
        self.label_group_tests = "-".join(group_tests)
        self.group_label_encoding = {test: idx for idx, test in enumerate(group_tests)}

    def set_extra_features(self, extra_features):
        self._set_list_string_attribute("extra_features_to_include", extra_features)

    def set_data_types(self, data_types):
        self._set_list_string_attribute("data_types", data_types)

    def set_prevalence_thresholds_min(self, prevalence_thresholds_min):
        if not isinstance(prevalence_thresholds_min, list):
            raise ValueError(
                "prevalence_thresholds must be a list of integers/float between 0-100."
            )
        setattr(self, "prevalence_thresholds_min", prevalence_thresholds_min)

    def set_prevalence_thresholds_max(self, prevalence_thresholds_max):
        if not isinstance(prevalence_thresholds_max, list):
            raise ValueError(
                "prevalence_thresholds must be a list of integers/float between 0-100."
            )
        setattr(self, "prevalence_thresholds_max", prevalence_thresholds_max)

    def set_with_oligos_options(self, value):
        self._set_list_boolean_attribute("with_oligos_options", value)

    def set_with_additional_features_options(self, value):
        self._set_list_boolean_attribute("with_additional_features_options", value)

    def set_with_run_plates_options(self, value):
        self._set_list_boolean_attribute("with_run_plates_options", value)

    def set_filter_by_entropy(self, value):
        self._set_list_boolean_attribute("filter_by_entropy", value)

    def set_filter_by_correlation(self, value):
        self._set_list_boolean_attribute("filter_by_correlation", value)

    def set_filters_metadata(self, value):
        self._set_dict_string_attribute("filters_metadata", value)

    def set_combined_filters_metadata(self, value):
        self._set_list_of_dict_string_attribute("combined_filters_metadata", value)

    def get_attribute(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            raise AttributeError(f"Config has no attribute named '{attr_name}'")


class MetadataHandler:
    def __init__(self, config):
        self.config = config
        self.imputed = (
            self.config.imputed
        )  # imputed if isinstance(imputed, bool) else False

    def set_imputed(self, imputed):
        if isinstance(imputed, bool):
            self.imputed = imputed
        else:
            raise ValueError("The imputed parameter must be a boolean.")

    @staticmethod
    def filter_metadata(ind_meta, filters_metadata):
        """
        Filters the metadata DataFrame based on specified filter conditions.
        """
        if not isinstance(filters_metadata, dict):
            raise ValueError(
                f"Value '{filters_metadata}' must be a dictionary "
                f"with key as column name and value as the attributes to subset"
            )

        for column, value in filters_metadata.items():
            if column in ind_meta.columns:
                if isinstance(value, list):
                    ind_meta = ind_meta[ind_meta[column].isin(value)]
                else:
                    ind_meta = ind_meta[ind_meta[column] == value]
            else:
                raise ValueError(f"Column '{column}' does not exist in the metadata.")

        return ind_meta

    @staticmethod
    def apply_combined_filters_metadata(ind_meta, combined_filters_metadata):
        """
        Applies combined filtering conditions to the metadata DataFrame.
        """
        if not isinstance(combined_filters_metadata, list):
            raise ValueError(
                f"Value '{combined_filters_metadata}' must be a list of dictionaries. "
                f"Each dictionary with key as column name and value as the attributes to subset"
            )

        combined_filter = pd.Series([False] * len(ind_meta), index=ind_meta.index)
        for condition in combined_filters_metadata:
            temp_filter = pd.Series([True] * len(ind_meta), index=ind_meta.index)
            for column, value in condition.items():
                if column in ind_meta.columns:
                    if isinstance(value, list):
                        temp_filter &= ind_meta[column].isin(value)
                    else:
                        temp_filter &= ind_meta[column] == value
                else:
                    raise ValueError(
                        f"Column '{column}' does not exist in the metadata."
                    )
            combined_filter |= temp_filter
        return ind_meta[combined_filter]

    def get_individuals_metadata_df(self):
        # Load metadata based on file type
        if self.config.meta_typefile == "excel":
            ind_meta = pd.read_excel(
                Path(self.config.metadata_dir) / f"{self.config.project}_metadata.xlsx",
                sheet_name=0,
                index_col=self.config.col_sample_name,
            )
        elif self.config.meta_typefile == "csv":
            ind_meta = pd.read_csv(
                Path(self.config.metadata_dir) / f"{self.config.project}_metadata.csv",
                index_col=self.config.col_sample_name,
                low_memory=False,
            )
        else:
            raise ValueError(
                f"Invalid file type: {self.config.meta_typefile}. Expected 'csv' or 'excel'."
            )

        # Drop unnamed numeric columns
        unnamed_col = ind_meta.columns[ind_meta.columns.str.match(r"^Unnamed")].tolist()
        if unnamed_col:
            # Check if the unnamed column contains a numeric range from 0 to number of rows - 1
            col_name = unnamed_col[0]
            if (
                ind_meta[col_name].dtype.kind in "iufc"
                and (ind_meta[col_name] == range(len(ind_meta))).all()
            ):
                ind_meta = ind_meta.drop(columns=[col_name])

        # Drop columns with 'Sample' or 'barcode'
        columns_to_drop = ind_meta.columns[
            ind_meta.columns.str.contains(r"Sample|barcode", case=False)
        ].tolist()
        if columns_to_drop:
            ind_meta = ind_meta.drop(columns=columns_to_drop)

        # if self.config.group_tests  is None:
        #    self.config.group_tests = ind_meta[self.config.COL_TARGET].unique()
        # ind_meta = ind_meta[ind_meta[self.config.COL_TARGET].isin(self.config.group_tests )]

        # Rename 'Gender' to 'Sex' and encode values if necessary
        if "Gender" in ind_meta.columns and "Sex" not in ind_meta.columns:
            ind_meta = ind_meta.rename(columns={"Gender": "Sex"})
        if "Sex" in ind_meta.columns and ind_meta["Sex"].isin(["F", "M"]).any():
            ind_meta["Sex"] = (
                ind_meta["Sex"].replace(self.config.sex_encoding).astype(int)
            )

        if self.config.filters_metadata:
            ind_meta = self.filter_metadata(ind_meta, self.config.filters_metadata)
        # Apply combined conditions if specified
        elif self.config.combined_filters_metadata:
            ind_meta = self.apply_combined_filters_metadata(
                ind_meta, self.config.combined_filters_metadata
            )

        return ind_meta

    def get_additional_features_df(self, **impute_kwargs):
        ind_meta_df = self.get_individuals_metadata_df()[
            self.config.extra_features_to_include
        ]
        if self.imputed:
            imputer = SimpleImputer(**impute_kwargs)
            ind_meta_df = pd.DataFrame(
                data=imputer.fit_transform(ind_meta_df),
                columns=ind_meta_df.columns,
                index=ind_meta_df.index,
            )
        return ind_meta_df

    def get_run_plates_df(self):
        meta_df = self.get_individuals_metadata_df().reset_index()
        meta_df["Run_Plate"] = meta_df[self.config.col_sample_name].str.extract(
            r"(^R\d+P\d+)"
        )
        run_plates_df = meta_df[[self.config.col_sample_name, "Run_Plate"]]

        # Perform one-hot encoding on the extracted Run_Plate column
        return pd.get_dummies(
            run_plates_df, columns=["Run_Plate"], prefix="Run_Plate", dtype=int
        ).set_index([self.config.col_sample_name])

    def get_additional_features_run_plates_df(self, **impute_kwargs):
        ind_meta_df = self.get_additional_features_df(**impute_kwargs)
        run_plate_df = self.get_run_plates_df()
        return pd.merge(
            ind_meta_df,
            run_plate_df,
            left_on=self.config.col_sample_name,
            right_on=self.config.col_sample_name,
        )


class OligosHandler:
    def __init__(self, config, data_type=None):
        self.config = config
        self.data_type = data_type if isinstance(data_type, str) else "exist"
        self.transposed = self.config.transposed

    def set_transposed(self, transposed):
        if isinstance(transposed, bool):
            self.transposed = transposed
        else:
            raise ValueError("transposed must be a boolean.")

    def set_data_type(self, data_type):
        if isinstance(data_type, str):
            self.data_type = data_type
        else:
            raise ValueError("data_type must be a str.")

    def get_oligos_df(self):
        if self.data_type not in ["fold", "exist", "p_val"]:
            raise ValueError(
                f"Invalid file type: {self.data_type}. Expected types: 'fold', 'exist', or 'p_val'."
            )
        oligos_df = pd.read_csv(
            Path(self.config.data_dir) / f"{self.data_type}.csv",
            index_col=0,
            low_memory=False,
        )
        return oligos_df.T if self.transposed else oligos_df

    def get_oligos_metadata_df(self):
        file_path = Path(self.config.metadata_dir) / self.config.lib_meta_data
        if file_path.suffix == ".pkl":
            df = pd.read_pickle(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError("The LIB_META_DATA must be a '.pkl' or '.csv' filename")
        return df.sort_index(ascending=True, inplace=False)


class FeatureManager:
    def __init__(
        self,
        config,
        metadata_handler,
        oligos_handler,
        subgroup=None,
        group_oligos=None,
        with_oligos=None,
        with_additional_features=None,
        with_run_plates=None,
        filter_by_correlation=None,
        filter_by_entropy=None,
        entropy_threshold=None,
        prevalence_threshold_max=None,
        prevalence_threshold_min=None,
    ):

        self.config = config
        self.metadata_handler = metadata_handler
        self.oligos_handler = oligos_handler

        self.le_classes = None

        self.fillna = (
            self.config.fillna
        )  # fillna if isinstance(fillna, bool) else False

        self.with_oligos = with_oligos if isinstance(with_oligos, bool) else True
        self.with_additional_features = (
            with_additional_features
            if isinstance(with_additional_features, bool)
            else False
        )
        self.with_run_plates = (
            with_run_plates if isinstance(with_run_plates, bool) else False
        )
        self.group_oligos = group_oligos if isinstance(group_oligos, bool) else False

        self.filter_by_correlation = (
            filter_by_correlation if isinstance(filter_by_correlation, bool) else False
        )
        self.filter_by_entropy = (
            filter_by_entropy if isinstance(filter_by_entropy, bool) else False
        )
        if self.filter_by_entropy:
            if entropy_threshold is not None and isinstance(entropy_threshold, float):
                self.entropy_threshold = entropy_threshold
            elif config.entropy_threshold is not None:
                self.entropy_threshold = config.entropy_threshold
            else:
                self.entropy_threshold = 0.4
        self.prevalence_threshold_min = (
            prevalence_threshold_min
            if isinstance(
                prevalence_threshold_min, (int, float, np.integer, np.floating)
            )
            else 2.0
        )
        self.prevalence_threshold_max = (
            prevalence_threshold_max
            if isinstance(
                prevalence_threshold_max, (int, float, np.integer, np.floating)
            )
            else 98.0
        )

        self.subgroup = subgroup if isinstance(subgroup, str) else "all"

    def _set_bool_attribute(self, attr_name, value):
        if not isinstance(value, bool):
            raise ValueError(f"{attr_name} must be a boolean.")
        setattr(self, attr_name, value)

    def set_fillna(self, fillna):
        self._set_bool_attribute("fillna", fillna)

    def set_with_oligos(self, with_oligos):
        self._set_bool_attribute("with_oligos", with_oligos)

    def set_with_additional_features(self, with_additional_features):
        self._set_bool_attribute("with_additional_features", with_additional_features)

    def set_with_run_plates(self, with_run_plates):
        self._set_bool_attribute("with_run_plates", with_run_plates)

    def set_filter_by_correlation(self, filter_by_correlation):
        self._set_bool_attribute("filter_by_correlation", filter_by_correlation)

    def set_filter_by_entropy(self, filter_by_entropy):
        self._set_bool_attribute("filter_by_entropy", filter_by_entropy)

    def set_group_oligos(self, group_oligos):
        self._set_bool_attribute("group_oligos", group_oligos)

    def set_entropy_threshold(self, entropy_threshold):
        if isinstance(entropy_threshold, float):
            self.entropy_threshold = entropy_threshold
        else:
            raise ValueError("The entropy_threshold must be a float.")

    def set_prevalence_threshold_min(self, prevalence_threshold_min):
        if (
            isinstance(prevalence_threshold_min, (int, float, np.integer, np.floating))
            and 0 <= prevalence_threshold_min <= 100
        ):
            self.prevalence_threshold_min = prevalence_threshold_min
        else:
            raise ValueError(
                "The prevalence threshold must be an integer or a float between 0-100."
            )

    def set_prevalence_threshold_max(self, prevalence_threshold_max):
        if (
            isinstance(prevalence_threshold_max, (int, float, np.integer, np.floating))
            and 0 <= prevalence_threshold_max <= 100
        ):
            self.prevalence_threshold_max = prevalence_threshold_max
        else:
            raise ValueError(
                "The prevalence threshold must be an integer or a float between 0-100."
            )

    def set_subgroup(self, subgroup):
        if isinstance(subgroup, str):
            self.subgroup = subgroup
        else:
            raise ValueError("subgroup must be a string.")

    def get_attribute(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            raise AttributeError(f"Config has no attribute named '{attr_name}'")

    def get_encoding(self, target_df):
        if self.config.group_label_encoding is None:
            le = LabelEncoder()
            target = le.fit_transform(target_df)
            self.le_classes = dict(zip(le.classes_, range(len(le.classes_))))
            return pd.DataFrame({self.config.col_target: target}, index=target_df.index)
        else:
            return target_df.map(self.config.group_label_encoding)

    def get_target_df(self):
        target_df = self.metadata_handler.get_individuals_metadata_df()[
            self.config.col_target
        ]

        try:
            # Try to convert the target column to integers
            target_df = target_df.astype(int)
            return target_df
        except ValueError:
            # If conversion fails, check for mixed values
            if target_df.apply(lambda x: isinstance(x, int)).any():
                raise ValueError(
                    "The target column must contain only the group test labels to encode or must already be encoded."
                )
            else:
                # If all are non-numeric, proceed to encoding
                return self.get_encoding(target_df)

    def filter_oligos_target_df(self, target_oligos_df):
        # Initialize the filters conditionally

        demog_cols = [
            c
            for c in target_oligos_df.columns
            if c in self.config.extra_features_to_include
        ]
        peptide_cols = [
            c
            for c in target_oligos_df.columns
            if c not in self.config.extra_features_to_include
        ]

        correlation_filter = (
            CorrelationFilter(threshold=0.9, method="phi")
            if self.filter_by_correlation
            else None
        )
        entropy_filter = (
            EntropyFilter(threshold=self.entropy_threshold)
            if self.filter_by_entropy
            else None
        )
        prevalence_filter = PrevalenceFilter(
            threshold_min=self.prevalence_threshold_min,
            threshold_max=self.prevalence_threshold_max,
        )  # Always apply prevalence filter based on given threshold

        # Initialize the pipeline steps based on active filters
        pipeline_steps = []

        if correlation_filter is not None:
            pipeline_steps.append(("correlation_filter", correlation_filter))
        if entropy_filter is not None:
            pipeline_steps.append(("entropy_filter", entropy_filter))

        # Prevalence filter is always added
        pipeline_steps.append(("prevalence_filter", prevalence_filter))

        # Create a feature selection pipeline with active steps
        pipeline = Pipeline(pipeline_steps)

        # 3) wrap that in a ColumnTransformer
        ct = ColumnTransformer(
            [
                ("peptides", pipeline, peptide_cols),
                ("demog", "passthrough", demog_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # if you want back a pandas DataFrame with nice columns
        # ct.set_output(transform="pandas")
        # Fit the pipeline and transform the DataFrame
        return ct.fit_transform(target_oligos_df)

    def get_oligos_with_target(self):
        target_df = self.get_target_df()  # Get target data
        oligos_df = self.oligos_handler.get_oligos_df()  # Get peptide data

        # Merge target and oligos dataframes
        target_oligos_df = (
            pd.merge(
                target_df, oligos_df, left_index=True, right_index=True, how="inner"
            )
            .set_index(self.config.col_target, append=True)
            .reorder_levels([1, 0])
        )
        target_oligos_df.index.rename(
            [self.config.col_target, self.config.col_sample_name], inplace=True
        )

        # Apply filters if provided
        return self.filter_oligos_target_df(target_oligos_df)

    def get_oligos_metadata_subgroup_with_target(self):
        target_oligos_df = self.get_oligos_with_target()

        if self.subgroup != "all":
            oligos_meta_df = self.oligos_handler.get_oligos_metadata_df()
            to_keep = oligos_meta_df[
                oligos_meta_df[self.subgroup].notna()
                & oligos_meta_df[self.subgroup].astype(bool)
            ].index
            to_keep = to_keep[to_keep.isin(target_oligos_df.columns)]
            target_oligos_df = target_oligos_df[
                to_keep
            ]  # all rows and only columns with true value

        return target_oligos_df

    def get_oligos_additional_features_run_plates_with_target(
        self, target_with_without_oligos_df, **impute_kwargs
    ):
        if self.with_additional_features and self.with_run_plates:
            ind_meta_df = self.metadata_handler.get_additional_features_run_plates_df(
                **impute_kwargs
            )
        elif self.with_additional_features:
            ind_meta_df = self.metadata_handler.get_additional_features_df(
                **impute_kwargs
            )
        elif self.with_run_plates:
            ind_meta_df = self.metadata_handler.get_run_plates_df()
        else:
            raise ValueError(
                "At least one of 'with_oligos', 'with_additional_features', or 'with_run_plates' must be True."
            )
            # return target_with_without_oligos_df

        if isinstance(target_with_without_oligos_df, pd.DataFrame):
            return pd.merge(
                target_with_without_oligos_df,
                ind_meta_df,
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:  # without oligos
            return (
                pd.merge(
                    target_with_without_oligos_df,
                    ind_meta_df,
                    left_on=self.config.col_sample_name,
                    right_on=self.config.col_sample_name,
                )
                .set_index(self.config.col_target, append=True)
                .reorder_levels([1, 0])
            )

    def get_data_with_target(self, **impute_kwargs):
        if self.with_oligos:
            target_oligos_df = self.get_oligos_metadata_subgroup_with_target()
            if self.with_additional_features or self.with_run_plates:
                return self.get_oligos_additional_features_run_plates_with_target(
                    target_oligos_df, **impute_kwargs
                )
            else:
                return target_oligos_df

        elif self.with_additional_features or self.with_run_plates:
            target_df = self.get_target_df()
            return self.get_oligos_additional_features_run_plates_with_target(
                target_df, **impute_kwargs
            )
        else:
            raise ValueError(
                "At least one of 'with_oligos', 'with_additional_features', or 'with_run_plates' must be True."
            )
            # return pd.DataFrame()

    def get_category_oligos_with_target(self, target_oligos_df):
        oligos_meta_df = self.oligos_handler.get_oligos_metadata_df()

        # Filter the metadata DataFrame to include only the desired subgroups
        valid_subgroups = [
            subgroup
            for subgroup in self.config.subgroups_to_include
            if subgroup in oligos_meta_df.columns
        ]
        metadata_filtered = oligos_meta_df.loc[
            oligos_meta_df.index.intersection(target_oligos_df.columns), valid_subgroups
        ]

        # Here we create a dictionary where each peptide maps to a list of subgroups it belongs to
        peptide_to_subgroups = metadata_filtered.apply(
            lambda row: [subgroup for subgroup in valid_subgroups if row[subgroup]],
            axis=1,
        )

        # Initialize an empty DataFrame to store subgroup counts for each sample
        target_category_oligos_df = pd.DataFrame(
            index=target_oligos_df.index, columns=valid_subgroups, data=0
        )

        # Iterate over each sample in the multi-index DataFrame and calculate counts
        for (group_test, sample_id), sample_data in target_oligos_df.groupby(
            level=[0, 1]
        ):
            # Get the list of peptides that are present in the sample
            present_peptides = sample_data.columns[sample_data.iloc[0] > 0]

            # Count presence in each subgroup for the sample
            for peptide in present_peptides:
                # Check which subgroups this peptide belongs to
                subgroups = peptide_to_subgroups.get(peptide, [])
                for subgroup in subgroups:
                    # Increment the count for the subgroup for this specific sample
                    target_category_oligos_df.at[(group_test, sample_id), subgroup] += 1

        entropy_filter = EntropyFilter(threshold=self.entropy_threshold)
        return entropy_filter.fit_transform(target_category_oligos_df)

    def get_aggregated_data_with_target(self, **impute_kwargs):
        target_oligos_df = self.get_oligos_with_target()
        target_category_oligos_df = self.get_category_oligos_with_target(
            target_oligos_df
        )

        if self.with_additional_features or self.with_run_plates:
            return self.get_oligos_additional_features_run_plates_with_target(
                target_category_oligos_df, **impute_kwargs
            )
        else:
            return target_category_oligos_df

    def split_features_target_df(self, target_features_df):
        target = target_features_df.reset_index(level=0)[self.config.col_target]
        features = target_features_df.reset_index(level=0, drop=True)

        if self.fillna:
            features = features.fillna(0).infer_objects(copy=False)
        if features.shape[1] == 0:
            return pd.DataFrame(), pd.Series()

        return features, target

    def get_features_target(self, **impute_kwargs):
        if self.group_oligos:
            target_features_df = self.get_aggregated_data_with_target(
                **impute_kwargs
            ).sort_index(level=self.config.col_target)
        else:
            target_features_df = self.get_data_with_target(**impute_kwargs).sort_index(
                level=self.config.col_target
            )

        features, target = self.split_features_target_df(target_features_df)
        # features = VarianceThreshold(threshold=0.0).fit_transform(features)

        return features, target
