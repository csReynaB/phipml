# ======================
# Standard library
# ======================
import logging
import time
from dataclasses import dataclass
from typing import Optional

# ======================
# Third-party libraries
# ======================
import joblib
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split

from phipml.classification.helpers import (
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)

# ======================
# Local / project imports
# ======================
from phipml.io.data_handler import FeatureManager, MetadataHandler, OligosHandler

# ======================
# Global configuration
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_config(transform_output="pandas")


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None

    @property
    def has_test(self) -> bool:
        return self.X_test is not None and self.y_test is not None


def apply_prevalence_filter_train_only(
    feature_manager, X_train: pd.DataFrame, args
) -> pd.DataFrame:
    """Apply prevalence filter to X_train, then reset thresholds (as your code does)."""
    feature_manager.prevalence_threshold_min = args.prevalence_threshold_min
    feature_manager.prevalence_threshold_max = args.prevalence_threshold_max
    X_train = feature_manager.filter_oligos_target_df(X_train)
    feature_manager.prevalence_threshold_min = 0.0
    feature_manager.prevalence_threshold_max = 100.0
    return X_train


def safe_concat_Xy(X1, y1, X2, y2):
    """Concat and keep indices consistent."""
    X = pd.concat([X1, X2])
    y = pd.concat([y1, y2])

    return X, y


def make_dataset(
    feature_manager,
    args,
    *,
    do_split: bool,
    apply_prevalence_filter: bool,
) -> SplitData:

    X, y = feature_manager.get_features_target()

    if do_split:
        stratify_param = y if y.nunique() > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            train_size=args.train_size,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify_param,
        )
    else:
        X_tr, y_tr = X, y
        X_te, y_te = None, None

    if apply_prevalence_filter:
        X_tr = apply_prevalence_filter_train_only(feature_manager, X_tr, args)

    return SplitData(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)


def run_and_save_nested_cv(X_train, y_train, args, config):
    """
    Build pipeline, run nested CV and save results.
    """

    start_time = time.time()

    pipeline = build_pipeline(
        X_train,
        model_type=args.model_type,
        random_state=args.seed,
    )

    (
        model_list,
        train_shap_values,
        scores_train,
        validation_indices,
        metrics_train,
    ) = nested_cv(
        X_train,
        y_train,
        pipeline=pipeline,
        param_grid=config.param_grid,
        n_splits=args.outer_cv_split,
        n_splits_inner=args.inner_cv_split,
        n_iter=50,
        model_type=args.model_type,
        random_state=args.seed,
        n_jobs=1,
        n_jobs_inner=-1,
    )

    results = {
        "model_list": model_list,
        "train_shap_values": train_shap_values,
        "scores_train": scores_train,
        "validation_indices_train": validation_indices,
        "metrics_train": metrics_train,
    }

    out_file = (
        f"{args.out_dir}/nested_{args.model_type}_{args.out_name}_{args.seed}.joblib"
    )
    joblib.dump(results, out_file)

    end_time = time.time()
    logger.info(
        f"nested cv runtime for {args.out_name}: {end_time - start_time:.2f} seconds"
    )

    return results


def fit_best_model(X_train, y_train, args, config):
    start_time = time.time()

    best_estimator = train_and_validate_model(
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        pipeline=None,
        param_grid=config.param_grid,
        n_splits=args.outer_cv_split,
        n_iter=50,
        model_type=args.model_type,
        random_state=args.seed,
        n_jobs=-1,
        get_only_model=True,
    )

    end_time = time.time()
    logger.info(
        f"train best model with {args.out_name} runtime: {end_time - start_time:.2f} seconds"
    )

    if args.only_train_model:
        joblib.dump(
            {"best_estimator": best_estimator},
            f"{args.out_dir}/training_{args.model_type}_{args.out_name}_{args.seed}.joblib",
        )

    return best_estimator


def validate_and_save(
    X_train, y_train, X_test, y_test, args, config, out_val, *, best_estimator=None
):
    start_time = time.time()

    best_estimator, test_shap_values, scores_test, metrics_test = (
        train_and_validate_model(
            X_train,
            y_train,
            X_test,
            y_test,
            param_grid=config.param_grid,
            best_estimator=best_estimator,
            n_splits=args.outer_cv_split,
            n_iter=50,
            model_type=args.model_type,
            random_state=args.seed,
            n_jobs=-1,
            get_only_model=False,
        )
    )

    results = {
        "best_estimator": best_estimator,
        "test_shap_values": test_shap_values,
        "scores_test": scores_test,
        "metrics_test": metrics_test,
    }

    out_file = (
        f"{args.out_dir}/validation_{args.model_type}_{out_val}_{args.seed}.joblib"
    )
    joblib.dump(results, out_file)

    end_time = time.time()
    logger.info(
        f"validation for {out_val} runtime: {end_time - start_time:.2f} seconds"
    )

    return results


def get_best_estimator(X_train, y_train, args, config):
    """
    Returns a fitted estimator, either loaded from disk or trained fresh.
    """
    if args.use_pretrained:
        input_file = f"{args.input_dir}/validation_{args.model_type}_{args.input_val}_{args.seed}.joblib"
        best_estimator = joblib.load(input_file)["best_estimator"]
        logger.info(f"Loaded best model from {input_file}")
        return best_estimator

    return fit_best_model(X_train, y_train, args, config)


def build_validation_set(
    feature_manager,
    config,
    filter_val,
    *,
    split_data=None,
):
    """
    Updates config.filters_metadata, fetches X_test/y_test, and optionally appends split_test.
    """
    config.filters_metadata = filter_val
    # IMPORTANT: build *test* from same FeatureManager obj cause filter_metadata is pointing to validation set now.
    X_test, y_test = feature_manager.get_features_target()

    if split_data is not None and split_data.X_test is not None:
        X_test, y_test = safe_concat_Xy(
            X_test, y_test, split_data.X_test, split_data.y_test
        )

    return X_test, y_test


def setup_feature_manager(config, filters_metadata, args):
    """
    Helper function to set up feature manager with common configuration.
    """
    config.filters_metadata = filters_metadata
    metadata_handler = MetadataHandler(config)
    oligos_handler = OligosHandler(config)
    feature_manager = FeatureManager(
        config,
        metadata_handler,
        oligos_handler,
        subgroup=args.subgroup,
        with_oligos=args.with_oligos,
        with_additional_features=args.with_additional_features,
        prevalence_threshold_min=0,  # args.prevalence_threshold_min,
        prevalence_threshold_max=100,  # args.prevalence_threshold_max,
    )
    return feature_manager
