"""Command-line entry point for PhIP-seq binary classification."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from phipml.classification.helpers import (
    ValidationResultWithReport,
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)
from phipml.classification.train_test_utils import (
    ClassificationRunSettings,
    SplitData,
    ValidationSpec,
    apply_training_prevalence,
    concatenate_datasets,
    make_dataset,
    setup_feature_manager,
)
from phipml.io.data_handler import Config

logger = logging.getLogger(__name__)


class _ArgParser(argparse.ArgumentParser):
    """Argument parser supporting one shell-like option per args-file line."""

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        line = arg_line.split("#", 1)[0].strip()
        return shlex.split(line) if line else []


def _json_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON mapping: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def parse_args_classification(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI overrides; unspecified settings are read from YAML."""
    parser = _ArgParser(
        description="Train and validate PhIP-seq classification models.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        "-c",
        "-cf",
        required=True,
        help="YAML configuration path",
    )

    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument(
        "--model-type",
        "--model_type",
        "-mt",
        dest="model_type",
        choices=("xgboost", "random-forest"),
        default=None,
    )
    parser.add_argument(
        "--run-nested-cv",
        "--run_nested_cv",
        "-ncv",
        dest="run_nested_cv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run nested cross-validation",
    )
    parser.add_argument(
        "--use-pretrained",
        "--use_pretrained",
        "-upr",
        dest="use_pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load a previously fitted pipeline",
    )
    parser.add_argument(
        "--only-train-model",
        "--only_train_model",
        "-otm",
        dest="only_train_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fit and save the full-cohort model without validation",
    )

    parser.add_argument("--subgroup", "-sub", default=None)
    parser.add_argument(
        "--oligo-filters",
        dest="oligo_filters",
        type=_json_mapping,
        default=None,
        help='Library metadata filters, for example {"Species":"Homo sapiens"}',
    )
    parser.add_argument(
        "--oligo-filter-mode",
        choices=("all", "any"),
        default=None,
    )
    parser.add_argument(
        "--with-oligos",
        "--with_oligos",
        "-wo",
        dest="with_oligos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include peptide features",
    )
    parser.add_argument(
        "--with-additional-features",
        "--with_additional_features",
        "-wa",
        dest="with_additional_features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include configured clinical features",
    )
    parser.add_argument(
        "--prevalence-threshold-min",
        "--prevalence_threshold_min",
        "-min",
        dest="prevalence_threshold_min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--prevalence-threshold-max",
        "--prevalence_threshold_max",
        "-max",
        dest="prevalence_threshold_max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--outer-cv-splits",
        "--outer_cv_split",
        "-ocv",
        dest="outer_cv_splits",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--inner-cv-splits",
        "--inner_cv_split",
        "-icv",
        dest="inner_cv_splits",
        type=int,
        default=None,
    )
    parser.add_argument("--n-iter", dest="n_iter", type=int, default=None)
    parser.add_argument("--n-jobs-outer", type=int, default=None)
    parser.add_argument("--n-jobs-inner", type=int, default=None)
    parser.add_argument("--param-grid-name", default=None)

    parser.add_argument(
        "--impute-extra-numeric",
        dest="impute_extra_numeric",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Impute continuous numeric clinical features",
    )
    parser.add_argument(
        "--extra-numeric-impute-strategy",
        choices=("mean", "median", "most_frequent", "constant"),
        default=None,
    )
    parser.add_argument(
        "--fill-missing-peptides-with-zero",
        dest="fill_missing_peptides_with_zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Zero-fill peptides absent from an external cohort",
    )

    parser.add_argument("--train", "-t", type=_json_mapping, default=None)
    parser.add_argument(
        "--train-test-split-data",
        "--train_test_split_data",
        "-sp",
        dest="train_test_split_data",
        type=_json_mapping,
        default=None,
    )
    parser.add_argument(
        "--train-size",
        "--train_size",
        "-ts",
        dest="train_size",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--split-only",
        "-nat",
        dest="split_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only the split cohort for training and hold-out validation",
    )
    parser.add_argument(
        "--validate",
        "-v",
        nargs=2,
        action="append",
        default=None,
        metavar=("FILTER_JSON", "OUTPUT_NAME"),
        help='Override YAML validations, for example -v {"cohort":"test"} external',
    )

    parser.add_argument(
        "--input-dir",
        "--input_dir",
        "-id",
        dest="input_dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        "--out_dir",
        "-d",
        dest="output_dir",
        default=None,
    )
    parser.add_argument(
        "--input-name",
        "--input_val",
        "-iv",
        dest="input_name",
        default=None,
    )
    parser.add_argument(
        "--output-name",
        "--out_name",
        "-o",
        dest="output_name",
        default=None,
    )
    return parser.parse_args(argv)

# Backward-compatible import name used by existing notebooks/scripts.
#parse_args_ML = parse_args_classification

def _save_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    logger.info("Saved %s", path)


def _load_pretrained(
    settings: ClassificationRunSettings,
) -> Pipeline:
    supplied = Path(settings.input_name)
    candidates: list[Path] = []
    if supplied.suffix == ".joblib":
        candidates.append(
            supplied if supplied.is_absolute() else settings.input_dir / supplied
        )
    candidates.extend(
        [
            settings.input_dir
            / (
                f"training_{settings.model_type}_"
                f"{settings.input_name}_{settings.seed}.joblib"
            ),
            settings.input_dir
            / (
                f"validation_{settings.model_type}_"
                f"{settings.input_name}_{settings.seed}.joblib"
            ),
        ]
    )
    path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "Could not find a pretrained model. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    loaded = joblib.load(path)
    estimator = loaded.get("best_estimator") if isinstance(loaded, dict) else loaded
    if not isinstance(estimator, Pipeline):
        raise TypeError(f"No sklearn Pipeline found in {path}")
    logger.info("Loaded pretrained model from %s", path)
    return estimator


def _param_grid(
    config: Config,
    settings: ClassificationRunSettings,
) -> dict[str, Any] | None:
    if not config.param_grid:
        logger.info("No hyperparameter grid configured")
        return None
    return config.get_bayesian_param_grid(settings.param_grid_name)


# move to train_test_utils.py and call it on top as from phipml.classification.train_test_utils import ( ... load_validation_cohort,)
def _load_validation_cohort(
    config: Config,
    settings: ClassificationRunSettings,
    validation: ValidationSpec,
    split_data: SplitData | None,
) -> tuple[pd.DataFrame, pd.Series]:
    manager = setup_feature_manager(
        config,
        validation.filters,
        settings,
    )
    dataset = make_dataset(manager, settings, split=False)
    X_test, y_test = dataset.X_train, dataset.y_train

    if split_data is not None and split_data.has_test:
        if split_data.X_test is None or split_data.y_test is None:
            raise RuntimeError("SplitData.has_test is inconsistent")
        X_test, y_test = concatenate_datasets(
            X_test,
            y_test,
            split_data.X_test,
            split_data.y_test,
        )
    return X_test, y_test


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args_classification(argv)
    config = Config(args.config)
    settings = ClassificationRunSettings.from_sources(config, args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    split_manager = None
    split_data: SplitData | None = None
    if settings.split_filters is not None:
        logger.info("Preparing the stratified split cohort")
        split_manager = setup_feature_manager(
            config,
            settings.split_filters,
            settings,
        )
        split_data = make_dataset(
            split_manager,
            settings,
            split=True,
        )

    if settings.split_only:
        if split_data is None or not split_data.has_test:
            raise RuntimeError("split_only requires a completed hold-out split")
        train_manager = split_manager
        if train_manager is None:
            raise RuntimeError("The split feature manager is unavailable")
        X_train = split_data.X_train
        y_train = split_data.y_train
    else:
        logger.info("Preparing the training cohort")
        train_manager = setup_feature_manager(
            config,
            settings.train_filters,
            settings,
        )
        training = make_dataset(
            train_manager,
            settings,
            split=False,
        )
        X_train, y_train = training.X_train, training.y_train

        if split_data is not None:
            X_train, y_train = concatenate_datasets(
                X_train,
                y_train,
                split_data.X_train,
                split_data.y_train,
            )

    X_train = apply_training_prevalence(
        train_manager,
        X_train,
        settings,
    )
    if y_train.nunique() != 2:
        raise ValueError(
            "The final training cohort must contain exactly two target classes"
        )
    logger.info(
        "Final training data: samples=%d, features=%d",
        X_train.shape[0],
        X_train.shape[1],
    )

    if settings.split_only:
        validation_sets = [
            ValidationSpec(
                filters={},
                name=settings.output_name,
            )
        ]
    else:
        validation_sets = list(settings.validation_sets)

    needs_model = settings.only_train_model or bool(validation_sets)
    if not settings.run_nested_cv and not needs_model:
        logger.info(
            "Nothing to run: nested CV, full-model training, and validation are disabled"
        )
        return 0

    pipeline: Pipeline | None = None
    param_grid: dict[str, Any] | None = None
    if settings.run_nested_cv or not settings.use_pretrained:
        pipeline = build_pipeline(
            X_train,
            model_type=settings.model_type,
            random_state=settings.seed,
            peptide_prefixes=config.peptide_prefixes,
            impute_extra_numeric=settings.impute_extra_numeric,
            extra_numeric_impute_strategy=(settings.extra_numeric_impute_strategy),
        )
        param_grid = _param_grid(config, settings)

    # ------------------------
    # Nested CV
    # ------------------------
    if settings.run_nested_cv:
        if pipeline is None:
            raise RuntimeError("Nested CV requires a pipeline")
        started = time.perf_counter()
        nested_result = nested_cv(
            X_train,
            y_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=settings.outer_cv_splits,
            n_splits_inner=settings.inner_cv_splits,
            n_iter=settings.n_iter,
            model_type=settings.model_type,
            random_state=settings.seed,
            n_jobs=settings.n_jobs_outer,
            n_jobs_inner=settings.n_jobs_inner,
            peptide_prefixes=config.peptide_prefixes,
            impute_extra_numeric=settings.impute_extra_numeric,
            extra_numeric_impute_strategy=(settings.extra_numeric_impute_strategy),
        )
        (
            models,
            train_shap,
            train_scores,
            validation_indices,
            train_metrics,
            selected_features,
        ) = nested_result
        _save_result(
            {
                "model_list": models,
                "train_shap_values": train_shap,
                "scores_train": train_scores,
                "validation_indices_train": validation_indices,
                "metrics_train": train_metrics,
                "roc_metrics_train": train_metrics["roc"],
                "pr_metrics_train": train_metrics["pr"],
                "selected_features_train": selected_features,
            },
            settings.output_dir
            / (
                f"nested_{settings.model_type}_"
                f"{settings.output_name}_{settings.seed}.joblib"
            ),
        )
        logger.info(
            "Nested CV completed in %.2f seconds",
            time.perf_counter() - started,
        )

    if not needs_model:
        return 0

    # -------------------------------
    # Full-model training
    # -------------------------------
    started = time.perf_counter()
    if settings.use_pretrained:
        best_estimator = _load_pretrained(settings)
    else:
        if pipeline is None:
            raise RuntimeError("Full-model training requires a pipeline")
        fitted = train_and_validate_model(
            X_train,
            y_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=settings.inner_cv_splits,
            n_iter=settings.n_iter,
            model_type=settings.model_type,
            random_state=settings.seed,
            n_jobs=settings.n_jobs_inner,
            get_only_model=True,
            peptide_prefixes=config.peptide_prefixes,
            impute_extra_numeric=settings.impute_extra_numeric,
            extra_numeric_impute_strategy=(settings.extra_numeric_impute_strategy),
        )
        if not isinstance(fitted, Pipeline):
            raise RuntimeError("Full-cohort training did not return a fitted Pipeline")
        best_estimator = fitted
    logger.info(
        "Full-cohort model ready in %.2f seconds",
        time.perf_counter() - started,
    )

    # Save every newly fitted full-cohort model independently of validation.
    if not settings.use_pretrained:
        _save_result(
            {"best_estimator": best_estimator},
            settings.output_dir
            / (
                f"training_{settings.model_type}_"
                f"{settings.output_name}_{settings.seed}.joblib"
            ),
        )

    # Stop here when validation was not requested.
    if settings.only_train_model:
        return 0


    #---------------------
    # Validation
    #---------------------
    for validation in validation_sets:
        started = time.perf_counter()
        if settings.split_only:
            if (
                split_data is None
                or split_data.X_test is None
                or split_data.y_test is None
            ):
                raise RuntimeError("The split validation set is unavailable")
            X_test = split_data.X_test
            y_test = split_data.y_test
        else:
            X_test, y_test = _load_validation_cohort(
                config,
                settings,
                validation,
                split_data,
            )

        validation_result = train_and_validate_model(
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            best_estimator=best_estimator,
            model_type=settings.model_type,
            random_state=settings.seed,
            get_only_model=False,
            peptide_prefixes=config.peptide_prefixes,
            fill_missing_peptides_with_zero=(settings.fill_missing_peptides_with_zero),
            return_feature_report=True,
        )
        if not isinstance(validation_result, tuple):
            raise RuntimeError(f"Validation {validation.name!r} returned no result")
        result_with_report = cast(
            ValidationResultWithReport,
            validation_result,
        )
        (
            fitted_model,
            test_shap,
            test_scores,
            test_metrics,
            selected_test_features,
            feature_report,
        ) = result_with_report
        _save_result(
            {
                "best_estimator": fitted_model,
                "test_shap_values": test_shap,
                "scores_test": test_scores,
                "metrics_test": test_metrics,
                "roc_metrics_test": test_metrics["roc"],
                "pr_metrics_test": test_metrics["pr"],
                "selected_features_test": selected_test_features,
                "feature_report": feature_report,
            },
            settings.output_dir
            / (
                f"validation_{settings.model_type}_"
                f"{validation.name}_{settings.seed}.joblib"
            ),
        )
        logger.info(
            "Validation %s completed in %.2f seconds",
            validation.name,
            time.perf_counter() - started,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())