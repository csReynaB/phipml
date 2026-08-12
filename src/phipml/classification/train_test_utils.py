"""Utilities shared by the classification CLI and notebooks."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from phipml.io.data_handler import (
    Config,
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationSpec:
    """Metadata filter and output label for one validation cohort."""

    filters: dict[str, Any]
    name: str


@dataclass(frozen=True)
class SplitData:
    """Optional stratified train/test partition for one metadata-defined cohort."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None

    @property
    def has_test(self) -> bool:
        return self.X_test is not None and self.y_test is not None


@dataclass(frozen=True)
class ClassificationRunSettings:
    """Validated execution settings resolved from YAML and CLI overrides."""

    seed: int
    model_type: str
    run_nested_cv: bool
    use_pretrained: bool
    only_train_model: bool

    subgroup: str
    oligo_filters: dict[str, Any] | None
    oligo_filter_mode: str
    with_oligos: bool
    with_additional_features: bool
    prevalence_threshold_min: float
    prevalence_threshold_max: float

    outer_cv_splits: int
    inner_cv_splits: int
    n_iter: int
    n_jobs_outer: int
    n_jobs_inner: int

    impute_extra_numeric: bool
    extra_numeric_impute_strategy: str
    fill_missing_peptides_with_zero: bool

    train_size: float
    split_only: bool
    train_filters: dict[str, Any] | None
    split_filters: dict[str, Any] | None
    validation_sets: tuple[ValidationSpec, ...]

    param_grid_name: str
    input_dir: Path
    output_dir: Path
    input_name: str
    output_name: str

    @classmethod
    def from_sources(
        cls,
        config: Config,
        args: Any,
    ) -> "ClassificationRunSettings":
        """Resolve defaults, then YAML classification, then explicit CLI values."""
        raw = config.classification
        if not isinstance(raw, Mapping):
            raise TypeError("The YAML 'classification' section must be a mapping")

        def choose(setting_name: str, default: Any) -> Any:
            cli_value = getattr(args, setting_name, None)
            return (
                cli_value if cli_value is not None else raw.get(setting_name, default)
            )

        config_dir = config.config_file.parent

        def resolve_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config_dir / path
            return path.resolve()

        train_filters = cls._choose_mapping(
            getattr(args, "train", None),
            raw.get("train_filters", config.filters_metadata),
            "train_filters",
        )
        split_filters = cls._choose_mapping(
            getattr(args, "train_test_split_data", None),
            raw.get("split_filters"),
            "split_filters",
        )
        subgroup = str(choose("subgroup", "all"))
        inherited_oligo_filters = config.oligo_filters if subgroup == "all" else None
        oligo_filters = cls._choose_mapping(
            getattr(args, "oligo_filters", None),
            raw.get("oligo_filters", inherited_oligo_filters),
            "oligo_filters",
        )

        cli_validations: list[list[str]] | None = getattr(args, "validate", None)
        if cli_validations is None:
            validation_sets = cls._parse_validation_sets(raw.get("validation_sets", []))
        else:
            parsed: list[ValidationSpec] = []
            for filter_json, validation_name in cli_validations:
                try:
                    filters = json.loads(filter_json)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON for validation {validation_name!r}: {error}"
                    ) from error
                if not isinstance(filters, Mapping):
                    raise TypeError("Each --validate filter must be a JSON object")
                parsed.append(
                    ValidationSpec(
                        filters=dict(filters),
                        name=validation_name,
                    )
                )
            validation_sets = tuple(parsed)

        model_type = str(choose("model_type", "xgboost"))
        settings = cls(
            seed=int(choose("seed", config.random_state)),
            model_type=model_type,
            run_nested_cv=cls._as_bool(
                choose("run_nested_cv", True),
                "run_nested_cv",
            ),
            use_pretrained=cls._as_bool(
                choose("use_pretrained", False),
                "use_pretrained",
            ),
            only_train_model=cls._as_bool(
                choose("only_train_model", False),
                "only_train_model",
            ),
            subgroup=subgroup,
            oligo_filters=oligo_filters,
            oligo_filter_mode=str(
                choose("oligo_filter_mode", config.oligo_filter_mode)
            ),
            with_oligos=cls._as_bool(
                choose("with_oligos", True),
                "with_oligos",
            ),
            with_additional_features=cls._as_bool(
                choose("with_additional_features", False),
                "with_additional_features",
            ),
            prevalence_threshold_min=float(choose("prevalence_threshold_min", 2.0)),
            prevalence_threshold_max=float(choose("prevalence_threshold_max", 98.0)),
            outer_cv_splits=int(choose("outer_cv_splits", 5)),
            inner_cv_splits=int(choose("inner_cv_splits", 5)),
            n_iter=int(choose("n_iter", 30)),
            n_jobs_outer=int(choose("n_jobs_outer", 1)),
            n_jobs_inner=int(choose("n_jobs_inner", -1)),
            impute_extra_numeric=cls._as_bool(
                choose("impute_extra_numeric", False),
                "impute_extra_numeric",
            ),
            extra_numeric_impute_strategy=str(
                choose("extra_numeric_impute_strategy", "median")
            ),
            fill_missing_peptides_with_zero=cls._as_bool(
                choose("fill_missing_peptides_with_zero", True),
                "fill_missing_peptides_with_zero",
            ),
            train_size=float(choose("train_size", 0.7)),
            split_only=cls._as_bool(
                choose(
                    "split_only",
                    choose("no_additional_train_test_data", False),
                ),
                "split_only",
            ),
            train_filters=train_filters,
            split_filters=split_filters,
            validation_sets=validation_sets,
            param_grid_name=str(choose("param_grid_name", model_type)),
            input_dir=resolve_path(choose("input_dir", ".")),
            output_dir=resolve_path(choose("output_dir", ".")),
            input_name=str(choose("input_name", "input_name")),
            output_name=str(choose("output_name", "out_name")),
        )
        settings.validate()
        return settings

    @staticmethod
    def _choose_mapping(
        cli_value: Any,
        yaml_value: Any,
        setting_name: str,
    ) -> dict[str, Any] | None:
        value = cli_value if cli_value is not None else yaml_value
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"{setting_name} must be a mapping or null")
        return dict(value)

    @staticmethod
    def _as_bool(value: Any, setting_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise TypeError(f"{setting_name} must be a YAML/CLI Boolean, not {value!r}")

    @staticmethod
    def _parse_validation_sets(raw: Any) -> tuple[ValidationSpec, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("classification.validation_sets must be a list")

        parsed: list[ValidationSpec] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise TypeError(f"validation_sets[{index}] must be a mapping")
            name = item.get("name")
            filters = item.get("filters")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"validation_sets[{index}].name must be non-empty")
            if not isinstance(filters, Mapping):
                raise TypeError(f"validation_sets[{index}].filters must be a mapping")
            parsed.append(
                ValidationSpec(
                    filters=dict(filters),
                    name=name,
                )
            )
        return tuple(parsed)

    def validate(self) -> None:
        if self.model_type not in {"xgboost", "random-forest"}:
            raise ValueError("model_type must be 'xgboost' or 'random-forest'")
        if not (self.with_oligos or self.with_additional_features):
            raise ValueError("Enable with_oligos, with_additional_features, or both")
        if self.subgroup != "all" and self.oligo_filters is not None:
            raise ValueError("Use subgroup or oligo_filters, not both")
        if self.oligo_filter_mode not in {"all", "any"}:
            raise ValueError("oligo_filter_mode must be 'all' or 'any'")
        if not (0 <= self.prevalence_threshold_min <= 100):
            raise ValueError("prevalence_threshold_min must be between 0 and 100")
        if not (0 <= self.prevalence_threshold_max <= 100):
            raise ValueError("prevalence_threshold_max must be between 0 and 100")
        if self.prevalence_threshold_min > self.prevalence_threshold_max:
            raise ValueError("Minimum prevalence cannot exceed maximum prevalence")
        if self.outer_cv_splits < 2 or self.inner_cv_splits < 2:
            raise ValueError("Both CV split counts must be at least 2")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if not 0 < self.train_size < 1:
            raise ValueError("train_size must be strictly between 0 and 1")
        if self.extra_numeric_impute_strategy not in {
            "mean",
            "median",
            "most_frequent",
            "constant",
        }:
            raise ValueError("Unsupported extra_numeric_impute_strategy")
        if self.split_only and self.split_filters is None:
            raise ValueError("split_only requires split_filters")
        if (
            self.split_filters is not None
            and not self.split_only
            and self.train_filters is None
        ):
            raise ValueError(
                "split_filters requires train_filters unless split_only is true"
            )
        if self.n_jobs_outer != 1 and self.n_jobs_inner != 1:
            logger.warning(
                "Both outer and inner CV parallelism are enabled; this can "
                "oversubscribe CPUs and memory."
            )


def setup_feature_manager(
    config: Config,
    filters_metadata: Mapping[str, Any] | None,
    settings: ClassificationRunSettings,
) -> FeatureManager:
    """Create isolated handlers so metadata caches cannot leak between cohorts."""
    analysis_config = copy.copy(config)
    # When filters_metadata is None, retain the top-level config.filters_metadata.
    if filters_metadata is not None:
        analysis_config.filters_metadata = dict(filters_metadata)

    return FeatureManager(
        analysis_config,
        MetadataHandler(analysis_config),
        OligosHandler(analysis_config),
        subgroup=settings.subgroup,
        oligo_filters=settings.oligo_filters,
        oligo_filter_mode=settings.oligo_filter_mode,
        with_oligos=settings.with_oligos,
        with_additional_features=settings.with_additional_features,
        # Prevalence is always learned from the final training matrix.
        prevalence_threshold_min=0,
        prevalence_threshold_max=100,
    )


def make_dataset(
    feature_manager: FeatureManager,
    settings: ClassificationRunSettings,
    *,
    split: bool,
) -> SplitData:
    """Load one cohort and optionally create a stratified hold-out split."""
    X, y = feature_manager.get_features_target()
    if not split:
        return SplitData(X_train=X, y_train=y)

    if y.nunique() < 2:
        logger.warning(
            "The split cohort has one target class; splitting without stratification"
        )
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=settings.train_size,
        random_state=settings.seed,
        shuffle=True,
        stratify=stratify,
    )
    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def apply_training_prevalence(
    feature_manager: FeatureManager,
    X_train: pd.DataFrame,
    settings: ClassificationRunSettings,
) -> pd.DataFrame:
    """Learn peptide prevalence from training samples only."""
    old_min = feature_manager.prevalence_threshold_min
    old_max = feature_manager.prevalence_threshold_max
    feature_manager.prevalence_threshold_min = settings.prevalence_threshold_min
    feature_manager.prevalence_threshold_max = settings.prevalence_threshold_max
    try:
        filtered = feature_manager.filter_prevalence(X_train)
    finally:
        feature_manager.prevalence_threshold_min = old_min
        feature_manager.prevalence_threshold_max = old_max

    if filtered.shape[1] == 0:
        raise ValueError("No features remain after training prevalence filtering")
    logger.info(
        "Training prevalence filter: %d -> %d features",
        X_train.shape[1],
        filtered.shape[1],
    )
    return filtered


def concatenate_datasets(
    first_X: pd.DataFrame,
    first_y: pd.Series,
    second_X: pd.DataFrame,
    second_y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Concatenate cohorts after validating feature and sample identities."""
    missing_from_second = first_X.columns.difference(second_X.columns).tolist()
    missing_from_first = second_X.columns.difference(first_X.columns).tolist()

    if missing_from_second or missing_from_first:
        raise ValueError(
            "Cohorts do not expose the same raw features. "
            f"Missing from second: {missing_from_second[:10]}; "
            f"missing from first: {missing_from_first[:10]}"
        )

    # Reorder the second cohort to match the first cohort exactly.
    column_order = first_X.columns.tolist()
    second_X_aligned: pd.DataFrame = second_X.loc[:, column_order].copy()

    X: pd.DataFrame = pd.concat(
        (first_X, second_X_aligned),
        axis="index",
    )
    y: pd.Series = pd.concat(
        (first_y, second_y),
        axis="index",
    )

    duplicated = X.index[X.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Cohorts contain duplicate sample IDs: {duplicated[:10]}"
        )

    # Reorder the target to match the combined feature matrix.
    y = y.reindex(X.index)

    return X, y