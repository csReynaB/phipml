"""Tests for :mod:`phipml.classification.train_test_utils`."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import phipml.classification.train_test_utils as utils
from phipml.classification.train_test_utils import (
    ClassificationRunSettings,
    SplitData,
    ValidationSpec,
    apply_training_prevalence,
    concatenate_datasets,
    make_dataset,
)


def _config(tmp_path: Path, classification: dict[str, Any] | None = None) -> Any:
    """Return the small Config-like object needed by from_sources()."""
    return SimpleNamespace(
        classification=classification or {},
        config_file=(tmp_path / "config.yaml").resolve(),
        random_state=420,
        filters_metadata={"cohort": "top-level"},
        oligo_filters=None,
        oligo_filter_mode="all",
    )


def _settings(**updates: Any) -> ClassificationRunSettings:
    values: dict[str, Any] = {
        "seed": 420,
        "model_type": "random-forest",
        "run_nested_cv": True,
        "use_pretrained": False,
        "only_train_model": False,
        "subgroup": "all",
        "oligo_filters": None,
        "oligo_filter_mode": "all",
        "with_oligos": True,
        "with_additional_features": False,
        "prevalence_threshold_min": 5.0,
        "prevalence_threshold_max": 95.0,
        "outer_cv_splits": 2,
        "inner_cv_splits": 2,
        "n_iter": 2,
        "n_jobs_outer": 1,
        "n_jobs_inner": 1,
        "impute_extra_numeric": False,
        "extra_numeric_impute_strategy": "median",
        "fill_missing_peptides_with_zero": True,
        "train_size": 0.5,
        "split_only": False,
        "train_filters": {"cohort": "training"},
        "split_filters": None,
        "validation_sets": (),
        "param_grid_name": "random-forest",
        "input_dir": Path("."),
        "output_dir": Path("."),
        "input_name": "input",
        "output_name": "output",
        "classification_threshold": 0.5,
        "bootstrap_validation": True,
        "bootstrap_n_resamples": 1000,
        "bootstrap_confidence_level": 0.95,
    }
    values.update(updates)
    return ClassificationRunSettings(**values)


def test_from_sources_resolves_yaml_cli_precedence_and_relative_paths(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        {
            "seed": 100,
            "model_type": "random-forest",
            "run_nested_cv": False,
            "with_oligos": True,
            "train_filters": {"cohort": "yaml-training"},
            "validation_sets": [
                {"name": "external", "filters": {"cohort": "external"}}
            ],
            "output_dir": "results",
            "classification_threshold": 0.4,
            "bootstrap_validation": True,
            "bootstrap_n_resamples": 500,
            "bootstrap_confidence_level": 0.90,
        },
    )
    args = Namespace(
        seed=999,
        run_nested_cv=True,
        classification_threshold=0.35,
        bootstrap_n_resamples=750,
    )

    settings = ClassificationRunSettings.from_sources(config, args)

    assert settings.seed == 999
    assert settings.model_type == "random-forest"
    assert settings.run_nested_cv is True
    assert settings.train_filters == {"cohort": "yaml-training"}
    assert settings.validation_sets == (
        ValidationSpec(filters={"cohort": "external"}, name="external"),
    )
    assert settings.output_dir == (tmp_path / "results").resolve()
    assert settings.classification_threshold == pytest.approx(0.35)
    assert settings.bootstrap_validation is True
    assert settings.bootstrap_n_resamples == 750
    assert settings.bootstrap_confidence_level == pytest.approx(0.90)


def test_cli_validation_definitions_replace_yaml_definitions(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {"validation_sets": [{"name": "yaml", "filters": {"cohort": "yaml"}}]},
    )
    args = Namespace(validate=[['{"cohort":"cli"}', "cli"]])

    settings = ClassificationRunSettings.from_sources(config, args)

    assert settings.validation_sets == (
        ValidationSpec(filters={"cohort": "cli"}, name="cli"),
    )


@pytest.mark.parametrize(
    ("classification", "message"),
    (
        ({"model_type": "svm"}, "model_type"),
        (
            {"with_oligos": False, "with_additional_features": False},
            "Enable with_oligos",
        ),
        (
            {"prevalence_threshold_min": 80, "prevalence_threshold_max": 20},
            "Minimum prevalence",
        ),
        ({"train_size": 1.0}, "train_size"),
        ({"split_only": True, "split_filters": None}, "requires split_filters"),
        ({"classification_threshold": 1.1}, "classification_threshold"),
        ({"bootstrap_n_resamples": 1}, "bootstrap_n_resamples"),
        ({"bootstrap_confidence_level": 1.0}, "bootstrap_confidence_level"),
    ),
)
def test_from_sources_rejects_invalid_settings(
    tmp_path: Path,
    classification: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassificationRunSettings.from_sources(
            _config(tmp_path, classification),
            Namespace(),
        )


def test_split_data_has_test_requires_both_objects() -> None:
    X = pd.DataFrame({"p": [0, 1]})
    y = pd.Series([0, 1])

    assert SplitData(X, y).has_test is False
    assert SplitData(X, y, X_test=X).has_test is False
    assert SplitData(X, y, X_test=X, y_test=y).has_test is True


class _FeatureManagerStub:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y

    def get_features_target(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.X.copy(), self.y.copy()


def test_make_dataset_without_split_preserves_all_samples() -> None:
    X = pd.DataFrame({"p": range(6)}, index=[f"S{i}" for i in range(6)])
    y = pd.Series([0, 1, 0, 1, 0, 1], index=X.index)

    result = make_dataset(_FeatureManagerStub(X, y), _settings(), split=False)

    pd.testing.assert_frame_equal(result.X_train, X)
    pd.testing.assert_series_equal(result.y_train, y)
    assert result.has_test is False


def test_make_dataset_creates_reproducible_stratified_split() -> None:
    X = pd.DataFrame({"p": range(12)}, index=[f"S{i}" for i in range(12)])
    y = pd.Series([0, 1] * 6, index=X.index)
    settings = _settings(seed=10, train_size=0.5)

    first = make_dataset(_FeatureManagerStub(X, y), settings, split=True)
    second = make_dataset(_FeatureManagerStub(X, y), settings, split=True)

    assert first.has_test is True
    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert set(first.y_train.unique()) == {0, 1}
    assert set(first.y_test.unique()) == {0, 1}
    assert set(first.X_train.index).isdisjoint(first.X_test.index)


class _PrevalenceManagerStub:
    def __init__(self, *, fail: bool = False) -> None:
        self.prevalence_threshold_min = 0.0
        self.prevalence_threshold_max = 100.0
        self.seen: tuple[float, float] | None = None
        self.fail = fail

    def filter_prevalence(self, X: pd.DataFrame) -> pd.DataFrame:
        self.seen = (
            self.prevalence_threshold_min,
            self.prevalence_threshold_max,
        )
        if self.fail:
            raise RuntimeError("filter failed")
        return X.loc[:, [X.columns[0]]].copy()


def test_apply_training_prevalence_uses_run_thresholds_and_restores_manager() -> None:
    manager = _PrevalenceManagerStub()
    X = pd.DataFrame({"keep": [0, 1], "drop": [0, 0]})

    filtered = apply_training_prevalence(manager, X, _settings())

    assert manager.seen == (5.0, 95.0)
    assert manager.prevalence_threshold_min == 0.0
    assert manager.prevalence_threshold_max == 100.0
    assert filtered.columns.tolist() == ["keep"]


def test_apply_training_prevalence_restores_manager_after_failure() -> None:
    manager = _PrevalenceManagerStub(fail=True)

    with pytest.raises(RuntimeError, match="filter failed"):
        apply_training_prevalence(
            manager,
            pd.DataFrame({"p": [0, 1]}),
            _settings(),
        )

    assert manager.prevalence_threshold_min == 0.0
    assert manager.prevalence_threshold_max == 100.0


def test_concatenate_datasets_reorders_columns_and_target() -> None:
    first_X = pd.DataFrame(
        {"p1": [1], "p2": [0]},
        index=["S1"],
    )
    second_X = pd.DataFrame(
        {"p2": [1], "p1": [0]},
        index=["S2"],
    )
    first_y = pd.Series([0], index=["S1"], name="target")
    second_y = pd.Series([1], index=["S2"], name="target")

    X, y = concatenate_datasets(first_X, first_y, second_X, second_y)

    assert X.columns.tolist() == ["p1", "p2"]
    assert X.index.tolist() == ["S1", "S2"]
    assert X.loc["S2"].tolist() == [0, 1]
    assert y.index.tolist() == X.index.tolist()
    assert y.tolist() == [0, 1]


def test_concatenate_datasets_rejects_mismatched_features() -> None:
    with pytest.raises(ValueError, match="same raw features"):
        concatenate_datasets(
            pd.DataFrame({"p1": [1]}, index=["S1"]),
            pd.Series([0], index=["S1"]),
            pd.DataFrame({"p2": [1]}, index=["S2"]),
            pd.Series([1], index=["S2"]),
        )


def test_concatenate_datasets_rejects_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        concatenate_datasets(
            pd.DataFrame({"p": [1]}, index=["S1"]),
            pd.Series([0], index=["S1"]),
            pd.DataFrame({"p": [0]}, index=["S1"]),
            pd.Series([1], index=["S1"]),
        )


def test_setup_feature_manager_copies_config_and_overrides_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(filters_metadata={"cohort": "original"})
    captured: dict[str, Any] = {}

    class HandlerStub:
        def __init__(self, handler_config: Any) -> None:
            self.config = handler_config

    class FeatureManagerStub:
        def __init__(self, manager_config: Any, *args: Any, **kwargs: Any) -> None:
            captured["config"] = manager_config
            captured["kwargs"] = kwargs

    monkeypatch.setattr(utils, "MetadataHandler", HandlerStub)
    monkeypatch.setattr(utils, "OligosHandler", HandlerStub)
    monkeypatch.setattr(utils, "FeatureManager", FeatureManagerStub)

    utils.setup_feature_manager(
        config,
        {"cohort": "training"},
        _settings(),
    )

    assert config.filters_metadata == {"cohort": "original"}
    assert captured["config"] is not config
    assert captured["config"].filters_metadata == {"cohort": "training"}
    assert captured["kwargs"]["prevalence_threshold_min"] == 0
    assert captured["kwargs"]["prevalence_threshold_max"] == 100
