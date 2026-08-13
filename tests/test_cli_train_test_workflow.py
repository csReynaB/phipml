"""Tests for the classification command-line entry point."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

import phipml.cli.train_test as cli
from phipml.classification.train_test_utils import SplitData, ValidationSpec

BOOLEAN_OPTIONS = {
    "--run-nested-cv": "run_nested_cv",
    "--use-pretrained": "use_pretrained",
    "--only-train-model": "only_train_model",
    "--with-oligos": "with_oligos",
    "--with-additional-features": "with_additional_features",
    "--impute-extra-numeric": "impute_extra_numeric",
    "--fill-missing-peptides-with-zero": "fill_missing_peptides_with_zero",
    "--split-only": "split_only",
}


def _pipeline() -> Pipeline:
    return Pipeline([("estimator", DummyClassifier(strategy="prior"))])


def _settings(tmp_path: Path, **updates: Any) -> Any:
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
        "prevalence_threshold_min": 0.0,
        "prevalence_threshold_max": 100.0,
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
        "validation_sets": (
            ValidationSpec(filters={"cohort": "external"}, name="external"),
        ),
        "param_grid_name": "random-forest",
        "input_dir": tmp_path,
        "output_dir": tmp_path / "results",
        "input_name": "comparison",
        "output_name": "comparison",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_parser_leaves_unspecified_booleans_for_yaml() -> None:
    args = cli.parse_args_classification(["--config", "config.yaml"])

    for destination in BOOLEAN_OPTIONS.values():
        assert getattr(args, destination) is None


@pytest.mark.parametrize("option,destination", BOOLEAN_OPTIONS.items())
def test_parser_supports_boolean_optional_actions(
    option: str,
    destination: str,
) -> None:
    positive = cli.parse_args_classification(["-c", "config.yaml", option])
    negative = cli.parse_args_classification(
        ["-c", "config.yaml", option.replace("--", "--no-", 1)]
    )

    assert getattr(positive, destination) is True
    assert getattr(negative, destination) is False


def test_parser_converts_json_filters() -> None:
    args = cli.parse_args_classification(
        [
            "-c",
            "config.yaml",
            "--train",
            '{"cohort":"training"}',
            "--oligo-filters",
            '{"Species":"Homo sapiens"}',
        ]
    )

    assert args.train == {"cohort": "training"}
    assert args.oligo_filters == {"Species": "Homo sapiens"}


def test_save_result_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.joblib"

    cli._save_result({"value": 7}, path)

    assert joblib.load(path) == {"value": 7}


def test_load_pretrained_prefers_training_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    training_model = _pipeline()
    validation_model = _pipeline()
    joblib.dump(
        {"best_estimator": training_model},
        tmp_path / "training_random-forest_comparison_420.joblib",
    )
    joblib.dump(
        {"best_estimator": validation_model},
        tmp_path / "validation_random-forest_comparison_420.joblib",
    )

    loaded = cli._load_pretrained(settings)

    assert isinstance(loaded, Pipeline)


def test_load_pretrained_accepts_explicit_joblib_filename(tmp_path: Path) -> None:
    path = tmp_path / "custom.joblib"
    joblib.dump(_pipeline(), path)
    settings = _settings(tmp_path, input_name="custom.joblib")

    assert isinstance(cli._load_pretrained(settings), Pipeline)


def test_load_pretrained_reports_all_attempted_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find"):
        cli._load_pretrained(_settings(tmp_path))


def test_param_grid_returns_none_or_selected_converted_grid() -> None:
    settings = SimpleNamespace(param_grid_name="random-forest")
    empty_config = SimpleNamespace(param_grid={})
    populated_config = SimpleNamespace(
        param_grid={"random-forest": {"x": {}}},
        get_bayesian_param_grid=lambda name: {"selected": name},
    )

    assert cli._param_grid(empty_config, settings) is None
    assert cli._param_grid(populated_config, settings) == {"selected": "random-forest"}


def test_load_validation_cohort_appends_split_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_X = pd.DataFrame({"p": [1]}, index=["V1"])
    validation_y = pd.Series([1], index=["V1"])
    split_X = pd.DataFrame({"p": [0]}, index=["H1"])
    split_y = pd.Series([0], index=["H1"])

    monkeypatch.setattr(cli, "setup_feature_manager", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "make_dataset",
        lambda *args, **kwargs: SplitData(validation_X, validation_y),
    )

    X, y = cli._load_validation_cohort(
        SimpleNamespace(),
        _settings(Path(".")),
        ValidationSpec(filters={"cohort": "external"}, name="external"),
        SplitData(
            validation_X,
            validation_y,
            X_test=split_X,
            y_test=split_y,
        ),
    )

    assert X.index.tolist() == ["V1", "H1"]
    assert y.index.tolist() == ["V1", "H1"]


def test_main_runs_nested_cv_trains_saves_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    config = SimpleNamespace(peptide_prefixes=("agilent_",), param_grid={})
    X_train = pd.DataFrame(
        {"agilent_p1": [0, 1, 0, 1]},
        index=["S1", "S2", "S3", "S4"],
    )
    y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
    X_test = pd.DataFrame({"agilent_p1": [0, 1]}, index=["E1", "E2"])
    y_test = pd.Series([0, 1], index=X_test.index)
    pipeline = _pipeline()

    monkeypatch.setattr(
        cli,
        "parse_args_classification",
        lambda argv: SimpleNamespace(config="unused.yaml"),
    )
    monkeypatch.setattr(cli, "Config", lambda path: config)
    monkeypatch.setattr(
        cli.ClassificationRunSettings,
        "from_sources",
        lambda config, args: settings,
    )
    monkeypatch.setattr(cli, "setup_feature_manager", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "make_dataset",
        lambda *args, **kwargs: SplitData(X_train, y_train),
    )
    monkeypatch.setattr(
        cli,
        "apply_training_prevalence",
        lambda manager, X, run_settings: X,
    )
    monkeypatch.setattr(cli, "build_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(cli, "_param_grid", lambda *args: None)

    metrics = {
        "roc": {"auc": 0.75},
        "pr": {"ap": 0.70},
    }
    monkeypatch.setattr(
        cli,
        "nested_cv",
        lambda *args, **kwargs: (
            [pipeline],
            pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns),
            pd.Series([0.1, 0.9, 0.2, 0.8], index=X_train.index),
            [np.array([0, 1]), np.array([2, 3])],
            metrics,
            [["agilent_p1"], ["agilent_p1"]],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_validation_cohort",
        lambda *args: (X_test, y_test),
    )

    def fake_train_validate(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("get_only_model"):
            return pipeline
        return (
            pipeline,
            pd.DataFrame(0.0, index=X_test.index, columns=X_test.columns),
            pd.Series([0.1, 0.9], index=X_test.index),
            metrics,
            ["agilent_p1"],
            {
                "missing_features": [],
                "missing_peptides": [],
                "missing_non_peptides": [],
                "extra_features": [],
            },
        )

    monkeypatch.setattr(cli, "train_and_validate_model", fake_train_validate)

    assert cli.main(["-c", "unused.yaml"]) == 0

    nested_path = settings.output_dir / "nested_random-forest_comparison_420.joblib"
    training_path = settings.output_dir / "training_random-forest_comparison_420.joblib"
    validation_path = (
        settings.output_dir / "validation_random-forest_external_420.joblib"
    )
    assert nested_path.is_file()
    assert training_path.is_file()
    assert validation_path.is_file()
    assert "metrics_train" in joblib.load(nested_path)
    assert "best_estimator" in joblib.load(training_path)
    assert "metrics_test" in joblib.load(validation_path)


def test_main_only_train_model_saves_model_and_skips_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path,
        run_nested_cv=False,
        only_train_model=True,
        validation_sets=(),
    )
    config = SimpleNamespace(peptide_prefixes=("agilent_",), param_grid={})
    X = pd.DataFrame({"agilent_p": [0, 1]}, index=["S1", "S2"])
    y = pd.Series([0, 1], index=X.index)
    pipeline = _pipeline()
    calls = {"validation": 0}

    monkeypatch.setattr(
        cli,
        "parse_args_classification",
        lambda argv: SimpleNamespace(config="unused.yaml"),
    )
    monkeypatch.setattr(cli, "Config", lambda path: config)
    monkeypatch.setattr(
        cli.ClassificationRunSettings,
        "from_sources",
        lambda config, args: settings,
    )
    monkeypatch.setattr(cli, "setup_feature_manager", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "make_dataset",
        lambda *args, **kwargs: SplitData(X, y),
    )
    monkeypatch.setattr(
        cli,
        "apply_training_prevalence",
        lambda manager, frame, run_settings: frame,
    )
    monkeypatch.setattr(cli, "build_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(cli, "_param_grid", lambda *args: None)

    def fake_train_validate(*args: Any, **kwargs: Any) -> Pipeline:
        if not kwargs.get("get_only_model"):
            calls["validation"] += 1
        return pipeline

    monkeypatch.setattr(cli, "train_and_validate_model", fake_train_validate)

    assert cli.main(["-c", "unused.yaml"]) == 0
    assert calls["validation"] == 0
    assert (
        settings.output_dir / "training_random-forest_comparison_420.joblib"
    ).is_file()


@pytest.mark.skipif(shutil.which("phipml") is None, reason="phipml is not installed")
def test_installed_phipml_command_displays_help() -> None:
    completed = subprocess.run(
        ["phipml", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "classification" in completed.stdout.lower()


def test_module_command_displays_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "phipml.cli.train_test", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--run-nested-cv" in completed.stdout
