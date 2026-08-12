"""Real integration tests for the phipml classification workflow.

Unlike the fast unit tests, this module does not mock nested CV, SHAP,
hyperparameter tuning, external validation, or the CLI workflow.  The data are
small and synthetic so the tests remain suitable for local development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real

from phipml.classification.helpers import (
    _build_and_fit_pipeline,
    _compute_shap_frame,
    bootstrap_auc,
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)
from phipml.cli.train_test import main


PEPTIDE_PREFIXES = ("agilent_", "twist_", "corona2_")
N_PEPTIDES = 50
MISSING_EXTERNAL_PEPTIDE = "corona2_noise_02"


def _peptide_columns(X: pd.DataFrame) -> list[str]:
    """Return peptide columns using the same prefixes as the real pipeline."""
    return [
        str(column)
        for column in X.columns
        if str(column).startswith(PEPTIDE_PREFIXES)
    ]


def _synthetic_classification_data(
    n_samples: int,
    *,
    seed: int,
    sample_prefix: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return balanced data with strong signal plus peptide/clinical noise."""
    if n_samples < 4 or n_samples % 2:
        raise ValueError("n_samples must be an even integer of at least four")

    rng = np.random.default_rng(seed)
    target = np.repeat([0, 1], n_samples // 2)
    rng.shuffle(target)
    index = pd.Index(
        [f"{sample_prefix}{number:02d}" for number in range(n_samples)],
        name="SampleName",
    )

    peptide_data: dict[str, np.ndarray] = {
        # Two deliberately strong peptide signals make expected performance
        # stable despite the small integration-test dataset.
        "agilent_signal": target,
        "twist_signal": 1 - target,
    }
    for number in range(N_PEPTIDES - len(peptide_data)):
        prefix = PEPTIDE_PREFIXES[number % len(PEPTIDE_PREFIXES)]
        peptide_data[f"{prefix}noise_{number:02d}"] = rng.integers(
            0,
            2,
            size=n_samples,
        )

    X = pd.DataFrame(peptide_data, index=index)
    # Exercise multiple binary and continuous additional-feature columns.
    X["Sex"] = rng.integers(0, 2, size=n_samples)
    X["Smoking"] = rng.integers(0, 2, size=n_samples)
    X["Age"] = rng.normal(55.0, 8.0, size=n_samples)
    X["BMI"] = rng.normal(26.0, 3.5, size=n_samples)

    assert len(_peptide_columns(X)) == N_PEPTIDES
    assert X.shape == (n_samples, N_PEPTIDES + 4)
    y = pd.Series(target, index=index, name="group_test", dtype=int)
    return X, y


def _small_random_forest_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build the production preprocessing pipeline with a small final forest."""
    pipeline = build_pipeline(
        X,
        model_type="random-forest",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    pipeline.set_params(
        estimator__n_estimators=12,
        estimator__max_depth=3,
        estimator__n_jobs=1,
    )
    return pipeline


def _small_joint_search_space() -> dict[str, Any]:
    """Return a fast, stable selector/forest space for integration tests."""
    return {
        "preprocessor__peptides__feature_selection__estimator__l1_ratio": Real(
            0.35,
            0.65,
            prior="uniform",
        ),
        "preprocessor__peptides__feature_selection__estimator__C": Real(
            0.5,
            2.0,
            prior="log-uniform",
        ),
        "estimator__n_estimators": Integer(20, 50),
        "estimator__max_depth": Integer(2, 5),
        "estimator__min_samples_leaf": Integer(1, 3),
    }


def _noisy_synthetic_classification_data(
    n_samples: int,
    *,
    seed: int,
    sample_prefix: str,
    flip_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return data whose two signal peptides are wrong for some samples."""
    if not 0.0 < flip_fraction < 0.5:
        raise ValueError("flip_fraction must be between zero and 0.5")

    X, y = _synthetic_classification_data(
        n_samples,
        seed=seed,
        sample_prefix=sample_prefix,
    )
    rng = np.random.default_rng(seed + 10_000)
    flipped_positions: list[int] = []
    flips_per_class = max(1, round(n_samples * flip_fraction / 2))
    for target_class in (0, 1):
        class_positions = np.flatnonzero(y.to_numpy() == target_class)
        flipped_positions.extend(
            rng.choice(
                class_positions,
                size=flips_per_class,
                replace=False,
            ).tolist()
        )

    flipped_index = X.index[flipped_positions]
    for signal_column in ("agilent_signal", "twist_signal"):
        X.loc[flipped_index, signal_column] = (
            1 - X.loc[flipped_index, signal_column]
        )
    return X, y


def _assert_probability_metric(value: object, name: str) -> None:
    numeric = float(value)
    assert np.isfinite(numeric), f"{name} is not finite: {numeric}"
    assert 0.0 <= numeric <= 1.0, f"{name} is outside [0, 1]: {numeric}"


def test_real_nested_cv_fits_models_calculates_metrics_and_shap() -> None:
    """Exercise real outer CV, fitting, predictions, metrics, and TreeSHAP."""
    X, y = _synthetic_classification_data(
        30,
        seed=1,
        sample_prefix="TRAIN",
    )

    pipeline = _small_random_forest_pipeline(X)
    preprocessor = pipeline.named_steps["preprocessor"]
    peptide_inputs = next(
        columns
        for name, _, columns in preprocessor.transformers
        if name == "peptides"
    )
    binary_inputs = next(
        columns
        for name, _, columns in preprocessor.transformers
        if name == "binary_extra"
    )
    continuous_inputs = next(
        columns
        for name, _, columns in preprocessor.transformers
        if name == "continuous_extra"
    )
    assert len(peptide_inputs) == N_PEPTIDES
    assert set(binary_inputs) == {"Sex", "Smoking"}
    assert set(continuous_inputs) == {"Age", "BMI"}

    (
        models,
        shap_values,
        scores,
        validation_folds,
        metrics,
        selected_feature_sets,
    ) = nested_cv(
        X,
        y,
        pipeline=pipeline,
        param_grid=None,
        n_splits=3,
        n_splits_inner=2,
        n_iter=2,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        n_jobs_inner=1,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert len(models) == 3
    assert all(isinstance(model, Pipeline) for model in models)
    assert all(hasattr(model.named_steps["estimator"], "classes_") for model in models)

    # Every sample must occur in exactly one outer validation fold.
    validation_positions = np.concatenate(validation_folds)
    assert sorted(validation_positions.tolist()) == list(range(len(X)))
    assert len(np.unique(validation_positions)) == len(X)

    assert scores.index.equals(X.index)
    assert scores.notna().all()
    assert scores.between(0.0, 1.0, inclusive="both").all()

    assert shap_values.index.equals(X.index)
    assert shap_values.columns.equals(X.columns)
    assert shap_values.notna().all().all()
    assert np.any(np.abs(shap_values.to_numpy()) > 0.0)

    assert len(selected_feature_sets) == 3
    assert all(selected for selected in selected_feature_sets)
    assert all(
        set(selected).issubset(set(X.columns))
        for selected in selected_feature_sets
    )

    roc = metrics["roc"]
    pr = metrics["pr"]
    _assert_probability_metric(roc["auc"], "mean ROC-AUC")
    _assert_probability_metric(pr["ap"], "mean PR-AUC")
    assert np.asarray(roc["fpr"]).shape == (200,)
    assert np.asarray(roc["tpr"]).shape == (200,)
    assert np.asarray(pr["recall"]).shape == (200,)
    assert np.asarray(pr["pr"]).shape == (200,)

    # The synthetic signal is intentionally strong; this also detects an
    # accidental target/probability inversion.
    assert float(roc["auc"]) >= 0.80
    assert float(pr["ap"]) >= 0.80


def test_real_nested_cv_with_noisy_signal_is_useful_but_not_perfect() -> None:
    """Check realistic non-perfect performance without weakening core tests."""
    X, y = _noisy_synthetic_classification_data(
        60,
        seed=11,
        sample_prefix="NOISY",
        flip_fraction=0.20,
    )
    observed_flip_fraction = X["agilent_signal"].ne(y).mean()
    assert observed_flip_fraction == pytest.approx(0.20)

    _, _, scores, _, metrics, _ = nested_cv(
        X,
        y,
        pipeline=_small_random_forest_pipeline(X),
        param_grid=None,
        n_splits=3,
        n_splits_inner=2,
        n_iter=2,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        n_jobs_inner=1,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    roc_auc = float(metrics["roc"]["auc"])
    pr_auc = float(metrics["pr"]["ap"])
    assert scores.notna().all()
    assert 0.65 <= roc_auc < 1.0
    assert 0.65 <= pr_auc < 1.0


def test_real_bayesian_tuning_returns_fitted_best_pipeline() -> None:
    """Run a small real BayesSearchCV through the production fitting helper."""
    X, y = _synthetic_classification_data(
        30,
        seed=2,
        sample_prefix="TUNE",
    )
    search_space = {
        "estimator__n_estimators": Integer(6, 10),
        "estimator__max_depth": Integer(2, 4),
    }

    fitted = _build_and_fit_pipeline(
        pipeline=_small_random_forest_pipeline(X),
        X_train=X,
        y_train=y,
        param_grid=search_space,
        n_splits=2,
        n_iter=2,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert isinstance(fitted, Pipeline)
    assert hasattr(fitted.named_steps["estimator"], "classes_")
    tuned_parameters = fitted.get_params()
    assert 6 <= tuned_parameters["estimator__n_estimators"] <= 10
    assert 2 <= tuned_parameters["estimator__max_depth"] <= 4
    probabilities = fitted.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_real_nested_cv_jointly_tunes_selector_and_random_forest() -> None:
    """Tune peptide selection and the classifier inside each outer fold."""
    X, y = _noisy_synthetic_classification_data(
        60,
        seed=11,
        sample_prefix="TUNED",
        flip_fraction=0.20,
    )
    search_space = _small_joint_search_space()

    models, shap_values, scores, folds, metrics, selected_features = nested_cv(
        X,
        y,
        pipeline=_small_random_forest_pipeline(X),
        param_grid=search_space,
        n_splits=3,
        n_splits_inner=2,
        n_iter=4,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        n_jobs_inner=1,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert len(models) == len(folds) == len(selected_features) == 3
    assert scores.notna().all()
    assert shap_values.notna().all().all()
    assert all(selected_features)

    for model in models:
        parameters = model.get_params()
        assert 0.35 <= parameters[
            "preprocessor__peptides__feature_selection__estimator__l1_ratio"
        ] <= 0.65
        assert 0.5 <= parameters[
            "preprocessor__peptides__feature_selection__estimator__C"
        ] <= 2.0
        assert 20 <= parameters["estimator__n_estimators"] <= 50
        assert 2 <= parameters["estimator__max_depth"] <= 5
        assert 1 <= parameters["estimator__min_samples_leaf"] <= 3

    roc_auc = float(metrics["roc"]["auc"])
    pr_auc = float(metrics["pr"]["ap"])
    assert 0.55 <= roc_auc <= 1.0
    assert 0.55 <= pr_auc <= 1.0


def test_real_xgboost_pipeline_fits_predicts_and_calculates_shap() -> None:
    """Exercise the second supported classifier and its TreeSHAP output."""
    X, y = _synthetic_classification_data(
        30,
        seed=7,
        sample_prefix="XGB",
    )
    pipeline = build_pipeline(
        X,
        model_type="xgboost",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    pipeline.set_params(
        estimator__n_estimators=8,
        estimator__max_depth=2,
        estimator__learning_rate=0.2,
        estimator__n_jobs=1,
    )
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]
    shap_frame, selected_features = _compute_shap_frame(
        pipeline,
        X.iloc[:6],
    )

    assert probabilities.shape == (len(X),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert shap_frame.shape[0] == 6
    assert shap_frame.notna().all().all()
    assert np.any(np.abs(shap_frame.to_numpy()) > 0.0)
    assert selected_features == shap_frame.columns.tolist()


def test_real_perfect_external_validation_without_tuning() -> None:
    """Validate perfect signal and external alignment without hyperparameter search."""
    X_train, y_train = _synthetic_classification_data(
        30,
        seed=3,
        sample_prefix="TRAIN",
    )
    fitted = train_and_validate_model(
        X_train,
        y_train,
        pipeline=_small_random_forest_pipeline(X_train),
        param_grid=None,
        n_splits=2,
        n_iter=2,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        get_only_model=True,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    assert isinstance(fitted, Pipeline)

    X_test, y_test = _synthetic_classification_data(
        10,
        seed=4,
        sample_prefix="EXTERNAL",
    )
    # A missing peptide may be safely zero-filled; an unused external feature
    # must be reported and discarded.
    X_test = X_test.drop(columns=MISSING_EXTERNAL_PEPTIDE)
    X_test["unused_external"] = 99

    result = train_and_validate_model(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        best_estimator=fitted,
        model_type="random-forest",
        random_state=17,
        get_only_model=False,
        peptide_prefixes=PEPTIDE_PREFIXES,
        fill_missing_peptides_with_zero=True,
        return_feature_report=True,
    )

    assert isinstance(result, tuple) and len(result) == 6
    model, shap_values, scores, metrics, selected_features, report = result
    assert model is fitted
    assert scores.index.equals(X_test.index)
    assert scores.between(0.0, 1.0, inclusive="both").all()
    assert shap_values.index.equals(X_test.index)
    assert shap_values.columns.tolist() == X_train.columns.tolist()
    assert shap_values.notna().all().all()
    assert np.any(np.abs(shap_values.to_numpy()) > 0.0)
    assert selected_features

    assert report["missing_peptides"] == [MISSING_EXTERNAL_PEPTIDE]
    assert report["missing_non_peptides"] == []
    assert report["extra_features"] == ["unused_external"]

    _assert_probability_metric(metrics["roc"]["auc"], "external ROC-AUC")
    _assert_probability_metric(metrics["pr"]["ap"], "external PR-AUC")
    assert float(metrics["roc"]["auc"]) == pytest.approx(1.0)
    assert float(metrics["pr"]["ap"]) == pytest.approx(1.0)


def test_real_noisy_external_validation_without_tuning() -> None:
    """Validate noisy signal with fitted selection but no hyperparameter search."""
    X_train, y_train = _noisy_synthetic_classification_data(
        60,
        seed=11,
        sample_prefix="NOISY_TRAIN",
        flip_fraction=0.20,
    )
    X_external, y_external = _noisy_synthetic_classification_data(
        40,
        seed=21,
        sample_prefix="NOISY_EXTERNAL",
        flip_fraction=0.20,
    )

    fitted = train_and_validate_model(
        X_train,
        y_train,
        pipeline=_small_random_forest_pipeline(X_train),
        param_grid=None,
        n_splits=2,
        n_iter=2,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        get_only_model=True,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    assert isinstance(fitted, Pipeline)

    result = train_and_validate_model(
        X_train,
        y_train,
        X_test=X_external,
        y_test=y_external,
        best_estimator=fitted,
        model_type="random-forest",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert isinstance(result, tuple) and len(result) == 5
    model, shap_values, scores, metrics, selected_features = result
    assert model is fitted
    assert scores.index.equals(X_external.index)
    assert scores.notna().all()
    assert shap_values.index.equals(X_external.index)
    assert shap_values.notna().all().all()
    assert selected_features

    roc_auc = float(metrics["roc"]["auc"])
    pr_auc = float(metrics["pr"]["ap"])
    assert 0.65 <= roc_auc < 0.95
    assert 0.65 <= pr_auc < 0.95


def test_real_noisy_external_validation_with_joint_tuning() -> None:
    """Tune selector/forest settings before evaluating an external cohort."""
    X_train, y_train = _noisy_synthetic_classification_data(
        60,
        seed=11,
        sample_prefix="TUNED_TRAIN",
        flip_fraction=0.20,
    )
    X_external, y_external = _noisy_synthetic_classification_data(
        40,
        seed=21,
        sample_prefix="TUNED_EXTERNAL",
        flip_fraction=0.20,
    )

    fitted = train_and_validate_model(
        X_train,
        y_train,
        pipeline=_small_random_forest_pipeline(X_train),
        param_grid=_small_joint_search_space(),
        n_splits=2,
        n_iter=4,
        model_type="random-forest",
        random_state=17,
        n_jobs=1,
        get_only_model=True,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    assert isinstance(fitted, Pipeline)

    parameters = fitted.get_params()
    assert 0.35 <= parameters[
        "preprocessor__peptides__feature_selection__estimator__l1_ratio"
    ] <= 0.65
    assert 0.5 <= parameters[
        "preprocessor__peptides__feature_selection__estimator__C"
    ] <= 2.0
    assert 20 <= parameters["estimator__n_estimators"] <= 50

    result = train_and_validate_model(
        X_train,
        y_train,
        X_test=X_external,
        y_test=y_external,
        best_estimator=fitted,
        model_type="random-forest",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert isinstance(result, tuple) and len(result) == 5
    _, shap_values, scores, metrics, selected_features = result
    assert scores.index.equals(X_external.index)
    assert scores.notna().all()
    assert shap_values.index.equals(X_external.index)
    assert shap_values.notna().all().all()
    assert selected_features

    roc_auc = float(metrics["roc"]["auc"])
    pr_auc = float(metrics["pr"]["ap"])
    assert 0.60 <= roc_auc < 0.98
    assert 0.60 <= pr_auc < 0.98


def test_real_bootstrap_auc_returns_finite_uncertainty() -> None:
    """Exercise the estimator-based bootstrap path with real predictions."""
    X, y = _synthetic_classification_data(
        30,
        seed=5,
        sample_prefix="BOOT",
    )
    fitted = _small_random_forest_pipeline(X).fit(X, y)
    grid = np.linspace(0.0, 1.0, 25)

    metrics = bootstrap_auc(
        mean_fpr=grid,
        estimator=fitted,
        X=X,
        y_true=y,
        n_bootstraps=20,
        random_state=17,
    )

    assert np.asarray(metrics["boot_mean_fpr"]).shape == grid.shape
    assert np.asarray(metrics["boot_mean_tpr"]).shape == grid.shape
    for key in (
        "boot_auc_mean",
        "boot_auc_std",
        "boot_auc_ci_lower",
        "boot_auc_ci_upper",
    ):
        assert np.isfinite(float(metrics[key]))
    _assert_probability_metric(metrics["boot_auc_mean"], "bootstrap mean AUC")
    _assert_probability_metric(metrics["boot_auc_ci_lower"], "bootstrap lower CI")
    _assert_probability_metric(metrics["boot_auc_ci_upper"], "bootstrap upper CI")
    assert float(metrics["boot_auc_std"]) >= 0.0
    assert float(metrics["boot_auc_ci_lower"]) <= float(
        metrics["boot_auc_ci_upper"]
    )


def test_real_cli_yaml_to_saved_training_model(tmp_path: Path) -> None:
    """Run the actual CLI main workflow using real CSV/YAML inputs."""
    X, y = _synthetic_classification_data(
        30,
        seed=6,
        sample_prefix="CLI",
    )

    metadata = pd.DataFrame(
        {
            "SampleName": X.index,
            "group_test": y.map({0: "Control", 1: "Case"}).to_numpy(),
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    # DataHandler expects peptide x sample when transposed is true.
    peptide_columns = _peptide_columns(X)
    assert len(peptide_columns) == N_PEPTIDES
    peptide_matrix = X.loc[:, peptide_columns].T
    peptide_matrix.to_csv(tmp_path / "data.csv", index_label="peptide_id")

    config_values = {
        "metadata_input": "metadata.csv",
        "data_input": "data.csv",
        "group_tests": ["Control", "Case"],
        "transposed": True,
        "param_grid": {},
        "classification": {
            "model_type": "random-forest",
            "run_nested_cv": False,
            "use_pretrained": False,
            "only_train_model": True,
            "with_oligos": True,
            "with_additional_features": False,
            "prevalence_threshold_min": 0,
            "prevalence_threshold_max": 100,
            "outer_cv_splits": 2,
            "inner_cv_splits": 2,
            "n_iter": 2,
            "n_jobs_outer": 1,
            "n_jobs_inner": 1,
            "output_dir": "results",
            "output_name": "synthetic",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_values),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 0

    model_path = (
        tmp_path
        / "results"
        / "training_random-forest_synthetic_420.joblib"
    )
    assert model_path.is_file()
    saved = joblib.load(model_path)
    assert set(saved) == {"best_estimator"}
    assert isinstance(saved["best_estimator"], Pipeline)
    assert hasattr(saved["best_estimator"].named_steps["estimator"], "classes_")