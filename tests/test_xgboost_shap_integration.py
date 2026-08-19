"""Integration tests for XGBoost and TreeSHAP in the production pipeline."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from phipml.classification.helpers import (
    _compute_shap_frame,
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)

PEPTIDE_PREFIXES = ("agilent_", "twist_", "corona2_")


def _classification_data(
    n_samples: int,
    *,
    seed: int,
    sample_prefix: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create balanced peptide data with stable signal and realistic noise."""
    if n_samples < 12 or n_samples % 2:
        raise ValueError("n_samples must be an even integer of at least 12")

    rng = np.random.default_rng(seed)
    target = np.repeat([0, 1], n_samples // 2)
    rng.shuffle(target)
    sample_index = pd.Index(
        [f"{sample_prefix}{position:03d}" for position in range(n_samples)],
        name="SampleName",
    )

    peptide_data: dict[str, np.ndarray] = {
        "agilent_signal": target.copy(),
        "twist_inverse_signal": 1 - target,
    }
    for position in range(28):
        prefix = PEPTIDE_PREFIXES[position % len(PEPTIDE_PREFIXES)]
        peptide_data[f"{prefix}noise_{position:02d}"] = rng.integers(
            0,
            2,
            size=n_samples,
        )

    features = pd.DataFrame(peptide_data, index=sample_index)
    features["Sex"] = rng.integers(0, 2, size=n_samples)
    features["Age"] = rng.normal(55.0, 8.0, size=n_samples)
    target_series = pd.Series(
        target,
        index=sample_index,
        name="group_test",
        dtype=int,
    )
    return features, target_series


def _small_xgboost_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build the production preprocessing pipeline with a fast XGBoost model."""
    pipeline = build_pipeline(
        features,
        model_type="xgboost",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    pipeline.set_params(
        estimator__n_estimators=16,
        estimator__max_depth=2,
        estimator__learning_rate=0.15,
        estimator__subsample=0.9,
        estimator__colsample_bytree=0.9,
        estimator__n_jobs=1,
    )
    return pipeline


def test_xgboost_pipeline_predictions_and_tree_shap_additivity() -> None:
    """Fit XGBoost and verify finite SHAP values reproduce raw model output."""
    features, target = _classification_data(
        36,
        seed=101,
        sample_prefix="XGB",
    )
    pipeline = _small_xgboost_pipeline(features)
    pipeline.fit(features, target)

    estimator = pipeline.named_steps["estimator"]
    assert isinstance(estimator, XGBClassifier)

    probabilities = pipeline.predict_proba(features)[:, 1]
    assert probabilities.shape == (len(features),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    evaluated = features.iloc[:8]
    shap_frame, selected_features = _compute_shap_frame(pipeline, evaluated)
    assert selected_features == shap_frame.columns.tolist()
    assert shap_frame.index.equals(evaluated.index)
    assert shap_frame.notna().all().all()
    assert np.isfinite(shap_frame.to_numpy()).all()
    assert np.any(np.abs(shap_frame.to_numpy()) > 0.0)

    transformed = pipeline.named_steps["preprocessor"].transform(evaluated)
    explainer = shap.TreeExplainer(estimator)
    direct_shap = np.asarray(explainer.shap_values(transformed))
    np.testing.assert_allclose(
        shap_frame.to_numpy(),
        direct_shap,
        rtol=1e-6,
        atol=1e-6,
    )
    # For binary XGBoost, TreeSHAP explains the raw log-odds margin.
    # Read expected_value after evaluating SHAP: newer SHAP releases update
    # the explainer baseline during the first evaluation.
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed_margin = base_value + shap_frame.sum(axis=1).to_numpy()
    model_margin = estimator.predict(transformed, output_margin=True)
    np.testing.assert_allclose(
        reconstructed_margin,
        model_margin,
        rtol=1e-5,
        atol=1e-5,
    )


def test_xgboost_nested_cv_and_external_validation_include_shap() -> None:
    """Exercise XGBoost through nested CV and independent validation."""
    features, target = _classification_data(
        36,
        seed=202,
        sample_prefix="TRAIN",
    )
    pipeline = _small_xgboost_pipeline(features)

    models, shap_values, scores, folds, metrics, selected_features = nested_cv(
        features,
        target,
        pipeline=pipeline,
        param_grid=None,
        n_splits=3,
        n_splits_inner=2,
        n_iter=2,
        model_type="xgboost",
        random_state=17,
        n_jobs=1,
        n_jobs_inner=1,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )

    assert len(models) == len(folds) == len(selected_features) == 3
    assert all(
        isinstance(model.named_steps["estimator"], XGBClassifier) for model in models
    )
    assert scores.index.equals(features.index)
    assert scores.between(0.0, 1.0, inclusive="both").all()
    assert shap_values.index.equals(features.index)
    assert shap_values.columns.equals(features.columns)
    assert shap_values.notna().all().all()
    assert np.any(np.abs(shap_values.to_numpy()) > 0.0)
    assert float(metrics["roc"]["auc"]) >= 0.80
    assert float(metrics["pr"]["ap"]) >= 0.80

    fitted = train_and_validate_model(
        features,
        target,
        pipeline=pipeline,
        param_grid=None,
        n_splits=2,
        n_iter=2,
        model_type="xgboost",
        random_state=17,
        n_jobs=1,
        get_only_model=True,
        peptide_prefixes=PEPTIDE_PREFIXES,
    )
    assert isinstance(fitted, Pipeline)

    external_features, external_target = _classification_data(
        18,
        seed=303,
        sample_prefix="EXTERNAL",
    )
    result = train_and_validate_model(
        features,
        target,
        X_test=external_features,
        y_test=external_target,
        best_estimator=fitted,
        model_type="xgboost",
        random_state=17,
        peptide_prefixes=PEPTIDE_PREFIXES,
        bootstrap_validation=False,
    )

    assert isinstance(result, tuple) and len(result) == 5
    model, external_shap, external_scores, external_metrics, external_selected = result
    assert model is fitted
    assert external_scores.index.equals(external_features.index)
    assert external_scores.between(0.0, 1.0, inclusive="both").all()
    assert external_shap.index.equals(external_features.index)
    assert external_shap.columns.equals(features.columns)
    assert external_shap.notna().all().all()
    assert np.any(np.abs(external_shap.to_numpy()) > 0.0)
    assert external_selected
    assert float(external_metrics["roc"]["auc"]) >= 0.80
    assert float(external_metrics["pr"]["ap"]) >= 0.80

    # These values are useful in verbose pytest output when testing an update.
    print(
        f"XGBoost {version('xgboost')}; SHAP {version('shap')}; "
        f"nested ROC-AUC={float(metrics['roc']['auc']):.3f}; "
        f"external ROC-AUC={float(external_metrics['roc']['auc']):.3f}"
    )
