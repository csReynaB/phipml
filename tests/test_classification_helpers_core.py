"""Fast behavioural tests for :mod:`phipml.classification.helpers`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import phipml.classification.helpers as helpers


def _binary_data(n_samples: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    index = [f"S{i}" for i in range(n_samples)]
    y = pd.Series([0, 1] * (n_samples // 2), index=index, name="target")
    X = pd.DataFrame(
        {
            "agilent_p1": y.to_numpy(),
            "twist_p2": 1 - y.to_numpy(),
            "Sex": [0, 1] * (n_samples // 2),
            "Age": np.linspace(40, 70, n_samples),
        },
        index=index,
    )
    return X, y


def _simple_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return Pipeline(
        [
            (
                "estimator",
                RandomForestClassifier(n_estimators=5, random_state=1),
            )
        ]
    ).fit(X, y)


def test_calculate_roc_metrics_aggregates_curves_and_auc() -> None:
    grid = np.array([0.0, 0.5, 1.0])
    metrics = helpers.calculate_mean_std_ci_tpr_auc(
        [0.7, 0.9],
        [[0.0, 0.6, 1.0], [0.0, 0.8, 1.0]],
        grid,
    )

    np.testing.assert_allclose(metrics["fpr"], grid)
    np.testing.assert_allclose(metrics["tpr"], [0.0, 0.7, 1.0])
    assert metrics["auc"] == pytest.approx(0.8)
    assert metrics["auc_std"] > 0


def test_calculate_roc_metrics_returns_bootstrap_keys() -> None:
    metrics = helpers.calculate_mean_std_ci_tpr_auc(
        [0.6, 0.7, 0.8, 0.9],
        np.tile([0.0, 0.5, 1.0], (4, 1)),
        [0.0, 0.5, 1.0],
        bootstrap=True,
    )

    assert "boot_auc_mean" in metrics
    assert "boot_auc_std" in metrics
    assert metrics["boot_auc_ci_lower"] <= metrics["boot_auc_ci_upper"]


def test_calculate_roc_metrics_rejects_one_run() -> None:
    with pytest.raises(ValueError, match="At least two"):
        helpers.calculate_mean_std_ci_tpr_auc(
            [0.8],
            [[0.0, 0.5, 1.0]],
            [0.0, 0.5, 1.0],
        )


def test_calculate_pr_metrics_aggregates_curves_and_ap() -> None:
    metrics = helpers.calculate_mean_std_ci_precision_ap(
        [0.6, 0.8],
        [[1.0, 0.7, 0.5], [1.0, 0.9, 0.4]],
        [0.0, 0.5, 1.0],
    )

    np.testing.assert_allclose(metrics["precision"], [1.0, 0.8, 0.45])
    assert metrics["ap"] == pytest.approx(0.7)
    assert metrics["ap_std"] > 0


def test_interpolated_roc_and_pr_have_expected_shapes() -> None:
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    grid = np.linspace(0, 1, 10)

    tpr, roc_auc = helpers.compute_interp_tpr_auc(y, scores, grid)
    precision, average_precision = helpers.compute_interp_pr_ap(y, scores, grid)

    assert tpr.shape == grid.shape
    assert precision.shape == grid.shape
    assert tpr[0] == 0.0
    assert precision[0] == 1.0
    assert roc_auc == pytest.approx(1.0)
    assert average_precision == pytest.approx(1.0)


def test_classification_metrics_use_the_recorded_threshold() -> None:
    metrics = helpers.calculate_classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.7, 0.8, 0.4],
        threshold=0.5,
    )

    assert metrics["threshold"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["negative_predictive_value"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["support_negative"] == 2
    assert metrics["support_positive"] == 2


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_classification_metrics_reject_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between"):
        helpers.calculate_classification_metrics(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            threshold=threshold,
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ([0, 1, 0, np.nan], True),
        ([0.0, 1.0], True),
        ([0, 2], False),
        (["F", "M"], False),
        ([np.nan, np.nan], False),
    ),
)
def test_is_binary_numeric_column(values: list[Any], expected: bool) -> None:
    assert helpers.is_binary_numeric_column(pd.Series(values)) is expected


def test_split_feature_columns_and_imputation_groups() -> None:
    X = pd.DataFrame(
        {
            "agilent_p1": [0, 1],
            "twist_p2": [1, 0],
            "Sex": [0, 1],
            "Age": [40.0, 50.0],
            "Center": ["A", "B"],
        }
    )

    peptides, extras = helpers.split_peptide_extra_columns(X)
    binary, continuous, non_numeric = helpers.split_extra_columns_for_imputation(
        X,
        extras,
    )

    assert peptides == ["agilent_p1", "twist_p2"]
    assert extras == ["Sex", "Age", "Center"]
    assert binary == ["Sex"]
    assert continuous == ["Age"]
    assert non_numeric == ["Center"]


def test_build_pipeline_has_expected_transformers_and_estimator() -> None:
    X, _ = _binary_data()

    pipeline = helpers.build_pipeline(
        X,
        model_type="random-forest",
        impute_extra_numeric=True,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert [name for name, _, _ in preprocessor.transformers] == [
        "peptides",
        "binary_extra",
        "continuous_extra",
    ]
    assert isinstance(
        pipeline.named_steps["estimator"],
        RandomForestClassifier,
    )


def test_build_pipeline_rejects_duplicate_or_non_numeric_features() -> None:
    duplicated = pd.DataFrame([[0, 1]], columns=["agilent_p", "agilent_p"])
    with pytest.raises(ValueError, match="duplicate feature"):
        helpers.build_pipeline(duplicated, model_type="random-forest")

    categorical = pd.DataFrame({"Center": ["A", "B"]})
    with pytest.raises(ValueError, match="non-numeric"):
        helpers.build_pipeline(categorical, model_type="random-forest")


def test_build_and_fit_pipeline_fits_directly_without_search() -> None:
    X = pd.DataFrame({"Age": [20, 30, 40, 50, 60, 70]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    preprocessor = ColumnTransformer(
        [("continuous_extra", "passthrough", ["Age"])],
        verbose_feature_names_out=False,
    )
    candidate = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("estimator", LogisticRegression()),
        ]
    )

    fitted = helpers._build_and_fit_pipeline(
        candidate,
        X,
        y,
        param_grid=None,
        n_splits=2,
        n_iter=2,
        model_type="random-forest",
        random_state=1,
        n_jobs=1,
    )

    assert isinstance(fitted, Pipeline)
    assert hasattr(fitted.named_steps["estimator"], "classes_")
    assert fitted is not candidate


def test_build_and_fit_pipeline_rejects_unknown_grid_parameter() -> None:
    X = pd.DataFrame({"Age": [20, 30, 40, 50]})
    y = pd.Series([0, 0, 1, 1])

    with pytest.raises(ValueError, match="do not match"):
        helpers._build_and_fit_pipeline(
            pipeline=None,
            X_train=X,
            y_train=y,
            param_grid={"not_a_parameter": [1, 2]},
            n_splits=2,
            n_iter=2,
            model_type="random-forest",
            random_state=1,
            n_jobs=1,
        )


def test_positive_class_shap_normalises_supported_shapes() -> None:
    class_zero = np.zeros((2, 3))
    class_one = np.ones((2, 3))

    from_list = helpers._positive_class_shap_values(
        [class_zero, class_one],
        n_samples=2,
        n_features=3,
    )
    from_samples_features_classes = helpers._positive_class_shap_values(
        np.stack([class_zero, class_one], axis=2),
        n_samples=2,
        n_features=3,
    )
    from_classes_samples_features = helpers._positive_class_shap_values(
        np.stack([class_zero, class_one], axis=0),
        n_samples=2,
        n_features=3,
    )

    np.testing.assert_array_equal(from_list, class_one)
    np.testing.assert_array_equal(from_samples_features_classes, class_one)
    np.testing.assert_array_equal(from_classes_samples_features, class_one)


def test_positive_class_shap_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected SHAP value shape"):
        helpers._positive_class_shap_values(
            np.zeros((2, 4)),
            n_samples=2,
            n_features=3,
        )


def test_align_external_zero_fills_only_missing_peptides() -> None:
    X, y = _binary_data()
    pipeline = _simple_pipeline(X, y)
    external = pd.DataFrame(
        {
            "agilent_p1": [1, 0],
            "Sex": [1, 0],
            "Age": [55.0, 60.0],
            "unused": [9, 9],
        },
        index=["E1", "E2"],
    )

    aligned, report = helpers.align_external_to_pipeline(external, pipeline)

    assert aligned.columns.tolist() == X.columns.tolist()
    assert aligned["twist_p2"].eq(0).all()
    assert report["missing_peptides"] == ["twist_p2"]
    assert report["extra_features"] == ["unused"]


def test_align_external_rejects_missing_clinical_feature() -> None:
    X, y = _binary_data()
    pipeline = _simple_pipeline(X, y)
    external = X.iloc[:2].drop(columns="Age")

    with pytest.raises(ValueError, match="non-peptide clinical"):
        helpers.align_external_to_pipeline(external, pipeline)


def test_train_and_validate_model_returns_metrics_shap_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X_train, y_train = _binary_data()
    fitted = _simple_pipeline(X_train, y_train)
    X_test = X_train.iloc[:4].copy()
    X_test.index = [f"E{i}" for i in range(4)]
    y_test = pd.Series([0, 1, 0, 1], index=X_test.index)

    def fake_shap(
        pipeline: Pipeline,
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        del pipeline
        selected = ["agilent_p1", "Age"]
        return pd.DataFrame(0.5, index=X.index, columns=selected), selected

    monkeypatch.setattr(helpers, "_compute_shap_frame", fake_shap)

    result = helpers.train_and_validate_model(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        best_estimator=fitted,
        return_feature_report=True,
    )

    assert isinstance(result, tuple)
    model, shap_values, scores, metrics, selected, report = result
    assert model is fitted
    assert shap_values.shape == X_test.shape
    assert scores.index.tolist() == X_test.index.tolist()
    assert set(metrics) == {"roc", "pr", "classification"}
    assert 0.0 <= metrics["roc"]["auc"] <= 1.0
    assert 0.0 <= metrics["pr"]["ap"] <= 1.0
    assert metrics["classification"]["threshold"] == pytest.approx(0.5)
    assert 0.0 <= metrics["classification"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["classification"]["f1"] <= 1.0
    assert selected == ["agilent_p1", "Age"]
    assert report["missing_features"] == []


def test_train_and_validate_model_requires_test_data() -> None:
    X, y = _binary_data()
    fitted = _simple_pipeline(X, y)

    with pytest.raises(ValueError, match="X_test and y_test"):
        helpers.train_and_validate_model(
            X,
            y,
            best_estimator=fitted,
        )