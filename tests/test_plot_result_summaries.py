"""Behavioural tests for performance and SHAP result plotting."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from phipml.plots.auc_shap_summary import (  # noqa: E402
    aggregate_result_metrics,
    aggregate_shap_values,
    bootstrap_validation_metrics,
    load_classification_result,
    plot_result_files,
)
from phipml.plots.helpers import (  # noqa: E402
    build_feature_importance_table,
    plot_performance_summary,
)


def _metrics(*, shift: float = 0.0) -> dict[str, dict[str, object]]:
    grid = np.linspace(0.0, 1.0, 5)
    tpr = np.clip(np.array([0.0, 0.45, 0.75, 0.92, 1.0]) + shift, 0.0, 1.0)
    precision = np.clip(
        np.array([1.0, 0.92, 0.84, 0.72, 0.5]) + shift,
        0.0,
        1.0,
    )
    return {
        "roc": {
            "fpr": grid,
            "tpr": tpr,
            "tprs_lower": np.clip(tpr - 0.08, 0.0, 1.0),
            "tprs_upper": np.clip(tpr + 0.08, 0.0, 1.0),
            "auc": 0.85 + shift,
            "auc_std": 0.04,
        },
        "pr": {
            "recall": grid,
            "pr": precision,
            "pr_lower": np.clip(precision - 0.08, 0.0, 1.0),
            "pr_upper": np.clip(precision + 0.08, 0.0, 1.0),
            "ap": 0.86 + shift,
            "ap_std": 0.03,
        },
        "classification": {
            "threshold": 0.5,
            "accuracy": 0.825 + shift,
            "balanced_accuracy": 0.825 + shift,
            "precision": 0.842 + shift,
            "recall": 0.800 + shift,
            "sensitivity": 0.800 + shift,
            "specificity": 0.850 + shift,
            "negative_predictive_value": 0.810 + shift,
            "f1": 0.821 + shift,
            "mcc": 0.651 + shift,
            "true_negatives": 17,
            "false_positives": 3,
            "false_negatives": 4,
            "true_positives": 16,
            "support_negative": 20,
            "support_positive": 20,
        },
    }


def _features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    index = [f"S{number}" for number in range(8)]
    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=index, name="target")
    features = pd.DataFrame(
        {
            "agilent_signal": [0, 0, 1, 0, 1, 1, 1, 0],
            "Sex": [0, 1, 0, 1, 0, 1, 0, 1],
            "Age": [40.0, 44.0, 47.0, 50.0, 55.0, 60.0, 63.0, 67.0],
        },
        index=index,
    )
    shap_values = pd.DataFrame(
        {
            "agilent_signal": np.linspace(-0.5, 0.5, 8),
            "Sex": np.linspace(0.2, -0.2, 8),
            "Age": np.linspace(-0.1, 0.3, 8),
        },
        index=index,
    )
    return features, target, shap_values


def _write_result(
    path: Path,
    *,
    shift: float = 0.0,
    split: str = "train",
) -> None:
    _, _, shap_values = _features()
    joblib.dump(
        {
            f"metrics_{split}": _metrics(shift=shift),
            f"scores_{split}": pd.Series(
                np.linspace(0.1, 0.9, len(shap_values)),
                index=shap_values.index,
            ),
            (
                "train_shap_values" if split == "train" else "test_shap_values"
            ): shap_values
            + shift,
            f"selected_features_{split}": shap_values.columns.tolist(),
        },
        path,
    )


def test_performance_summary_renders_all_four_panels(tmp_path: Path) -> None:
    output = tmp_path / "performance.pdf"
    figure, axes = plot_performance_summary(
        _metrics(),
        class_labels=("Control", "Case"),
        output_path=output,
    )

    assert axes.shape == (2, 2)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert len(figure.axes) == 4


def test_feature_table_distinguishes_prevalence_from_continuous_means() -> None:
    features, target, shap_values = _features()
    table = build_feature_importance_table(
        shap_values,
        features,
        target,
        group_labels=("Control", "Case"),
    ).set_index("Feature")

    assert table.loc["agilent_signal", "Feature type"] == "peptide"
    assert table.loc["agilent_signal", "Statistic"] == "Prevalence (%)"
    assert table.loc["Sex", "Feature type"] == "binary clinical"
    assert table.loc["Sex", "Statistic"] == "Prevalence (%)"
    assert table.loc["Age", "Feature type"] == "continuous clinical"
    assert table.loc["Age", "Statistic"] == "Mean"
    assert table.loc["Age", "Control"] == pytest.approx(45.25)
    assert table.loc["Age", "Case"] == pytest.approx(61.25)


def test_repeated_results_aggregate_curves_metrics_and_shap(tmp_path: Path) -> None:
    first_path = tmp_path / "nested_rf_demo_1.joblib"
    second_path = tmp_path / "nested_rf_demo_2.joblib"
    _write_result(first_path, shift=0.0)
    _write_result(second_path, shift=0.02)

    results = [
        load_classification_result(first_path),
        load_classification_result(second_path),
    ]
    metrics = aggregate_result_metrics(results)
    shap_values = aggregate_shap_values(results)

    assert metrics["roc"]["auc"] == pytest.approx(0.86)
    assert metrics["pr"]["ap"] == pytest.approx(0.87)
    assert metrics["classification"]["accuracy"] == pytest.approx(0.835)
    assert metrics["roc"]["n_runs"] == 2
    assert "empirical interval" in metrics["roc"]["uncertainty_label"]
    assert shap_values is not None
    _, _, expected = _features()
    pd.testing.assert_frame_equal(shap_values, expected + 0.01)


def test_plot_result_files_creates_performance_and_shap_outputs(tmp_path: Path) -> None:
    first_path = tmp_path / "nested_rf_demo_1.joblib"
    second_path = tmp_path / "nested_rf_demo_2.joblib"
    _write_result(first_path, shift=0.0)
    _write_result(second_path, shift=0.02)
    features, target, _ = _features()
    output_dir = tmp_path / "plots"

    output = plot_result_files(
        [first_path, second_path],
        class_labels=("Control", "Case"),
        title="Repeated nested CV",
        output_dir=output_dir,
        output_prefix="demo",
        features=features,
        target=target,
        max_display=3,
    )

    expected_files = {
        "demo_performance.pdf",
        "demo_shap_importance.pdf",
        "demo_shap_heatmap.pdf",
        "demo_shap_beeswarm.pdf",
        "demo_feature_importance.csv",
        "demo_feature_table.pdf",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert output.shap_values is not None
    assert output.shap_beeswarm_figure is not None
    assert output.shap_heatmap_figure is not None
    assert output.feature_table is not None
    assert set(output.metrics) == {"roc", "pr", "classification"}


def test_result_loader_rejects_a_training_model_without_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_rf_demo.joblib"
    joblib.dump({"best_estimator": object()}, path)

    with pytest.raises(KeyError, match="neither metrics_train nor metrics_test"):
        load_classification_result(path)


def test_single_validation_can_add_stratified_bootstrap_intervals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_rf_demo.joblib"
    _write_result(path, split="test")
    result = load_classification_result(path)
    _, target, _ = _features()

    metrics = bootstrap_validation_metrics(
        result,
        target,
        class_labels=("Control", "Case"),
        n_bootstraps=25,
        random_state=7,
    )

    assert "stratified bootstrap" in metrics["roc"]["uncertainty_label"]
    assert metrics["roc"]["auc_ci_low"] <= metrics["roc"]["auc_ci_high"]
    assert metrics["pr"]["ap_ci_low"] <= metrics["pr"]["ap_ci_high"]
    assert (
        metrics["classification"]["accuracy_ci_low"]
        <= metrics["classification"]["accuracy_ci_high"]
    )
