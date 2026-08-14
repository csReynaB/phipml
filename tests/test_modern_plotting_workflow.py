"""Tests for normalized result plots, repeated runs, heatmaps, and CLIs."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phipml.cli.metric_heatmap import main as heatmap_main  # noqa: E402
from phipml.cli.plot_results import main as plot_main  # noqa: E402
from phipml.io.data_handler import Config  # noqa: E402
from phipml.plots.metric_heatmap import (  # noqa: E402
    build_metric_matrix,
    plot_metric_heatmap,
)
from phipml.plots.helpers import plot_classification_metric_bars  # noqa: E402
from phipml.plots.result_summary import (  # noqa: E402
    aggregate_result_metrics,
    aggregate_shap_summary,
    load_classification_result,
    plot_result_files,
    rank_features_by_top_k_shap,
    selected_feature_frequency,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Keep the plotting tests isolated and avoid accumulating open figures."""
    yield
    plt.close("all")


def _classification(value: float = 0.75) -> dict[str, object]:
    metrics: dict[str, object] = {
        "threshold": 0.5,
        "accuracy": value,
        "balanced_accuracy": value,
        "precision": value,
        "recall": value,
        "sensitivity": value,
        "specificity": value,
        "negative_predictive_value": value,
        "f1": value,
        "mcc": value - 0.2,
        "true_negatives": 8,
        "false_positives": 2,
        "false_negatives": 3,
        "true_positives": 7,
        "support_negative": 10,
        "support_positive": 10,
    }
    return metrics


def test_legacy_plot_modules_reexport_the_modern_generic_apis() -> None:
    """Old notebook imports remain valid without duplicating modern behavior."""
    from phipml.plots.auc_heatmap import (  # noqa: PLC0415
        build_metric_matrix as legacy_build_metric_matrix,
    )
    from phipml.plots.auc_shap_summary import (  # noqa: PLC0415
        plot_result_files as legacy_plot_result_files,
    )

    assert legacy_build_metric_matrix is build_metric_matrix
    assert legacy_plot_result_files is plot_result_files


def _metrics(
    *,
    auc: float = 0.80,
    ap: float = 0.78,
    external_ci: bool = False,
    nested_variability: bool = False,
) -> dict[str, dict[str, object]]:
    grid = np.linspace(0.0, 1.0, 5)
    classification = _classification()
    roc: dict[str, object] = {
        "fpr": grid,
        "tpr": np.array([0.0, 0.45, 0.72, 0.92, 1.0]),
        "auc": auc,
    }
    pr: dict[str, object] = {
        "recall": grid,
        "precision": np.array([1.0, 0.90, 0.78, 0.68, 0.50]),
        "ap": ap,
        "positive_prevalence": 0.5,
    }
    if external_ci:
        roc.update(
            {
                "tprs_lower": np.array([0.0, 0.30, 0.60, 0.80, 1.0]),
                "tprs_upper": np.array([0.0, 0.60, 0.85, 1.0, 1.0]),
                "auc_ci_lower": auc - 0.12,
                "auc_ci_upper": auc + 0.10,
                "uncertainty_label": "95% stratified bootstrap interval",
            }
        )
        pr.update(
            {
                "precision_lower": np.array([1.0, 0.75, 0.62, 0.55, 0.50]),
                "precision_upper": np.array([1.0, 1.0, 0.90, 0.82, 0.50]),
                "ap_ci_lower": ap - 0.11,
                "ap_ci_upper": ap + 0.09,
                "uncertainty_label": "95% stratified bootstrap interval",
            }
        )
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "negative_predictive_value",
            "f1",
            "mcc",
        ):
            classification[f"{key}_ci_lower"] = float(classification[key]) - 0.1
            classification[f"{key}_ci_upper"] = float(classification[key]) + 0.1
        classification["uncertainty_label"] = "95% stratified bootstrap interval"
    if nested_variability:
        roc.update({"auc_std": 0.05, "tprs_lower": grid * 0.75, "tprs_upper": grid})
        pr.update(
            {
                "ap_std": 0.07,
                "pr_lower": np.clip(np.asarray(pr["precision"]) - 0.1, 0, 1),
                "pr_upper": np.clip(np.asarray(pr["precision"]) + 0.1, 0, 1),
            }
        )
        fold_mean = {
            key: float(classification[key])
            for key in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "sensitivity",
                "specificity",
                "negative_predictive_value",
                "f1",
                "mcc",
            )
        }
        classification["fold_mean"] = fold_mean
        classification["fold_std"] = {key: 0.06 for key in fold_mean}
    return {"roc": roc, "pr": pr, "classification": classification}


def _write_result(
    path: Path,
    *,
    split: str,
    metrics: dict[str, dict[str, object]],
    shap_columns: tuple[str, ...] = ("agilent_a", "Age"),
    shift: float = 0.0,
) -> None:
    samples = [f"S{i}" for i in range(6)]
    shap_values = pd.DataFrame(
        np.arange(len(samples) * len(shap_columns), dtype=float).reshape(
            len(samples), len(shap_columns)
        )
        / 20.0
        + shift,
        index=samples,
        columns=shap_columns,
    )
    joblib.dump(
        {
            f"metrics_{split}": metrics,
            f"scores_{split}": pd.Series(np.linspace(0.1, 0.9, 6), index=samples),
            (
                "train_shap_values" if split == "train" else "test_shap_values"
            ): shap_values,
            f"selected_features_{split}": list(shap_columns),
        },
        path,
    )


def test_single_validation_keeps_saved_bootstrap_intervals(tmp_path: Path) -> None:
    path = tmp_path / "validation.joblib"
    _write_result(path, split="test", metrics=_metrics(external_ci=True))
    result = load_classification_result(path)

    assert result.metrics["roc"]["auc_ci_lower"] == pytest.approx(0.68)
    assert result.metrics["pr"]["ap_ci_upper"] == pytest.approx(0.87)
    assert result.metrics["classification"]["specificity_ci_low"] == pytest.approx(0.65)


def test_missing_validation_intervals_can_be_reconstructed_for_plotting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_without_intervals.joblib"
    _write_result(path, split="test", metrics=_metrics())
    target = pd.Series([0, 0, 0, 1, 1, 1], index=[f"S{i}" for i in range(6)])

    output = plot_result_files(
        [path],
        split="test",
        target=target,
        validation_bootstraps=20,
        output_dir=tmp_path / "plots",
        plots=["performance"],
        save_standalone=False,
    )

    assert len(output.metrics["roc"]["fpr"]) == 200
    assert len(output.metrics["roc"]["tprs_lower"]) == 200
    assert len(output.metrics["pr"]["recall"]) == 200
    assert "auc_ci_lower" in output.metrics["roc"]
    assert "ap_ci_lower" in output.metrics["pr"]


def test_nested_classification_plot_uses_all_metrics_and_fold_sd() -> None:
    metrics = _metrics(nested_variability=True)["classification"]
    figure, axis = plot_classification_metric_bars(metrics)

    assert len(axis.patches) == 8
    assert "outer-fold mean" in axis.get_title()
    assert figure is axis.figure


def test_repeated_metrics_have_empirical_intervals_not_formal_ci(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"nested_{index}.joblib" for index in range(3)]
    for index, path in enumerate(paths):
        _write_result(
            path,
            split="train",
            metrics=_metrics(auc=0.76 + index * 0.04, ap=0.74 + index * 0.04),
            shift=index * 0.02,
        )
    results = [load_classification_result(path) for path in paths]
    summary = aggregate_result_metrics(results)

    assert summary["roc"]["auc"] == pytest.approx(0.80)
    assert summary["roc"]["n_runs"] == 3
    assert summary["roc"]["formal_confidence_interval"] is False
    assert "empirical interval" in summary["roc"]["uncertainty_label"]
    assert len(summary["classification"]["specificity_runs"]) == 3


def test_shap_intersection_and_stability_summary(tmp_path: Path) -> None:
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    _write_result(first, split="train", metrics=_metrics())
    _write_result(
        second,
        split="train",
        metrics=_metrics(),
        shap_columns=("agilent_a", "BMI"),
        shift=0.1,
    )
    results = [load_classification_result(first), load_classification_result(second)]

    with pytest.raises(ValueError, match="same samples and features"):
        aggregate_shap_summary(results)
    summary = aggregate_shap_summary(results, alignment="intersection")
    assert summary is not None
    assert summary.mean_values.columns.tolist() == ["agilent_a"]
    assert summary.n_runs == 2
    assert summary.nonzero_run_frequency.loc["agilent_a"] == pytest.approx(100.0)


def test_shap_importance_averages_absolute_values_before_runs_cancel(
    tmp_path: Path,
) -> None:
    first = tmp_path / "positive.joblib"
    second = tmp_path / "negative.joblib"
    _write_result(first, split="train", metrics=_metrics())
    _write_result(second, split="train", metrics=_metrics())
    loaded = joblib.load(second)
    loaded["train_shap_values"] = -loaded["train_shap_values"]
    joblib.dump(loaded, second)

    summary = aggregate_shap_summary(
        [load_classification_result(first), load_classification_result(second)]
    )
    assert summary is not None
    assert summary.mean_values.to_numpy().sum() == pytest.approx(0.0)
    assert summary.mean_absolute.loc["agilent_a"] > 0.0


def test_top_k_frequency_ranking_prefers_consistency_over_one_large_run(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"run_{number}.joblib" for number in range(3)]
    run_importances = [
        {"stable": 5.0, "spike": 100.0, "noise": 1.0},
        {"stable": 5.0, "spike": 0.0, "noise": 6.0},
        {"stable": 5.0, "spike": 0.0, "noise": 1.0},
    ]
    for path, importances in zip(paths, run_importances):
        _write_result(
            path,
            split="train",
            metrics=_metrics(),
            shap_columns=("stable", "spike", "noise"),
        )
        saved = joblib.load(path)
        saved["train_shap_values"] = pd.DataFrame(
            {
                feature: np.full(6, value, dtype=float)
                for feature, value in importances.items()
            },
            index=[f"S{number}" for number in range(6)],
        )
        joblib.dump(saved, path)

    results = [load_classification_result(path) for path in paths]
    ranking = rank_features_by_top_k_shap(results, top_k=2).set_index("Feature")

    assert ranking.loc["stable", "Top-k SHAP frequency (%)"] == pytest.approx(100)
    assert ranking.loc["spike", "Top-k SHAP frequency (%)"] == pytest.approx(
        100 / 3
    )
    assert ranking.loc["spike", "Ranking mean |SHAP|"] > ranking.loc[
        "stable", "Ranking mean |SHAP|"
    ]
    assert ranking.loc["stable", "Top-k frequency rank"] == 1

    output = plot_result_files(
        paths,
        reconstruct_data=False,
        save_standalone=False,
        plots=["shap-importance"],
        feature_ranking="auto",
        ranking_top_k=2,
        min_top_k_frequency=50,
        max_display=2,
    )
    assert output.display_features == ["stable", "noise"]
    assert output.feature_ranking is not None


def test_nested_selected_features_are_summarized_across_outer_folds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.joblib"
    _write_result(path, split="train", metrics=_metrics())
    saved = joblib.load(path)
    saved["selected_features_train"] = [
        ["agilent_a", "Age"],
        ["agilent_a"],
        ["agilent_a", "BMI"],
    ]
    joblib.dump(saved, path)

    frequency = selected_feature_frequency([load_classification_result(path)])

    assert frequency is not None
    assert frequency.loc["agilent_a"] == pytest.approx(100.0)
    assert frequency.loc["Age"] == pytest.approx(100.0 / 3.0)
    assert frequency.loc["BMI"] == pytest.approx(100.0 / 3.0)


def test_embedded_config_reconstructs_features_target_and_library_annotations(
    tmp_path: Path,
) -> None:
    samples = [f"S{i}" for i in range(6)]
    metadata = pd.DataFrame(
        {
            "SampleName": samples,
            "group_test": ["Control"] * 3 + ["Case"] * 3,
            "Age": [40, 45, 50, 55, 60, 65],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    pd.DataFrame(
        [[0, 1, 0, 1, 1, 0]],
        index=["agilent_a"],
        columns=samples,
    ).to_csv(tmp_path / "data.csv")
    pd.DataFrame(
        {
            "Feature": ["agilent_a"],
            "Description": ["Demonstration peptide"],
            "Species": ["Example species"],
        }
    ).to_csv(tmp_path / "library.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata_input": "metadata.csv",
                "data_input": "data.csv",
                "lib_metadata_input": "library.csv",
                "group_tests": ["Control", "Case"],
                "extra_features_to_include": ["Age"],
                "transposed": True,
            }
        ),
        encoding="utf-8",
    )

    result_path = tmp_path / "validation.joblib"
    _write_result(result_path, split="test", metrics=_metrics())
    saved = joblib.load(result_path)
    config = Config(config_path)
    saved["target_test"] = pd.Series([0, 0, 0, 1, 1, 1], index=samples)
    saved["selected_features_test"] = ["agilent_a", "Age"]
    saved["data_context"] = {
        "schema_version": 1,
        "artifact_type": "validation",
        "config_file": str(config.config_file),
        "resolved_config": config.to_mapping(),
    }
    joblib.dump(saved, result_path)

    output = plot_result_files(
        [result_path],
        split="test",
        class_labels=("Control", "Case"),
        output_dir=tmp_path / "plots",
        save_standalone=False,
        table_annotation_columns=("Description", "Species"),
    )

    assert output.feature_table is not None
    peptide = output.feature_table.set_index("Feature").loc["agilent_a"]
    assert peptide["Description"] == "Demonstration peptide"
    assert peptide["Species"] == "Example species"
    assert peptide["Selection frequency (%)"] == pytest.approx(100.0)
    assert (tmp_path / "plots" / "phipml_feature_importance.csv").is_file()


def test_metric_matrix_supports_repeats_and_native_validation_ci(
    tmp_path: Path,
) -> None:
    nested_1 = tmp_path / "nested_1.joblib"
    nested_2 = tmp_path / "nested_2.joblib"
    validation = tmp_path / "validation.joblib"
    _write_result(nested_1, split="train", metrics=_metrics(auc=0.74))
    _write_result(nested_2, split="train", metrics=_metrics(auc=0.82))
    _write_result(
        validation,
        split="test",
        metrics=_metrics(auc=0.77, external_ci=True),
    )
    records = pd.DataFrame(
        {
            "training": ["A", "A", "A"],
            "validation": ["A", "A", "B"],
            "path": [nested_1, nested_2, validation],
            "split": ["train", "train", "test"],
        }
    )
    summary = build_metric_matrix(records, metric="roc.auc", order=["A", "B"])

    assert summary.mean.loc["A", "A"] == pytest.approx(0.78)
    assert summary.standard_deviation.loc["A", "A"] > 0
    assert summary.mean.loc["B", "A"] == pytest.approx(0.77)
    assert summary.lower.loc["B", "A"] == pytest.approx(0.65)
    output = tmp_path / "heatmap.pdf"
    figure, _ = plot_metric_heatmap(summary, output_path=output)
    assert output.is_file() and output.stat().st_size > 0
    assert figure.axes


def test_plot_and_heatmap_cli_create_outputs(tmp_path: Path) -> None:
    result = tmp_path / "nested.joblib"
    _write_result(result, split="train", metrics=_metrics(nested_variability=True))
    output_dir = tmp_path / "plots"
    assert (
        plot_main(
            [
                str(result),
                "--split",
                "train",
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                "demo",
                "--feature-ranking",
                "top-k-frequency",
                "--ranking-top-k",
                "1",
                "--min-top-k-frequency",
                "50",
                "--max-display",
                "1",
                "--plots",
                "performance",
                "roc",
                "pr",
                "confusion",
                "classification",
                "shap-importance",
                "shap-heatmap",
            ]
        )
        == 0
    )
    assert (output_dir / "demo_performance.pdf").is_file()
    assert (output_dir / "demo_performance.svg").is_file()
    assert (output_dir / "demo_performance.png").is_file()
    assert (output_dir / "demo_roc.pdf").is_file()

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "training": ["A"],
            "validation": ["A"],
            "path": [result.name],
            "split": ["train"],
        }
    ).to_csv(manifest, index=False)
    heatmap = tmp_path / "metric_heatmap.pdf"
    assert (
        heatmap_main(
            [
                "--manifest",
                str(manifest),
                "--metric",
                "pr.ap",
                "--output",
                str(heatmap),
            ]
        )
        == 0
    )
    assert heatmap.is_file() and heatmap.stat().st_size > 0


def test_plot_selection_formats_and_colors_are_independent(tmp_path: Path) -> None:
    result = tmp_path / "nested.joblib"
    _write_result(result, split="train", metrics=_metrics())
    output_dir = tmp_path / "selected"

    output = plot_result_files(
        [result],
        split="train",
        plots=["roc"],
        output_formats=["svg", "png"],
        output_dir=output_dir,
        output_prefix="selected",
        plot_colors={"roc": "#123456", "roc_band": "#ABCDEF"},
        reconstruct_data=False,
    )

    assert output.performance_figure is None
    assert set(output.standalone_figures) == {"roc"}
    assert output.standalone_figures["roc"].axes[0].lines[0].get_color() == "#123456"
    assert (output_dir / "selected_roc.svg").is_file()
    assert (output_dir / "selected_roc.png").is_file()
    assert not (output_dir / "selected_roc.pdf").exists()
    assert not (output_dir / "selected_performance.svg").exists()


def test_plot_yaml_and_cli_override_render_curated_feature_table(
    tmp_path: Path,
) -> None:
    result = tmp_path / "nested.joblib"
    _write_result(result, split="train", metrics=_metrics())
    curated = pd.DataFrame(
        {
            "Feature": ["agilent_a", "Age"],
            "Short description": ["Short viral name", "Age"],
            "Feature type": ["peptide", "continuous clinical"],
            "Statistic": ["Prevalence (%)", "Mean"],
            "Control": [20.0, 48.0],
            "Case": [75.0, 61.0],
            "Mean |SHAP|": [0.42, 0.18],
        }
    )
    curated.to_csv(tmp_path / "curated.csv", index=False)
    plot_config = tmp_path / "plotting.yaml"
    plot_config.write_text(
        yaml.safe_dump(
            {
                "plotting": {
                    "results": [result.name],
                    "split": "train",
                    "plots": ["feature-table"],
                    "formats": ["svg", "png"],
                    "output_dir": "plots",
                    "output_prefix": "curated",
                    "class_labels": ["Control", "Case"],
                    "feature_table": {
                        "input": "curated.csv",
                        "annotation_columns": ["Short description"],
                        "title": "Curated top features",
                        "header_color": "#CCCCCC",
                        "prevalence_cmap": "YlOrBr",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    # Explicit CLI formats override the SVG/PNG formats in plotting.yaml.
    assert plot_main(["--plot-config", str(plot_config), "--formats", "pdf"]) == 0

    output_dir = tmp_path / "plots"
    assert (output_dir / "curated_feature_table.pdf").is_file()
    assert not (output_dir / "curated_feature_table.svg").exists()
    assert not (output_dir / "curated_feature_table.png").exists()
    saved = pd.read_csv(output_dir / "curated_feature_importance.csv")
    assert saved["Short description"].tolist()[0] == "Short viral name"


def test_repeated_feature_table_contains_shap_stability_columns(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.joblib", tmp_path / "second.joblib"
    _write_result(first, split="train", metrics=_metrics())
    _write_result(second, split="train", metrics=_metrics(), shift=0.05)
    samples = [f"S{i}" for i in range(6)]
    features = pd.DataFrame(
        {
            "agilent_a": [0, 1, 0, 1, 1, 0],
            "Age": [40, 45, 50, 55, 60, 65],
        },
        index=samples,
    )
    target = pd.Series([0, 0, 0, 1, 1, 1], index=samples)
    output = plot_result_files(
        [first, second],
        features=features,
        target=target,
        class_labels=("Control", "Case"),
        output_dir=tmp_path / "plots",
        save_standalone=False,
    )
    assert output.feature_table is not None
    assert "Run SD |SHAP|" in output.feature_table.columns
    assert "Non-zero SHAP runs (%)" in output.feature_table.columns
    assert "Top-k SHAP frequency (%)" in output.feature_table.columns
    assert "Mean rank when in top K" in output.feature_table.columns
    assert "Feature ranking method" in output.feature_table.columns
