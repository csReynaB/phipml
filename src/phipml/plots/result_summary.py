"""Load, aggregate, and plot current phipml result artifacts.

The functions in this module distinguish three different uncertainty sources:

* one nested-CV run: variability across outer folds;
* one external validation: bootstrap uncertainty saved during validation;
* repeated runs: empirical variability across run-level point estimates.

No uncertainty interval is silently relabelled as another type.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phipml.classification.helpers import bootstrap_classification_metrics
from phipml.io.data_handler import Config, MetadataHandler, OligosHandler
from phipml.plots.helpers import (
    PHIPML_PREVALENCE_CMAP_NAME,
    PHIPML_SHAP_CMAP_NAME,
    PHIPML_SHAP_HEATMAP_CMAP_NAME,
    build_feature_importance_table,
    plot_classification_metric_bars,
    plot_confusion_matrix_metrics,
    plot_feature_importance_table,
    plot_performance_summary,
    plot_precision_recall_metrics,
    plot_roc_metrics,
    plot_shap_heatmap,
    plot_shap_importance_bar,
    plot_shap_values,
)

ResultSplit = Literal["auto", "train", "test"]
ShapAlignment = Literal["strict", "intersection"]
FeatureRanking = Literal["auto", "top-k-frequency", "mean-abs-shap"]

PLOT_NAMES = (
    "performance",
    "roc",
    "pr",
    "confusion",
    "classification",
    "shap-beeswarm",
    "shap-importance",
    "shap-heatmap",
    "feature-table",
)
PLOT_ALIASES = {
    "ap": "pr",
    "classification-metrics": "classification",
    "shap-bar": "shap-importance",
    "table": "feature-table",
}
DEFAULT_OUTPUT_FORMATS = ("pdf", "svg", "png")
DEFAULT_PLOT_COLORS = {
    "roc": "#2A6F97",
    "roc_band": "#89C2D9",
    "pr": "#8A5A44",
    "pr_band": "#DDBEA9",
    "confusion_cmap": "Blues",
    "classification": "#52796F",
    "shap_cmap": PHIPML_SHAP_CMAP_NAME,
    "shap_heatmap_cmap": PHIPML_SHAP_HEATMAP_CMAP_NAME,
    "shap_importance": "#6D597A",
    "negative_class": "black",
    "positive_class": "black",
    "table_header": "#D9D9D9",
    "table_row_odd": "#F6F6F6",
    "table_row_even": "white",
    "table_prevalence_cmap": PHIPML_PREVALENCE_CMAP_NAME,
}

logger = logging.getLogger(__name__)

CLASSIFICATION_RATE_KEYS = (
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
CONFUSION_COUNT_KEYS = (
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "support_negative",
    "support_positive",
)


@dataclass(frozen=True)
class ClassificationResult:
    """Validated contents extracted from one nested-CV or validation file."""

    path: Path
    split: Literal["train", "test"]
    metrics: dict[str, dict[str, Any]]
    scores: pd.Series | None
    shap_values: pd.DataFrame | None
    selected_features: list[str] | list[list[str]] | None
    target: pd.Series | None = None
    data_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class ShapAggregate:
    """Repeated-run SHAP summary and feature stability information."""

    mean_values: pd.DataFrame
    mean_absolute: pd.Series
    run_standard_deviation: pd.Series
    nonzero_run_frequency: pd.Series
    n_runs: int


@dataclass(frozen=True)
class ResultPlotOutput:
    """Data and figures created from one or several result artifacts."""

    metrics: dict[str, dict[str, Any]]
    performance_figure: plt.Figure | None
    shap_values: pd.DataFrame | None
    shap_beeswarm_figure: plt.Figure | None
    shap_importance_figure: plt.Figure | None
    shap_heatmap_figure: plt.Figure | None
    feature_table: pd.DataFrame | None
    feature_table_figure: plt.Figure | None
    standalone_figures: dict[str, plt.Figure] = field(default_factory=dict)
    shap_summary: ShapAggregate | None = None
    feature_ranking: pd.DataFrame | None = None
    display_features: list[str] = field(default_factory=list)


def _mapping(value: object, *, name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping in {path}")
    return value


def _choose_result_split(
    result: Mapping[str, Any],
    split: ResultSplit,
    path: Path,
) -> Literal["train", "test"]:
    available = {name for name in ("train", "test") if f"metrics_{name}" in result}
    if split != "auto":
        if split not in available:
            raise KeyError(
                f"{path} has no metrics_{split}; available splits: {sorted(available)}"
            )
        return split
    if len(available) == 1:
        return "train" if "train" in available else "test"
    if not available:
        raise KeyError(f"{path} contains neither metrics_train nor metrics_test")
    raise ValueError(
        f"{path} contains both train and test metrics; select split='train' or 'test'"
    )


def _normalise_metrics(
    raw: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, dict[str, Any]]:
    missing = sorted({"roc", "pr", "classification"} - set(raw))
    if missing:
        raise KeyError(f"{path} is missing metric sections: {missing}")
    metrics = {
        section: dict(_mapping(raw[section], name=section, path=path))
        for section in ("roc", "pr", "classification")
    }

    roc = metrics["roc"]
    aliases = {
        "boot_mean_fpr": "fpr",
        "mean_fpr": "fpr",
        "boot_mean_tpr": "tpr",
        "bootstrap_mean_tpr": "tpr",
        "mean_tpr": "tpr",
        "boot_auc_mean": "auc",
        "auc_mean": "auc",
        "boot_auc_std": "auc_std",
        "boot_auc_ci_lower": "auc_ci_lower",
        "boot_auc_ci_upper": "auc_ci_upper",
    }
    for old, canonical in aliases.items():
        if canonical not in roc and old in roc:
            roc[canonical] = roc[old]
    if "auc_ci_lower" in roc and "auc_ci_low" not in roc:
        roc["auc_ci_low"] = roc["auc_ci_lower"]
    if "auc_ci_upper" in roc and "auc_ci_high" not in roc:
        roc["auc_ci_high"] = roc["auc_ci_upper"]

    pr = metrics["pr"]
    pr_aliases = {
        "mean_recall": "recall",
        "pr": "precision",
        "mean_precision": "precision",
        "bootstrap_mean_precision": "precision",
        "ap_mean": "ap",
    }
    for old, canonical in pr_aliases.items():
        if canonical not in pr and old in pr:
            pr[canonical] = pr[old]
    if "ap_ci_lower" in pr and "ap_ci_low" not in pr:
        pr["ap_ci_low"] = pr["ap_ci_lower"]
    if "ap_ci_upper" in pr and "ap_ci_high" not in pr:
        pr["ap_ci_high"] = pr["ap_ci_upper"]

    classification = metrics["classification"]
    for metric in CLASSIFICATION_RATE_KEYS:
        for suffix in ("lower", "upper"):
            canonical = f"{metric}_ci_{suffix}"
            short = f"{metric}_ci_{'low' if suffix == 'lower' else 'high'}"
            if canonical not in classification and short in classification:
                classification[canonical] = classification[short]
            if short not in classification and canonical in classification:
                classification[short] = classification[canonical]
    uncertainty = raw.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        metrics["uncertainty"] = dict(uncertainty)
    return metrics


def load_classification_result(
    path: str | Path,
    *,
    split: ResultSplit = "auto",
) -> ClassificationResult:
    """Load one result while normalizing compatible metric-key variants."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Result file does not exist: {source}")
    loaded = joblib.load(source)
    loaded_mapping = _mapping(loaded, name="result", path=source)
    selected_split = _choose_result_split(loaded_mapping, split, source)
    raw_metrics = _mapping(
        loaded_mapping[f"metrics_{selected_split}"],
        name=f"metrics_{selected_split}",
        path=source,
    )
    metrics = _normalise_metrics(raw_metrics, path=source)

    score_key = f"scores_{selected_split}"
    scores = loaded_mapping.get(score_key)
    if scores is not None and not isinstance(scores, pd.Series):
        raise TypeError(f"{score_key} must be a pandas Series in {source}")

    shap_key = "train_shap_values" if selected_split == "train" else "test_shap_values"
    shap_values = loaded_mapping.get(shap_key)
    if shap_values is not None and not isinstance(shap_values, pd.DataFrame):
        raise TypeError(f"{shap_key} must be a pandas DataFrame in {source}")
    if isinstance(shap_values, pd.DataFrame):
        if shap_values.index.has_duplicates or shap_values.columns.has_duplicates:
            raise ValueError(f"{shap_key} has duplicate labels in {source}")

    target_key = f"target_{selected_split}"
    target = loaded_mapping.get(target_key)
    if target is not None and not isinstance(target, pd.Series):
        raise TypeError(f"{target_key} must be a pandas Series in {source}")
    if isinstance(target, pd.Series) and target.index.has_duplicates:
        raise ValueError(f"{target_key} has duplicate sample IDs in {source}")

    raw_context = loaded_mapping.get("data_context", loaded_mapping.get("run_context"))
    if raw_context is not None and not isinstance(raw_context, Mapping):
        raise TypeError(f"data_context must be a mapping in {source}")

    return ClassificationResult(
        path=source,
        split=selected_split,
        metrics=metrics,
        scores=scores,
        shap_values=shap_values,
        selected_features=loaded_mapping.get(f"selected_features_{selected_split}"),
        target=target,
        data_context=dict(raw_context) if isinstance(raw_context, Mapping) else None,
    )


def _curve(
    metrics: Mapping[str, Any],
    *,
    section: Literal["roc", "pr"],
    grid: np.ndarray,
) -> np.ndarray:
    x_key, y_key = ("fpr", "tpr") if section == "roc" else ("recall", "precision")
    if x_key not in metrics or y_key not in metrics:
        raise KeyError(
            f"Incomplete {section.upper()} curve: missing {x_key} or {y_key}"
        )
    x = np.asarray(metrics[x_key], dtype=np.float64)
    y = np.asarray(metrics[y_key], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError(f"{section.upper()} coordinates must be equal 1D arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{section.upper()} coordinates contain non-finite values")
    order = np.argsort(x, kind="stable")
    unique_x, unique_indices = np.unique(x[order], return_index=True)
    return np.interp(grid, unique_x, y[order][unique_indices])


def _empirical_curve_summary(
    results: Sequence[ClassificationResult],
    *,
    section: Literal["roc", "pr"],
    grid: np.ndarray,
    interval: tuple[float, float],
) -> dict[str, Any]:
    score_key = "auc" if section == "roc" else "ap"
    y_key = "tpr" if section == "roc" else "precision"
    x_key = "fpr" if section == "roc" else "recall"
    lower_key = "tprs_lower" if section == "roc" else "precision_lower"
    upper_key = "tprs_upper" if section == "roc" else "precision_upper"
    curves = np.vstack(
        [
            _curve(result.metrics[section], section=section, grid=grid)
            for result in results
        ]
    )
    scores = np.asarray(
        [float(result.metrics[section][score_key]) for result in results],
        dtype=np.float64,
    )
    low, high = interval
    summary: dict[str, Any] = {
        x_key: grid.copy(),
        y_key: curves.mean(axis=0),
        lower_key: np.percentile(curves, low, axis=0),
        upper_key: np.percentile(curves, high, axis=0),
        score_key: scores.mean().item(),
        f"{score_key}_std": scores.std(ddof=1).item(),
        f"{score_key}_ci_lower": np.percentile(scores, low).item(),
        f"{score_key}_ci_upper": np.percentile(scores, high).item(),
        f"{score_key}_runs": scores.tolist(),
        "n_runs": len(results),
        "summary_scope": "mean across repeated run-level point estimates",
        "uncertainty_label": (
            f"{low:g}–{high:g}% empirical interval across {len(results)} runs"
        ),
        "formal_confidence_interval": False,
    }
    if section == "roc":
        summary["tpr"][0], summary["tpr"][-1] = 0.0, 1.0
    else:
        prevalence = [
            result.metrics["pr"].get("positive_prevalence") for result in results
        ]
        available = [float(value) for value in prevalence if value is not None]
        if available:
            summary["positive_prevalence"] = float(np.mean(available))
    return summary


def _empirical_classification_summary(
    results: Sequence[ClassificationResult],
    *,
    interval: tuple[float, float],
) -> dict[str, Any]:
    sections = [result.metrics["classification"] for result in results]
    thresholds = np.asarray(
        [float(section.get("threshold", 0.5)) for section in sections],
        dtype=np.float64,
    )
    if not np.allclose(thresholds, thresholds[0]):
        raise ValueError("Cannot aggregate runs evaluated at different thresholds")
    low, high = interval
    summary: dict[str, Any] = {"threshold": thresholds[0].item()}
    for key in CLASSIFICATION_RATE_KEYS:
        values = np.asarray([float(section[key]) for section in sections])
        summary[key] = values.mean().item()
        summary[f"{key}_std"] = values.std(ddof=1).item()
        summary[f"{key}_ci_lower"] = np.percentile(values, low).item()
        summary[f"{key}_ci_upper"] = np.percentile(values, high).item()
        summary[f"{key}_ci_low"] = summary[f"{key}_ci_lower"]
        summary[f"{key}_ci_high"] = summary[f"{key}_ci_upper"]
        summary[f"{key}_runs"] = values.tolist()
    for key in CONFUSION_COUNT_KEYS:
        values = [section.get(key) for section in sections]
        if all(value is not None for value in values):
            summary[key] = float(np.mean([float(value) for value in values]))
    summary.update(
        {
            "n_runs": len(results),
            "summary_scope": "mean across repeated run-level point estimates",
            "uncertainty_label": (
                f"{low:g}–{high:g}% empirical interval across {len(results)} runs"
            ),
            "formal_confidence_interval": False,
        }
    )
    return summary


def aggregate_result_metrics(
    results: Sequence[ClassificationResult],
    *,
    grid_size: int = 200,
    interval: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, Any]]:
    """Use native uncertainty for one file or summarize repeated run estimates."""
    if not results:
        raise ValueError("At least one result is required")
    if len({result.split for result in results}) != 1:
        raise ValueError("Train and test results cannot be aggregated together")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    low, high = interval
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("interval must contain increasing percentiles within [0, 100]")
    if len(results) == 1:
        return {section: dict(values) for section, values in results[0].metrics.items()}
    grid = np.linspace(0.0, 1.0, grid_size)
    return {
        "roc": _empirical_curve_summary(
            results, section="roc", grid=grid, interval=interval
        ),
        "pr": _empirical_curve_summary(
            results, section="pr", grid=grid, interval=interval
        ),
        "classification": _empirical_classification_summary(results, interval=interval),
    }


def add_validation_bootstrap_if_missing(
    result: ClassificationResult,
    target: pd.Series,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 420,
) -> dict[str, dict[str, Any]]:
    """Reconstruct validation bootstrap intervals only when not already saved."""
    if result.split != "test":
        raise ValueError("Validation bootstrap requires a test result")
    if result.scores is None:
        raise ValueError("Validation result does not contain scores_test")
    existing = result.metrics["roc"]
    if "auc_ci_lower" in existing or "auc_ci_low" in existing:
        return {section: dict(values) for section, values in result.metrics.items()}
    missing = result.scores.index.difference(target.index)
    if len(missing):
        raise ValueError(f"Target is missing samples: {missing.tolist()[:5]}")
    aligned_target = pd.to_numeric(
        target.loc[result.scores.index], errors="raise"
    ).astype(int)
    grid = np.linspace(0.0, 1.0, 200)
    bootstrapped = bootstrap_classification_metrics(
        aligned_target,
        result.scores,
        threshold=float(result.metrics["classification"].get("threshold", 0.5)),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        interpolation_grid=grid,
    )
    metrics = {section: dict(values) for section, values in result.metrics.items()}
    for section in ("roc", "pr", "classification"):
        metrics[section].update(bootstrapped[section])
    metrics["roc"]["fpr"] = grid
    metrics["roc"]["tpr"] = _curve(result.metrics["roc"], section="roc", grid=grid)
    metrics["pr"]["recall"] = grid
    metrics["pr"]["precision"] = _curve(result.metrics["pr"], section="pr", grid=grid)
    metrics["uncertainty"] = bootstrapped["uncertainty"]
    return _normalise_metrics(metrics, path=result.path)


def aggregate_shap_summary(
    results: Sequence[ClassificationResult],
    *,
    alignment: ShapAlignment = "strict",
) -> ShapAggregate | None:
    """Average repeated SHAP matrices and retain run-level stability summaries."""
    frames = [
        result.shap_values for result in results if result.shap_values is not None
    ]
    if not frames:
        return None
    if alignment not in {"strict", "intersection"}:
        raise ValueError("alignment must be 'strict' or 'intersection'")
    reference = frames[0]
    assert reference is not None
    if alignment == "strict":
        samples = reference.index
        features = reference.columns
        for frame in frames[1:]:
            assert frame is not None
            if set(frame.index) != set(samples) or set(frame.columns) != set(features):
                raise ValueError(
                    "Repeated SHAP matrices must contain the same samples and "
                    "features; use alignment='intersection' only when a shared "
                    "subset is intended"
                )
    else:
        samples = reference.index
        features = reference.columns
        for frame in frames[1:]:
            assert frame is not None
            samples = samples.intersection(frame.index, sort=False)
            features = features.intersection(frame.columns, sort=False)
        if len(samples) == 0 or len(features) == 0:
            raise ValueError("Repeated SHAP matrices have no common samples/features")

    arrays = np.stack(
        [
            frame.loc[samples, features].to_numpy(dtype=np.float64, copy=False)
            for frame in frames
            if frame is not None
        ],
        axis=0,
    )
    mean_values = pd.DataFrame(arrays.mean(axis=0), index=samples, columns=features)
    per_run_importance = np.abs(arrays).mean(axis=1)
    ddof = 1 if len(frames) > 1 else 0
    return ShapAggregate(
        mean_values=mean_values,
        mean_absolute=pd.Series(
            per_run_importance.mean(axis=0), index=features, name="Mean |SHAP|"
        ),
        run_standard_deviation=pd.Series(
            per_run_importance.std(axis=0, ddof=ddof),
            index=features,
            name="Run SD |SHAP|",
        ),
        nonzero_run_frequency=pd.Series(
            (per_run_importance > 0.0).mean(axis=0) * 100.0,
            index=features,
            name="Non-zero SHAP runs (%)",
        ),
        n_runs=len(frames),
    )


def rank_features_by_top_k_shap(
    results: Sequence[ClassificationResult],
    *,
    top_k: int = 30,
) -> pd.DataFrame:
    """Rank features by repeated top-K SHAP occurrence and conditional rank.

    One ranking opportunity is one result artifact (normally one repeated
    nested-CV or external-validation run). For each run, mean absolute SHAP is
    calculated across its samples. Only positive-importance features can enter
    that run's top K, preventing arbitrary zero-SHAP ties from being counted.

    Features are ordered by top-K frequency, mean rank while present, repeated
    mean absolute SHAP, and finally feature name. Consequently, a feature with
    one unusually large SHAP value cannot outrank a consistently top-ranked
    feature merely because its across-run mean is large.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    run_importances: list[pd.Series] = []
    for result in results:
        if result.shap_values is None:
            continue
        importance = result.shap_values.abs().mean(axis=0).astype(float)
        importance.index = importance.index.astype(str)
        run_importances.append(importance)
    if not run_importances:
        raise ValueError("No SHAP values are available for feature ranking")

    all_features = sorted(
        set().union(*(set(importance.index) for importance in run_importances))
    )
    importance_by_run = pd.DataFrame(
        {
            run_number: importance.reindex(all_features, fill_value=0.0)
            for run_number, importance in enumerate(run_importances, start=1)
        }
    )
    appearances = pd.Series(0, index=all_features, dtype=int)
    rank_sum = pd.Series(0.0, index=all_features, dtype=float)

    for importance in run_importances:
        ranked = importance.rename("importance").rename_axis("Feature").reset_index()
        ranked = ranked[
            np.isfinite(ranked["importance"]) & ranked["importance"].gt(0.0)
        ]
        ranked = ranked.sort_values(
            ["importance", "Feature"],
            ascending=[False, True],
            kind="stable",
        ).head(top_k)
        for rank, feature in enumerate(ranked["Feature"], start=1):
            appearances.loc[feature] += 1
            rank_sum.loc[feature] += rank

    n_runs = len(run_importances)
    mean_rank = rank_sum.div(appearances.where(appearances.gt(0)))
    ranking = pd.DataFrame(
        {
            "Feature": all_features,
            "Top-k SHAP appearances": appearances.to_numpy(),
            "Top-k SHAP opportunities": n_runs,
            "Top-k SHAP frequency (%)": appearances.to_numpy() * 100.0 / n_runs,
            "Mean rank when in top K": mean_rank.to_numpy(),
            "Ranking mean |SHAP|": importance_by_run.mean(axis=1).to_numpy(),
        }
    )
    ranking["_rank_sort"] = ranking["Mean rank when in top K"].fillna(np.inf)
    ranking = ranking.sort_values(
        [
            "Top-k SHAP frequency (%)",
            "_rank_sort",
            "Ranking mean |SHAP|",
            "Feature",
        ],
        ascending=[False, True, False, True],
        kind="stable",
    ).drop(columns="_rank_sort")
    ranking.insert(
        1,
        "Top-k frequency rank",
        np.arange(1, len(ranking) + 1, dtype=int),
    )
    ranking["Ranking top K"] = top_k
    return ranking.reset_index(drop=True)


def _features_for_display(
    results: Sequence[ClassificationResult],
    shap_summary: ShapAggregate,
    *,
    feature_ranking: FeatureRanking,
    ranking_top_k: int,
    min_top_k_frequency: float,
    max_display: int,
) -> tuple[list[str], pd.DataFrame, str]:
    """Resolve the requested ranking and return an ordered display subset."""
    if feature_ranking not in {"auto", "top-k-frequency", "mean-abs-shap"}:
        raise ValueError(
            "feature_ranking must be 'auto', 'top-k-frequency', or " "'mean-abs-shap'"
        )
    if max_display < 1:
        raise ValueError("max_display must be at least 1")
    if not 0.0 <= min_top_k_frequency <= 100.0:
        raise ValueError("min_top_k_frequency must be between 0 and 100")

    ranking = rank_features_by_top_k_shap(results, top_k=ranking_top_k)
    available = set(shap_summary.mean_values.columns.astype(str))
    ranking = ranking[ranking["Feature"].isin(available)].copy()
    eligible_ranking = ranking[
        ranking["Top-k SHAP frequency (%)"].ge(min_top_k_frequency)
    ].copy()
    if eligible_ranking.empty:
        raise ValueError(
            "No features satisfy min_top_k_frequency=" f"{min_top_k_frequency:g}%"
        )

    resolved_method = (
        "top-k-frequency"
        if feature_ranking == "auto" and shap_summary.n_runs > 1
        else "mean-abs-shap" if feature_ranking == "auto" else feature_ranking
    )
    if resolved_method == "top-k-frequency":
        ordered = eligible_ranking["Feature"].astype(str).tolist()
    else:
        eligible = set(eligible_ranking["Feature"].astype(str))
        ordered = [
            str(feature)
            for feature in shap_summary.mean_absolute.sort_values(
                ascending=False,
                kind="stable",
            ).index
            if str(feature) in eligible
        ]
    return ordered[:max_display], ranking, resolved_method


def aggregate_shap_values(
    results: Sequence[ClassificationResult],
    *,
    alignment: ShapAlignment = "strict",
) -> pd.DataFrame | None:
    """Compatibility helper returning only the repeated-run mean SHAP matrix."""
    summary = aggregate_shap_summary(results, alignment=alignment)
    return summary.mean_values if summary is not None else None


def selected_feature_frequency(
    results: Sequence[ClassificationResult],
) -> pd.Series | None:
    """Percentage of saved outer-fold/run models selecting each feature."""
    selected_sets: list[set[str]] = []
    for result in results:
        selected = result.selected_features
        if not selected:
            continue
        if all(isinstance(feature, str) for feature in selected):
            selected_sets.append(set(selected))
            continue
        for fold_features in selected:
            if isinstance(fold_features, Sequence) and not isinstance(
                fold_features, (str, bytes)
            ):
                selected_sets.append({str(feature) for feature in fold_features})
    if not selected_sets:
        return None
    features = sorted(set().union(*selected_sets))
    return pd.Series(
        [
            100.0
            * sum(feature in selected_set for selected_set in selected_sets)
            / len(selected_sets)
            for feature in features
        ],
        index=features,
        name="Selection frequency (%)",
        dtype=float,
    )


def bootstrap_validation_metrics(
    result: ClassificationResult,
    target: pd.Series,
    *,
    class_labels: Sequence[str] | None = None,
    n_bootstraps: int = 1000,
    random_state: int = 420,
    grid_size: int = 200,
    interval: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper that explicitly recalculates validation bootstrap."""
    if len(class_labels) != 2:
        raise ValueError("class_labels must contain exactly two labels")
    if result.split != "test" or result.scores is None:
        raise ValueError("Validation bootstrap requires a test result with scores_test")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    low, high = interval
    confidence_level = (high - low) / 100.0
    if not np.isclose(low, (100.0 - 100.0 * confidence_level) / 2.0):
        raise ValueError("interval must be a central percentile interval")
    missing = result.scores.index.difference(target.index)
    if len(missing):
        raise ValueError(f"Target is missing samples: {missing.tolist()[:5]}")
    aligned = target.loc[result.scores.index]
    numeric = pd.to_numeric(aligned, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) == {0, 1}:
        encoded = numeric.astype(int)
    else:
        encoded = aligned.map({label: code for code, label in enumerate(class_labels)})
        if encoded.isna().any() or set(encoded.astype(int).unique()) != {0, 1}:
            raise ValueError("target must contain both configured binary classes")
        encoded = encoded.astype(int)
    grid = np.linspace(0.0, 1.0, grid_size)
    bootstrapped = bootstrap_classification_metrics(
        encoded,
        result.scores,
        threshold=float(result.metrics["classification"].get("threshold", 0.5)),
        n_resamples=n_bootstraps,
        confidence_level=confidence_level,
        random_state=random_state,
        interpolation_grid=grid,
    )
    metrics = {section: dict(values) for section, values in result.metrics.items()}
    for section in ("roc", "pr", "classification"):
        metrics[section].update(bootstrapped[section])
    metrics["roc"]["fpr"] = grid
    metrics["roc"]["tpr"] = _curve(result.metrics["roc"], section="roc", grid=grid)
    metrics["pr"]["recall"] = grid
    metrics["pr"]["precision"] = _curve(result.metrics["pr"], section="pr", grid=grid)
    metrics["uncertainty"] = bootstrapped["uncertainty"]
    return _normalise_metrics(metrics, path=result.path)


def _align_features(
    features: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> pd.DataFrame:
    missing_samples = shap_values.index.difference(features.index)
    missing_features = shap_values.columns.difference(features.columns)
    if len(missing_samples) or len(missing_features):
        raise ValueError(
            "features does not cover the SHAP matrix; "
            f"missing samples={missing_samples.tolist()[:5]}, "
            f"missing features={missing_features.tolist()[:5]}"
        )
    return features.loc[shap_values.index, shap_values.columns]


def _config_for_plotting(
    result: ClassificationResult,
    *,
    config_file: str | Path | None,
) -> Config | None:
    """Load explicit inputs first, otherwise use the embedded resolved snapshot."""
    if config_file is not None:
        return Config(config_file)
    if result.data_context is None:
        return None
    snapshot = result.data_context.get("resolved_config")
    if not isinstance(snapshot, Mapping):
        raise TypeError("data_context.resolved_config must be a mapping")
    return Config.from_mapping(
        snapshot,
        config_file=result.data_context.get("config_file"),
    )


def _encode_plot_target(metadata: pd.DataFrame, config: Config) -> pd.Series:
    """Encode configured textual or already numeric targets as integer codes."""
    labels = metadata[config.col_target]
    mapped = labels.map(config.group_label_encoding)
    numeric = pd.to_numeric(labels, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric, numeric.round())
    valid_codes = set(range(len(config.group_tests)))
    encoded = mapped.where(mapped.notna(), numeric.where(integer_like))
    invalid = encoded.isna() | ~encoded.isin(valid_codes)
    if invalid.any():
        raise ValueError(
            "Could not encode target values for plotting samples: "
            f"{labels.index[invalid].tolist()[:10]}"
        )
    return encoded.astype(int).rename(config.col_target)


def reconstruct_plot_data(
    result: ClassificationResult,
    *,
    config_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None] | None:
    """Recover values by aligning original inputs to saved SHAP labels.

    The SHAP matrix is authoritative for both samples and feature names. This
    means metadata filters, split filters, prevalence thresholds, and outer-CV
    fold assignments do not need to be replayed merely to make plots.
    """
    config = _config_for_plotting(result, config_file=config_file)
    if config is None:
        return None

    reference_index: pd.Index | None = None
    expected_columns: pd.Index | None = None
    if result.shap_values is not None:
        reference_index = result.shap_values.index
        expected_columns = result.shap_values.columns
    elif result.scores is not None:
        reference_index = result.scores.index
    elif result.target is not None:
        reference_index = result.target.index
    if reference_index is None:
        raise ValueError(f"{result.path} has no sample-indexed plotting data")

    analysis_config = copy.copy(config)
    # Saved SHAP sample IDs already define the evaluated cohort. Clear metadata
    # filters so CLI-only training/validation choices do not need to be replayed.
    analysis_config.filters_metadata = None
    analysis_config.combined_filters_metadata = None
    metadata = MetadataHandler(analysis_config).get_individuals_metadata_df()
    missing_metadata_samples = reference_index.difference(metadata.index)
    if len(missing_metadata_samples):
        raise ValueError(
            "Metadata does not contain saved result samples: "
            f"{missing_metadata_samples.tolist()[:10]}"
        )
    metadata = metadata.loc[reference_index].copy()
    reconstructed_target = _encode_plot_target(metadata, analysis_config)

    features = pd.DataFrame(index=reference_index)
    if expected_columns is not None:
        prefixes = tuple(analysis_config.peptide_prefixes)
        peptide_columns = [
            column for column in expected_columns if str(column).startswith(prefixes)
        ]
        clinical_columns = [
            column for column in expected_columns if column not in peptide_columns
        ]

        if peptide_columns:
            peptides = OligosHandler(analysis_config).get_oligos_df()
            missing_peptide_samples = reference_index.difference(peptides.index)
            if len(missing_peptide_samples):
                raise ValueError(
                    "Peptide data does not contain saved result samples: "
                    f"{missing_peptide_samples.tolist()[:10]}"
                )
            for column in peptide_columns:
                # External alignment uses zero for a peptide absent from the
                # external raw matrix, so reproduce that convention here.
                features[column] = (
                    peptides.loc[reference_index, column]
                    if column in peptides.columns
                    else 0
                )
        missing_clinical = [
            column for column in clinical_columns if column not in metadata.columns
        ]
        if missing_clinical:
            raise ValueError(
                "Metadata is missing saved clinical SHAP features: "
                f"{missing_clinical[:10]}"
            )
        for column in clinical_columns:
            features[column] = metadata[column]
        features = features.loc[:, expected_columns].copy()

    target = reconstructed_target
    if result.target is not None:
        saved_target = result.target.reindex(reference_index)
        if saved_target.isna().any():
            raise ValueError("Saved target does not cover all result samples")
        if not np.array_equal(saved_target.to_numpy(), target.to_numpy()):
            raise ValueError(
                "Saved and reconstructed targets differ; input data or filters "
                "have changed since model evaluation"
            )
        target = saved_target

    library_metadata = None
    if analysis_config.lib_metadata_input is not None:
        library_metadata = OligosHandler(analysis_config).get_oligos_metadata_df()
    return features, target, library_metadata


def normalise_plot_names(plots: Sequence[str] | None) -> tuple[str, ...]:
    """Validate plot names, expand ``all``, and retain the requested order."""
    if plots is None:
        return PLOT_NAMES
    if isinstance(plots, str):
        plots = [plots]
    requested: list[str] = []
    for raw_name in plots:
        name = str(raw_name).strip().lower().replace("_", "-")
        if name == "all":
            requested.extend(PLOT_NAMES)
            continue
        name = PLOT_ALIASES.get(name, name)
        if name not in PLOT_NAMES:
            raise ValueError(
                f"Unknown plot {raw_name!r}; choose from {list(PLOT_NAMES)!r}"
            )
        requested.append(name)
    unique = tuple(dict.fromkeys(requested))
    if not unique:
        raise ValueError("At least one plot must be requested")
    return unique


def normalise_output_formats(formats: Sequence[str] | None) -> tuple[str, ...]:
    """Validate requested figure formats and remove duplicates."""
    values = DEFAULT_OUTPUT_FORMATS if formats is None else formats
    if isinstance(values, str):
        values = [values]
    normalised = tuple(
        dict.fromkeys(str(value).strip().lower().lstrip(".") for value in values)
    )
    invalid = [value for value in normalised if value not in {"pdf", "svg", "png"}]
    if invalid:
        raise ValueError(
            f"Unsupported output formats {invalid}; choose PDF, SVG, and/or PNG"
        )
    if not normalised:
        raise ValueError("At least one output format must be requested")
    return normalised


def _resolve_plot_colors(colors: Mapping[str, str] | None) -> dict[str, str]:
    resolved = dict(DEFAULT_PLOT_COLORS)
    if colors is None:
        return resolved
    unknown = sorted(set(colors) - set(resolved))
    if unknown:
        raise ValueError(f"Unknown plot color/style keys: {unknown}")
    resolved.update({str(key): str(value) for key, value in colors.items()})
    return resolved


def _save_figure_formats(
    figure: plt.Figure,
    *,
    destination: Path | None,
    output_prefix: str,
    suffix: str,
    formats: Sequence[str],
    dpi: int,
) -> None:
    if destination is None:
        return
    for output_format in formats:
        path = destination / f"{output_prefix}_{suffix}.{output_format}"
        save_kwargs: dict[str, object] = {
            "format": output_format,
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if output_format != "svg":
            save_kwargs["dpi"] = dpi
        figure.savefig(path, **save_kwargs)


def plot_result_files(
    result_files: Sequence[str | Path],
    *,
    split: ResultSplit = "auto",
    class_labels: Sequence[str] | None = ("Negative", "Positive"),
    title: str | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str = "phipml",
    plots: Sequence[str] | None = None,
    output_formats: Sequence[str] | None = None,
    dpi: int = 600,
    plot_colors: Mapping[str, str] | None = None,
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    oligos_metadata: pd.DataFrame | None = None,
    feature_importance_table: pd.DataFrame | None = None,
    peptide_prefixes: Sequence[str] | None = None,
    table_annotation_columns: Sequence[str] | None = None,
    table_extra_columns: Sequence[str] | None = None,
    feature_table_title: str = "Top features by mean absolute SHAP value",
    max_display: int = 20,
    feature_ranking: FeatureRanking = "auto",
    ranking_top_k: int = 30,
    min_top_k_frequency: float = 0.0,
    shap_alignment: ShapAlignment = "strict",
    validation_bootstraps: int = 0,
    random_state: int = 420,
    save_standalone: bool = True,
    config_file: str | Path | None = None,
    reconstruct_data: bool = True,
) -> ResultPlotOutput:
    """Plot one result or an empirical summary of repeated result files.

    All available plots and PDF/SVG/PNG output are enabled by default. Passing
    ``plots`` or ``output_formats`` selects a smaller reproducible subset.
    """
    if not result_files:
        raise ValueError("result_files cannot be empty")
    if not output_prefix:
        raise ValueError("output_prefix cannot be empty")
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    requested_plots = list(normalise_plot_names(plots))
    if not save_standalone:
        requested_plots = [
            name
            for name in requested_plots
            if name not in {"roc", "pr", "confusion", "classification"}
        ]
    if not requested_plots:
        raise ValueError("No plots remain after applying save_standalone=False")
    requested_plot_set = set(requested_plots)
    formats = normalise_output_formats(output_formats)
    colors = _resolve_plot_colors(plot_colors)
    if feature_importance_table is not None and feature_importance_table.empty:
        raise ValueError("feature_importance_table cannot be empty")
    results = [load_classification_result(path, split=split) for path in result_files]
    plot_config = _config_for_plotting(results[0], config_file=config_file)
    if class_labels is None:
        class_labels = (
            tuple(str(label) for label in plot_config.group_tests[:2])
            if plot_config is not None
            else ("Negative", "Positive")
        )
    if len(class_labels) != 2:
        raise ValueError("class_labels must contain exactly two labels")
    if peptide_prefixes is None:
        peptide_prefixes = (
            tuple(plot_config.peptide_prefixes)
            if plot_config is not None
            else ("agilent_", "twist_", "corona2_")
        )
    if target is None and results[0].target is not None:
        target = results[0].target.copy()
    if features is None and reconstruct_data:
        reconstructed = reconstruct_plot_data(
            results[0],
            config_file=config_file,
        )
        if reconstructed is not None:
            features, reconstructed_target, reconstructed_library = reconstructed
            if target is None:
                target = reconstructed_target
            if oligos_metadata is None:
                oligos_metadata = reconstructed_library
    if (
        oligos_metadata is None
        and features is not None
        and target is not None
        and reconstruct_data
    ):
        if plot_config is not None and plot_config.lib_metadata_input is not None:
            oligos_metadata = OligosHandler(plot_config).get_oligos_metadata_df()
    metrics = aggregate_result_metrics(results)
    if (
        len(results) == 1
        and results[0].split == "test"
        and target is not None
        and validation_bootstraps > 0
    ):
        metrics = add_validation_bootstrap_if_missing(
            results[0],
            target,
            n_resamples=validation_bootstraps,
            random_state=random_state,
        )

    destination = Path(output_dir).expanduser() if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    performance_figure = None
    if "performance" in requested_plot_set:
        performance_figure, _ = plot_performance_summary(
            metrics,
            class_labels=class_labels,
            title=title,
            roc_color=colors["roc"],
            roc_band_color=colors["roc_band"],
            pr_color=colors["pr"],
            pr_band_color=colors["pr_band"],
            confusion_cmap=colors["confusion_cmap"],
            classification_color=colors["classification"],
        )
        _save_figure_formats(
            performance_figure,
            destination=destination,
            output_prefix=output_prefix,
            suffix="performance",
            formats=formats,
            dpi=dpi,
        )
    standalone: dict[str, plt.Figure] = {}
    if "roc" in requested_plot_set:
        standalone["roc"], _ = plot_roc_metrics(
            metrics["roc"],
            color=colors["roc"],
            band_color=colors["roc_band"],
        )
    if "pr" in requested_plot_set:
        standalone["pr"], _ = plot_precision_recall_metrics(
            metrics["pr"],
            positive_prevalence=metrics["pr"].get("positive_prevalence"),
            color=colors["pr"],
            band_color=colors["pr_band"],
        )
    if "confusion" in requested_plot_set:
        standalone["confusion"], _ = plot_confusion_matrix_metrics(
            metrics["classification"],
            class_labels=class_labels,
            cmap=colors["confusion_cmap"],
        )
    if "classification" in requested_plot_set:
        standalone["classification"], _ = plot_classification_metric_bars(
            metrics["classification"],
            color=colors["classification"],
        )
    for name, figure in standalone.items():
        _save_figure_formats(
            figure,
            destination=destination,
            output_prefix=output_prefix,
            suffix=name,
            formats=formats,
            dpi=dpi,
        )

    needs_shap = bool(
        requested_plot_set & {"shap-beeswarm", "shap-importance", "shap-heatmap"}
    ) or ("feature-table" in requested_plot_set and feature_importance_table is None)
    shap_summary = (
        aggregate_shap_summary(results, alignment=shap_alignment)
        if needs_shap
        else None
    )
    shap_values = shap_summary.mean_values if shap_summary is not None else None
    shap_importance_figure = None
    shap_heatmap_figure = None
    shap_beeswarm_figure = None
    feature_table = (
        feature_importance_table.copy()
        if feature_importance_table is not None
        else None
    )
    feature_table_figure = None
    feature_ranking_table = None
    display_features: list[str] = []
    if shap_values is not None:
        assert shap_summary is not None
        (
            display_features,
            feature_ranking_table,
            resolved_feature_ranking,
        ) = _features_for_display(
            results,
            shap_summary,
            feature_ranking=feature_ranking,
            ranking_top_k=ranking_top_k,
            min_top_k_frequency=min_top_k_frequency,
            max_display=max_display,
        )
        display_shap_values = shap_values.loc[:, display_features]
        importance_values = pd.DataFrame(
            [shap_summary.mean_absolute.to_numpy()],
            columns=shap_summary.mean_absolute.index,
            index=["Repeated-run mean"],
        )
        importance_title = "Global SHAP feature importance"
        if resolved_feature_ranking == "top-k-frequency":
            importance_title += f"\n(ranked by top-{ranking_top_k} frequency)"
        if "shap-importance" in requested_plot_set:
            shap_importance_figure, _ = plot_shap_importance_bar(
                importance_values,
                feature_order=display_features,
                max_display=max_display,
                title=importance_title,
                color=colors["shap_importance"],
            )
            _save_figure_formats(
                shap_importance_figure,
                destination=destination,
                output_prefix=output_prefix,
                suffix="shap_importance",
                formats=formats,
                dpi=dpi,
            )
        if "shap-heatmap" in requested_plot_set:
            shap_heatmap_figure, _ = plot_shap_heatmap(
                shap_values,
                target=target,
                class_labels=class_labels,
                feature_order=display_features,
                max_display=max_display,
                cmap=colors["shap_heatmap_cmap"],
            )
            _save_figure_formats(
                shap_heatmap_figure,
                destination=destination,
                output_prefix=output_prefix,
                suffix="shap_heatmap",
                formats=formats,
                dpi=dpi,
            )
        if features is not None:
            aligned_features = _align_features(features, shap_values)
            display_features_frame = aligned_features.loc[:, display_features]
            if "shap-beeswarm" in requested_plot_set:
                shap_beeswarm_figure, _ = plot_shap_values(
                    display_shap_values.to_numpy(),
                    display_features_frame,
                    cmap=colors["shap_cmap"],
                    max_display=len(display_features),
                    group_tests=list(class_labels),
                    group_label_colors=[
                        colors["negative_class"],
                        colors["positive_class"],
                    ],
                    filename_label=output_prefix,
                    add_binary_legend=None,
                    save_fig=False,
                    sort=False,
                )
                _save_figure_formats(
                    shap_beeswarm_figure,
                    destination=destination,
                    output_prefix=output_prefix,
                    suffix="shap_beeswarm",
                    formats=formats,
                    dpi=dpi,
                )
            if target is not None and feature_table is None:
                missing_target = shap_values.index.difference(target.index)
                if len(missing_target):
                    raise ValueError(
                        f"target is missing SHAP samples: {missing_target.tolist()[:5]}"
                    )
                feature_table = build_feature_importance_table(
                    shap_values,
                    aligned_features,
                    target.loc[shap_values.index],
                    group_labels=class_labels,
                    oligos_metadata=oligos_metadata,
                    peptide_prefixes=peptide_prefixes,
                )
                importance = shap_summary.mean_absolute.rename(
                    "Repeated-run mean |SHAP|"
                )
                stability = pd.concat(
                    [
                        importance,
                        shap_summary.run_standard_deviation,
                        shap_summary.nonzero_run_frequency,
                    ],
                    axis=1,
                )
                selection_frequency = selected_feature_frequency(results)
                if selection_frequency is not None:
                    stability = pd.concat(
                        [
                            stability,
                            selection_frequency.reindex(
                                stability.index,
                                fill_value=0.0,
                            ),
                        ],
                        axis=1,
                    )
                feature_table = feature_table.merge(
                    stability.reset_index(names="Feature"),
                    on="Feature",
                    how="left",
                    validate="one_to_one",
                )
                assert feature_ranking_table is not None
                ranking_columns = [
                    "Feature",
                    "Top-k frequency rank",
                    "Top-k SHAP appearances",
                    "Top-k SHAP opportunities",
                    "Top-k SHAP frequency (%)",
                    "Mean rank when in top K",
                    "Ranking top K",
                ]
                feature_table = feature_table.merge(
                    feature_ranking_table.loc[:, ranking_columns],
                    on="Feature",
                    how="left",
                    validate="one_to_one",
                )
                feature_table["Mean |SHAP|"] = feature_table["Repeated-run mean |SHAP|"]
                feature_table["Displayed"] = feature_table["Feature"].isin(
                    display_features
                )
                feature_table["Feature ranking method"] = resolved_feature_ranking
                if resolved_feature_ranking == "top-k-frequency":
                    feature_table = feature_table.sort_values(
                        ["Top-k frequency rank", "Mean |SHAP|"],
                        ascending=[True, False],
                        kind="stable",
                    ).reset_index(drop=True)
                else:
                    feature_table = feature_table.sort_values(
                        "Mean |SHAP|",
                        ascending=False,
                        kind="stable",
                    ).reset_index(drop=True)

    if "feature-table" in requested_plot_set and feature_table is not None:
        if destination is not None:
            feature_table.to_csv(
                destination / f"{output_prefix}_feature_importance.csv",
                index=False,
            )
        if feature_importance_table is not None:
            displayed_table = feature_table.head(max_display).copy()
        elif display_features:
            displayed_table = (
                feature_table.set_index("Feature").loc[display_features].reset_index()
            )
        else:
            displayed_table = feature_table.head(max_display).copy()
        feature_table_figure, _ = plot_feature_importance_table(
            displayed_table,
            group_labels=class_labels,
            max_display=min(max_display, len(displayed_table)),
            annotation_columns=table_annotation_columns,
            extra_columns=table_extra_columns,
            title=feature_table_title,
            header_color=colors["table_header"],
            row_colors=[colors["table_row_odd"], colors["table_row_even"]],
            prevalence_cmap=colors["table_prevalence_cmap"],
        )
        _save_figure_formats(
            feature_table_figure,
            destination=destination,
            output_prefix=output_prefix,
            suffix="feature_table",
            formats=formats,
            dpi=dpi,
        )
    elif "feature-table" in requested_plot_set:
        logger.warning(
            "Feature-table plot was requested but raw features/targets or a "
            "curated feature-importance table were unavailable"
        )

    if "shap-beeswarm" in requested_plot_set and shap_beeswarm_figure is None:
        logger.warning(
            "SHAP beeswarm was requested but raw feature values were unavailable"
        )
    if "shap-importance" in requested_plot_set and shap_importance_figure is None:
        logger.warning("SHAP importance was requested but SHAP values were unavailable")
    if "shap-heatmap" in requested_plot_set and shap_heatmap_figure is None:
        logger.warning("SHAP heatmap was requested but SHAP values were unavailable")

    return ResultPlotOutput(
        metrics=metrics,
        performance_figure=performance_figure,
        shap_values=shap_values,
        shap_beeswarm_figure=shap_beeswarm_figure,
        shap_importance_figure=shap_importance_figure,
        shap_heatmap_figure=shap_heatmap_figure,
        feature_table=feature_table,
        feature_table_figure=feature_table_figure,
        standalone_figures=standalone,
        shap_summary=shap_summary,
        feature_ranking=feature_ranking_table,
        display_features=display_features,
    )
