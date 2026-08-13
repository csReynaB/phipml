# ======================
# Standard library
# ======================
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd

# ======================
# Local libraries
# ======================
from phipml.classification.helpers import (
    calculate_classification_metrics,
    compute_interp_pr_ap,
    compute_interp_tpr_auc,
)
from phipml.io.data_handler import (
    Config,
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)
from phipml.plots.helpers import (
    build_feature_importance_table,
    generate_feature_importance_table,
    plot_feature_importance_table,
    plot_performance_summary,
    plot_roc_summary,
    plot_shap_heatmap,
    plot_shap_importance_bar,
    plot_shap_values,
)


# =====================
# ROC AUC helpers
# =====================
def _load_auc_metrics(fn: Path, fpr_grid: np.ndarray):
    """
    Load ROC TPR + AUC from saved joblib results.

    Supports:
    - old format: roc_metrics_train / roc_metrics_test
    - new format: metrics_train["roc"] / metrics_test["roc"]
    """

    d = joblib.load(fn)

    # --- NEW STRUCTURE -------------------------------------------------
    if "metrics_train" in d or "metrics_test" in d:
        metrics = d.get("metrics_train", d.get("metrics_test"))
        roc = metrics["roc"]
    # --- OLD STRUCTURE -------------------------------------------------
    else:
        roc = d.get("roc_metrics_train", d.get("roc_metrics_test"))

    if roc is None:
        raise KeyError(f"No ROC metrics found in {fn}")

    tpr = np.asarray(roc["tpr"])
    auc = roc["auc"]

    # If saved tpr length already matches fpr_grid, return directly
    if tpr.shape[0] == fpr_grid.shape[0]:
        # enforce ROC endpoints (safe)
        tpr[0] = 0.0
        tpr[-1] = 1.0
        return tpr, auc

    # Otherwise, if you stored fpr too, interpolate:
    if "fpr" in roc:
        fpr = np.asarray(roc["fpr"])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        return tpr_interp, auc

    raise ValueError(
        f"{fn} TPR length {tpr.shape[0]} does not match fpr_grid {fpr_grid.shape[0]} "
        "and no 'fpr' was saved for interpolation."
    )


def summarize_roc_runs(tprs: np.ndarray, aucs: np.ndarray) -> Dict[str, Any]:
    """
    tprs: shape (n_runs, n_grid)
    aucs: shape (n_runs,)
    """
    mean_tpr = tprs.mean(axis=0)
    low_tpr = np.percentile(tprs, 2.5, axis=0)
    high_tpr = np.percentile(tprs, 97.5, axis=0)

    auc_mean = aucs.mean()
    auc_ci_low, auc_ci_high = np.percentile(aucs, [2.5, 97.5])

    return {
        "mean_tpr": mean_tpr,
        "low_tpr": low_tpr,
        "high_tpr": high_tpr,
        "auc_mean": auc_mean,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
    }


def plot_nested_aucs(
    joblib_dir: str,
    group1: str,
    size1: int,
    group2: str,
    size2: int,
    colors: Dict[str, str],
    out_dir: str,
    out_base: str,
    prefix_base: str = "nested_xgboost_",
    fpr_grid: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, plt.Figure], Dict[str, plt.Axes]]:
    """
    Scan `joblib_dir` for nested_predictions_*.joblib, group by prefix,
    and for each prefix produce & save an ROC plot PDF and serialized fig.
    Returns a dict mapping prefix -> Axes object.
    """
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 200)

    joblib_dir = Path(joblib_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_loc = (
        "upper left" if group1 == "HCC" else "lower right"
    )  # special case, can be deleted

    files = sorted(
        joblib_dir.glob(f"{prefix_base}*.joblib")
    )  # or "random-forest_*.joblib"
    prefix_re = re.compile(rf"^{re.escape(prefix_base)}(.+?)_\d+$")

    buckets: Dict[str, List[Path]] = {}
    for fn in files:
        m = prefix_re.match(fn.stem)
        if not m:
            continue
        buckets.setdefault(m.group(1), []).append(fn)

    figs: Dict[str, plt.Figure] = {}
    axes: Dict[str, plt.Axes] = {}
    for prefix, fns in buckets.items():
        with ThreadPoolExecutor() as exe:
            results = list(exe.map(lambda p: _load_auc_metrics(p, fpr_grid), fns))
            # futures = [exe.submit(_load_auc_metrics, fn, fpr_grid) for fn in fns]
            # results = [f.result() for f in as_completed(futures)]

        tprs, aucs = zip(*results)
        tprs_arr = np.vstack(tprs)

        summary = summarize_roc_runs(tprs_arr, aucs)

        title = f"{group1} (n={size1}) vs. {group2} (n={size2})"
        pdf_path = out_dir / f"{out_base}_{prefix}.pdf"

        fig, ax = plot_roc_summary(
            fpr_grid=fpr_grid,
            tprs=tprs_arr,
            summary=summary,
            colors=colors,
            title=title,
            label_loc=label_loc,
            pdf_path=pdf_path,
        )

        figs[prefix] = fig
        axes[prefix] = ax

    return figs, axes


# =====================
# SHAP helpers
# =====================
def _list_files(file_dir: Union[str, Path], file_pattern: str) -> List[Path]:
    file_dir = Path(file_dir)
    files = sorted(file_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{file_pattern}' in '{file_dir}'."
        )
    return files


def _load_shap_df(fn: Path, shap_key: str) -> pd.DataFrame:
    obj = joblib.load(fn)
    if shap_key not in obj:
        raise KeyError(f"Key '{shap_key}' not found in joblib file: {fn}")
    df = obj[shap_key]
    # if not isinstance(df, pd.DataFrame):
    #    raise TypeError(f"Expected '{shap_key}' to be a pandas DataFrame in {fn}, got {type(df)}")
    return df


def mean_shap_across_files(files, shap_key: str) -> pd.DataFrame:
    ref_index = None
    ref_cols = None
    acc = None
    for fn in files:
        df = _load_shap_df(fn, shap_key)  # returns a DataFrame

        if ref_index is None:
            ref_index = df.index
            ref_cols = df.columns
            acc = np.zeros(df.shape, dtype=np.float32)
        else:
            # alignment checks (fast + safe)
            if not df.index.equals(ref_index) or not df.columns.equals(ref_cols):
                raise ValueError(f"SHAP matrices not aligned. Offending file: {fn}")

        # accumulate (convert each df once)
        acc += df.to_numpy(dtype=np.float32, copy=False)

    acc /= len(files)

    return pd.DataFrame(acc, index=ref_index, columns=ref_cols)


def run_shap_summary_and_feature_table(
    config_file,
    file_dir,
    file_pattern,
    output_dir,
    output_name="shap_values",
    cmap="viridis",
    max_display=30,
    figure_size=(6, 5),
    shap_key=None,
    shap_fontsize=None,
    legend_labels=None,
    label_groups=None,
):
    files = _list_files(file_dir, file_pattern)
    if shap_key is None:
        shap_key = (
            "test_shap_values"
            if "validation" in file_pattern.lower()
            else "train_shap_values"
        )
    shap_values = mean_shap_across_files(files, shap_key)

    # Load data
    config = Config(config_file)
    metadata_handler = MetadataHandler(config)
    oligos_handler = OligosHandler(config)
    feature_manager = FeatureManager(
        config,
        metadata_handler,
        oligos_handler,
        subgroup="all",
        with_oligos=True,
        with_additional_features=False,
        filter_by_entropy=False,
        prevalence_threshold_min=0,
        prevalence_threshold_max=100,
    )
    X_train, y_train = feature_manager.get_features_target()

    # Align X/y to SHAP df (rows and columns)
    # missing_rows = shap_values.index.difference(X_train.index)
    # missing_cols = shap_values.columns.difference(X_train.columns)
    # if len(missing_rows) or len(missing_cols):
    #     raise KeyError(
    #         "SHAP df contains samples/features not found in X_train.\n"
    #         f"Missing rows in X_train: {list(missing_rows[:10])}{' ...' if len(missing_rows) > 10 else ''}\n"
    #         f"Missing cols in X_train: {list(missing_cols[:10])}{' ...' if len(missing_cols) > 10 else ''}"
    #     )

    X_train = X_train.loc[shap_values.index, shap_values.columns]
    y_train = y_train.loc[shap_values.index]

    group_tests = label_groups if label_groups is not None else config.group_tests

    plot_shap_values(
        shap_values.values,
        X_train,
        cmap=cmap,
        max_display=max_display,
        group_tests=group_tests,
        filename_label=output_name,
        fontsize=shap_fontsize,
        figure_size=figure_size,
        legend_labels=legend_labels,
        save_fig=True,
        figures_dir=output_dir,
    )

    oligos_metadata = oligos_handler.get_oligos_metadata_df()
    keep_cols = ["Description", "species", "genus", "family", "order", "pos", "len_seq"]
    # oligos_metadata.set_index(oligos_metadata.columns[0], inplace=True)
    missing = [c for c in keep_cols if c not in oligos_metadata.columns]
    if missing:
        raise KeyError(f"Oligos metadata missing expected columns: {missing}")
    oligos_metadata = oligos_metadata[keep_cols]

    generate_feature_importance_table(
        shap_values.values,
        X_train,
        y_train,
        oligos_metadata,
        group_tests=group_tests,
        filename_label=output_name,
        figures_dir=output_dir,
    )


# =====================================================================
# Modern result-file API
# =====================================================================
ResultSplit = Literal["auto", "train", "test"]


@dataclass(frozen=True)
class ClassificationResult:
    """Validated contents extracted from one phipml result file."""

    path: Path
    split: Literal["train", "test"]
    metrics: dict[str, dict[str, Any]]
    scores: pd.Series | None
    shap_values: pd.DataFrame | None
    selected_features: list[str] | list[list[str]] | None


@dataclass(frozen=True)
class ResultPlotOutput:
    """Objects created from one or multiple result files."""

    metrics: dict[str, dict[str, Any]]
    performance_figure: plt.Figure
    shap_values: pd.DataFrame | None
    shap_beeswarm_figure: plt.Figure | None
    shap_importance_figure: plt.Figure | None
    shap_heatmap_figure: plt.Figure | None
    feature_table: pd.DataFrame | None
    feature_table_figure: plt.Figure | None


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


def load_classification_result(
    path: str | Path,
    *,
    split: ResultSplit = "auto",
) -> ClassificationResult:
    """Load and validate one current-format nested-CV or validation result."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Result file does not exist: {source}")
    loaded = joblib.load(source)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Result file must contain a mapping: {source}")
    selected_split = _choose_result_split(loaded, split, source)
    raw_metrics = loaded[f"metrics_{selected_split}"]
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(f"metrics_{selected_split} must be a mapping in {source}")
    missing_sections = sorted({"roc", "pr", "classification"} - set(raw_metrics))
    if missing_sections:
        raise KeyError(f"{source} is missing metric sections: {missing_sections}")
    metrics: dict[str, dict[str, Any]] = {}
    for section in ("roc", "pr", "classification"):
        values = raw_metrics[section]
        if not isinstance(values, Mapping):
            raise TypeError(f"Metric section {section!r} must be a mapping in {source}")
        metrics[section] = dict(values)

    score_key = f"scores_{selected_split}"
    scores = loaded.get(score_key)
    if scores is not None and not isinstance(scores, pd.Series):
        raise TypeError(f"{score_key} must be a pandas Series in {source}")
    shap_key = "train_shap_values" if selected_split == "train" else "test_shap_values"
    shap_values = loaded.get(shap_key)
    if shap_values is not None and not isinstance(shap_values, pd.DataFrame):
        raise TypeError(f"{shap_key} must be a pandas DataFrame in {source}")
    selected_key = f"selected_features_{selected_split}"
    selected_features = loaded.get(selected_key)
    return ClassificationResult(
        path=source,
        split=selected_split,
        metrics=metrics,
        scores=scores,
        shap_values=shap_values,
        selected_features=selected_features,
    )


def _interpolate_curve(
    x: object,
    y: object,
    grid: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError(
            f"{name} coordinates must be equal-length one-dimensional arrays"
        )
    if x_values.size < 2:
        raise ValueError(f"{name} requires at least two curve coordinates")
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    y_values = y_values[order]
    unique_x, unique_indices = np.unique(x_values, return_index=True)
    unique_y = y_values[unique_indices]
    return np.interp(grid, unique_x, unique_y)


def _aggregate_curve_metrics(
    results: Sequence[ClassificationResult],
    *,
    section: Literal["roc", "pr"],
    grid: np.ndarray,
    interval: tuple[float, float],
) -> dict[str, Any]:
    if section == "roc":
        x_key, y_candidates, score_key = "fpr", ("tpr", "mean_tpr"), "auc"
        lower_key, upper_key = "tprs_lower", "tprs_upper"
    else:
        x_key, y_candidates, score_key = "recall", ("precision", "pr"), "ap"
        lower_key, upper_key = "pr_lower", "pr_upper"

    curves: list[np.ndarray] = []
    scores: list[float] = []
    for result in results:
        metrics = result.metrics[section]
        y_key = next((key for key in y_candidates if key in metrics), None)
        if x_key not in metrics or y_key is None or score_key not in metrics:
            raise KeyError(
                f"Incomplete {section.upper()} metrics in {result.path}; "
                f"required {x_key}, one of {list(y_candidates)}, and {score_key}"
            )
        curves.append(
            _interpolate_curve(
                metrics[x_key],
                metrics[y_key],
                grid,
                name=section.upper(),
            )
        )
        scores.append(float(metrics[score_key]))

    curve_array = np.vstack(curves)
    score_array = np.asarray(scores, dtype=np.float64)
    low_percentile, high_percentile = interval
    output: dict[str, Any] = {
        x_key: grid.copy(),
        y_candidates[0]: curve_array.mean(axis=0),
        lower_key: np.percentile(curve_array, low_percentile, axis=0),
        upper_key: np.percentile(curve_array, high_percentile, axis=0),
        score_key: float(score_array.mean()),
        f"{score_key}_std": float(score_array.std(ddof=1)),
        f"{score_key}_ci_low": float(np.percentile(score_array, low_percentile)),
        f"{score_key}_ci_high": float(np.percentile(score_array, high_percentile)),
        "uncertainty_label": (
            f"{low_percentile:g}â€“{high_percentile:g}% empirical interval "
            "across repeated runs"
        ),
        "n_runs": len(results),
    }
    if section == "roc":
        output["tpr"][0] = 0.0
        output["tpr"][-1] = 1.0
        output[lower_key][0] = 0.0
        output[upper_key][-1] = 1.0
    return output


def _aggregate_classification_metrics(
    results: Sequence[ClassificationResult],
    *,
    interval: tuple[float, float],
) -> dict[str, Any]:
    scalar_keys = (
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
    count_keys = (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "support_negative",
        "support_positive",
    )
    low_percentile, high_percentile = interval
    output: dict[str, Any] = {}
    thresholds = np.asarray(
        [result.metrics["classification"].get("threshold", 0.5) for result in results],
        dtype=np.float64,
    )
    if not np.allclose(thresholds, thresholds[0]):
        raise ValueError("Cannot aggregate runs evaluated at different thresholds")
    output["threshold"] = float(thresholds[0])

    for key in scalar_keys:
        values = np.asarray(
            [result.metrics["classification"][key] for result in results],
            dtype=np.float64,
        )
        output[key] = float(values.mean())
        output[f"{key}_std"] = float(values.std(ddof=1))
        output[f"{key}_ci_low"] = float(np.percentile(values, low_percentile))
        output[f"{key}_ci_high"] = float(np.percentile(values, high_percentile))
    for key in count_keys:
        values = np.asarray(
            [result.metrics["classification"][key] for result in results],
            dtype=np.float64,
        )
        output[key] = float(values.mean())
    output["n_runs"] = len(results)
    output["uncertainty_label"] = (
        f"{low_percentile:g}â€“{high_percentile:g}% empirical interval "
        "across repeated runs"
    )
    return output


def aggregate_result_metrics(
    results: Sequence[ClassificationResult],
    *,
    grid_size: int = 200,
    interval: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, Any]]:
    """Return native single-run metrics or aggregate repeated-run metrics."""
    if not results:
        raise ValueError("At least one result is required")
    splits = {result.split for result in results}
    if len(splits) != 1:
        raise ValueError("Train and test result files cannot be aggregated together")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    low, high = interval
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("interval must contain increasing percentiles within [0, 100]")
    if len(results) == 1:
        return {section: dict(values) for section, values in results[0].metrics.items()}

    grid = np.linspace(0.0, 1.0, grid_size)
    return {
        "roc": _aggregate_curve_metrics(
            results,
            section="roc",
            grid=grid,
            interval=interval,
        ),
        "pr": _aggregate_curve_metrics(
            results,
            section="pr",
            grid=grid,
            interval=interval,
        ),
        "classification": _aggregate_classification_metrics(
            results,
            interval=interval,
        ),
    }


def _encode_binary_target(
    target: pd.Series,
    class_labels: Sequence[str],
) -> pd.Series:
    if len(class_labels) != 2:
        raise ValueError("class_labels must contain exactly two labels")
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) == {0, 1}:
        return numeric.astype(int)
    mapping = {label: code for code, label in enumerate(class_labels)}
    encoded = target.map(mapping)
    if encoded.isna().any():
        invalid = target.loc[encoded.isna()].drop_duplicates().tolist()[:5]
        raise ValueError(
            f"Target values {invalid!r} do not match class_labels={list(class_labels)!r}"
        )
    if set(encoded.astype(int).unique()) != {0, 1}:
        raise ValueError("Validation bootstrap requires both target classes")
    return encoded.astype(int)


def bootstrap_validation_metrics(
    result: ClassificationResult,
    target: pd.Series,
    *,
    class_labels: Sequence[str] = ("Negative", "Positive"),
    n_bootstraps: int = 1000,
    random_state: int = 420,
    grid_size: int = 200,
    interval: tuple[float, float] = (2.5, 97.5),
) -> dict[str, dict[str, Any]]:
    """Add stratified-bootstrap uncertainty to one external validation result."""
    if result.split != "test":
        raise ValueError(
            "Bootstrap uncertainty is intended for a test/validation result"
        )
    if result.scores is None:
        raise ValueError("The validation result does not contain scores_test")
    if n_bootstraps < 2:
        raise ValueError("n_bootstraps must be at least 2")
    missing = result.scores.index.difference(target.index)
    if len(missing):
        raise ValueError(
            f"Target is missing validation samples: {missing.tolist()[:5]}"
        )
    encoded = _encode_binary_target(
        target.loc[result.scores.index],
        class_labels,
    )
    scores = result.scores.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain NaN or infinite values")

    low_percentile, high_percentile = interval
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("interval must contain increasing percentiles within [0, 100]")
    grid = np.linspace(0.0, 1.0, grid_size)
    target_values = encoded.to_numpy(dtype=np.int64, copy=False)
    class_positions = [np.flatnonzero(target_values == code) for code in (0, 1)]
    rng = np.random.default_rng(random_state)
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    precisions: list[np.ndarray] = []
    average_precisions: list[float] = []
    classification_runs: list[dict[str, float | int]] = []
    threshold = float(result.metrics["classification"].get("threshold", 0.5))
    for _ in range(n_bootstraps):
        sampled_positions = np.concatenate(
            [
                rng.choice(positions, size=len(positions), replace=True)
                for positions in class_positions
            ]
        )
        rng.shuffle(sampled_positions)
        sampled_target = target_values[sampled_positions]
        sampled_scores = scores[sampled_positions]
        tpr, auc_value = compute_interp_tpr_auc(
            sampled_target,
            sampled_scores,
            grid,
        )
        precision, ap_value = compute_interp_pr_ap(
            sampled_target,
            sampled_scores,
            grid,
        )
        tprs.append(np.asarray(tpr, dtype=np.float64))
        aucs.append(float(auc_value))
        precisions.append(np.asarray(precision, dtype=np.float64))
        average_precisions.append(float(ap_value))
        classification_runs.append(
            calculate_classification_metrics(
                sampled_target,
                sampled_scores,
                threshold=threshold,
            )
        )

    metrics = {section: dict(values) for section, values in result.metrics.items()}
    tpr_array = np.vstack(tprs)
    auc_array = np.asarray(aucs)
    precision_array = np.vstack(precisions)
    ap_array = np.asarray(average_precisions)
    uncertainty_label = (
        f"{low_percentile:g}â€“{high_percentile:g}% stratified bootstrap interval "
        f"(n={n_bootstraps})"
    )
    metrics["roc"].update(
        {
            "tprs_lower": np.percentile(tpr_array, low_percentile, axis=0),
            "tprs_upper": np.percentile(tpr_array, high_percentile, axis=0),
            "auc_ci_low": float(np.percentile(auc_array, low_percentile)),
            "auc_ci_high": float(np.percentile(auc_array, high_percentile)),
            "uncertainty_label": uncertainty_label,
        }
    )
    metrics["pr"].update(
        {
            "pr_lower": np.percentile(precision_array, low_percentile, axis=0),
            "pr_upper": np.percentile(precision_array, high_percentile, axis=0),
            "ap_ci_low": float(np.percentile(ap_array, low_percentile)),
            "ap_ci_high": float(np.percentile(ap_array, high_percentile)),
            "uncertainty_label": uncertainty_label,
        }
    )
    scalar_keys = (
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
    for key in scalar_keys:
        values = np.asarray([run[key] for run in classification_runs], dtype=np.float64)
        metrics["classification"][f"{key}_ci_low"] = float(
            np.percentile(values, low_percentile)
        )
        metrics["classification"][f"{key}_ci_high"] = float(
            np.percentile(values, high_percentile)
        )
    metrics["classification"]["uncertainty_label"] = uncertainty_label
    return metrics


def aggregate_shap_values(
    results: Sequence[ClassificationResult],
) -> pd.DataFrame | None:
    """Average aligned SHAP matrices, preserving sample and feature labels."""
    frames: list[pd.DataFrame] = [
        frame for result in results if (frame := result.shap_values) is not None
    ]
    if not frames:
        return None
    reference = frames[0]
    if reference.index.has_duplicates or reference.columns.has_duplicates:
        raise ValueError("SHAP matrices must have unique sample and feature labels")
    total = np.zeros(reference.shape, dtype=np.float64)
    for frame in frames:
        if set(frame.index) != set(reference.index) or set(frame.columns) != set(
            reference.columns
        ):
            raise ValueError(
                "Repeated SHAP matrices must contain the same samples and features"
            )
        aligned = frame.loc[reference.index, reference.columns]
        total += aligned.to_numpy(dtype=np.float64, copy=False)
    total /= len(frames)
    return pd.DataFrame(total, index=reference.index, columns=reference.columns)


def plot_result_files(
    result_files: Sequence[str | Path],
    *,
    split: ResultSplit = "auto",
    class_labels: Sequence[str] = ("Negative", "Positive"),
    title: str | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str = "phipml",
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    oligos_metadata: pd.DataFrame | None = None,
    peptide_prefixes: Sequence[str] = ("agilent_", "twist_", "corona2_"),
    max_display: int = 20,
    validation_bootstraps: int = 1000,
    random_state: int = 420,
) -> ResultPlotOutput:
    """Create performance and SHAP summaries from one or repeated result files.

    ``features`` is optional for performance and global SHAP-importance plots,
    but required for a beeswarm because its colors represent observed feature
    values. ``target`` is additionally required for the feature-statistics table.
    """
    if not result_files:
        raise ValueError("result_files cannot be empty")
    results = [load_classification_result(path, split=split) for path in result_files]
    metrics = aggregate_result_metrics(results)
    if (
        len(results) == 1
        and results[0].split == "test"
        and target is not None
        and validation_bootstraps > 0
    ):
        metrics = bootstrap_validation_metrics(
            results[0],
            target,
            class_labels=class_labels,
            n_bootstraps=validation_bootstraps,
            random_state=random_state,
        )
    output_path = Path(output_dir).expanduser() if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    performance_file = (
        output_path / f"{output_prefix}_performance.pdf"
        if output_path is not None
        else None
    )
    performance_figure, _ = plot_performance_summary(
        metrics,
        class_labels=class_labels,
        title=title,
        output_path=performance_file,
    )

    shap_values = aggregate_shap_values(results)
    shap_importance_figure: plt.Figure | None = None
    shap_heatmap_figure: plt.Figure | None = None
    shap_beeswarm_figure: plt.Figure | None = None
    feature_table: pd.DataFrame | None = None
    feature_table_figure: plt.Figure | None = None
    if shap_values is not None:
        shap_bar_file = (
            output_path / f"{output_prefix}_shap_importance.pdf"
            if output_path is not None
            else None
        )
        shap_importance_figure, _ = plot_shap_importance_bar(
            shap_values,
            max_display=max_display,
            output_path=shap_bar_file,
        )
        shap_heatmap_file = (
            output_path / f"{output_prefix}_shap_heatmap.pdf"
            if output_path is not None
            else None
        )
        shap_heatmap_figure, _ = plot_shap_heatmap(
            shap_values,
            target=target,
            max_display=max_display,
            output_path=shap_heatmap_file,
        )

        if features is not None:
            missing_samples = shap_values.index.difference(features.index)
            missing_features = shap_values.columns.difference(features.columns)
            if len(missing_samples) or len(missing_features):
                raise ValueError(
                    "features does not cover the SHAP matrix; "
                    f"missing samples={missing_samples.tolist()[:5]}, "
                    f"missing features={missing_features.tolist()[:5]}"
                )
            aligned_features = features.loc[shap_values.index, shap_values.columns]
            shap_beeswarm_figure, _ = plot_shap_values(
                shap_values.to_numpy(),
                aligned_features,
                max_display=max_display,
                group_tests=list(class_labels),
                filename_label=output_prefix,
                add_binary_legend=False,
                save_fig=output_path is not None,
                figures_dir=output_path or ".",
            )

            if target is not None:
                feature_csv = (
                    output_path / f"{output_prefix}_feature_importance.csv"
                    if output_path is not None
                    else None
                )
                feature_table = build_feature_importance_table(
                    shap_values,
                    aligned_features,
                    target,
                    group_labels=class_labels,
                    oligos_metadata=oligos_metadata,
                    peptide_prefixes=peptide_prefixes,
                    output_csv=feature_csv,
                )
                feature_table_file = (
                    output_path / f"{output_prefix}_feature_table.pdf"
                    if output_path is not None
                    else None
                )
                feature_table_figure, _ = plot_feature_importance_table(
                    feature_table,
                    group_labels=class_labels,
                    max_display=max_display,
                    output_path=feature_table_file,
                )

    return ResultPlotOutput(
        metrics=metrics,
        performance_figure=performance_figure,
        shap_values=shap_values,
        shap_beeswarm_figure=shap_beeswarm_figure,
        shap_importance_figure=shap_importance_figure,
        shap_heatmap_figure=shap_heatmap_figure,
        feature_table=feature_table,
        feature_table_figure=feature_table_figure,
    )