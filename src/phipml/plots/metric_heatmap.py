"""Generic cohort-by-cohort heatmaps for saved classification metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from phipml.plots.result_summary import (
    load_classification_result,
    normalise_output_formats,
)


@dataclass(frozen=True)
class MetricMatrixSummary:
    """Point estimates and repeated-run variability for a metric matrix."""

    mean: pd.DataFrame
    standard_deviation: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    n_runs: pd.DataFrame
    metric: str


def _nested_metric(metrics: dict, metric: str) -> float:
    """Extract a dotted metric such as ``roc.auc`` or ``classification.f1``."""
    section, separator, key = metric.partition(".")
    if not separator or section not in metrics or key not in metrics[section]:
        raise KeyError(
            f"Metric {metric!r} was not found; use section.key, for example "
            "'roc.auc', 'pr.ap', or 'classification.balanced_accuracy'"
        )
    value = float(metrics[section][key])
    if not np.isfinite(value):
        raise ValueError(f"Metric {metric!r} must be finite")
    return value


def _native_metric_interval(
    metrics: dict,
    metric: str,
) -> tuple[float, float] | None:
    section, _, key = metric.partition(".")
    values = metrics[section]
    lower = values.get(f"{key}_ci_lower", values.get(f"{key}_ci_low"))
    upper = values.get(f"{key}_ci_upper", values.get(f"{key}_ci_high"))
    if lower is None or upper is None:
        return None
    return float(lower), float(upper)


def _native_metric_standard_deviation(
    metrics: dict,
    metric: str,
) -> float | None:
    """Return saved outer-fold SD for one nested-CV artifact, when available."""
    section, _, key = metric.partition(".")
    values = metrics[section]
    standard_deviation = values.get(f"{key}_std")
    if standard_deviation is None and section == "classification":
        fold_standard_deviation = values.get("fold_std")
        if isinstance(fold_standard_deviation, Mapping):
            standard_deviation = fold_standard_deviation.get(key)
    if standard_deviation is None:
        return None
    value = float(standard_deviation)
    return value if np.isfinite(value) and value >= 0.0 else None


def build_metric_matrix(
    records: pd.DataFrame,
    *,
    metric: str = "roc.auc",
    training_column: str = "training",
    validation_column: str = "validation",
    path_column: str = "path",
    split_column: str | None = "split",
    order: Sequence[str] | None = None,
    training_order: Sequence[str] | None = None,
    validation_order: Sequence[str] | None = None,
    interval: tuple[float, float] = (2.5, 97.5),
) -> MetricMatrixSummary:
    """Summarize result files listed in a tidy train/validation manifest.

    Multiple files for one cell are summarized across run-level point
    estimates. A single validation file retains its native bootstrap interval
    when available; a single nested-CV file retains its outer-fold SD.

    By default the output is rectangular: rows contain observed validation
    cohorts and columns contain observed training cohorts. ``order`` retains
    the historical square-matrix behavior. Use ``training_order`` and
    ``validation_order`` to control the two axes independently.
    """
    required = {training_column, validation_column, path_column}
    missing = sorted(required - set(records.columns))
    if missing:
        raise KeyError(f"Metric manifest is missing columns: {missing}")
    low_percentile, high_percentile = interval
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("interval must contain increasing values within [0, 100]")

    if order is not None and (
        training_order is not None or validation_order is not None
    ):
        raise ValueError(
            "order cannot be combined with training_order or validation_order"
        )

    observations: dict[
        tuple[str, str],
        list[tuple[float, tuple[float, float] | None, float | None]],
    ] = defaultdict(list)
    for row in records.to_dict(orient="records"):
        training = str(row[training_column])
        validation = str(row[validation_column])
        requested_split: Literal["auto", "train", "test"] = "auto"
        if split_column and split_column in row and pd.notna(row[split_column]):
            raw_split = str(row[split_column]).lower()
            if raw_split not in {"auto", "train", "test"}:
                raise ValueError(f"Invalid split {raw_split!r} in metric manifest")
            requested_split = raw_split  # type: ignore[assignment]
        result = load_classification_result(row[path_column], split=requested_split)
        observations[(training, validation)].append(
            (
                _nested_metric(result.metrics, metric),
                _native_metric_interval(result.metrics, metric),
                _native_metric_standard_deviation(result.metrics, metric),
            )
        )

    if not observations:
        raise ValueError("Metric manifest contains no result files")

    observed_training = list(dict.fromkeys(training for training, _ in observations))
    observed_validation = list(
        dict.fromkeys(validation for _, validation in observations)
    )
    if order is not None:
        training_labels = [str(label) for label in order]
        validation_labels = training_labels.copy()
    else:
        training_labels = (
            [str(label) for label in training_order]
            if training_order is not None
            else observed_training
        )
        validation_labels = (
            [str(label) for label in validation_order]
            if validation_order is not None
            else observed_validation
        )
    unknown_training = sorted(set(observed_training) - set(training_labels))
    unknown_validation = sorted(set(observed_validation) - set(validation_labels))
    if unknown_training or unknown_validation:
        raise ValueError(
            "Heatmap orders do not contain all manifest labels; "
            f"missing training={unknown_training}, "
            f"missing validation={unknown_validation}"
        )

    def empty() -> pd.DataFrame:
        return pd.DataFrame(
            np.nan,
            index=validation_labels,
            columns=training_labels,
            dtype=float,
        )

    mean, std, lower, upper, n_runs = empty(), empty(), empty(), empty(), empty()
    for (training, validation), entries in observations.items():
        values = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        mean.loc[validation, training] = values.mean()
        n_runs.loc[validation, training] = len(values)
        if len(values) > 1:
            std.loc[validation, training] = values.std(ddof=1)
            lower.loc[validation, training] = np.percentile(values, low_percentile)
            upper.loc[validation, training] = np.percentile(values, high_percentile)
        elif entries[0][1] is not None:
            lower.loc[validation, training], upper.loc[validation, training] = entries[
                0
            ][1]
        elif entries[0][2] is not None:
            std.loc[validation, training] = entries[0][2]
    return MetricMatrixSummary(
        mean=mean,
        standard_deviation=std,
        lower=lower,
        upper=upper,
        n_runs=n_runs,
        metric=metric,
    )


def plot_metric_heatmap(
    summary: MetricMatrixSummary,
    *,
    title: str | None = None,
    palette: str = "inferno",
    vmin: float = 0.5,
    vmax: float = 1.0,
    annotate_uncertainty: bool = True,
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    output_formats: Sequence[str] | None = None,
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a metric matrix with mean ± SD or native-CI annotations.

    When ``output_formats`` is omitted, ``output_path`` is saved exactly as
    supplied for backward compatibility. When formats are provided, the
    suffix of ``output_path`` is treated as an optional format hint and the
    figure is written once per requested format using the same filename stem.
    """
    if not vmin < vmax:
        raise ValueError("vmin must be smaller than vmax")
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    width = max(4.8, 1.25 * len(summary.mean.columns) + 2.5)
    height = max(4.2, 1.05 * len(summary.mean.index) + 2.4)
    fig, ax = plt.subplots(figsize=figsize or (width, height))
    cmap = plt.get_cmap(palette).copy()
    cmap.set_bad("#E6E6E6")
    label = {
        "roc.auc": "ROC-AUC",
        "pr.ap": "Average precision",
    }.get(summary.metric, summary.metric.replace("_", " ").title())
    sns.heatmap(
        summary.mean,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": label, "shrink": 0.8},
        annot=False,
    )
    for row, validation in enumerate(summary.mean.index):
        for column, training in enumerate(summary.mean.columns):
            value = summary.mean.loc[validation, training]
            if not np.isfinite(value):
                continue
            annotation = f"{value:.2f}"
            count = summary.n_runs.loc[validation, training]
            sd = summary.standard_deviation.loc[validation, training]
            low = summary.lower.loc[validation, training]
            high = summary.upper.loc[validation, training]
            if annotate_uncertainty and np.isfinite(sd):
                if count > 1:
                    annotation += f"\n±{sd:.2f} ({int(count)} runs)"
                else:
                    annotation += f"\n±{sd:.2f}\n(outer-fold SD)"
            elif annotate_uncertainty and np.isfinite(low) and np.isfinite(high):
                annotation += f"\n95% CI [{low:.2f}, {high:.2f}]"
            normalized = np.clip((float(value) - vmin) / (vmax - vmin), 0.0, 1.0)
            cell_color = cmap(normalized)
            luminance = (
                0.2126 * cell_color[0] + 0.7152 * cell_color[1] + 0.0722 * cell_color[2]
            )
            ax.text(
                column + 0.5,
                row + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=9,
                color="black" if luminance > 0.55 else "white",
            )
    ax.set_xlabel("Training cohort")
    ax.set_ylabel("Validation cohort")
    ax.set_title(title or f"{label} across training and validation cohorts")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if output_formats is None:
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=dpi)
        else:
            formats = normalise_output_formats(output_formats)
            known_suffixes = {".pdf", ".svg", ".png"}
            stem = (
                path.with_suffix("") if path.suffix.lower() in known_suffixes else path
            )
            for output_format in formats:
                destination = stem.parent / f"{stem.name}.{output_format}"
                save_kwargs: dict[str, object] = {
                    "format": output_format,
                    "bbox_inches": "tight",
                    "facecolor": "white",
                }
                if output_format != "svg":
                    save_kwargs["dpi"] = dpi
                fig.savefig(destination, **save_kwargs)
    return fig, ax
