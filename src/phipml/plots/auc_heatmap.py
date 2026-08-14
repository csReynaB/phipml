"""Legacy AUC heatmap utilities.

This module retains the original AUC-specific helpers used by older notebooks
and by the legacy ``phipml.cli.auc_heatmap`` command.  New code that plots any
saved scalar metric (ROC-AUC, average precision, F1, balanced accuracy, and so
on) should import from :mod:`phipml.plots.metric_heatmap` instead.

The generic names are re-exported at the end of this module so existing imports
continue to work while sharing the single implementation in
``metric_heatmap.py``.
"""

import glob
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Third-party
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from phipml.plots.result_summary import load_classification_result


def _extract_auc(d: dict, split: str) -> float:
    """
    Extract AUC value from different possible metric structures.

    Supported structures:
        d['roc_metrics_train']['auc']
        d['metrics_train']['roc']['auc']
    """

    # Case 1: roc_metrics_train / roc_metrics_test
    key1 = f"roc_metrics_{split}"
    if key1 in d and "auc" in d[key1]:
        return d[key1]["auc"]

    # Case 2: metrics_train['roc']['auc']
    key2 = f"metrics_{split}"
    if key2 in d and "roc" in d[key2] and "auc" in d[key2]["roc"]:
        return d[key2]["roc"]["auc"]

    raise KeyError(
        f"Could not find AUC for split '{split}'. " f"Available keys: {list(d.keys())}"
    )


def load_auc(fn):
    name = os.path.basename(fn).split("_")
    d = joblib.load(fn)

    if name[0] == "nested":
        auc = _extract_auc(d, split="train")
        return name[2], name[2], auc

    if name[0] == "validation":
        auc = _extract_auc(d, split="test")
        return name[2], name[3], auc

    return None


def collect_cohort_files(parent_dirs, subdir):
    cohort_paths = {}
    for parent in parent_dirs:
        full_path = os.path.join(parent, subdir) if subdir else parent
        if not os.path.exists(full_path):
            continue
        cohorts = [
            d
            for d in os.listdir(full_path)
            if os.path.isdir(os.path.join(full_path, d))
        ]
        for cohort in cohorts:
            print(cohort)
            path = os.path.join(full_path, cohort)
            cohort_label = f"{os.path.basename(parent)}:{cohort}"
            cohort_paths[cohort_label] = path
    return cohort_paths


def sort_cohorts_by_structure(cohort_labels, parent_dirs, cohort_order):
    ordered = []
    for parent in parent_dirs:
        group = [
            label
            for label in cohort_labels
            if label.startswith(f"{os.path.basename(parent)}:")
        ]
        group = sorted(
            group,
            key=lambda x: (
                cohort_order.index(x.split(":")[1])
                if x.split(":")[1] in cohort_order
                else 999
            ),
        )
        ordered.extend(group)
    return ordered


def format_label(label, sizes, subtract_map=None):
    size = sizes[label]
    base = label.split(":")[1]
    parent = label.split(":")[0]
    group = parent.split("_")[0]  # e.g., "Controls_HCC" ->  "Controls"

    if subtract_map and group in subtract_map:
        size -= subtract_map[group]

    return f"{base}\n(n={size})"


def add_suffix_first_line(s, suffix):
    parts = s.split("\n", 1)
    if len(parts) == 1:
        return s + suffix
    return parts[0] + suffix + "\n" + parts[1]


def append_extra_n(label, extra_value):
    """
    Appends a second (n=extra_value) to the label after the existing (n=XX).
    """
    match = re.search(r"\(n=\d+\)", label)
    if match:
        return label + f"  (n={extra_value})"
    return label


def add_to_n(label, add_value):
    """
        Takes a label like 'Cirrhosis\n(n=72)' and adds add_value to 72.
        Returns updated label string.
    n"""
    match = re.search(r"\(n=(\d+)\)", label)
    if match:
        n_val = int(match.group(1))
        new_val = n_val + add_value
        return label.replace(f"(n={n_val})", f"(n={new_val})")
    return label


def heatmap_aucs(
    parent_dirs,
    subdir,
    cohort_order,
    title,
    outname,
    palette,
    object_filename,
    subtract_sizes=None,
):
    cohort_paths = collect_cohort_files(parent_dirs, subdir)
    allowed_labels = set(cohort_paths.keys())
    print(allowed_labels)
    # cohorts = list(cohort_paths.keys())

    # 1) Gather all joblib files and assign to cohort pairs
    aucs = defaultdict(list)
    sizes = {}

    # Preload train sizes
    for cohort_lbl, cohort_path in cohort_paths.items():
        for fn in glob.glob(os.path.join(cohort_path, "nested_*.joblib")):
            d = joblib.load(fn)
            sizes[cohort_lbl] = len(d["scores_train"])
            break

    # Preload test sizes
    for cohort_lbl, cohort_path in cohort_paths.items():
        cohort_name = cohort_lbl.split(":")[1]
        for fn in glob.glob(os.path.join(cohort_path, "validation_*.joblib")):
            parts = os.path.basename(fn).split("_")
            test_cohort = os.path.splitext(parts[3])[0] if len(parts) >= 4 else ""
            if test_cohort == cohort_name:
                d = joblib.load(fn)
                # test sizes no longer used
                # sizes[cohort_lbl]["test"] = len(d['scores_test'])
                break

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {}
        for train_lbl, train_path in cohort_paths.items():
            for fn in glob.glob(os.path.join(train_path, "*.joblib")):
                futures[exe.submit(load_auc, fn)] = (train_lbl, fn)

        for fut in as_completed(futures):
            train_lbl, fn = futures[fut]
            result = fut.result()
            if result:
                tr, te, auc = result
                test_lbl = f"{train_lbl.split(':')[0]}:{te}"

                if test_lbl not in allowed_labels:
                    continue
                aucs[(train_lbl, test_lbl)].append(auc)

    # 2) Prepare all combinations
    all_labels = sorted(set([lbl for pair in aucs.keys() for lbl in pair]))
    ordered_labels = sort_cohorts_by_structure(all_labels, parent_dirs, cohort_order)

    df_med = pd.DataFrame(index=ordered_labels, columns=ordered_labels)
    df_q1 = df_med.copy()
    df_q3 = df_med.copy()
    df_mean = df_med.copy()
    for (tr, te), vals in aucs.items():
        q1, m, q3 = np.percentile(vals, [25, 50, 75])
        df_med.loc[te, tr] = m
        df_q1.loc[te, tr] = q1
        df_q3.loc[te, tr] = q3
        df_mean.loc[te, tr] = np.mean(vals)
    df_med = df_med.astype(float)
    df_q1 = df_q1.astype(float)
    df_q3 = df_q3.astype(float)
    df_mean = df_mean.astype(float)

    # Reverse y-axis for bottom-left to top-right diagonal
    df_med = df_med.reindex(index=df_med.index[::-1])
    df_q1 = df_q1.reindex(index=df_q1.index[::-1])
    df_q3 = df_q3.reindex(index=df_q3.index[::-1])
    df_mean = df_mean.reindex(index=df_mean.index[::-1])

    # Save
    with open(f"{object_filename}.pkl", "wb") as f:
        pickle.dump(
            {"df_med": df_med, "df_q1": df_q1, "df_q3": df_q3, "df_mean": df_mean}, f
        )

    # 3) Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    cmap = plt.get_cmap(palette)
    cmap.set_bad(color="lightgray")
    low_color = cmap(0.0)
    cmap.set_under(low_color)

    heatmap = sns.heatmap(
        df_mean,
        ax=ax,
        cmap=cmap,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Mean AUC", "shrink": 0.8, "pad": 0.005},
        linewidths=0.5,
        linecolor="white",
        annot=False,
    )

    n = df_mean.shape[0]
    for k in range(n):
        # draw a 1x1 rectangle around the anti-diagonal cell
        ax.add_patch(
            plt.Rectangle(
                (n - 1 - k, k), 1, 1, fill=False, ec="black", lw=2.5, clip_on=False
            )
        )

    # Change font size of colorbar label and ticks
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Mean AUC", fontsize=12)  # Change label font size

    split_indices = [3, 4, 7]  # tile indices in data coords
    n = df_mean.shape[0]
    for idx in split_indices:
        # Vertical line: bottom axis → intersection
        y_corner = n - idx
        ax.plot(
            [idx, idx],
            [n, y_corner],
            color="black",
            linewidth=1.8,
            clip_on=True,
            solid_capstyle="butt",
        )
        # Horizontal line: left axis → intersection
        ax.plot(
            [0, idx],
            [y_corner, y_corner],
            color="black",
            linewidth=1.8,
            clip_on=True,
            solid_capstyle="butt",
        )

    for i, te in enumerate(df_mean.index):
        for j, tr in enumerate(df_mean.columns):
            m = df_mean.loc[te, tr]
            if np.isfinite(m):
                q1 = df_q1.loc[te, tr]
                q3 = df_q3.loc[te, tr]
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{m:.2f}",
                    ha="center",
                    va="center",
                    color="white" if m > 0.745 else "black",
                    fontsize=14,
                    # , fontweight='bold'
                )

    # add specific colors to same groups
    diag_indices_bottom = [7, 6, 3, 0]
    edge_colors = {
        7: "#66a61e",  # green
        6: "#bc80bd",  # purple
        3: "#d95f02",  # orange
        0: "#cb7060",  # red
    }
    # draw colored borders for specified diagonal tiles
    for k in diag_indices_bottom:
        ax.add_patch(
            plt.Rectangle(
                (n - 1 - k, k),
                1,
                1,
                fill=False,
                ec=edge_colors[k],
                lw=5,
                clip_on=False,
                zorder=10,
            )
        )

    xticklabels = [
        format_label(label, sizes, subtract_sizes) for label in df_mean.columns
    ]
    yticklabels = [
        format_label(label, sizes, subtract_sizes) for label in df_mean.index
    ]

    ax.set_xticklabels(xticklabels, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(yticklabels, rotation=0)

    ax.set_title(title, fontsize=13)

    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Training set", fontsize=13, labelpad=10)
    ax.set_ylabel("Test set", fontsize=13, labelpad=10)

    plt.tight_layout()
    plt.savefig(outname, bbox_inches="tight", dpi=600)

    return 0


# =====================================================================
# Manifest-driven metric heatmaps
# =====================================================================


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


def _native_metric_interval(metrics: dict, metric: str) -> tuple[float, float] | None:
    section, _, key = metric.partition(".")
    values = metrics[section]
    lower = values.get(f"{key}_ci_lower", values.get(f"{key}_ci_low"))
    upper = values.get(f"{key}_ci_upper", values.get(f"{key}_ci_high"))
    if lower is None or upper is None:
        return None
    return float(lower), float(upper)


def build_metric_matrix(
    records: pd.DataFrame,
    *,
    metric: str = "roc.auc",
    training_column: str = "training",
    validation_column: str = "validation",
    path_column: str = "path",
    split_column: str | None = "split",
    order: list[str] | None = None,
    interval: tuple[float, float] = (2.5, 97.5),
) -> MetricMatrixSummary:
    """Summarize result files listed in a tidy train/validation manifest.

    Multiple files for one cell are summarized across run-level point
    estimates. A single validation file retains its native bootstrap interval
    when available; a single nested-CV file retains its point estimate only.
    """
    required = {training_column, validation_column, path_column}
    missing = sorted(required - set(records.columns))
    if missing:
        raise KeyError(f"Metric manifest is missing columns: {missing}")
    low_percentile, high_percentile = interval
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("interval must contain increasing values within [0, 100]")

    observations: dict[
        tuple[str, str], list[tuple[float, tuple[float, float] | None]]
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
            )
        )

    if not observations:
        raise ValueError("Metric manifest contains no result files")
    labels = order or sorted({label for pair in observations for label in pair})
    unknown = sorted({label for pair in observations for label in pair} - set(labels))
    if unknown:
        raise ValueError(f"order does not contain manifest labels: {unknown}")

    def empty() -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

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
    palette: str = "YlGnBu",
    vmin: float = 0.5,
    vmax: float = 1.0,
    annotate_uncertainty: bool = True,
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a metric matrix with mean ± SD or native-CI annotations."""
    size = max(6.0, 1.05 * len(summary.mean.columns))
    fig, ax = plt.subplots(figsize=figsize or (size, size))
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
    midpoint = (vmin + vmax) / 2.0
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
                annotation += f"\n±{sd:.2f} (n={int(count)})"
            elif annotate_uncertainty and np.isfinite(low) and np.isfinite(high):
                annotation += f"\n[{low:.2f}, {high:.2f}]"
            ax.text(
                column + 0.5,
                row + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > midpoint else "black",
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
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=600)
    return fig, ax


# The first generic metric-heatmap implementation lived in this historically
# AUC-named module.  Keep those import paths working, but make the accurately
# named module authoritative for all new code.
from phipml.plots.metric_heatmap import (  # noqa: E402,F401
    MetricMatrixSummary,
    build_metric_matrix,
    plot_metric_heatmap,
)
