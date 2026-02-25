# ======================
# Standard library
# ======================
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from phipml.io.data_handler import (
    Config,
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)
from phipml.plots.helpers import (
    generate_feature_importance_table,
    plot_roc_summary,
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
