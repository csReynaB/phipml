# ======================
# Standard library
# ======================
import logging
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBClassifier

# ======================
# Global configuration
# ======================
logger = logging.getLogger(__name__)
MetricValue = NDArray[np.float64] | float
RocMetrics = dict[str, MetricValue]
CurveMetrics = dict[str, MetricValue]
ClassificationMetrics = dict[str, float | int]
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
BOOTSTRAP_CLASSIFICATION_METRICS = (
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

# ======================
# Hyper tuning
# ======================


def search_best_model(
    estimator: Any,
    param_grid: dict[str, Any],
    X_train,
    y_train,
    method: str = "bayesian",  # "random", "grid", or "bayesian"
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 420,
    n_jobs: int = -1,
    **kwargs,
) -> Pipeline:
    """
    Tune hyperparameters for a model using one of three methods:
      - RandomizedSearchCV ("random")
      - GridSearchCV ("grid")
      - BayesSearchCV ("bayesian", if scikit-optimize is installed)

    Parameters
    ----------
    estimator : Any
        A scikit-learn-compatible binary classification pipeline.
    param_grid : Dict
        - For 'grid', a dict of parameter lists, e.g. {'param': [1, 2, 3]}.
        - For 'random', a dict of parameter distributions or lists.
        - For 'bayesian', a dict of parameter search spaces (from skopt.space).
    X_train : array-like or DataFrame
        Training feature data.
    y_train : array-like
        Binary training target.
    method : str, default="random"
        Which search method to use: "random", "grid", or "bayesian".
    n_splits : int, default=5
        Number of folds for stratified cross-validation.
    n_iter : int, default=30
        - For 'random', number of draws from param distributions.
        - For 'bayesian', number of parameter settings to sample.
        - Ignored for 'grid'.
    random_state : int, default=420
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    **kwargs :
        Additional keyword arguments passed to the underlying search class.

    Returns
    -------
    best_estimator_ : Any
        The best-fitted estimator from the search.
    """

    # scoring = 'auc'
    scoring = make_scorer(
        average_precision_score,
        response_method="predict_proba",
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    method = method.lower()
    if method == "bayesian":
        # BayesSearchCV from scikit-optimize
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "random":
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "grid":
        # GridSearchCV
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            **kwargs,
        )
    else:
        raise ValueError("method must be 'bayesian', 'random' or 'grid'.")

    # Run the search
    search.fit(X_train, y_train)

    # Return the best model
    best_estimator = search.best_estimator_
    if not isinstance(best_estimator, Pipeline):
        raise TypeError(
            "Hyperparameter search did not return an sklearn Pipeline; "
            f"found {type(best_estimator).__name__}"
        )
    return best_estimator


# ##############################
#           Metrics            #
# ##############################
def calculate_classification_metrics(
    y_true: Iterable[int] | pd.Series,
    positive_class_scores: Iterable[float] | pd.Series,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> ClassificationMetrics:
    """Calculate threshold-dependent metrics for binary classification.

    ROC-AUC and average precision evaluate the continuous scores across all
    possible thresholds. These complementary metrics first convert the
    positive-class scores to predictions using ``score >= threshold``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    target: NDArray[np.int64] = np.asarray(list(y_true), dtype=np.int64)
    scores: NDArray[np.float64] = np.asarray(
        list(positive_class_scores),
        dtype=np.float64,
    )
    if target.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and positive_class_scores must be one-dimensional")
    if target.size != scores.size:
        raise ValueError("y_true and positive_class_scores must have equal length")
    if target.size == 0:
        raise ValueError("Cannot calculate classification metrics for empty inputs")
    if not np.isfinite(scores).all():
        raise ValueError("positive_class_scores contains NaN or infinite values")

    classes = set(np.unique(target).tolist())
    if classes != {0, 1}:
        raise ValueError(
            "Binary classification metrics require targets encoded as 0 and 1; "
            f"found {sorted(classes)}"
        )

    predictions: NDArray[np.int64] = np.asarray(
        scores >= threshold,
        dtype=np.int64,
    )
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    )
    specificity = tn / (tn + fp)
    negative_predictive_value = tn / (tn + fn) if tn + fn else 0.0
    recall = float(recall_score(target, predictions, zero_division=0))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": recall,
        "sensitivity": recall,
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, predictions)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
    }


def _log_classification_metrics(
    context: str,
    metrics: ClassificationMetrics,
) -> None:
    """Log threshold metrics consistently for nested and external validation."""
    logger.info(
        "%s at threshold %.2f: accuracy=%.3f, balanced_accuracy=%.3f, "
        "precision=%.3f, recall/sensitivity=%.3f, specificity=%.3f, "
        "F1=%.3f, MCC=%.3f",
        context,
        metrics["threshold"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["specificity"],
        metrics["f1"],
        metrics["mcc"],
    )
    logger.info(
        "%s confusion matrix: TN=%d, FP=%d, FN=%d, TP=%d",
        context,
        metrics["true_negatives"],
        metrics["false_positives"],
        metrics["false_negatives"],
        metrics["true_positives"],
    )


def _summarise_classification_folds(
    fold_metrics: Sequence[ClassificationMetrics],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return mean and sample SD for threshold metrics across outer folds."""
    if len(fold_metrics) < 2:
        raise ValueError("At least two outer folds are required to summarize metrics")

    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for metric in BOOTSTRAP_CLASSIFICATION_METRICS:
        values = np.asarray(
            [fold[metric] for fold in fold_metrics],
            dtype=np.float64,
        )
        means[metric] = values.mean().item()
        standard_deviations[metric] = values.std(ddof=1).item()
    return means, standard_deviations


def _linear_percentile_1d(
    values: NDArray[np.float64],
    percentile: float,
) -> float:
    """Return NumPy's default linear percentile for one-dimensional values.

    This small typed implementation avoids the scalar-or-array return union in
    ``numpy.percentile`` stubs, which otherwise produces false-positive IDE
    warnings when confidence-interval bounds are indexed.
    """
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered: NDArray[np.float64] = np.sort(values)
    position = (ordered.size - 1) * percentile / 100.0
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    weight = position - lower_index
    lower_value = float(ordered[lower_index])
    upper_value = float(ordered[upper_index])
    return lower_value + weight * (upper_value - lower_value)


def _linear_percentile_columns(
    values: NDArray[np.float64],
    percentile: float,
) -> NDArray[np.float64]:
    """Return the linear percentile independently for every array column."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty two-dimensional array")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered: NDArray[np.float64] = np.sort(values, axis=0)
    position = (ordered.shape[0] - 1) * percentile / 100.0
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    weight = position - lower_index
    result = ordered[lower_index, :] + weight * (
        ordered[upper_index, :] - ordered[lower_index, :]
    )
    return np.asarray(result, dtype=np.float64)


def calculate_mean_std_ci_tpr_auc(
    auc_list: Sequence[float] | NDArray[np.float64],
    tpr_list: Sequence[Sequence[float]] | NDArray[np.float64],
    mean_fpr: Sequence[float] | NDArray[np.float64],
    bootstrap: bool = False,
) -> RocMetrics:
    """Aggregate ROC curves and AUC values across folds or bootstrap runs."""
    aucs: NDArray[np.float64] = np.asarray(
        auc_list,
        dtype=np.float64,
    )
    tprs: NDArray[np.float64] = np.asarray(
        tpr_list,
        dtype=np.float64,
    )
    mean_fpr_array: NDArray[np.float64] = np.asarray(
        mean_fpr,
        dtype=np.float64,
    )

    if aucs.ndim != 1:
        raise ValueError("auc_list must be one-dimensional")
    if tprs.ndim != 2:
        raise ValueError("tpr_list must be a two-dimensional collection")
    if aucs.size != tprs.shape[0]:
        raise ValueError("auc_list and tpr_list must contain the same number of runs")
    if aucs.size < 2:
        raise ValueError(
            "At least two runs are required to calculate a sample standard deviation"
        )
    if tprs.shape[1] != mean_fpr_array.size:
        raise ValueError("Each TPR curve must have the same length as mean_fpr")

    # Aggregate TPR curves.
    mean_tpr: NDArray[np.float64] = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    # Aggregate AUC values.
    auc_mean = aucs.mean().item()
    auc_std = aucs.std(ddof=1).item()

    if bootstrap:
        curve_lower = _linear_percentile_columns(tprs, 2.5)
        curve_upper = _linear_percentile_columns(tprs, 97.5)
        auc_lower = _linear_percentile_1d(aucs, 2.5)
        auc_upper = _linear_percentile_1d(aucs, 97.5)

        return {
            "boot_mean_fpr": mean_fpr_array,
            "boot_mean_tpr": mean_tpr,
            "boot_tprs_upper": curve_upper,
            "boot_tprs_lower": curve_lower,
            "boot_auc_mean": auc_mean,
            "boot_auc_std": auc_std,
            "boot_auc_ci_lower": auc_lower,
            "boot_auc_ci_upper": auc_upper,
        }

    std_tpr: NDArray[np.float64] = tprs.std(axis=0, ddof=1)
    tprs_lower: NDArray[np.float64] = np.maximum(
        mean_tpr - std_tpr,
        0.0,
    )
    tprs_upper: NDArray[np.float64] = np.minimum(
        mean_tpr + std_tpr,
        1.0,
    )

    return {
        "fpr": mean_fpr_array,
        "tpr": mean_tpr,
        "tprs_upper": tprs_upper,
        "tprs_lower": tprs_lower,
        "auc": auc_mean,
        "auc_std": auc_std,
    }


def calculate_mean_std_ci_precision_ap(
    ap_list: Sequence[float] | NDArray[np.float64],
    pr_list: Sequence[Sequence[float]] | NDArray[np.float64],
    mean_recall: Sequence[float] | NDArray[np.float64],
) -> CurveMetrics:
    """Aggregate precision-recall curves and average-precision values."""
    precisions: NDArray[np.float64] = np.asarray(
        pr_list,
        dtype=np.float64,
    )
    ap_array: NDArray[np.float64] = np.asarray(
        ap_list,
        dtype=np.float64,
    )
    mean_recall_array: NDArray[np.float64] = np.asarray(
        mean_recall,
        dtype=np.float64,
    )

    if precisions.ndim != 2:
        raise ValueError("pr_list must be a two-dimensional collection")
    if ap_array.ndim != 1:
        raise ValueError("ap_list must be one-dimensional")
    if precisions.shape[0] != ap_array.size:
        raise ValueError("pr_list and ap_list must contain the same number of runs")
    if precisions.shape[1] != mean_recall_array.size:
        raise ValueError(
            "Each precision curve must have the same length as mean_recall"
        )
    if ap_array.size < 2:
        raise ValueError("At least two runs are required to calculate variability")

    mean_precision: NDArray[np.float64] = np.clip(
        precisions.mean(axis=0),
        0.0,
        1.0,
    )
    std_precision: NDArray[np.float64] = precisions.std(
        axis=0,
        ddof=1,
    )

    prec_lower: NDArray[np.float64] = np.maximum(
        mean_precision - std_precision,
        0.0,
    )
    prec_upper: NDArray[np.float64] = np.minimum(
        mean_precision + std_precision,
        1.0,
    )

    ap_mean = ap_array.mean().item()
    ap_std = ap_array.std(ddof=1).item()

    return {
        "recall": mean_recall_array,
        "precision": mean_precision,
        "precision_upper": prec_upper,
        "precision_lower": prec_lower,
        "ap": ap_mean,
        "ap_std": ap_std,
    }


def compute_interp_tpr_auc(y_true, y_pred_proba, mean_fpr):
    """
    Interpolate an ROC curve onto a common false-positive-rate grid.

    Returns
    -------
    interp_tpr : np.ndarray
        TPR interpolated onto the common FPR grid (for mean AUC curve).
    auc_value : float
        AUC.
    """

    # Compute FPR and TPR
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Interpolate TPR to the common mean FPR grid
    interp_tpr = np.interp(mean_fpr, fpr, tpr)

    # Ensure the curve starts at (0, 0)
    interp_tpr[0] = 0.0

    # Compute AUC
    auc_value = auc(fpr, tpr)

    return interp_tpr, auc_value


def compute_interp_pr_ap(y_true, y_pred_proba, mean_recall):
    """
    Interpolate precision onto a common recall grid and calculate AP.

    Returns
    -------
    interp_precision : np.ndarray
        Precision interpolated onto the common recall grid (for mean PR curve).
    ap_value : float
        Average Precision (standard PR-AUC in ML).
    """

    # Compute precision–recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    # sklearn returns recall from 1 -> 0.
    # Reverse it so recall increases from 0 -> 1 for np.interp.
    recall = recall[::-1]
    precision = precision[::-1]

    # Ensure recall is strictly increasing for interpolation
    recall_unique, idx = np.unique(recall, return_index=True)
    precision_unique = precision[idx]

    # Interpolate precision to the common mean recall grid
    interp_precision = np.interp(mean_recall, recall_unique, precision_unique)

    # Ensure curve starts correctly at (0,1)
    if mean_recall[0] == 0.0:
        interp_precision[0] = 1.0

    # pr_auc = auc(recall, precision) this uses trapezoid
    # Compute Average Precision (AP)
    ap_value = average_precision_score(y_true, y_pred_proba)

    return interp_precision, ap_value


def _compute_metrics_test(
    y_test: Iterable[int] | pd.Series,
    y_pred: Iterable[float] | pd.Series,
    common_grid: Sequence[float] | NDArray[np.float64],
) -> dict[str, MetricValue]:
    """Calculate point estimates for one external validation cohort."""

    grid: NDArray[np.float64] = np.asarray(
        common_grid,
        dtype=np.float64,
    )

    interp_tpr, auc_value = compute_interp_tpr_auc(
        y_test,
        y_pred,
        grid,
    )
    interp_precision, ap_value = compute_interp_pr_ap(
        y_test,
        y_pred,
        grid,
    )

    tpr: NDArray[np.float64] = np.asarray(
        interp_tpr,
        dtype=np.float64,
    )
    precision: NDArray[np.float64] = np.asarray(
        interp_precision,
        dtype=np.float64,
    )
    auc_score: float = float(auc_value)
    average_precision: float = float(ap_value)

    metrics: dict[str, MetricValue] = {
        "fpr": grid,
        "tpr": tpr,
        "auc": auc_score,
        "recall": grid,
        "precision": precision,
        "ap": average_precision,
    }
    return metrics


def _classification_rates_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, float]:
    """Calculate bootstrap-friendly classification rates from a 2x2 table."""
    total = tn + fp + fn + tp
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    negative_predictive_value = tn / (tn + fn) if tn + fn else 0.0
    f1_denominator = 2 * tp + fp + fn
    mcc_denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "f1": (2 * tp / f1_denominator) if f1_denominator else 0.0,
        "mcc": ((tp * tn - fp * fn) / mcc_denominator) if mcc_denominator else 0.0,
    }


def bootstrap_classification_metrics(
    y_true: Iterable[int] | pd.Series,
    positive_class_scores: Iterable[float] | pd.Series,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 420,
    interpolation_grid: Sequence[float] | NDArray[np.float64] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bootstrap uncertainty for all external-validation metrics.

    Samples are resampled as paired ``(target, score)`` observations within
    each target class. The fitted model is intentionally kept fixed: this
    estimates uncertainty caused by the finite external validation cohort,
    not uncertainty from retraining the model.

    The returned dictionary contains pointwise percentile bands for the ROC
    and precision-recall curves, percentile confidence intervals for ROC-AUC
    and average precision, and intervals for threshold-dependent metrics.
    Raw bootstrap draws are not returned, keeping saved joblib files compact.
    """
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    target = np.asarray(list(y_true), dtype=np.int64)
    scores = np.asarray(list(positive_class_scores), dtype=np.float64)
    # This validates dimensions, finite scores, binary coding, and threshold.
    calculate_classification_metrics(
        target,
        scores,
        threshold=threshold,
    )

    if interpolation_grid is None:
        grid: NDArray[np.float64] = np.linspace(0.0, 1.0, 200)
    else:
        grid = np.asarray(interpolation_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("interpolation_grid must be one-dimensional with >=2 points")
    if not np.isfinite(grid).all() or (np.diff(grid) < 0).any():
        raise ValueError("interpolation_grid must be finite and non-decreasing")
    if grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("interpolation_grid values must lie between 0 and 1")

    class_positions: list[NDArray[np.intp]] = [
        np.flatnonzero(target == class_code) for class_code in (0, 1)
    ]
    rng = np.random.default_rng(random_state)
    tpr_samples: NDArray[np.float64] = np.empty(
        (n_resamples, grid.size),
        dtype=np.float64,
    )
    precision_samples: NDArray[np.float64] = np.empty(
        (n_resamples, grid.size),
        dtype=np.float64,
    )
    auc_samples: NDArray[np.float64] = np.empty(n_resamples, dtype=np.float64)
    ap_samples: NDArray[np.float64] = np.empty(n_resamples, dtype=np.float64)

    classification_samples: dict[str, NDArray[np.float64]] = {}

    for metric_name in BOOTSTRAP_CLASSIFICATION_METRICS:
        metric_key: str = metric_name
        metric_values: NDArray[np.float64] = np.empty(
            n_resamples,
            dtype=np.float64,
        )
        classification_samples[metric_key] = metric_values

    for bootstrap_index in range(n_resamples):
        sampled_positions: NDArray[np.intp] = np.concatenate(
            [
                rng.choice(
                    positions,
                    size=positions.size,
                    replace=True,
                )
                for positions in class_positions
            ]
        )
        rng.shuffle(sampled_positions)
        sampled_target = target[sampled_positions]
        sampled_scores = scores[sampled_positions]

        sampled_tpr, sampled_auc = compute_interp_tpr_auc(
            sampled_target,
            sampled_scores,
            grid,
        )
        sampled_precision, sampled_ap = compute_interp_pr_ap(
            sampled_target,
            sampled_scores,
            grid,
        )
        tpr_samples[bootstrap_index] = sampled_tpr
        precision_samples[bootstrap_index] = sampled_precision
        auc_samples[bootstrap_index] = sampled_auc
        ap_samples[bootstrap_index] = sampled_ap

        sampled_predictions = sampled_scores >= threshold
        sampled_negative = sampled_target == 0
        sampled_positive = ~sampled_negative
        rates = _classification_rates_from_counts(
            tn=int(np.count_nonzero(sampled_negative & ~sampled_predictions)),
            fp=int(np.count_nonzero(sampled_negative & sampled_predictions)),
            fn=int(np.count_nonzero(sampled_positive & ~sampled_predictions)),
            tp=int(np.count_nonzero(sampled_positive & sampled_predictions)),
        )
        for metric, value in rates.items():
            classification_samples[metric][bootstrap_index] = value

    alpha = (1.0 - confidence_level) / 2.0
    low_percentile = 100.0 * alpha
    high_percentile = 100.0 * (1.0 - alpha)
    interval_label = (
        f"{confidence_level:.0%} stratified bootstrap interval " f"(n={n_resamples})"
    )

    tpr_lower = _linear_percentile_columns(tpr_samples, low_percentile)
    tpr_upper = _linear_percentile_columns(tpr_samples, high_percentile)
    auc_lower = _linear_percentile_1d(auc_samples, low_percentile)
    auc_upper = _linear_percentile_1d(auc_samples, high_percentile)
    precision_lower = _linear_percentile_columns(
        precision_samples,
        low_percentile,
    )
    precision_upper = _linear_percentile_columns(
        precision_samples,
        high_percentile,
    )
    ap_lower = _linear_percentile_1d(ap_samples, low_percentile)
    ap_upper = _linear_percentile_1d(ap_samples, high_percentile)
    classification_intervals: dict[str, Any] = {"uncertainty_label": interval_label}
    for metric, values in classification_samples.items():
        classification_intervals[f"{metric}_ci_lower"] = _linear_percentile_1d(
            values,
            low_percentile,
        )
        classification_intervals[f"{metric}_ci_upper"] = _linear_percentile_1d(
            values,
            high_percentile,
        )

    return {
        "roc": {
            "bootstrap_mean_tpr": tpr_samples.mean(axis=0),
            "tprs_lower": tpr_lower,
            "tprs_upper": tpr_upper,
            "auc_bootstrap_mean": auc_samples.mean().item(),
            "auc_bootstrap_std": auc_samples.std(ddof=1).item(),
            "auc_ci_lower": auc_lower,
            "auc_ci_upper": auc_upper,
            "uncertainty_label": interval_label,
        },
        "pr": {
            "bootstrap_mean_precision": precision_samples.mean(axis=0),
            "precision_lower": precision_lower,
            "precision_upper": precision_upper,
            "ap_bootstrap_mean": ap_samples.mean().item(),
            "ap_bootstrap_std": ap_samples.std(ddof=1).item(),
            "ap_ci_lower": ap_lower,
            "ap_ci_upper": ap_upper,
            "uncertainty_label": interval_label,
        },
        "classification": classification_intervals,
        "uncertainty": {
            "method": "class-stratified paired bootstrap",
            "resampling_unit": "external-validation samples",
            "model_refitted": False,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "random_state": random_state,
            "curve_interval": "pointwise percentile",
        },
    }


#
# def bootstrap_auc(
#     mean_fpr: Sequence[float] | NDArray[np.float64] | None = None,
#     estimator: Any = None,
#     X: Any = None,
#     y_true: Iterable[int] | pd.Series | None = None,
#     y_pred: Iterable[float] | pd.Series | None = None,
#     n_bootstraps: int = 200,
#     random_state: int = 420,
# ) -> RocMetrics:
#     """Compatibility wrapper returning the historical ROC bootstrap keys.
#
#     New code should use :func:`bootstrap_classification_metrics`, which obtains
#     ROC, PR, AP, and threshold-dependent intervals from the same resamples.
#     """
#     if y_true is None:
#         raise ValueError("y_true is required for bootstrapping")
#     if y_pred is None:
#         if estimator is None or X is None:
#             raise ValueError("Provide y_pred, or provide both estimator and X")
#         y_pred = estimator.predict_proba(X)[:, 1]
#     grid = (
#         np.linspace(0.0, 1.0, 200)
#         if mean_fpr is None
#         else np.asarray(mean_fpr, dtype=np.float64)
#     )
#     bootstrapped = bootstrap_classification_metrics(
#         y_true,
#         y_pred,
#         n_resamples=n_bootstraps,
#         random_state=random_state,
#         interpolation_grid=grid,
#     )
#     roc_metrics = bootstrapped["roc"]
#     return {
#         "boot_mean_fpr": grid,
#         "boot_mean_tpr": np.asarray(
#             roc_metrics["bootstrap_mean_tpr"],
#             dtype=np.float64,
#         ),
#         "boot_tprs_lower": np.asarray(
#             roc_metrics["tprs_lower"],
#             dtype=np.float64,
#         ),
#         "boot_tprs_upper": np.asarray(
#             roc_metrics["tprs_upper"],
#             dtype=np.float64,
#         ),
#         "boot_auc_mean": float(roc_metrics["auc_bootstrap_mean"]),
#         "boot_auc_std": float(roc_metrics["auc_bootstrap_std"]),
#         "boot_auc_ci_lower": float(roc_metrics["auc_ci_lower"]),
#         "boot_auc_ci_upper": float(roc_metrics["auc_ci_upper"]),
#     }


# ##############################
#      Build Pipeline          #
# ##############################


def is_binary_numeric_column(series: pd.Series) -> bool:
    """Return whether a numeric column contains only 0/1, ignoring missing data."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = pd.Series(series.dropna().unique())
    return not values.empty and bool(values.isin([0, 1]).all())


def split_peptide_extra_columns(
    X: pd.DataFrame,
    peptide_prefixes: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Separate peptide features from all additional clinical variables."""
    prefixes = tuple(peptide_prefixes or ("agilent_", "corona2_", "twist_"))
    peptide_columns = [
        str(column) for column in X.columns if str(column).startswith(prefixes)
    ]
    peptide_set = set(peptide_columns)
    extra_columns = [
        str(column) for column in X.columns if str(column) not in peptide_set
    ]
    return peptide_columns, extra_columns


def split_extra_columns_for_imputation(
    X: pd.DataFrame,
    extra_columns: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split extras into binary numeric, continuous numeric, and non-numeric."""
    binary: list[str] = []
    continuous: list[str] = []
    non_numeric: list[str] = []
    for column in extra_columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            non_numeric.append(column)
        elif is_binary_numeric_column(X[column]):
            binary.append(column)
        else:
            continuous.append(column)
    return binary, continuous, non_numeric


def make_pipeline(
    X: pd.DataFrame,
    peptide_columns: Sequence[str],
    extra_columns: Sequence[str],
    estimator: Any,
    random_state: int,
    *,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """Build peptide feature selection and clinical-feature preprocessing."""
    transformers: list[tuple[str, Any, Sequence[str]]] = []

    if peptide_columns:
        peptide_pipeline = Pipeline(
            [
                ("variance_removal", VarianceThreshold(threshold=0.0)),
                (
                    "feature_selection",
                    SelectFromModel(
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=0.5,
                            C=1.0,
                            max_iter=10_000,
                            random_state=random_state,
                        ),
                        threshold=1e-5,
                    ),
                ),
            ]
        )
        transformers.append(("peptides", peptide_pipeline, list(peptide_columns)))

    if extra_columns:
        binary, continuous, non_numeric = split_extra_columns_for_imputation(
            X,
            extra_columns,
        )
        if non_numeric:
            raise ValueError(
                "These additional clinical features are non-numeric. One-hot "
                f"encode them before building the classification pipeline: {non_numeric}"
            )
        if binary:
            transformers.append(("binary_extra", "passthrough", binary))
        if continuous:
            continuous_transformer: Any = "passthrough"
            if impute_extra_numeric:
                continuous_transformer = SimpleImputer(
                    strategy=extra_numeric_impute_strategy
                )
            transformers.append(
                (
                    "continuous_extra",
                    continuous_transformer,
                    continuous,
                )
            )

    if not transformers:
        raise ValueError("No peptide or additional features were available")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )


def build_pipeline(
    X_train: pd.DataFrame,
    model_type: str = "xgboost",
    random_state: int = 420,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """Build a binary classifier for peptide and numeric clinical features."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if X_train.columns.has_duplicates:
        duplicated = X_train.columns[X_train.columns.duplicated()].tolist()[:10]
        raise ValueError(f"Training data has duplicate feature names: {duplicated}")

    peptide_columns, extra_columns = split_peptide_extra_columns(
        X_train,
        peptide_prefixes,
    )
    logger.info("Peptide features: %d", len(peptide_columns))
    logger.info("Additional features: %s", extra_columns)

    if model_type == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
    elif model_type == "random-forest":
        estimator = RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError("model_type must be 'xgboost' or 'random-forest'")

    return make_pipeline(
        X=X_train,
        peptide_columns=peptide_columns,
        extra_columns=extra_columns,
        estimator=estimator,
        random_state=random_state,
        impute_extra_numeric=impute_extra_numeric,
        extra_numeric_impute_strategy=extra_numeric_impute_strategy,
    )


def _build_and_fit_pipeline(
    pipeline: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, Any] | None,
    n_splits: int,
    n_iter: int,
    model_type: str,
    random_state: int,
    n_jobs: int,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """Build or clone, validate its search space, and fit one pipeline."""
    if pipeline is None:
        candidate = build_pipeline(
            X_train,
            model_type=model_type,
            random_state=random_state,
            peptide_prefixes=peptide_prefixes,
            impute_extra_numeric=impute_extra_numeric,
            extra_numeric_impute_strategy=extra_numeric_impute_strategy,
        )
    else:
        candidate = clone(pipeline)

    if not isinstance(candidate, Pipeline):
        raise TypeError(
            "build_pipeline/clone did not produce an sklearn Pipeline; "
            f"found {type(candidate).__name__}"
        )
    if param_grid is None:
        candidate.fit(X_train, y_train)
        return candidate

    search_grid = dict(param_grid)
    valid_parameters = set(candidate.get_params())
    unknown = set(search_grid) - valid_parameters

    inactive_peptide_parameters: set[str] = set()
    preprocessor = candidate.named_steps.get("preprocessor")
    if unknown and isinstance(preprocessor, ColumnTransformer):
        transformer_names = {
            transformer_name for transformer_name, _, _ in preprocessor.transformers
        }
        inactive_peptide_parameters = {
            parameter
            for parameter in unknown
            if parameter.startswith("preprocessor__peptides__")
            and "peptides" not in transformer_names
        }

    if inactive_peptide_parameters:
        logger.info(
            "Ignoring %d peptide search parameters because this run has no "
            "peptide features",
            len(inactive_peptide_parameters),
        )
        search_grid = {
            parameter: space
            for parameter, space in search_grid.items()
            if parameter not in inactive_peptide_parameters
        }
        unknown -= inactive_peptide_parameters

    if unknown:
        raise ValueError(
            "Parameter-grid entries do not match the classification pipeline: "
            f"{sorted(unknown)}"
        )
    if not search_grid:
        logger.info("No active search parameters; fitting the pipeline directly")
        candidate.fit(X_train, y_train)
        return candidate

    return search_best_model(
        candidate,
        search_grid,
        X_train,
        y_train,
        method="bayesian",
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _transform_features(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps.get("preprocessor")
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("Pipeline step 'preprocessor' must be a ColumnTransformer")

    transformed = preprocessor.transform(X)
    if isinstance(transformed, pd.DataFrame):
        return transformed

    return pd.DataFrame(
        transformed,
        index=X.index,
        columns=preprocessor.get_feature_names_out(),
    )


def _positive_class_shap_values(
    shap_result: Any,
    *,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Normalize SHAP output from XGBoost and RandomForest to a 2D class-1 array."""
    if isinstance(shap_result, list):
        if len(shap_result) < 2:
            values = np.asarray(shap_result[0])
        else:
            values = np.asarray(shap_result[1])
    else:
        values = np.asarray(shap_result)

    if values.ndim == 3:
        if values.shape[:2] == (n_samples, n_features):
            values = values[:, :, -1]
        elif values.shape[1:] == (n_samples, n_features):
            values = values[-1, :, :]
    if values.shape != (n_samples, n_features):
        raise ValueError(
            "Unexpected SHAP value shape "
            f"{values.shape}; expected ({n_samples}, {n_features})"
        )
    return values


def _compute_shap_frame(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    transformed = _transform_features(fitted_pipeline, X)
    estimator = fitted_pipeline.named_steps["estimator"]
    explainer = shap.TreeExplainer(estimator)
    values = _positive_class_shap_values(
        explainer.shap_values(transformed),
        n_samples=transformed.shape[0],
        n_features=transformed.shape[1],
    )
    return (
        pd.DataFrame(
            values,
            index=transformed.index,
            columns=transformed.columns,
        ),
        transformed.columns.astype(str).tolist(),
    )


def nested_cv_single(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline | None = None,
    param_grid: dict[str, Any] | None = None,
    n_splits: int = 5,
    n_iter: int = 30,
    model_type: str = "xgboost",
    random_state: int = 420,
    n_jobs: int = -1,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    Pipeline,
    np.ndarray,
    float,
    np.ndarray,
    float,
    list[str],
]:
    """Fit one outer fold and return predictions, metrics, SHAP, and its model."""
    X_fit = X_train.iloc[train_indices]
    X_validation = X_train.iloc[validation_indices]
    y_fit = y_train.iloc[train_indices]
    y_validation = y_train.iloc[validation_indices]

    best_estimator = _build_and_fit_pipeline(
        pipeline,
        X_fit,
        y_fit,
        param_grid,
        n_splits,
        n_iter,
        model_type,
        random_state,
        n_jobs,
        peptide_prefixes,
        impute_extra_numeric,
        extra_numeric_impute_strategy,
    )

    scores = best_estimator.predict_proba(X_validation)[:, 1]
    interpolation_grid = np.linspace(0, 1, 200)
    interpolated_tpr, auc_value = compute_interp_tpr_auc(
        y_validation,
        scores,
        interpolation_grid,
    )
    interpolated_precision, average_precision = compute_interp_pr_ap(
        y_validation,
        scores,
        interpolation_grid,
    )
    shap_frame, selected_features = _compute_shap_frame(
        best_estimator,
        X_validation,
    )

    return (
        validation_indices,
        scores,
        shap_frame,
        best_estimator,
        interpolated_tpr,
        float(auc_value),
        interpolated_precision,
        float(average_precision),
        selected_features,
    )


def nested_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline | None = None,
    param_grid: dict[str, Any] | None = None,
    n_splits: int = 10,
    n_splits_inner: int = 5,
    n_iter: int = 30,
    model_type: str = "xgboost",
    random_state: int = 420,
    n_jobs: int = 1,
    n_jobs_inner: int = -1,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> tuple[
    list[Pipeline],
    pd.DataFrame,
    pd.Series,
    list[np.ndarray],
    dict[str, dict[str, Any]],
    list[list[str]],
]:
    """Run nested stratified CV and preserve outer-fold variability.

    ROC and precision-recall curves are summarized as the outer-fold mean
    with a pointwise ±1 sample-SD band.

    Top-level classification metrics are calculated from the pooled
    out-of-fold predictions. Per-fold classification metrics and their
    mean and standard deviation are stored separately.

    These variability summaries describe differences between outer folds;
    they are not formal confidence intervals.
    """
    if y_train.nunique() != 2:
        raise ValueError("Binary classification requires exactly two target classes")
    if not 0.0 <= classification_threshold <= 1.0:
        raise ValueError("classification_threshold must be between 0 and 1")

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            fit_indices,
            validation_indices,
            X_train,
            y_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=n_splits_inner,
            n_iter=n_iter,
            model_type=model_type,
            random_state=random_state,
            n_jobs=n_jobs_inner,
            peptide_prefixes=peptide_prefixes,
            impute_extra_numeric=impute_extra_numeric,
            extra_numeric_impute_strategy=extra_numeric_impute_strategy,
        )
        for fit_indices, validation_indices in outer_cv.split(
            X_train,
            y_train,
        )
    )

    models: list[Pipeline] = []
    validation_folds: list[np.ndarray] = []
    selected_feature_sets: list[list[str]] = []

    interpolated_tprs: list[np.ndarray] = []
    auc_values: list[float] = []

    interpolated_precisions: list[np.ndarray] = []
    average_precisions: list[float] = []

    # Keep the narrowly typed metrics separate from the richer records
    # containing fold identifiers and sample counts.
    fold_classification_metrics: list[ClassificationMetrics] = []
    fold_classification_records: list[dict[str, Any]] = []

    scores = pd.Series(
        np.nan,
        index=X_train.index,
        name="Score",
        dtype=float,
    )

    shap_values = pd.DataFrame(
        0.0,
        index=X_train.index,
        columns=X_train.columns,
    )

    for fold_number, (
        validation_indices,
        fold_scores,
        fold_shap,
        model,
        interpolated_tpr,
        auc_value,
        interpolated_precision,
        average_precision,
        selected_features,
    ) in enumerate(fold_results, start=1):
        models.append(model)
        validation_folds.append(validation_indices)
        selected_feature_sets.append(selected_features)

        scores.iloc[validation_indices] = fold_scores

        shap_values.loc[
            fold_shap.index,
            fold_shap.columns,
        ] = fold_shap

        interpolated_tprs.append(interpolated_tpr)
        auc_values.append(float(auc_value))

        interpolated_precisions.append(interpolated_precision)
        average_precisions.append(float(average_precision))

        fold_metrics = calculate_classification_metrics(
            y_train.iloc[validation_indices],
            fold_scores,
            threshold=classification_threshold,
        )
        fold_classification_metrics.append(fold_metrics)

        fold_record: dict[str, Any] = {
            **fold_metrics,
            "fold": fold_number,
            "n_samples": len(validation_indices),
        }
        fold_classification_records.append(fold_record)

    # Every training sample should receive exactly one outer-fold
    # validation prediction.
    missing_mask = scores.isna()
    if missing_mask.any():
        missing_count = int(np.count_nonzero(missing_mask.to_numpy()))
        missing_samples = scores.index[missing_mask].tolist()[:10]

        raise RuntimeError(
            "Nested CV did not produce valid out-of-fold scores for "
            f"{missing_count} samples. First missing samples: "
            f"{missing_samples}"
        )

    interpolation_grid = np.linspace(0.0, 1.0, 200)

    roc_metrics = calculate_mean_std_ci_tpr_auc(
        auc_values,
        interpolated_tprs,
        interpolation_grid,
    )

    pr_metrics = calculate_mean_std_ci_precision_ap(
        average_precisions,
        interpolated_precisions,
        interpolation_grid,
    )

    uncertainty_label = f"Outer-fold ±1 SD (n={len(fold_results)})"

    # These summary dictionaries intentionally contain heterogeneous
    # values: arrays, scalars, lists, integers, and explanatory strings.
    roc_summary: dict[str, Any] = {
        **roc_metrics,
        "auc_folds": [float(value) for value in auc_values],
        "n_outer_folds": len(fold_results),
        "summary_scope": "mean across outer validation folds",
        "uncertainty_label": uncertainty_label,
    }

    positive_count = int(np.count_nonzero(y_train.to_numpy() == 1))
    positive_prevalence = positive_count / len(y_train)

    pr_summary: dict[str, Any] = {
        **pr_metrics,
        "ap_folds": [float(value) for value in average_precisions],
        "n_outer_folds": len(fold_results),
        "positive_prevalence": positive_prevalence,
        "summary_scope": "mean across outer validation folds",
        "uncertainty_label": uncertainty_label,
    }

    # These are calculated from all pooled out-of-fold predictions.
    pooled_classification = calculate_classification_metrics(
        y_train,
        scores,
        threshold=classification_threshold,
    )

    fold_mean, fold_std = _summarise_classification_folds(fold_classification_metrics)

    classification_metrics: dict[str, Any] = {
        **pooled_classification,
        "fold_metrics": fold_classification_records,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "n_outer_folds": len(fold_results),
        "summary_scope": "pooled out-of-fold predictions",
        "threshold_source": ("pre-specified before outer-CV evaluation"),
        "threshold_selected_on_evaluation_data": False,
        "uncertainty_label": ("Outer-fold variability stored as fold_mean/fold_std"),
    }

    metrics: dict[str, dict[str, Any]] = {
        "roc": roc_summary,
        "pr": pr_summary,
        "classification": classification_metrics,
        "uncertainty": {
            "method": "outer-fold variability",
            "n_outer_folds": len(fold_results),
            "model_refitted_per_fold": True,
            "inner_hyperparameter_tuning": param_grid is not None,
            "curve_interval": "pointwise mean ±1 sample SD",
            "formal_confidence_interval": False,
        },
    }

    logger.info(
        "Mean ROC-AUC across folds: %.3f ± %.3f",
        roc_summary["auc"],
        roc_summary["auc_std"],
    )
    logger.info(
        "Mean Average Precision (AP) across folds: %.3f ± %.3f",
        pr_summary["ap"],
        pr_summary["ap_std"],
    )

    _log_classification_metrics(
        "Out-of-fold classification metrics",
        pooled_classification,
    )

    logger.info(
        "Outer-fold classification variability: "
        "accuracy=%.3f ± %.3f, "
        "F1=%.3f ± %.3f, "
        "MCC=%.3f ± %.3f",
        fold_mean["accuracy"],
        fold_std["accuracy"],
        fold_mean["f1"],
        fold_std["f1"],
        fold_mean["mcc"],
        fold_std["mcc"],
    )

    return (
        models,
        shap_values,
        scores,
        validation_folds,
        metrics,
        selected_feature_sets,
    )


def align_external_to_pipeline(
    X_external: pd.DataFrame,
    fitted_pipeline: Pipeline,
    peptide_prefixes: Sequence[str] = (
        "agilent_",
        "twist_",
        "corona2_",
    ),
    fill_missing_peptides_with_zero: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Align external raw inputs to the feature schema learned during training."""
    if not isinstance(X_external, pd.DataFrame):
        raise TypeError("X_external must be a pandas DataFrame")
    if X_external.columns.has_duplicates:
        duplicated = X_external.columns[X_external.columns.duplicated()].tolist()[:10]
        raise ValueError(f"External data has duplicate feature names: {duplicated}")

    expected_raw: Any = getattr(
        fitted_pipeline,
        "feature_names_in_",
        None,
    )
    if expected_raw is None:
        preprocessor = fitted_pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            expected_raw = getattr(
                preprocessor,
                "feature_names_in_",
                None,
            )
    if expected_raw is None:
        raise ValueError(
            "The fitted pipeline does not expose feature_names_in_. It must "
            "be fitted with a pandas DataFrame before external alignment."
        )
    if isinstance(expected_raw, (str, bytes)) or not isinstance(
        expected_raw,
        Iterable,
    ):
        raise TypeError(
            "feature_names_in_ must be an iterable of feature names; "
            f"found {type(expected_raw).__name__}"
        )

    expected = [str(column) for column in expected_raw]
    expected_set = set(expected)
    external_columns = set(X_external.columns)
    missing = [column for column in expected if column not in external_columns]
    extra = [str(column) for column in X_external.columns if column not in expected_set]

    prefixes = tuple(peptide_prefixes)
    missing_peptides = [column for column in missing if column.startswith(prefixes)]
    missing_non_peptides = [
        column for column in missing if not column.startswith(prefixes)
    ]
    if missing_non_peptides:
        raise ValueError(
            "External data is missing required non-peptide clinical features; "
            f"these cannot be zero-filled safely: {missing_non_peptides[:20]}"
        )
    if missing_peptides and not fill_missing_peptides_with_zero:
        raise ValueError(
            "External data is missing peptide features and zero filling is "
            f"disabled: {missing_peptides[:20]}"
        )

    aligned = X_external.copy()
    if missing_peptides:
        aligned = pd.concat(
            [
                aligned,
                pd.DataFrame(
                    0,
                    index=aligned.index,
                    columns=missing_peptides,
                ),
            ],
            axis=1,
        )
    aligned = aligned.loc[:, expected].copy()

    report = {
        "missing_features": missing,
        "missing_peptides": missing_peptides,
        "missing_non_peptides": missing_non_peptides,
        "extra_features": extra,
    }
    logger.info(
        "External alignment: expected=%d, missing peptides=%d, extras=%d",
        len(expected),
        len(missing_peptides),
        len(extra),
    )
    return aligned, report


ValidationResult: TypeAlias = tuple[
    Pipeline,
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
    list[str],
]
ValidationResultWithReport: TypeAlias = tuple[
    Pipeline,
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
    list[str],
    dict[str, list[str]],
]


def train_and_validate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    pipeline: Pipeline | None = None,
    param_grid: dict[str, Any] | None = None,
    best_estimator: Pipeline | None = None,
    n_splits: int = 10,
    n_iter: int = 30,
    model_type: str = "xgboost",
    random_state: int = 420,
    n_jobs: int = -1,
    get_only_model: bool = False,
    peptide_prefixes: Sequence[str] = (
        "agilent_",
        "twist_",
        "corona2_",
    ),
    fill_missing_peptides_with_zero: bool = True,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
    return_feature_report: bool = False,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    bootstrap_validation: bool = True,
    bootstrap_n_resamples: int = 1000,
    bootstrap_confidence_level: float = 0.95,
) -> Pipeline | ValidationResult | ValidationResultWithReport:
    """Fit the full training cohort or evaluate a fitted model externally.

    External validation uses one fixed fitted model. By default, paired target
    and score observations are resampled within each class to save uncertainty
    for ROC, PR, AUC, AP, and threshold-dependent classification metrics.
    """
    if best_estimator is None:
        best_estimator = _build_and_fit_pipeline(
            pipeline,
            X_train,
            y_train,
            param_grid,
            n_splits,
            n_iter,
            model_type,
            random_state,
            n_jobs,
            peptide_prefixes,
            impute_extra_numeric,
            extra_numeric_impute_strategy,
        )

    if get_only_model:
        return best_estimator
    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test are required for validation")

    aligned_test, feature_report = align_external_to_pipeline(
        X_test,
        best_estimator,
        peptide_prefixes=peptide_prefixes,
        fill_missing_peptides_with_zero=fill_missing_peptides_with_zero,
    )
    if y_test.index.has_duplicates:
        duplicated = y_test.index[y_test.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Validation target has duplicate sample IDs: {duplicated}")
    missing_targets = aligned_test.index.difference(y_test.index).tolist()
    if missing_targets:
        raise ValueError(
            f"Validation targets are missing for samples: {missing_targets[:10]}"
        )
    y_test = y_test.loc[aligned_test.index]
    if y_test.nunique() != 2:
        raise ValueError("Validation ROC/PR metrics require exactly two target classes")
    scores_array = best_estimator.predict_proba(aligned_test)[:, 1]
    scores = pd.Series(
        scores_array,
        index=aligned_test.index,
        name="Score",
    )
    interpolation_grid = np.linspace(0.0, 1.0, 200)
    flat_metrics = _compute_metrics_test(
        y_test,
        scores_array,
        interpolation_grid,
    )
    classification_metrics = calculate_classification_metrics(
        y_test,
        scores_array,
        threshold=classification_threshold,
    )

    metrics: dict[str, dict[str, Any]] = {
        "roc": {
            "fpr": flat_metrics["fpr"],
            "tpr": flat_metrics["tpr"],
            "auc": flat_metrics["auc"],
        },
        "pr": {
            "recall": flat_metrics["recall"],
            "precision": flat_metrics["precision"],
            "ap": flat_metrics["ap"],
            "positive_prevalence": float(y_test.mean()),
        },
        "classification": dict(classification_metrics),
    }
    metrics["classification"].update(
        {
            "threshold_source": "pre-specified before external validation",
            "threshold_selected_on_evaluation_data": False,
        }
    )
    if bootstrap_validation:
        bootstrap_metrics = bootstrap_classification_metrics(
            y_test,
            scores_array,
            threshold=classification_threshold,
            n_resamples=bootstrap_n_resamples,
            confidence_level=bootstrap_confidence_level,
            random_state=random_state,
            interpolation_grid=interpolation_grid,
        )
        for section in ("roc", "pr", "classification"):
            metrics[section].update(bootstrap_metrics[section])
        metrics["uncertainty"] = bootstrap_metrics["uncertainty"]

    selected_shap, selected_features = _compute_shap_frame(
        best_estimator,
        aligned_test,
    )
    shap_values = pd.DataFrame(
        0.0,
        index=aligned_test.index,
        columns=aligned_test.columns,
    )
    shap_values.loc[
        selected_shap.index,
        selected_shap.columns,
    ] = selected_shap

    if bootstrap_validation:
        logger.info(
            "ROC-AUC in validation set: %.3f (%.0f%% bootstrap CI %.3f-%.3f)",
            metrics["roc"]["auc"],
            100.0 * bootstrap_confidence_level,
            metrics["roc"]["auc_ci_lower"],
            metrics["roc"]["auc_ci_upper"],
        )
        logger.info(
            "Average Precision (AP) in validation set: %.3f "
            "(%.0f%% bootstrap CI %.3f-%.3f)",
            metrics["pr"]["ap"],
            100.0 * bootstrap_confidence_level,
            metrics["pr"]["ap_ci_lower"],
            metrics["pr"]["ap_ci_upper"],
        )
    else:
        logger.info("ROC-AUC in validation set: %.3f", metrics["roc"]["auc"])
        logger.info(
            "Average Precision (AP) in validation set: %.3f",
            metrics["pr"]["ap"],
        )
    _log_classification_metrics(
        "Validation classification metrics",
        classification_metrics,
    )

    result: ValidationResult = (
        best_estimator,
        shap_values,
        scores,
        metrics,
        selected_features,
    )
    if return_feature_report:
        result_with_report: ValidationResultWithReport = (
            best_estimator,
            shap_values,
            scores,
            metrics,
            selected_features,
            feature_report,
        )
        return result_with_report
    return result
