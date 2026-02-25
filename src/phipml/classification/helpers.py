# ======================
# Standard library
# ======================
import logging
from typing import Any, Dict, Optional

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    make_scorer,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from skopt import BayesSearchCV
from xgboost import XGBClassifier

# ======================
# Global configuration
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_config(transform_output="pandas")


# ======================
# Hyper tuning
# ======================


def search_best_model(
    estimator: Any,
    param_grid: Dict,
    X_train,
    y_train,
    method: str = "bayesian",  # "random", "grid", or "bayesian"
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 420,
    n_jobs: int = -1,
    **kwargs,
) -> Any:
    """
    Tune hyperparameters for a model using one of three methods:
      - RandomizedSearchCV ("random")
      - GridSearchCV ("grid")
      - BayesSearchCV ("bayesian", if scikit-optimize is installed)

    Parameters
    ----------
    estimator : Any
        A scikit-learn style survival estimator.
    param_grid : Dict
        - For 'grid', a dict of parameter lists, e.g. {'param': [1, 2, 3]}.
        - For 'random', a dict of parameter distributions or lists.
        - For 'bayesian', a dict of parameter search spaces (from skopt.space).
    X_train : array-like or DataFrame
        Training feature data.
    y_train : array-like or structured array
        Training survival target (time + event).
    method : str, default="random"
        Which search method to use: "random", "grid", or "bayesian".
    n_splits : int, default=5
        Number of folds for StratifiedKFoldSurv cross-validation.
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
    scoring = make_scorer(average_precision_score, needs_proba=True)
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
    return search.best_estimator_


# ##############################
#           Metrics            #
# ##############################


def calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_fpr, bootstrap=False):
    """
    Calculate mean FPR, mean TPR, and std TPR for ROC curves.
    """
    # Aggregate TPRs
    tprs = np.array(tpr_list)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0  # Ensure curve ends at (1, 1)

    std_tpr = tprs.std(axis=0, ddof=1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    # Aggregate AUCs
    aucs = np.array(auc_list)
    auc_mean = aucs.mean()
    auc_std = aucs.std(ddof=1)

    if bootstrap:
        # se = auc_std / np.sqrt(len(aucs))
        auc_ci_lower = np.percentile(aucs, 2.5)
        auc_ci_upper = np.percentile(aucs, 97.5)
        roc_metrics = {
            "boot_mean_fpr": mean_fpr,
            "boot_mean_tpr": mean_tpr,
            #'boot_std_tpr': std_tpr,
            "boot_tprs_upper": tprs_upper,
            "boot_tprs_lower": tprs_lower,
            "boot_auc_mean": auc_mean,
            "boot_auc_std": auc_std,
            "boot_auc_ci_lower": auc_ci_lower,
            "boot_auc_ci_upper": auc_ci_upper,
        }
        return roc_metrics
    else:
        # t_value = t.ppf(0.975, len(aucs) - 1)
        # se = auc_std / np.sqrt(len(aucs))
        # auc_ci_lower = np.maximum(auc_mean - t_value * se, 0)  # 95% CI lower
        # auc_ci_upper = np.minimum(auc_mean + t_value * se, 1)  # 95% CI upper
        roc_metrics = {
            "fpr": mean_fpr,
            "tpr": mean_tpr,
            #'std_tpr': std_tpr,
            "tprs_upper": tprs_upper,
            "tprs_lower": tprs_lower,
            "auc": auc_mean,
            "auc_std": auc_std,
            #'auc_ci_lower': auc_ci_lower,
            #'auc_ci_upper': auc_ci_upper
        }

        return roc_metrics


def calculate_mean_std_ci_precision_ap(ap_list, pr_list, mean_recall):
    """
    Calculate mean recall, mean precision, and variability for PR curves.
    """

    precisions = np.array(pr_list)
    mean_precision = precisions.mean(axis=0)
    mean_precision = np.clip(mean_precision, 0, 1)

    std_precision = precisions.std(axis=0, ddof=1)
    prec_lower = np.maximum(mean_precision - std_precision, 0)
    prec_upper = np.minimum(mean_precision + std_precision, 1)

    ap_array = np.array(ap_list)
    ap_mean = ap_array.mean()
    ap_std = ap_array.std(ddof=1)

    pr_metrics = {
        "recall": mean_recall,
        "pr": mean_precision,
        "pr_upper": prec_upper,
        "pr_lower": prec_lower,
        "ap": ap_mean,
        "ap_std": ap_std,
    }

    return pr_metrics


def bootstrap_auc(
    mean_fpr=None,
    estimator=None,
    X=None,
    y_true=None,
    y_pred=None,
    n_bootstraps=200,
    random_state=420,
):
    tpr_bootstraps = []
    auc_bootstraps = []

    if mean_fpr is None:
        mean_fpr = np.linspace(0, 1, 200)  # Define default mean_fpr here

    for i in range(n_bootstraps):
        if estimator is not None and X is not None and y_true is not None:
            X_resampled, y_resampled = resample(
                X, y_true, stratify=y_true, random_state=random_state + i
            )
            y_pred_resampled = estimator.predict_proba(X_resampled)[:, 1]
        elif y_pred is not None and y_true is not None:
            y_resampled, y_pred_resampled = resample(
                y_true, y_pred, stratify=y_true, random_state=random_state + i
            )
        else:
            raise ValueError(
                "Missing arguments. Estimator 'estimator', features 'X' and their true target 'y_true' must be provided for bootstrapping auc for test data."
                "For loocv, true target 'y_true' and predictions 'y_pred' must be provided."
            )

        interp_tpr, auc_value = compute_interp_tpr_auc(
            y_resampled, y_pred_resampled, mean_fpr
        )
        tpr_bootstraps.append(interp_tpr)
        auc_bootstraps.append(auc_value)

    roc_metrics = calculate_mean_std_ci_tpr_auc(
        auc_bootstraps, tpr_bootstraps, mean_fpr, bootstrap=True
    )

    return roc_metrics


def compute_interp_tpr_auc(y_true, y_pred_proba, mean_fpr):
    """
    PR-curve analogue of ROC interpolation.

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
    PR-curve analogue of ROC interpolation.

    Returns
    -------
    interp_precision : np.ndarray
        Precision interpolated onto the common recall grid (for mean PR curve).
    ap_value : float
        Average Precision (standard PR-AUC in ML).
    """

    # Compute precision–recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    # Ensure recall is strictly increasing for interpolation
    recall_unique, idx = np.unique(recall, return_index=True)
    precision_unique = precision[idx]

    # Interpolate precision to the common mean recall grid
    interp_precision = np.interp(mean_recall, recall_unique, precision_unique)

    # Ensure curve starts correctly at (0,1)
    if mean_recall[0] == 0.0:
        interp_precision[0] = 1.0

    # pr_auc = auc(recall, precision) this uses trapezoid
    # Compute PR-AUC using Average Precision (ML standard)
    ap_value = average_precision_score(y_true, y_pred_proba)

    return interp_precision, ap_value


######TEST############


def _compute_metrics_test(y_test, y_pred, mean_ip):
    """Compute metrics for the test set."""
    interp_tpr, auc_value = compute_interp_tpr_auc(y_test, y_pred, mean_ip)
    interp_pr, ap_value = compute_interp_pr_ap(y_test, y_pred, mean_ip)
    metrics = {
        "fpr": mean_ip,
        "tpr": interp_tpr,
        "auc": auc_value,
        "recall": mean_ip,
        "pr": interp_pr,
        "ap": ap_value,
    }

    return metrics


def _compute_roc_metrics_test(
    estimator, X_test, y_test, predicted_probs_test, random_state=420
):
    """Compute ROC metrics for the test set."""
    fpr, tpr, _ = roc_curve(y_test, predicted_probs_test)

    roc_metrics = bootstrap_auc(
        estimator=estimator, X=X_test, y_true=y_test, random_state=random_state
    )
    roc_metrics.update({"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "auc_std": None})

    return roc_metrics


# ##############################
#      Build Pipeline          #
# ##############################


def make_pipeline(peptide_cols, demog_cols, estimator, random_state):
    """
    Build a pipeline which:
      - on peptide_cols: does variance threshold + SelectFromModel(elastic net regression)
      - on demog_cols : just passes them through untouched
      - then fits whatever `estimator` you give it
    """

    transformers = []
    if peptide_cols:
        # fix_mi = partial(mutual_info_classif, random_state=random_state)
        enet_selector = SelectFromModel(
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,  # initial value; can be tuned in BayesSearchCV
                solver="saga",
                C=1.0,  # initial regularization; can be tuned
                max_iter=10000,
                random_state=random_state,
            ),
            threshold=1e-5,  # minimal coefficient to keep a feature
        )

        transformers.append(
            (
                "peptides",
                Pipeline(
                    [
                        # ("prevalence_filter", PrevalenceSelector(min_pct=0.02, max_pct=0.98)),
                        ("variance_removal", VarianceThreshold(threshold=0.0)),
                        (
                            "feature_selection",
                            enet_selector,
                        ),  # SelectPercentile(fix_mi, percentile=20)),
                    ]
                ),
                peptide_cols,
            )
        )
    if demog_cols:
        transformers.append(("demographics", "passthrough", demog_cols))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",  # drop anything not in peptide_cols or demog_cols
        verbose_feature_names_out=False,  # <— disable automatic "peptides__…" prefixes
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    return pipe


def build_pipeline(X_train, model_type="xgboost", random_state=420, all_demog=None):
    """
    Create a pipeline using demographic and peptide columns for XGBoost or RandomForest.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data with features (including 'Age' and 'Sex').
    model_type : str
        One of {'xgboost', 'random_forest'}.
    random_state : int
        Seed for reproducibility.
    all_demog : list of extra features

    Returns
    -------
    pipeline : sklearn.Pipeline
        Preprocessing + model pipeline.
    """

    if all_demog is None:
        all_demog = {"Sex", "Age"}

    # Column split
    peptide_cols = [c for c in X_train.columns if c not in all_demog]
    demog_cols = [c for c in X_train.columns if c in all_demog]

    # Define estimator
    if model_type == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=1,
        )
    elif model_type == "random-forest":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=1)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Build pipeline
    return make_pipeline(
        peptide_cols=peptide_cols,
        demog_cols=demog_cols,
        estimator=estimator,
        random_state=random_state,
    )


#####################
#     Fit model     #
#####################


def _build_and_fit_pipeline(
    pipeline,
    X_train,
    y_train,
    param_grid,
    n_splits,
    n_iter,
    model_type,
    random_state,
    n_jobs,
):
    """
    Helper function to build pipeline and perform hyperparameter tuning.

    Parameters
    ----------
    pipeline : Pipeline or None
        Existing pipeline or None to build default
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    param_grid : dict or None
        Hyperparameter grid for tuning
    n_splits : int
        Number of CV splits for tuning
    n_iter : int
        Number of iterations for Bayesian optimization
    model_type : str
        Type of ML model: xgboost or random-forest
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    best_estimator : Pipeline
        Fitted pipeline
    """
    if pipeline is None:
        pipeline = build_pipeline(
            X_train, model_type=model_type, random_state=random_state
        )

    # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        valid_params = set(pipeline.get_params().keys())
        # keep only those entries whose key is in valid_params
        param_grid = {k: v for k, v in param_grid.items() if k in valid_params}
        # Use search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_model(
            pipeline,
            param_grid,
            X_train,
            y_train,
            method="bayesian",
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train, y_train)

    return best_estimator


######################
#   Nested models    #
######################


def nested_cv_single(
    train_idx,
    valid_idx,
    X_train,
    y_train,
    pipeline=None,
    param_grid=None,
    n_splits=5,
    n_iter=30,
    model_type="xgboost",
    random_state=420,
    n_jobs=-1,
):

    mean_ip = np.linspace(0, 1, 200)

    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    best_estimator = _build_and_fit_pipeline(
        pipeline,
        X_train_fold,
        y_train_fold,
        param_grid,
        n_splits,
        n_iter,
        model_type,
        random_state,
        n_jobs,
    )

    # Predict scores on the validation fold
    scores_fold = best_estimator.predict_proba(X_valid_fold)[:, 1]
    interp_tpr, auc_value = compute_interp_tpr_auc(y_valid_fold, scores_fold, mean_ip)
    interp_pr, ap_value = compute_interp_pr_ap(y_valid_fold, scores_fold, mean_ip)

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
            logger.info(f"shape valid fold:{X_valid_fold.shape}")
        except Exception as e:
            logger.error(f"Error transforming validation data in fold: {e}")
            return None

    # Compute SHAP values using the estimator (last step)
    explainer = shap.TreeExplainer(best_estimator[-1])
    shap_values_fold = explainer.shap_values(X_valid_fold)

    if model_type == "random-forest":
        # Binary classification, select SHAP values for class 1
        shap_values_fold = shap_values_fold[:, :, 1]
    shap_values_fold_df = pd.DataFrame(
        shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns
    )

    return (
        valid_idx,
        scores_fold,
        shap_values_fold_df,
        best_estimator,
        interp_tpr,
        auc_value,
        interp_pr,
        ap_value,
    )


def nested_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Optional[Pipeline] = None,
    param_grid: Optional[Dict] = None,
    n_splits: int = 10,
    n_splits_inner: int = 5,
    n_iter: int = 30,
    model_type: str = "xgboost",
    random_state: int = 420,
    n_jobs: int = 1,
    n_jobs_inner: int = -1,
):
    """
    Perform nested cross-validation to tune hyperparameters and feature selection,
    and aggregate SHAP values and risk scores for each outer fold.

    For each outer fold:
      - Split data into training and validation sets.
      - Run hyperparameter tuning (BayesSearchCV) on the inner folds (if param_grid is provided)
        to select the best estimator.
      - Use the best estimator to predict risk scores on the validation set.
      - Compute SHAP values on the validation set. Note: because the pipeline's
        feature selection step may select a different subset of features per fold,
        the returned SHAP values DataFrame may have different columns.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Classes for training samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_splits_inner : int, default 5
        Number of inner CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    model_type : str, default xgboost
        Estimator model [xgboost | random-forest]
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    n_jobs_inner : int, default 1
        Number of parallel jobs to run in the inner CV (default is 1).
    Returns
    -------
    model_list : List[Pipeline]
        List of best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    scores : pd.Series
        Series of predicted scores from all folds, indexed by sample.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    metrics : List[float]
        List of dictionaries containing ROC and PR metrics for each outer fold.
    """
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    # Run the outer folds in parallel if n_jobs > 1.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx,
            valid_idx,
            X_train,
            y_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=n_splits_inner,
            n_iter=n_iter,
            model_type=model_type,
            random_state=random_state,
            n_jobs=n_jobs_inner,
        )
        for train_idx, valid_idx in outer_cv.split(X_train, y_train)
    )

    # Filter out any folds that returned None (e.g., due to transformation errors).
    fold_results = [result for result in fold_results if result is not None]

    # Initialize master containers.
    model_list = []
    validation_indices = []
    tpr_list, auc_list, pr_list, ap_list = [], [], [], []

    mean_ip = np.linspace(0, 1, 200)  # Define default mean_fpr here
    scores = pd.Series(index=X_train.index, name="score")

    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)

    # Aggregate results from each fold.
    i = 0
    for fold_result in fold_results:
        (
            valid_idx,
            fold_scores,
            fold_shap_df,
            model,
            interp_tpr,
            auc_value,
            interp_pr,
            ap_value,
        ) = fold_result

        model_list.append(model)
        validation_indices.append(valid_idx)

        # Assign scores for the validation fold.
        scores.iloc[valid_idx] = fold_scores

        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df
        del fold_shap_df

        # ROC-AUC
        tpr_list.append(interp_tpr)
        auc_list.append(auc_value)

        # PR-AUC
        pr_list.append(interp_pr)
        ap_list.append(ap_value)

        i = i + 1

    # roc_metrics = bootstrap_auc(mean_fpr=mean_fpr, y_true=y_train, y_pred=scores.values, n_bootstraps = 200, random_state=420)
    # kfold_metrics = calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_ip, bootstrap=False)
    # roc_metrics.update(kfold_metrics)

    roc_metrics = calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_ip)
    pr_metrics = calculate_mean_std_ci_precision_ap(ap_list, pr_list, mean_ip)

    metrics = {
        "roc": roc_metrics,
        "pr": pr_metrics,
    }

    logger.info(f"Mean AUC across folds: {metrics['roc']['auc']:.2f}")
    logger.info(f"Mean PR-AUC across folds: {metrics['pr']['ap']:.2f}")

    return {model_list, shap_values, scores, validation_indices, metrics}


################################
#  External Validation model   #
################################


def align_external_to_train(X_train, X_ext, fill_value=0, min_overlap=0.7):
    train_cols = list(X_train.columns)
    overlap = len(set(train_cols) & set(X_ext.columns)) / len(train_cols)
    if overlap < min_overlap:
        raise ValueError(
            f"External set overlaps only {overlap:.1%} of training features."
        )
    X_ext_aligned = X_ext.reindex(columns=train_cols, fill_value=fill_value)
    return X_ext_aligned


def train_and_validate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    pipeline: Optional[Pipeline] = None,
    param_grid: Optional[Dict] = None,
    best_estimator: Optional[Pipeline] = None,  # need to be pipeline
    n_splits: int = 10,
    n_iter: int = 30,
    model_type: str = "xgboost",
    random_state: int = 420,
    n_jobs: int = -1,
    get_only_model: bool = False,
):
    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Target for training samples.
    X_test : pd.DataFrame, optional
        Feature matrix for testing.
    y_test : pd.Series, optional
        Target for testing samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    best_estimator : Pipeline, optional
        Best estimator to predict on test data
    n_splits : int, default 10
        Number of outer CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    model_type : str, default xgboost
        Estimator model
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    get_only_model : bool, default False
        return only the fitted model
    Returns
    -------
    best_estimator : Pipeline
        best estimator
    shap_values : pd.DataFrame
        DataFrame SHAP values, indexed by sample.
    scores : pd.Series
        Series of scores, indexed by sample.
    metrics_test : Dict
        Dict containing metrics for testing set.
    """

    # if the user handed us an already‐trained model, use that
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
        )

        if get_only_model:
            return best_estimator

    X_test = align_external_to_train(X_train, X_test, fill_value=0, min_overlap=0.7)
    shap_values_df = pd.DataFrame(0.0, index=X_test.index, columns=X_test.columns)

    # Predict scores on the validation
    scores = best_estimator.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics_test(y_test, scores, np.linspace(0, 1, 200))

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_test = best_estimator[:-1].transform(X_test)
            logger.info(f"shape test data:{X_test.shape}")
        except Exception as e:
            logger.error(f"Error transforming validation data: {e}")
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator[-1])
    shap_values = explainer.shap_values(X_test)
    #    if model_type == "random-forest" and shap_values.ndim == 3 and shap_values.shape[2] == 2:
    if model_type == "random-forest":
        shap_values = shap_values[:, :, 1]

    shap_values = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    shap_values_df.loc[shap_values.index, shap_values.columns] = shap_values

    scores = pd.Series(scores, index=X_test.index, name="Score")

    logger.info(f"AUC in testing set: {metrics.get('auc'):.4f}")

    return {best_estimator, shap_values_df, scores, metrics}
