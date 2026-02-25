#!/usr/bin/env python3
"""
CLI tool to generate and preserve ROC AUC plots from nested_predictions joblib files.
Usage example:
  python roc_aucs_plot.py \
    --joblib-dir /path/to/Diagnostics/Cirrhosis_HCC/HCC \
    --group1 Cirrhosis \
    --size1 50 \
    --group2 HCC \
    --size2 46 \
    --out-dir /path/to/output/plots \
    --out-base ROC_Cirrhosis_HCC_HCC
"""

# ======================
# Standard library
# ======================
import os

# Limit OpenMP/BLAS threads to avoid thread creation errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse

# ======================
# Third-party library
# ======================
import matplotlib

matplotlib.use("Agg")

# ======================
# Local library
# ======================
from phipml.plots.auc_shap_summary import plot_nested_aucs


def parse_args_roc() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROC AUC plots from nested_predictions joblib files."
    )
    parser.add_argument(
        "--joblib-dir", required=True, help="Directory with nested_predictions_*.joblib"
    )
    parser.add_argument("--group1", required=True, help="Label for group1")
    parser.add_argument(
        "--size1", type=int, required=True, help="Sample size of group1"
    )
    parser.add_argument("--group2", required=True, help="Label for group2")
    parser.add_argument(
        "--size2", type=int, required=True, help="Sample size of group2"
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory to save PDFs and figures"
    )
    parser.add_argument("--out-base", required=True, help="Base filename for outputs")
    parser.add_argument(
        "--color-indiv", default="moccasin", help="Color for individual curves"
    )
    parser.add_argument("--color-mean", default="#d95f02", help="Color for mean curve")
    parser.add_argument("--color-ci", default="moccasin", help="Color for CI band")
    parser.add_argument(
        "--color-rand", default="gray", help="Color for random diagonal"
    )
    parser.add_argument(
        "--prefix-base",
        default="nested_random-forest_",
        help="Base prefix filename for input files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_roc()

    colors = {
        "indiv": args.color_indiv,
        "mean": args.color_mean,
        "ci": args.color_ci,
        "rand": args.color_rand,
    }

    figs, axes = plot_nested_aucs(
        joblib_dir=args.joblib_dir,
        group1=args.group1,
        size1=args.size1,
        group2=args.group2,
        size2=args.size2,
        colors=colors,
        out_dir=args.out_dir,
        out_base=args.out_base,
        prefix_base=args.prefix_base,
    )
    print("Generated plots for prefixes:", list(axes.keys()))
