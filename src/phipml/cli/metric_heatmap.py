"""Create cohort-by-cohort metric heatmaps from a tidy result manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phipml.plots.metric_heatmap import build_metric_matrix, plot_metric_heatmap


def parse_args_metric_heatmap(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phipml-heatmap",
        description="Plot ROC-AUC, AP, or classification metrics across cohorts.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV/TSV with training, validation, path, and optional split columns",
    )
    parser.add_argument("--metric", default="roc.auc")
    parser.add_argument(
        "--order",
        nargs="+",
        default=None,
        help="Use one shared order for a square matrix (legacy behavior)",
    )
    parser.add_argument(
        "--training-order",
        nargs="+",
        default=None,
        help="Optional order of training-cohort columns",
    )
    parser.add_argument(
        "--validation-order",
        nargs="+",
        default=None,
        help="Optional order of validation-cohort rows",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--palette", default="YlGnBu")
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--annotate-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _read_manifest(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".csv":
        records = pd.read_csv(source)
    elif source.suffix.lower() in {".tsv", ".txt"}:
        records = pd.read_csv(source, sep="\t")
    else:
        raise ValueError("Heatmap manifest must be CSV, TSV, or TXT")
    if "path" in records.columns:
        records["path"] = records["path"].map(
            lambda value: str(
                (source.parent / Path(str(value))).resolve()
                if not Path(str(value)).expanduser().is_absolute()
                else Path(str(value)).expanduser().resolve()
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args_metric_heatmap(argv)
    summary = build_metric_matrix(
        _read_manifest(args.manifest),
        metric=args.metric,
        order=args.order,
        training_order=args.training_order,
        validation_order=args.validation_order,
    )
    plot_metric_heatmap(
        summary,
        title=args.title,
        palette=args.palette,
        vmin=args.vmin,
        vmax=args.vmax,
        annotate_uncertainty=args.annotate_uncertainty,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
