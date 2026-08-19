"""Command-line plotting for one or repeated phipml result files."""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from phipml import __version__
from phipml.io.data_handler import Config
from phipml.plots.result_summary import DEFAULT_OUTPUT_FORMATS, plot_result_files


def _read_table(path: str | Path, *, index_column: str | None) -> pd.DataFrame:
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(source)
    elif suffix in {".tsv", ".txt"}:
        table = pd.read_csv(source, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(source)
    else:
        raise ValueError(f"Unsupported table format: {source}")
    if index_column is not None:
        if index_column not in table.columns:
            raise KeyError(f"Index column {index_column!r} not found in {source}")
        table = table.set_index(index_column)
    table.index = table.index.astype(str)
    return table


def _expand_results(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(str(Path(value).expanduser())))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value).expanduser())
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("No result files were supplied")
    return unique


def _build_plot_results_parser() -> argparse.ArgumentParser:
    """Build the plot-results parser for both parsing and help display."""
    parser = argparse.ArgumentParser(
        prog="phipml-plot",
        description=(
            f"phipml {__version__} - plot performance and SHAP summaries from "
            "one result or aggregate repeated phipml runs."
        ),
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "results",
        nargs="*",
        help=(
            "Result paths or quoted glob patterns; may instead be set in "
            "--plot-config"
        ),
    )
    parser.add_argument(
        "--plot-config",
        default=None,
        help="Dedicated plotting YAML; explicit CLI arguments take precedence",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=None,
        metavar="PLOT",
        help=(
            "Plots to create. Use all (default), performance, roc, pr, confusion, "
            "classification, shap-beeswarm, shap-importance, shap-heatmap, or "
            "feature-table"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=None,
        help="Figure formats; defaults to PDF, SVG, and PNG",
    )
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--split", choices=("auto", "train", "test"), default=None)
    parser.add_argument(
        "--class-labels",
        nargs=2,
        default=None,
        metavar=("NEGATIVE", "POSITIVE"),
        help="Display labels; defaults to group_tests from the configuration",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional phipml YAML used to load feature values, targets, and "
            "peptide-library annotations. New artifacts embed this input context."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Plot directory. Defaults to classification.plot_output_dir, then "
            "the result directory's plots/ subdirectory."
        ),
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--max-display",
        type=int,
        default=None,
        help="Number of ranked features shown in SHAP plots and the compact table",
    )
    parser.add_argument(
        "--feature-ranking",
        choices=("auto", "top-k-frequency", "mean-abs-shap"),
        default=None,
        help=(
            "Feature ranking: auto uses repeated top-K frequency for multiple "
            "files and mean absolute SHAP for one file"
        ),
    )
    parser.add_argument(
        "--ranking-top-k",
        "--frequency-top-k",
        dest="ranking_top_k",
        type=int,
        default=None,
        help="Number of features considered top-ranked within each repeated run",
    )
    parser.add_argument(
        "--min-top-k-frequency",
        type=float,
        default=None,
        help="Minimum percentage of runs in whose top K a displayed feature occurs",
    )
    parser.add_argument(
        "--shap-alignment", choices=("strict", "intersection"), default=None
    )
    parser.add_argument(
        "--save-standalone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save ROC, PR, confusion, and classification panels separately",
    )
    parser.add_argument(
        "--features-table",
        default=None,
        help="Optional sample-by-feature CSV/TSV/Excel table for beeswarm/table",
    )
    parser.add_argument("--target-table", default=None)
    parser.add_argument("--sample-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--library-metadata", default=None)
    parser.add_argument("--library-id-column", default=None)
    parser.add_argument(
        "--feature-importance-table",
        default=None,
        help=(
            "Curated feature-importance CSV/TSV/Excel table. Its row order and "
            "edited annotations are used when rendering feature-table."
        ),
    )
    parser.add_argument(
        "--table-annotation-columns",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=(
            "Library columns displayed in the compact table, for example "
            "Description Species. The CSV retains every annotation column."
        ),
    )
    parser.add_argument(
        "--table-extra-columns",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=(
            "Optional audit columns displayed in the compact table, for "
            "example 'Feature type', 'Statistic', 'Top-k SHAP frequency (%%)', "
            "'Mean rank when in top K', or 'Selection frequency (%%)'. They "
            "remain available in the CSV when omitted from the figure."
        ),
    )
    parser.add_argument(
        "--validation-bootstraps",
        type=int,
        default=None,
        help="Reconstruct bootstrap CIs only when absent and targets are supplied",
    )
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--reconstruct-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load feature values from explicit or embedded configuration inputs",
    )
    parser.add_argument("--roc-color", default=None)
    parser.add_argument("--roc-band-color", default=None)
    parser.add_argument("--pr-color", default=None)
    parser.add_argument("--pr-band-color", default=None)
    parser.add_argument("--confusion-cmap", default=None)
    parser.add_argument("--classification-color", default=None)
    parser.add_argument("--shap-cmap", default=None)
    parser.add_argument("--shap-heatmap-cmap", default=None)
    parser.add_argument("--shap-importance-color", default=None)
    parser.add_argument(
        "--class-colors",
        nargs=2,
        default=None,
        metavar=("NEGATIVE", "POSITIVE"),
        help="Colors for the two prediction-direction labels above the SHAP plot",
    )
    parser.add_argument("--table-header-color", default=None)
    parser.add_argument(
        "--table-row-colors",
        nargs=2,
        default=None,
        metavar=("ODD", "EVEN"),
    )
    parser.add_argument("--table-prevalence-cmap", default=None)
    parser.add_argument("--feature-table-title", default=None)
    return parser


def parse_args_plot_results(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse plotting arguments."""
    return _build_plot_results_parser().parse_args(argv)


def _load_plot_config(
    value: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if value is None:
        return {}, None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Plot configuration does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("Plot configuration must contain a YAML mapping")
    raw = loaded.get("plotting", loaded)
    if not isinstance(raw, Mapping):
        raise TypeError("The YAML 'plotting' section must be a mapping")
    return dict(raw), path


def _choose(cli_value: Any, raw: Mapping[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else raw.get(key, default)


def _optional_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Plot setting {name!r} must be a mapping")
    return dict(value)


def _optional_sequence(value: Any, *, name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Plot setting {name!r} must be a YAML list")
    return list(value)


def _name_sequence(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    values = _optional_sequence(value, name=name)
    assert values is not None
    return [str(item) for item in values]


def _yaml_path(value: Any, *, plot_config: Path | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if plot_config is not None and not path.is_absolute():
        path = plot_config.parent / path
    return str(path.resolve())


def _yaml_paths(value: Any, *, plot_config: Path | None) -> list[str]:
    values = _name_sequence(value, name="results")
    if values is None:
        return []
    return [_yaml_path(item, plot_config=plot_config) or "" for item in values]


def _plot_output_dir(
    explicit: str | None,
    *,
    config_file: str | None,
    first_result: Path,
) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if config_file is not None:
        config = Config(config_file)
        raw = config.classification
        configured = raw.get("plot_output_dir")
        append_plots = False
        if configured is None:
            configured = raw.get("output_dir")
            append_plots = configured is not None
        if configured is not None:
            destination = Path(configured).expanduser()
            if not destination.is_absolute():
                destination = config.config_file.parent / destination
            destination = destination.resolve()
            return destination / "plots" if append_plots else destination
    return first_result.parent / "plots"


def main(argv: list[str] | None = None) -> int:
    argument_values = sys.argv[1:] if argv is None else argv
    parser = _build_plot_results_parser()
    if not argument_values:
        parser.print_help()
        return 0

    args = parser.parse_args(argument_values)
    raw, plot_config_path = _load_plot_config(args.plot_config)
    feature_table_settings = _optional_mapping(
        raw.get("feature_table"),
        name="feature_table",
    )

    if args.results:
        result_values = args.results
    else:
        result_values = _yaml_paths(raw.get("results"), plot_config=plot_config_path)
    if not result_values:
        parser.error(
            "result files are required either as positional arguments or in "
            "--plot-config"
        )
    result_paths = _expand_results(result_values)

    split = str(_choose(args.split, raw, "split", "auto"))
    class_labels = _name_sequence(
        _choose(args.class_labels, raw, "class_labels", None),
        name="class_labels",
    )
    if class_labels is not None and len(class_labels) != 2:
        raise ValueError("class_labels must contain exactly two values")
    plots = _name_sequence(
        _choose(args.plots, raw, "plots", ["all"]),
        name="plots",
    )
    output_formats = _name_sequence(
        _choose(args.formats, raw, "formats", list(DEFAULT_OUTPUT_FORMATS)),
        name="formats",
    )

    def configured_path(cli_value: Any, key: str) -> str | None:
        if cli_value is not None:
            return str(cli_value)
        return _yaml_path(raw.get(key), plot_config=plot_config_path)

    model_config = configured_path(args.config, "config")
    features_path = configured_path(args.features_table, "features_table")
    target_path = configured_path(args.target_table, "target_table")
    library_path = configured_path(args.library_metadata, "library_metadata")
    curated_table_value = (
        args.feature_importance_table
        if args.feature_importance_table is not None
        else feature_table_settings.get(
            "input",
            raw.get("feature_importance_table"),
        )
    )
    curated_table_path = (
        str(curated_table_value)
        if args.feature_importance_table is not None
        else _yaml_path(curated_table_value, plot_config=plot_config_path)
    )

    features = None
    target = None
    sample_column = str(_choose(args.sample_column, raw, "sample_column", "SampleName"))
    target_column = _choose(args.target_column, raw, "target_column", None)
    if features_path is not None:
        features = _read_table(features_path, index_column=sample_column)
    if target_path is not None:
        if target_column is None:
            raise ValueError("--target-column is required with --target-table")
        target_table = _read_table(target_path, index_column=sample_column)
        if target_column not in target_table.columns:
            raise KeyError(f"Target column {target_column!r} was not found")
        target = target_table[target_column]
    elif features is not None and target_column is not None:
        if target_column not in features.columns:
            raise KeyError(f"Target column {target_column!r} was not found")
        target = features.pop(target_column)

    library_metadata = None
    library_id_column = _choose(
        args.library_id_column,
        raw,
        "library_id_column",
        None,
    )
    if library_path is not None:
        library_metadata = _read_table(
            library_path,
            index_column=library_id_column,
        )

    curated_table = (
        _read_table(curated_table_path, index_column=None)
        if curated_table_path is not None
        else None
    )

    explicit_output_dir = args.output_dir
    if explicit_output_dir is None and raw.get("output_dir") is not None:
        explicit_output_dir = _yaml_path(
            raw["output_dir"],
            plot_config=plot_config_path,
        )

    output_dir = _plot_output_dir(
        explicit_output_dir,
        config_file=model_config,
        first_result=result_paths[0],
    )

    colors = _optional_mapping(raw.get("colors"), name="colors")
    cli_colors = {
        "roc": args.roc_color,
        "roc_band": args.roc_band_color,
        "pr": args.pr_color,
        "pr_band": args.pr_band_color,
        "confusion_cmap": args.confusion_cmap,
        "classification": args.classification_color,
        "shap_cmap": args.shap_cmap,
        "shap_heatmap_cmap": args.shap_heatmap_cmap,
        "shap_importance": args.shap_importance_color,
        "table_header": args.table_header_color,
        "table_prevalence_cmap": args.table_prevalence_cmap,
    }
    colors.update(
        {key: value for key, value in cli_colors.items() if value is not None}
    )
    class_colors = _name_sequence(
        args.class_colors if args.class_colors is not None else raw.get("class_colors"),
        name="class_colors",
    )
    if class_colors is not None:
        if len(class_colors) != 2:
            raise ValueError("class_colors must contain exactly two values")
        colors["negative_class"], colors["positive_class"] = class_colors
    row_colors = _name_sequence(
        (
            args.table_row_colors
            if args.table_row_colors is not None
            else feature_table_settings.get("row_colors")
        ),
        name="feature_table.row_colors",
    )
    if row_colors is not None:
        if len(row_colors) != 2:
            raise ValueError("feature-table row colors must contain two values")
        colors["table_row_odd"], colors["table_row_even"] = row_colors
    if args.table_header_color is None and feature_table_settings.get("header_color"):
        colors["table_header"] = str(feature_table_settings["header_color"])
    if args.table_prevalence_cmap is None and feature_table_settings.get(
        "prevalence_cmap"
    ):
        colors["table_prevalence_cmap"] = str(feature_table_settings["prevalence_cmap"])

    annotation_columns = _name_sequence(
        (
            args.table_annotation_columns
            if args.table_annotation_columns is not None
            else feature_table_settings.get(
                "annotation_columns",
                raw.get("table_annotation_columns"),
            )
        ),
        name="feature_table.annotation_columns",
    )
    extra_columns = _name_sequence(
        (
            args.table_extra_columns
            if args.table_extra_columns is not None
            else feature_table_settings.get("extra_columns")
        ),
        name="feature_table.extra_columns",
    )

    plot_result_files(
        result_paths,
        split=split,
        class_labels=class_labels,
        title=_choose(args.title, raw, "title", None),
        output_dir=output_dir,
        output_prefix=str(_choose(args.output_prefix, raw, "output_prefix", "phipml")),
        plots=plots,
        output_formats=output_formats,
        dpi=int(_choose(args.dpi, raw, "dpi", 600)),
        plot_colors=colors,
        features=features,
        target=target,
        oligos_metadata=library_metadata,
        feature_importance_table=curated_table,
        table_annotation_columns=annotation_columns,
        table_extra_columns=extra_columns,
        feature_table_title=str(
            args.feature_table_title
            if args.feature_table_title is not None
            else feature_table_settings.get(
                "title",
                "Top features by mean absolute SHAP value",
            )
        ),
        max_display=int(_choose(args.max_display, raw, "max_display", 20)),
        feature_ranking=str(
            _choose(args.feature_ranking, raw, "feature_ranking", "auto")
        ),
        ranking_top_k=int(_choose(args.ranking_top_k, raw, "ranking_top_k", 30)),
        min_top_k_frequency=float(
            _choose(args.min_top_k_frequency, raw, "min_top_k_frequency", 0.0)
        ),
        shap_alignment=str(
            _choose(args.shap_alignment, raw, "shap_alignment", "strict")
        ),
        validation_bootstraps=int(
            _choose(args.validation_bootstraps, raw, "validation_bootstraps", 0)
        ),
        random_state=int(_choose(args.random_state, raw, "random_state", 420)),
        save_standalone=bool(
            _choose(args.save_standalone, raw, "save_standalone", True)
        ),
        config_file=model_config,
        reconstruct_data=bool(
            _choose(args.reconstruct_data, raw, "reconstruct_data", True)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
