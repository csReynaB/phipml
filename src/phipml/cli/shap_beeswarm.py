# ======================
# Standard library
# ======================
import argparse
from typing import List

# ======================
# Third-party libraries
# ======================
from matplotlib.colors import LinearSegmentedColormap

# ======================
# Local libraries
# ======================
from phipml.plots.auc_shap_summary import run_shap_summary_and_feature_table


def parse_cmap_arg(cmap: List[str]):
    """
    If a single str is provided, treat it as a matplotlib cmap name.
    If multiple color str are provided, treat them as colors to build a LinearSegmentedColormap.
    """
    if len(cmap) == 1:
        return cmap[0]
    return LinearSegmentedColormap.from_list("", cmap)


def parse_args_shap() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP summary plot and feature table"
    )

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--file_dir", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, required=True)
    parser.add_argument("--output_name", type=str, default=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_display", type=int, default=30)
    parser.add_argument(
        "--figure_size",
        nargs=2,
        type=float,
        default=[6, 5],
        metavar=("WIDTH", "HEIGHT"),
    )

    # Either pass a matplotlib cmap name (e.g., viridis) or a list of hex colors
    parser.add_argument(
        "--cmap_colors",
        nargs="+",
        default=["#6699CC", "#CC6677"],
        help="Either a single matplotlib colormap name (e.g. 'viridis') or a list of colors (e.g. '#6699CC #CC6677').",
    )

    parser.add_argument(
        "--shap_key",
        type=str,
        default=None,
        help="Optional override for SHAP key: e.g. 'train_shap_values' or 'test_shap_values'.",
    )

    # Font sizes for plot_shap_values (kept compatible with your dict)
    parser.add_argument("--xlabel_fontsize", type=int, default=13)
    parser.add_argument("--xticks_fontsize", type=int, default=12)
    parser.add_argument("--yticks_fontsize", type=int, default=12)
    parser.add_argument("--colorbar_fontsize", type=int, default=12)
    parser.add_argument("--legend_fontsize", type=int, default=10)

    parser.add_argument("--legend_labels", nargs="+", default=["0", "1"])
    parser.add_argument("--label_groups", nargs="+", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_shap()

    font_sizes = {
        "xlabel": args.xlabel_fontsize,
        "xticks": args.xticks_fontsize,
        "yticks": args.yticks_fontsize,
        "legend": args.legend_fontsize,
        "colorbar": args.colorbar_fontsize,
    }
    cmap = parse_cmap_arg(args.cmap)

    run_shap_summary_and_feature_table(
        config_file=args.config_file,
        file_dir=args.file_dir,
        file_pattern=args.file_pattern,
        output_dir=args.output_dir,
        output_name=args.output_name,
        cmap=cmap,
        max_display=args.max_display,
        figure_size=tuple(args.figure_size),
        shap_key=args.shap_key,
        shap_fontsize=font_sizes,
        label_groups=args.label_groups,
        legend_labels=args.legend_labels,
    )
