# ======================
# Standard library
# ======================
import argparse

from phipml.plots.auc_heatmap import heatmap_aucs


def parse_args_heatmap_auc() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AUC heatmap from joblib cohort outputs"
    )
    parser.add_argument(
        "--parents",
        "-p",
        nargs="+",
        required=True,
        help="Parent folders like Controls_HCC Cirrhosis_HCC etc",
    )
    parser.add_argument(
        "--subdir",
        "-s",
        default="",
        help="Subdirectory like withSexAge or onlySexAge (default: base)",
    )
    parser.add_argument(
        "--cohorts", "-c", nargs="+", required=True, help="Ordered list of cohort names"
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Mean AUC from 100 repeated runs (nested CVs on the diagonal)",
        help="Plot title",
    )
    parser.add_argument(
        "--outname", "-o", default="heatmap_auc.pdf", help="Output PDF filename"
    )
    parser.add_argument(
        "--palette",
        default="YlGnBu",
        help="Colormap to use (e.g., viridis, viridis_r, plasma, YlGnBu, etc)",
    )
    parser.add_argument(
        "--object",
        default="heatmap_data.pkl",
        help="name of the returned pkl object containing the auc values",
    )
    parser.add_argument(
        "--subtract_sizes",
        nargs="*",
        default=[],
        help="Optional parent cohort sizes to subtract (e.g., Controls 72 Cirrhosis 77)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args_heatmap_auc()

    subtract_map = (
        dict(zip(args.subtract_sizes[::2], map(int, args.subtract_sizes[1::2])))
        if args.subtract_sizes
        else None
    )

    heatmap_aucs(
        args.parents,
        args.subdir,
        args.cohorts,
        args.title,
        args.outname,
        args.palette,
        args.object,
        subtract_map,
    )
