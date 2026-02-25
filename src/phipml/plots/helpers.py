# ======================
# Standard library
# ======================
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.axes as maxes

# ======================
# Third-party libraries
# ======================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

# ======================
# Global configuration
# ======================
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.titlecolor"] = "black"


def format_pval(p, alpha=0.05):
    """Format p-value similar to automatic R report."""

    if isinstance(p, str):
        try:
            p = float(p)
        except ValueError:
            return "NA"

    if p is None or math.isnan(p):
        return "NA"

    # 1. Non-significant (p > alpha)
    if p > alpha:
        raw = f"{p:.2f}".rstrip("0").rstrip(".")
        return f"ns [{raw}]"

    # 2. Normal fixed-decimal formatting (0.001 ≤ p ≤ alpha)
    if p >= 0.001:
        raw = f"{p:.3f}".rstrip("0").rstrip(".")
        return raw

    # 3. Scientific notation (< 0.001)
    # Format like 1e-4 instead of 1.00e-04
    raw = f"{p:.2e}"

    # Clean up trailing zeros in mantissa
    raw = raw.replace(".00e", "e")
    raw = raw.replace(".0e", "e")

    # Also trim e- notation like in R:
    # "1.10e-04" → "1.1e-04"
    if "e" in raw:
        mantissa, exp = raw.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        raw = mantissa + "e" + exp

    return raw


##############################
#        ROC-AUC curve       #
##############################
def plot_roc_summary(
    fpr_grid: np.ndarray,
    tprs: np.ndarray,
    summary: Dict,
    colors: Dict[str, str],
    title: str,
    label_loc: str,
    pdf_path: Path,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:

    mean_tpr = summary["mean_tpr"]
    low_tpr = summary["low_tpr"]
    high_tpr = summary["high_tpr"]
    auc_mean = summary["auc_mean"]
    auc_ci_low = summary["auc_ci_low"]
    auc_ci_high = summary["auc_ci_high"]

    fig, ax = plt.subplots(figsize=(6, 6))

    for row in tprs:
        ax.plot(fpr_grid, row, color=colors["indiv"], alpha=0.65, lw=1)

    ax.plot(
        fpr_grid,
        mean_tpr,
        color=colors["mean"],
        lw=2.5,
        label=f"Mean AUC={auc_mean:.2f}",
    )

    ax.fill_between(
        fpr_grid,
        low_tpr,
        high_tpr,
        color=colors["ci"],
        alpha=0.45,
        label=f"95% CI [{auc_ci_low:.2f},{auc_ci_high:.2f}]",
    )

    ax.plot([0, 1], [0, 1], "--", color=colors["rand"], lw=1, label="Random Classifier")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(labelsize=15)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=15, loc=label_loc)
    fig.tight_layout()

    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")

    return fig, ax


##############################
#       SHAP helpers         #
##############################


def generate_feature_importance_table(
    values: Union[pd.DataFrame, np.ndarray],
    features: pd.DataFrame,
    target: pd.Series,
    oligos_metadata: pd.DataFrame,
    group_tests: Optional[List[str]] = None,
    filename_label: Optional[str] = None,
    with_oligos: bool = True,
    with_additional_features: bool = False,
    with_run_plates: bool = False,
    figures_dir: str = "./",
) -> pd.DataFrame:
    """Generates feature importance table from SHAP values with metadata and group statistics.

    Args:
        values: SHAP values array
        features: Feature values DataFrame with SampleName index
        target: Target values Series with same index as features
        oligos_metadata: Oligos metadata DataFrame indexed by peptide ID
        group_tests: List of two group names for computing prevalence statistics
        filename_label: Name of the ouptut file name
        suffix_file: Model identifier for output filename
        with_oligos: Include oligos metadata
        with_additional_features: Include additional feature columns
        with_run_plates: Include run plate information
        figures_dir: Output directory for saved CSV

    Returns:
        DataFrame with feature importance stats and metadata
    """

    # Join features with target and get feature names
    features = features.join(target).set_index(target.name, append=True)
    feature_names = features.columns

    # Calculate mean absolute SHAP values
    importance_df = pd.DataFrame(
        {"SHAP value": np.abs(values).mean(axis=0)}, index=feature_names
    ).sort_values("SHAP value", ascending=False)

    # Add metadata if available
    if with_oligos:
        importance_df = importance_df.merge(
            oligos_metadata, left_index=True, right_index=True, how="left"
        ).rename(
            columns={
                "species": "Species",
                "genus": "Genus",
                "family": "Family",
                "order": "Order",
                #'len_seq' : 'Length'
            }
        )

    # Add group statistics if requested
    if group_tests is not None:
        # Get relevant feature columns
        feat_pattern = "|".join(
            ["agilent", "corona2", "twist", "Sex", "Age", "run_plate"]
        )
        relevant_cols = feature_names[
            feature_names.str.contains(feat_pattern, case=False)
        ]

        # Calculate group means
        group_means = features[relevant_cols].groupby(level=1).mean()

        # Convert relevant columns to percentages
        perc_cols = group_means.columns[
            group_means.columns.str.contains(
                "agilent|corona2|twist|sex|run_plate", case=False
            )
        ]
        group_means[perc_cols] *= 100

        # Transpose and rename columns to group names
        group_stats = group_means.T.rename(columns=dict(enumerate(group_tests)))

        # Add to importance DataFrame
        importance_df = pd.merge(
            importance_df, group_stats, left_index=True, right_index=True, how="left"
        )

        # Calculate log2 ratio between groups
        importance_df["Ratio"] = (
            importance_df[group_tests[0]] / importance_df[group_tests[1]]
        )

    # Format output
    importance_df.reset_index(inplace=True)
    importance_df.rename(columns={"index": "Peptide ID"}, inplace=True)

    # Save to CSV
    filename_parts = [
        "Table_shap",
        f"{'-'.join(group_tests)}" if group_tests else None,
        filename_label or "model",
        "with_oligos" if with_oligos else None,
        "with_additional_features" if with_additional_features else None,
        "with_run_plates" if with_run_plates else None,
    ]
    filename = "_".join(filter(None, filename_parts)) + ".csv"

    importance_df.to_csv(Path(figures_dir, filename), index=False)

    return importance_df


def _get_top_features(
    feature_shap_values_df: pd.DataFrame,
    group_tests: List[str] = None,
    to_select_features: int = 30,
    max_length_text: int = 75,
) -> pd.DataFrame:
    """
    Generate a table of top feature importances from a SHAP values DataFrame.

    The function selects the top features based on a hard-coded column order,
    rounds specified columns (based on group_tests and 'Ratio (log10)'),
    and truncates long text in the 'Description' column.

    Parameters
    ----------
    feature_shap_values_df : pd.DataFrame
        DataFrame containing feature importance values and additional metadata.
        Important to have Ratio (log 10), Description and Group_test columns. Run generate_feature_importance_df() first.
    group_tests : list of str
        A list of two group labels for binary classification. These labels are
        used as column names for rounding values. Must be provided with exactly two elements.
    to_select_features : int, default 30
        The number of top features to select.
    max_length_text : int, default 90
        Maximum allowed length for text in the 'Description' column. Longer texts are truncated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top features with rounded group test values,
        a computed ratio (in log10), and a truncated description.

    Raises
    ------
    ValueError
        If group_tests is not provided or does not have exactly two elements,
        or if the required columns are not found in the DataFrame.
    """
    # Check that group_tests is provided correctly
    if group_tests is None or len(group_tests) != 2:
        raise ValueError(
            "group_tests is expected to be provided as a list of two labels for binary classification."
        )

    # Helper function to truncate text
    def truncate_description(text: Optional[str], max_length: int) -> Optional[str]:
        if text is None:
            return None
        return text if len(text) <= max_length else text[:max_length] + "..."

    # Select top features based on a hard-coded column order.
    # Adjust these indices as necessary for your DataFrame.
    top_features = feature_shap_values_df[
        [
            "Peptide ID",
            "Description",
            group_tests[0],
            group_tests[1],
            "Ratio",
            "SHAP value",
        ]
    ].iloc[:to_select_features]
    # Round columns specified by group_tests and the 'Ratio' column.
    round_columns = [group_tests[0], group_tests[1], "Ratio"]
    for col in round_columns:
        if col in top_features.columns:
            top_features[col] = top_features[col].round(2)
        else:
            raise ValueError(
                f"Expected column '{col}' not found in the DataFrame. Check group_tests or the column order."
            )

    # Truncate text in the 'Description' column if it exists.
    if "Description" in top_features.columns:
        top_features["Description"] = top_features["Description"].apply(
            lambda x: truncate_description(x, max_length_text)
        )
    else:
        raise ValueError("Expected column 'Description' not found in the DataFrame.")
    return top_features


def _colorize(val: float, min_val: float, max_val: float, palette: str = "Reds"):
    """
    Returns a color (as an RGBA tuple) based on a normalized value using a given palette.

    Parameters
    ----------
    val : float
        The value to colorize.
    min_val : float
        The minimum value for normalization.
    max_val : float
        The maximum value for normalization.
    palette : str, default 'Reds'
        The palette name to use. If 'Reds', then 'RdYlGn' colormap is used; otherwise 'BuPu' is used.

    Returns
    -------
    color : tuple or np.ndarray
        The RGBA color corresponding to the normalized value.
    """
    # Avoid division by zero if min_val equals max_val
    # if max_val == min_val:
    #   norm_val = 0.5
    # else:
    val = float(val) if isinstance(val, str) else val

    norm_val = (val - min_val) / (max_val - min_val)

    if palette == "Reds":
        color = colormaps["RdYlGn"](norm_val)
    else:
        color = colormaps["BuPu"](norm_val)
    return color


def _render_main_table(
    df: pd.DataFrame,
    col_widths: List[float],
    row_height: float = 0.625,
    font_size: int = 9,
    header_color: str = "lightgray",
    row_colors: Optional[List[str]] = None,
    edge_color: str = "w",
    bbox: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Render a DataFrame as a matplotlib table with custom cell coloring based on feature values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to display.
    col_widths : list of float
        List of widths for each column.
    row_height : float, default 0.625
        Height of each row.
    font_size : int, default 11
        Base font size for the table.
    header_color : str, default 'lightgray'
        Background color for the header cells.
    row_colors : list of str, optional
        List of alternating row colors. Defaults to ['#f1f1f2', 'w'].
    edge_color : str, default 'w'
        Color of the cell edges.
    bbox : list of float, optional
        Bounding box for the table in figure coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes on which to render the table. If None, a new figure and axes are created.
    **kwargs : dict
        Additional keyword arguments to pass to ax.table.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the rendered table.
    """
    if bbox is None:
        bbox = [0, 0, 1, 1]
    if row_colors is None:
        row_colors = ["#f1f1f2", "w"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(np.sum(col_widths), len(df) * row_height + 1))
        ax.axis("off")

    # Predefined scaling values for percentage columns
    min_val = 0
    max_val = 100
    # For the ratio, using a symmetric scale (can be adjusted as needed)
    # min_val_ratio = -2
    # max_val_ratio = 2

    # Create the matplotlib table.
    mpl_table = ax.table(
        cellText=df.values,
        bbox=bbox,
        colLabels=df.columns,
        cellLoc="center",
        colWidths=col_widths,
        **kwargs,
    )

    # Which column is "Description"?
    desc_col = df.columns.get_loc("Peptide ID")  # 0-based index
    # Left-align every cell in that column (header + body)
    for row in range(len(df) + 1):  # +1 for header row
        cell = mpl_table[(row, desc_col)]
        cell.set_text_props(ha="left", va="center")
        cell._loc = "left"  # ensure the cell itself is left-located
        cell.PAD = 0.01  # optional: add a tiny left padding

    # Which column is "Description"?
    desc_col = df.columns.get_loc("Description")  # 0-based index
    # Left-align every cell in that column (header + body)
    for row in range(len(df) + 1):  # +1 for header row
        cell = mpl_table[(row, desc_col)]
        cell.set_text_props(ha="left", va="center")
        cell._loc = "left"  # ensure the cell itself is left-located
        cell.PAD = 0.01  # optional: add a tiny left padding

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Iterate over table cells
    for (i, j), cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        # Set a slightly smaller font for cell text (adjust as needed)
        cell.set_text_props(fontsize=8.5)

        if i == 0:
            # Header formatting
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor(header_color)
        else:
            # For specific columns, apply a gradient color
            # Note: Adjust the column indices based on your DataFrame layout.
            if j == 2:  # Assuming column 2 corresponds to 'Disease'
                cell_val = df.iloc[i - 1, 2]
                cell.set_facecolor(
                    _colorize(cell_val, min_val, max_val, palette="Reds")
                )
                cell.set_text_props(color="black")
            elif j == 3:  # Assuming column 3 corresponds to 'Ctrl'
                cell_val = df.iloc[i - 1, 3]
                cell.set_facecolor(_colorize(cell_val, min_val, max_val))
                cell.set_text_props(color="black")
            # elif j == 4:  # Assuming column 4 corresponds to 'Ratio'
            # cell_val = df.iloc[i - 1, 4]
            # cell.set_facecolor(_colorize(cell_val, min_val_ratio, max_val_ratio, palette='Blues'))
            # cell.set_text_props(color='black')
            else:
                # Alternate row colors for remaining cells.
                cell.set_facecolor(row_colors[i % len(row_colors)])

    return ax


def _render_header_main_table(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Render the header portion of a table on the given matplotlib axis.

    This function draws a header table using fixed column widths and then adds
    custom text labels above the header cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes on which to render the header. If None, the current axes (plt.gca()) is used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the rendered header table.
    """
    if ax is None:
        ax = plt.gca()

    # Define header colors and column widths.
    col_colors = ["lightgray", "lightgray", "lightgray"]
    col_widths = [0.563, 0.292, 0.107]
    bbox = [0, 1, 1, 0.12]

    # Create header table using matplotlib's table function.
    header_table = plt.table(
        cellLoc="center", colWidths=col_widths, bbox=bbox, cellColours=[col_colors]
    )

    # Customize cells in the header table.
    for (i, j), cell in header_table._cells.items():
        cell.set_edgecolor("w")
        if i == 0:  # Header row
            cell.set_text_props(weight="bold", color="black")

    # Add additional header text labels above the table.
    ax.text(
        0.29,
        1.05,
        "Peptide details",
        ha="center",
        color="black",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        0.74,
        1.035,
        "Antibody responses\nappearing % in ...",
        ha="center",
        fontsize=10,
        color="black",
        weight="bold",
    )
    ax.text(
        0.945,
        1.035,
        "Feature\nimportance",
        ha="center",
        fontsize=10,
        color="black",
        weight="bold",
    )

    return ax


def plot_table_top_features(
    feature_shap_values_df: pd.DataFrame,
    group_tests: List[str] = None,
    ax: Optional[plt.Axes] = None,
    to_select_features: int = 10,
    fig_size=(8.5, 5),
    figure_dir: str = "./",
    filename_label: str = "plot",
    save_fig: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Render a table of top features (with SHAP values) on a matplotlib axis.

    This function selects the top features using get_top_features(), renders a header table,
    renders the main table with custom column widths and formatting, hides the axes, and optionally saves the figure.

    Parameters
    ----------
    feature_shap_values_df : pd.DataFrame
        DataFrame with feature importance and metadata. Run generate_feature_importance_table() first to get the df.
    group_tests : list of str
        List of two group labels. Used in get_top_features() to add prevalence and ratio columns.
    ax : matplotlib.axes.Axes, optional
        The axis on which to render the tables. If None, a new figure and axis are created.
    to_select_features : int, default 10
        Number of top features to select.
    fig_size : tuple, default (8.5, 5)
        Size of the figure.
    figure_dir : str, default './'
        Directory to save the figure.
    filename_label : str, optional
        Suffix for the filename. If not provided, it is constructed from prefix_label, and group_tests.
    save_fig : bool, default False
        If True, the figure is saved to figure_dir.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the rendered table.

    Raises
    ------
    ValueError
        If group_tests is provided but does not contain exactly two elements.
    """
    # Get top features with optional group tests. (Assumes get_top_features is defined elsewhere.)
    top_features_df = _get_top_features(
        feature_shap_values_df,
        group_tests=group_tests,
        to_select_features=to_select_features,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Render the header table.
    _render_header_main_table(ax)

    # Ensure SHAP values are formatted as strings (rounded) for display.
    if "SHAP value" in top_features_df.columns:
        top_features_df["SHAP value"] = (
            top_features_df["SHAP value"].round(3).astype(str)
        )
    else:
        raise ValueError("Expected column 'SHAP value' not found in the DataFrame.")

    # Render the main data table.
    # You can adjust col_widths as needed.
    _render_main_table(
        top_features_df, col_widths=[0.35, 1.5, 0.32, 0.32, 0.32, 0.35], ax=ax
    )
    #                      col_widths=[0.35, 2.15, 0.32, 0.32, 0.32, 0.35], ax=ax)

    # Hide axes.
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    plt.tight_layout()

    # Construct filename and save the figure if required.
    if save_fig:
        save_path = (
            Path(figure_dir)
            / f"SHAP_table_top{to_select_features}_{filename_label}_{'-'.join(group_tests)}.pdf"
        )
        plt.savefig(save_path, dpi=600, format="pdf", bbox_inches="tight")

    return fig, ax


def plot_shap_values(
    values: np.ndarray,
    features: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: Union[str, LinearSegmentedColormap] = "viridis",
    max_display: int = 30,
    group_tests: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    filename_label: str = "plot",
    plot_title: str = "",
    fontsize: Optional[Dict[str, int]] = None,
    figure_size: Tuple[float, float] = (6, 5),
    x_label: str = "Prediction toward...",
    legend_title: str = "Feature\nvalue",
    legend_labels: Optional[List[str]] = None,
    add_group_labels: bool = True,
    add_binary_legend: bool = True,
    save_fig: bool = False,
    figures_dir: Union[str, Path] = "./",
    **shap_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Create a SHAP summary (beeswarm) plot with top x-axis and group labels.
    """

    # -------------------------
    # Defaults
    # -------------------------
    fs_default: Dict[str, int] = {
        "title": 12,
        "xlabel": 13,
        "ylabel": 13,
        "xticks": 12,
        "yticks": 12,
        "legend": 10,
        "colorbar": 13,
    }
    if fontsize:
        fs_default.update(fontsize)
    fontsize = fs_default

    if group_tests is None:
        group_tests = ["group1", "group2"]

    # -------------------------
    # Figure / axis handling
    # -------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure
        plt.figure(fig.number)
        plt.sca(ax)

    # -------------------------
    # SHAP summary plot
    # -------------------------
    shap.summary_plot(
        values,
        features=features,
        plot_type="dot",
        cmap=cmap,
        max_display=max_display,
        plot_size=[figure_size[0], figure_size[1]],
        show=False,
        **shap_kwargs,
    )

    # Re-bind to the axis SHAP actually used
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("")
    # =====================================================
    # X-AXIS (TOP) + GROUP LABELS
    # =====================================================
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize=fontsize["xticks"],
    )
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    # ax.set_xlabel( x_label, fontsize=fontsize["xlabel"], fontweight="bold", labelpad=18, color="black", )
    if add_group_labels:
        xmin, xmax = ax.get_xlim()
        x0 = 0.0 if (xmin < 0 < xmax) else (xmin + xmax) / 2.0
        x_left = (xmin + x0) / 2.0
        x_right = (x0 + xmax) / 2.0
        x_mid = 0.5 * (x_left + x_right)
        trans = ax.get_xaxis_transform()

        ax.text(
            x_left,
            1.08,
            group_tests[0],
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            color="black",
            transform=trans,
            clip_on=False,
        )
        ax.text(
            x_right,
            1.08,
            group_tests[1],
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            color="black",
            transform=trans,
            clip_on=False,
        )

        # --- Center xlabel between the two groups (data-driven) ---
        ax.text(
            x_mid,
            1.14,  # above group labels
            x_label,
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            fontweight="bold",
            color="black",
            transform=trans,
            clip_on=False,
        )

        # fig.subplots_adjust(top=0.82)

    # -------------------------
    # Y tick labels cleanup
    # -------------------------
    if pattern:
        yticklabels = [re.sub(pattern, lbl.get_text()) for lbl in ax.get_yticklabels()]
    else:
        yticklabels = [lbl.get_text() for lbl in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, fontsize=fontsize["yticks"])

    # -------------------------
    # Title
    # -------------------------
    if plot_title:
        ax.set_title(plot_title, fontsize=fontsize["title"], weight="bold", pad=10)

    # -------------------------
    # Grid + aesthetics
    # -------------------------
    ax.tick_params(axis="y", pad=-10)
    ax.yaxis.set_label_coords(-0.3, 0.5)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="lightgrey")
    ax.grid(False, axis="x")
    ax.patch.set_alpha(0.0)

    # -------------------------
    # Binary legend (right side)
    # -------------------------
    if add_binary_legend:

        # Remove SHAP colorbar axis
        for a in fig.axes:
            if a is not ax:
                a.remove()

        if legend_labels is None:
            legend_labels = ["0", "1"]

        handles = [
            Patch(facecolor=plt.get_cmap(cmap)(0.0), label=legend_labels[0]),
            Patch(facecolor=plt.get_cmap(cmap)(1.0), label=legend_labels[1]),
        ]
        legend = ax.legend(
            handles=handles,
            title=legend_title,
            frameon=False,
            bbox_to_anchor=(0.97, 0.5),
            loc="center left",
            fontsize=fontsize["legend"],
            title_fontsize=fontsize["legend"],
        )
        for txt in legend.get_texts():
            txt.set_color("black")
        legend.get_title().set_color("black")
    else:
        if legend_labels is None:
            legend_labels = ["Low", "High"]
        cbar = fig.axes[-1]
        cbar.set_yticklabels(
            legend_labels, color="black", fontsize=fontsize["colorbar"]
        )
        cbar.set_ylabel(legend_title, fontsize=fontsize["legend"], color="black")

    # -------------------------
    # Save
    # -------------------------
    if save_fig:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        safe_groups = "-".join(group_tests) if group_tests else "groups"
        save_path = figures_dir / f"shap_{safe_groups}_{filename_label}.pdf"
        fig.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")

    return fig, ax


def plot_shap_values_deprecated(
    values: np.ndarray,
    features: pd.DataFrame,
    ax: Optional[maxes.Axes] = None,
    cmap: str = "viridis",
    max_display: int = 30,
    group_tests: Optional[List[str]] = None,
    pattern: str = None,
    filename_label: str = "plot",
    plot_title: str = "",
    save_fig: bool = False,
    figures_dir: str = "./",
    fontsize: dict = None,
    figure_size: tuple = (6, 5),
    **shap_kwargs,
) -> maxes.Axes:
    """
    Create a custom SHAP summary plot with configurable font sizes and improved visualizations.

    Parameters
    ----------
    values : np.ndarray
        Array of SHAP values.
    features : pd.DataFrame
        DataFrame of features corresponding to the SHAP values.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    cmap : str or matplotlib.colors.Colormap, default 'viridis'
        Colormap used in the summary plot.
    max_display : int, default 30
        Maximum number of features to display.
    group_tests : list of str, optional
        List of two labels to annotate the x-axis.
    pattern : str, default r'(bloodb|BloodB)'
        Regular expression pattern to remove from y-axis tick labels.
    filename_label : str, default 'plot'
        Suffix to add to the filename when saving the figure.
    plot_title : str, default ""
        Title of the plot.
    save_fig : bool, default False
        Whether to save the figure.
    figures_dir : str, default './'
        Directory to save the figure.
    fontsize : dict, optional
        Dictionary with font sizes for different elements:
        {
            'title': 13,
            'xlabel': 13,
            'ylabel': 13,
            'xticks': 12,
            'yticks': 8,
            'legend': 10,
            'colorbar': 13
        }
    figure_size : tuple, default (6, 5)
        Figure size in inches (width, height)
    **shap_kwargs : dict
        Additional keyword arguments to pass to shap.summary_plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the SHAP plot.
    """
    # Default font sizes
    default_fontsize = {
        "title": 12,
        "xlabel": 13,
        "ylabel": 13,
        "xticks": 12,
        "yticks": 12,
        "legend": 10,
        "colorbar": 13,
    }

    if fontsize is None:
        fontsize = default_fontsize
    else:
        # Update defaults with provided values
        default_fontsize.update(fontsize)
        fontsize = default_fontsize

    # Default group labels if not provided
    if group_tests is None:
        group_tests = ["group1", "group2"]

    # Create an axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)

    # Create the SHAP summary plot
    shap.summary_plot(
        values,
        features=features,
        plot_type="dot",
        cmap=cmap,
        max_display=max_display,
        plot_size=[figure_size[0], figure_size[1]],
        show=False,
        **shap_kwargs,
    )

    # Customize colorbar
    fig = plt.gcf()
    if fig.axes:
        cbar = fig.axes[-1]
        cbar.set_ylabel("Feature value", fontsize=fontsize["colorbar"])
        cbar.tick_params(labelsize=fontsize["colorbar"])

    # Get and set x-ticks
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xlabel("")

    # Add custom x-axis labels
    ax.text(
        0.2,
        -0.06,
        group_tests[0],
        ha="center",
        va="top",
        fontsize=fontsize["xlabel"],
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        -0.06,
        group_tests[1],
        ha="center",
        va="top",
        fontsize=fontsize["xlabel"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.14,
        "Prediction toward...",
        ha="center",
        va="top",
        fontsize=fontsize["xlabel"],
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Customize tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize["xticks"])

    # Process y-tick labels only if pattern is provided
    if pattern:
        yticklabels = [
            re.sub(pattern, "", label.get_text()) for label in ax.get_yticklabels()
        ]
    else:
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]

    ax.set_yticklabels(yticklabels, fontsize=fontsize["yticks"])

    # Adjust layout
    ax.tick_params(axis="y", pad=-10)
    ax.yaxis.set_label_coords(-0.3, 0.5)
    ax.set_xlim((values.min(), values.max()))

    # Add title if provided
    if plot_title:
        ax.set_title(plot_title, fontsize=fontsize["title"], weight="bold", pad=10)

    # Enhance grid appearance
    ax.grid(True, axis="y", linestyle="--", color="lightgrey", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.patch.set_alpha(0.0)

    # Save the figure if requested
    if save_fig:
        save_path = Path(
            figures_dir, f"shap_{'-'.join(group_tests)}_{filename_label}.pdf"
        )
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return ax


def test_contingency(contingency_table, method="fisher"):
    """
    Perform Fisher's Exact test or Chi-square test.

    Parameters
    ----------
    contingency_table : 2×2 numpy array or list of lists
    method : "fisher" or "chisquare"

    Returns
    -------
    odds_ratio : float or None
    p_value : float
    """

    contingency_table = np.array(contingency_table)

    if method.lower() == "fisher":
        odds_ratio, p_value = fisher_exact(contingency_table)
        annotation = (
            f"Fisher Exact p = {format_pval(p_value)}\nOdds Ratio = {odds_ratio:.2f}"
        )

        return p_value, annotation

    elif method.lower() in ["chi", "chisq", "chisquare", "chi-square"]:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        annotation = f"X² p = {format_pval(p_value)}"

        return p_value, annotation

    else:
        raise ValueError("method must be 'fisher' or 'chisquare'")


def barplot_counts_fisher_test(
    df,
    cat_var1,
    cat_var2,
    method="fisher",
    legend_title=None,
    y_label="Count",
    x_label=None,
    title=None,
    mapping1=None,
    mapping2=None,
    figsize=(6, 4),
    palette="viridis",
    save_path=None,
    suffix_file=None,
    fontsize=None,
):
    """
    Create a grouped bar plot from a 2x2 contingency table of two categorical variables,
    run Fisher's exact test, and annotate the plot with the p-value and odds ratio.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    cat_var1 : str
        The name of the first categorical variable (e.g., 'DCR').
    cat_var2 : str
        The name of the second categorical variable (e.g., 'Antigen Score (Dichotomized)').
    method : str, default "fisher"
        Method to use for the test. Can be "fisher" or "chisquare".
    legend_title : str, optional
        Title for the legend corresponding to cat_var1 (default: cat_var1).
    y_label : str, optional
        Label for the y-axis (default is 'Count').
    title : str, optional
        Title for the plot.
    mapping1 : dict, optional
        A dictionary to map the values of cat_var1 to descriptive labels (e.g., {0.0: 'Non-Responder', 1.0: 'Responder'}).
    mapping2 : dict, optional
        A dictionary to map the values of cat_var2 to descriptive labels (e.g., {0: 'Low', 1: 'High'}).
    figsize : tuple, optional
        Figure size (default is (6, 4)).
    palette : str or list, optional
        Color palette for the plot (default is 'viridis').
    save_path : str or Path, optional
        If provided, the plot will be saved to this path.
    suffix_file : str, default None
        Suffix name for to save the plot. If None, default is fisherTest.png
    fontsize : dict, optional
        Dictionary of font sizes: {"title": int, "xlabel": int, "ylabel": int, xtick": int, ytick": int,  "legend": int, "annotation": int}.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    if fontsize is None:
        fontsize = {
            "title": 12,
            "xlabel": 10,
            "ylabel": 10,
            "xtick": 10,
            "ytick": 10,
            "legend": 10,
            "annotation": 10,
        }

    # Compute the contingency table (rows: cat_var1, columns: cat_var2)
    contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])

    # Run Chi2 or Fisher's exact test (works for 2x2 tables)
    p_value, annotation = test_contingency(contingency_table, method)

    # Convert the contingency table into tidy (long) format
    df_plot = contingency_table.stack().reset_index()
    df_plot.columns = [cat_var1, cat_var2, "Count"]

    # Map numeric codes to descriptive labels if mapping dictionaries are provided
    if mapping1 is not None:
        df_plot[cat_var1] = df_plot[cat_var1].map(mapping1)
    if mapping2 is not None:
        df_plot[cat_var2] = df_plot[cat_var2].map(mapping2)

    # Use the first categorical variable as the legend title if not provided
    if legend_title is None:
        legend_title = cat_var1

    # Create the grouped bar plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_plot, x=cat_var2, y="Count", hue=cat_var1, palette=palette, ax=ax
    )

    if title is not None:
        ax.set_title(title, fontsize=fontsize["title"])
    if x_label is None:
        ax.set_xlabel(cat_var2, fontsize=fontsize["xlabel"])
    else:
        ax.set_xlabel(x_label, fontsize=fontsize["xlabel"])
    ax.set_ylabel(y_label, fontsize=fontsize["ylabel"])
    ax.legend(loc="lower left", fontsize=fontsize["legend"])

    ax.tick_params(axis="x", labelsize=fontsize["xtick"])
    ax.tick_params(axis="y", labelsize=fontsize["ytick"])

    # Annotate the plot with Fisher's test results in the top center of the axis
    ax.text(
        0.5,
        0.1,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize["annotation"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
    )

    # Place legend outside upper left, no background, no border/margin
    plt.legend(
        loc="upper left",  # anchor to upper left
        bbox_to_anchor=(0.01, 1.22),  # shift outside the axes
        frameon=False,  # no frame
        borderaxespad=0,  # no extra margin
        fontsize=fontsize["legend"],
    )

    plt.tight_layout()

    if save_path is not None:
        if suffix_file is not None:
            plt.savefig(
                Path(save_path) / f"fisherTest_{suffix_file}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                Path(save_path) / "fisherTest.pdf", dpi=600, bbox_inches="tight"
            )

    return fig


def boxplot_compare_distribution_by_category(
    df,
    cont_var,
    cat_var,
    cat_mapping=None,
    figsize=(6, 4),
    palette="viridis",
    x_label=None,
    y_label=None,
    title=None,
    save_path=None,
    suffix_file=None,
):
    """
    Plot the distribution of a continuous variable by a categorical variable,
    perform a Mann–Whitney U test if there are exactly 2 groups or a
    Kruskal–Wallis H test if there are more than 2 groups, and annotate
    the plot with the p-value and test statistic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    cont_var : str
        Name of the continuous variable (e.g., "Antigen Score (Scaled)").
    cat_var : str
        Name of the categorical variable (e.g., "DCR" or "ORR").
    cat_mapping : dict, optional
        Mapping for the categorical variable values to descriptive labels
        (e.g., {0: 'Non-Responder', 1: 'Responder'}). If provided, a new
        column "cat_label" is created for plotting.
    figsize : tuple, optional
        Figure size (default is (6,4)).
    palette : str or list, optional
        Color palette for the boxplot (default is "viridis").
    x_label : str, optional
        Label for the x-axis. Defaults to the (mapped) categorical variable.
    y_label : str, optional
        Label for the y-axis. Defaults to the continuous variable name.
    title : str, optional
        Plot title.
    save_path : str or Path, optional
        If provided, the plot will be saved to this path.
    suffix_file : str, optional
        Suffix to append to the saved filename.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # If a mapping is provided, add a new column for descriptive labels.
    if cat_mapping is not None:
        df = df.copy()  # Avoid modifying the original DataFrame
        df["cat_label"] = df[cat_var].map(cat_mapping)
        plot_cat = "cat_label"
        groups = list(cat_mapping.keys())
    else:
        plot_cat = cat_var
        groups = df[
            cat_var
        ].unique()  # Get the unique groups from the original categorical variable.

    # Depending on the number of groups, run the appropriate test.
    if len(groups) == 2:
        test_name = "Mann–Whitney"
        group0 = df.loc[df[cat_var] == groups[0], cont_var]
        group1 = df.loc[df[cat_var] == groups[1], cont_var]
        stat, p_value = mannwhitneyu(group0, group1, alternative="two-sided")
        # stat_str = f"U = {stat}"
    elif len(groups) > 2:
        test_name = "Kruskal–Wallis"
        group_data = [df.loc[df[cat_var] == g, cont_var] for g in groups]
        stat, p_value = kruskal(*group_data)
        # stat_str = f"H = {stat:.2f}"
    else:
        raise ValueError("The categorical variable must have at least 2 unique groups.")

    print(f"{test_name} Test statistic: {stat}, p-value: {p_value:.3f}")

    # Set default axis labels if not provided.
    if x_label is None:
        x_label = plot_cat
    if y_label is None:
        y_label = cont_var

    # Create the plot.
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x=plot_cat, y=cont_var, data=df, palette=palette, ax=ax, showfliers=False
    )
    sns.swarmplot(x=plot_cat, y=cont_var, data=df, color="0.2", ax=ax)

    # Annotate the plot with the p-value and test statistic.
    # max_val = df[cont_var].max()
    annotation = f"{test_name} p = {format_pval(p_value)}"
    ax.text(
        0.5,
        0.85,
        annotation,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    # Save the figure if a save path is provided.
    if save_path is not None:
        if suffix_file is not None:
            file_name = f"{test_name}_Test_{suffix_file}.png"
        else:
            file_name = f"{test_name}_Test.png"
        plt.savefig(Path(save_path) / file_name, dpi=600, bbox_inches="tight")

    return fig
