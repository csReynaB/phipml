import glob
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
