"""
This script plots the evolution of the spinal cord cross-sectional area (CSA) over time, in the
lesion level versus in a lesion-free spinal cord region outside it, for one or several subjects.

Each subject is given as a pair of AUC csvs (output of compute_lesion_auc.py, or the same csv
format computed on a healthy region): one for the lesion area and one for the healthy region.
The plotted quantity is the CSA AUC (area under the CSA curve over the region, in mm2 x slices,
as computed by compute_lesion_auc.py). The x-axis is the time in months since the baseline
(oldest) scan of that subject. Lesion levels are drawn in red, levels outside the lesion in blue,
and each subject has its own marker.

Two figures are saved:
    - atrophy_csa_auc.png: CSA AUC vs. months since baseline
    - atrophy_csa_auc_percent_change.png: percent change of the CSA AUC relative to baseline
      (100 * (AUC / baseline AUC - 1)) vs. months since baseline

Input:
    --subject: "<subject_id> <lesion_area_csv> <healthy_region_csv>". Repeat once per subject.
        If a csv holds several lesion areas (lesion_area_id), each is drawn as its own line.
    -o / --output_dir: folder where the two figures are saved

Example:
    python plot_atrophy_evolution.py -o figures \
        --subject sub-003 sub-003_lesion_auc_smooth10.csv sub-003_healthy_region_850_890_auc_smooth10.csv \
        --subject sub-006 sub-006_lesion_auc_smooth10.csv sub-006_healthy_region_auc_smooth10.csv

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_subject_csa import LABELS_FONT_SIZE, TICKS_FONT_SIZE
from compute_lesion_auc import parse_session_date

DAYS_PER_MONTH = 30.4375
LESION_COLOR = "#c44e52"
HEALTHY_COLOR = "#4c72b0"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the evolution of the CSA in the lesion area vs. healthy spinal cord.")
    parser.add_argument("--subject", type=str, nargs=3, action="append", required=True,
                        metavar=("SUBJECT", "LESION_AREA_CSV", "HEALTHY_REGION_CSV"),
                        help="Subject id, its lesion-area AUC csv and its healthy-region AUC csv. Repeat for each subject.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Folder where the figures are saved.")
    return parser.parse_args()


def load_series(csv_path):
    """
    Reads an AUC csv and returns one DataFrame per lesion area, sorted chronologically, with the
    columns months_since_baseline and auc_percent_change (alongside the csv's AUC).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"WARNING: {csv_path} has no rows, skipped.")
        return []

    dates = df["session_id"].map(parse_session_date)
    if dates.isna().any():
        raise SystemExit(f"Could not parse a YYYYMMDD date from every session_id in {csv_path}")
    df["date"] = pd.to_datetime(dates)

    series = []
    for _, df_area in df.groupby("lesion_area_id"):
        df_area = df_area.sort_values("date").copy()
        df_area["months_since_baseline"] = (df_area["date"] - df_area["date"].iloc[0]).dt.days / DAYS_PER_MONTH
        df_area["auc_percent_change"] = 100 * (df_area["AUC"] / df_area["AUC"].iloc[0] - 1)
        series.append(df_area)
    return series


def plot_evolution(subject_series, y_column, ylabel, output_png, zero_line=False):
    """
    subject_series: list of (subject_id, marker, lesion_series, healthy_series).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for subject_id, marker, lesion_series, healthy_series in subject_series:
        for color, series in [(LESION_COLOR, lesion_series), (HEALTHY_COLOR, healthy_series)]:
            for df_area in series:
                ax.plot(df_area["months_since_baseline"], df_area[y_column], color=color, marker=marker,
                        markersize=7, linewidth=2)

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    region_handles = [Line2D([0], [0], color=LESION_COLOR, linewidth=2, label="At lesion level"),
                      Line2D([0], [0], color=HEALTHY_COLOR, linewidth=2, label="Outside lesion level")]
    subject_handles = [Line2D([0], [0], color="grey", marker=marker, linestyle="", markersize=7, label=subject_id)
                       for subject_id, marker, _, _ in subject_series]
    region_legend = ax.legend(handles=region_handles, loc="lower left", fontsize=TICKS_FONT_SIZE)
    ax.add_artist(region_legend)
    ax.legend(handles=subject_handles, loc="lower right", fontsize=TICKS_FONT_SIZE)

    ax.set_xlabel("Months since baseline scan", fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABELS_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICKS_FONT_SIZE)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_png}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    subject_series = []
    for idx, (subject_id, lesion_csv, healthy_csv) in enumerate(args.subject):
        subject_series.append((subject_id, MARKERS[idx % len(MARKERS)],
                               load_series(lesion_csv), load_series(healthy_csv)))

    plot_evolution(subject_series, "AUC", "CSA AUC (mm² × slices)",
                   os.path.join(args.output_dir, "atrophy_csa_auc.png"))
    plot_evolution(subject_series, "auc_percent_change", "CSA AUC change since baseline (%)",
                   os.path.join(args.output_dir, "atrophy_csa_auc_percent_change.png"), zero_line=True)


if __name__ == "__main__":
    main()
