"""
Plots the evolution of the CSA since baseline at lesion level vs. outside the lesion (average of
the slices just above and just below it), for one lesion per subject, using the lesion-level csvs
of compute_lesion_level_csa.py. Two panels side by side: mean CSA (left) and median CSA (right).

Each curve is the percent change since the subject's baseline scan of the CSA at lesion level
(red) or of the average of the above and below CSA (blue), vs. months since baseline. Each
subject has its own marker.

A second figure, saved next to the first with "_yearly_change" added to the file name, shows the
yearly CSA change per subject (slope of the percent change vs. years since baseline, fitted
through the baseline), outside vs. at lesion level: one dot per subject and level, a grey line
linking each subject's two values, and the median of each level marked by a black bar.

Example:
    python plot_lesion_vs_outside_csa.py -o figure.png \
        --subject sub-003 sub-003_lesion_level_csa_smooth10.csv 1 \
        --subject sub-005 sub-005_lesion_level_csa_smooth10.csv 2

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compute_lesion_level_csa import parse_session_date

LESION_COLOR = "#c44e52"
OUTSIDE_COLOR = "#4c72b0"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def yearly_change(months, percent_change):
    """Slope of the percent change since baseline vs. years, fitted through the baseline (0, 0%)."""
    years = months.to_numpy() / 12
    percent = percent_change.to_numpy()
    valid = np.isfinite(years) & np.isfinite(percent)
    denominator = (years[valid] ** 2).sum()
    return (years[valid] * percent[valid]).sum() / denominator if denominator > 0 else np.nan


parser = argparse.ArgumentParser(description="Plot the CSA evolution at lesion level vs. outside the lesion (mean and median).")
parser.add_argument("--subject", nargs=3, action="append", required=True, metavar=("SUBJECT", "CSV", "LESION_ID"),
                    help="Subject id, its lesion-level csv and the lesion_area_id to plot. Repeat per subject.")
parser.add_argument("-o", "--output", required=True, help="Path to the output png file.")
args = parser.parse_args()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
yearly = {"mean": {}, "median": {}}

for idx, (subject_id, csv_path, lesion_id) in enumerate(args.subject):
    df = pd.read_csv(csv_path)
    df = df[df["lesion_area_id"] == int(lesion_id)].copy()
    df["date"] = pd.to_datetime(df["session_id"].map(parse_session_date))
    df = df.sort_values("date")
    months = (df["date"] - df["date"].iloc[0]).dt.days / 30.4375

    for ax, stat in zip(axes, ["mean", "median"]):
        lesion = df[f"CSA_{stat}_mm2"]
        outside = df[[f"CSA_above_{stat}_mm2", f"CSA_below_{stat}_mm2"]].mean(axis=1)
        slopes = {}
        for color, values in [(LESION_COLOR, lesion), (OUTSIDE_COLOR, outside)]:
            percent_change = 100 * (values / values.iloc[0] - 1)
            ax.plot(months, percent_change, color=color, marker=MARKERS[idx % len(MARKERS)],
                    markersize=7, linewidth=2)
            slopes[color] = yearly_change(months, percent_change)
        yearly[stat][subject_id] = (slopes[OUTSIDE_COLOR], slopes[LESION_COLOR])

for ax, stat in zip(axes, ["Mean", "Median"]):
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{stat} CSA", fontweight="bold")
    ax.set_xlabel("Months since baseline scan")
    ax.yaxis.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
axes[0].set_ylabel("CSA change since baseline (%)")

color_handles = [Line2D([0], [0], color=LESION_COLOR, linewidth=2, label="At lesion level"),
                 Line2D([0], [0], color=OUTSIDE_COLOR, linewidth=2, label="Outside lesion level: above + below average")]
marker_handles = [Line2D([0], [0], color="grey", marker=MARKERS[idx % len(MARKERS)], linestyle="", markersize=7, label=subject_id)
                  for idx, (subject_id, _, _) in enumerate(args.subject)]
for ax in axes:
    color_legend = ax.legend(loc="lower left", handles=color_handles)
    ax.add_artist(color_legend)
    ax.legend(loc="lower right", handles=marker_handles)

fig.tight_layout()
fig.savefig(args.output, dpi=300, bbox_inches="tight")

# Second figure: yearly change per subject, outside vs. at lesion level, linked by a grey line
fig_yearly, axes_yearly = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
median_handle = Line2D([0], [0], color="black", linewidth=3, label="Median")
for ax, stat in zip(axes_yearly, ["mean", "median"]):
    for idx, (subject_id, _, _) in enumerate(args.subject):
        outside_change, lesion_change = yearly[stat][subject_id]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot([0, 1], [outside_change, lesion_change], color="grey", linewidth=1.5, zorder=1)
        ax.scatter([0], [outside_change], color=OUTSIDE_COLOR, marker=marker, s=70, zorder=2)
        ax.scatter([1], [lesion_change], color=LESION_COLOR, marker=marker, s=70, zorder=2)
    for x, position in [(0, 0), (1, 1)]:
        median = np.nanmedian([changes[position] for changes in yearly[stat].values()])
        ax.hlines(median, x - 0.08, x + 0.08, color="black", linewidth=3, zorder=3)
        ax.text(x + 0.12, median, f"{median:.1f} %", ha="left", va="center", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Outside lesion", "At lesion"])
    ax.set_title(f"{stat.capitalize()} CSA", fontweight="bold")
    ax.yaxis.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(handles=[median_handle], loc="lower left")
axes_yearly[0].set_ylabel("Yearly CSA change (% per year)")
fig_yearly.tight_layout()
yearly_path, extension = os.path.splitext(args.output)
fig_yearly.savefig(f"{yearly_path}_yearly_change{extension}", dpi=300, bbox_inches="tight")
