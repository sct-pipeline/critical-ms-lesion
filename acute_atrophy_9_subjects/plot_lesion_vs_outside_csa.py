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

A log file ("_yearly_change_stats.log" next to the first figure) compares the yearly changes at
lesion level vs. outside the lesion, for mean and median CSA, with tests suited to paired
data from a small sample: paired t-test (comparison of means, with the mean difference and its
95% CI and Cohen's dz), Wilcoxon signed-rank test, and Shapiro-Wilk test of the differences'
normality to help choose between them. Requires scipy.

Font sizes are set with --labels_font_size (titles, axis labels, median value labels; default 20)
and --ticks_font_size (tick labels and legends; default 16).

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
from scipy import stats

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


def describe(values):
    q1, q3 = np.percentile(values, [25, 75])
    sd = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return f"mean {values.mean():.2f} (SD {sd:.2f}), median {np.median(values):.2f} (IQR {q1:.2f} to {q3:.2f})"


def paired_comparison(subject_ids, outside, lesion):
    """Log lines comparing the yearly changes at lesion level vs. outside it (paired by subject)."""
    valid = np.isfinite(outside) & np.isfinite(lesion)
    ids = np.array(subject_ids)[valid]
    outside, lesion = outside[valid], lesion[valid]
    difference = lesion - outside
    n = len(difference)

    lines = [f"  Subjects with both values (n = {n})", "",
             f"  {'subject':<10}{'outside':>10}{'at lesion':>12}{'difference':>12}"]
    lines += [f"  {s:<10}{o:>10.2f}{l:>12.2f}{d:>12.2f}" for s, o, l, d in zip(ids, outside, lesion, difference)]
    lines += ["", f"  Outside lesion : {describe(outside)}",
              f"  At lesion      : {describe(lesion)}",
              f"  Difference     : {describe(difference)}   (at lesion - outside)", ""]
    if n < 3:
        return lines + ["  Not enough subjects for statistical tests (n >= 3 needed)."]

    shapiro_w, shapiro_p = stats.shapiro(difference)
    t_stat, t_p = stats.ttest_rel(lesion, outside)
    sd_difference = difference.std(ddof=1)
    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=difference.mean(), scale=sd_difference / np.sqrt(n))
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(lesion, outside)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    lines += [
        f"  Shapiro-Wilk normality test of the paired differences: W = {shapiro_w:.3f}, p = {shapiro_p:.4f}"
        + ("  (differences not normal: prefer the Wilcoxon test)" if shapiro_p < 0.05 else "  (no evidence against normality)"),
        f"  Paired t-test, comparison of means: t({n - 1}) = {t_stat:.3f}, p = {t_p:.4f}",
        f"  Mean difference (at lesion - outside): {difference.mean():.2f} % per year, 95% CI {ci_low:.2f} to {ci_high:.2f}",
        f"  Effect size, Cohen's dz: {difference.mean() / sd_difference if sd_difference > 0 else np.nan:.2f}",
        f"  Wilcoxon signed-rank test, no normality assumption: W = {wilcoxon_stat:.1f}, p = {wilcoxon_p:.4f}",
    ]
    return lines


def write_stats_log(subjects, yearly, log_path):
    lines = ["Yearly CSA change (% per year): at lesion level vs. outside the lesion (above + below average)",
             "Yearly change = slope of the percent change since baseline vs. years, fitted through the baseline.",
             "Subjects (lesion_area_id): " + ", ".join(f"{subject_id} ({lesion_id})" for subject_id, _, lesion_id in subjects),
             "All tests are two-sided and paired by subject, alpha = 0.05. Two comparisons (mean and median CSA) are",
             "not adjusted for multiplicity (Bonferroni threshold: 0.025). Small sample: interpret with caution.", ""]
    subject_ids = [subject_id for subject_id, _, _ in subjects]
    for stat in ["mean", "median"]:
        outside = np.array([yearly[stat][subject_id][0] for subject_id in subject_ids])
        lesion = np.array([yearly[stat][subject_id][1] for subject_id in subject_ids])
        lines += [f"=== {stat.capitalize()} CSA ===", ""] + paired_comparison(subject_ids, outside, lesion) + [""]

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Statistics log saved: {log_path}")


parser = argparse.ArgumentParser(description="Plot the CSA evolution at lesion level vs. outside the lesion (mean and median).")
parser.add_argument("--subject", nargs=3, action="append", required=True, metavar=("SUBJECT", "CSV", "LESION_ID"),
                    help="Subject id, its lesion-level csv and the lesion_area_id to plot. Repeat per subject.")
parser.add_argument("-o", "--output", required=True, help="Path to the output png file.")
parser.add_argument("--labels_font_size", type=int, default=20,
                    help="Font size of the titles, axis labels and value labels. Default 20.")
parser.add_argument("--ticks_font_size", type=int, default=16,
                    help="Font size of the tick labels and legends. Default 16.")
args = parser.parse_args()

plt.rcParams.update({"font.size": args.labels_font_size, "axes.titlesize": args.labels_font_size,
                     "axes.labelsize": args.labels_font_size, "xtick.labelsize": args.ticks_font_size,
                     "ytick.labelsize": args.ticks_font_size, "legend.fontsize": args.ticks_font_size})

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
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
fig_yearly, axes_yearly = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
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
write_stats_log(args.subject, yearly, f"{yearly_path}_yearly_change_stats.log")
