"""
Summarizes, across all subjects, the sensitivity of the per-lesion-area CSA AUC to the
moving-average smoothing window (-s/--smooth_window of compute_lesion_auc.py).

Expects a folder holding the per-subject, per-window AUC csvs produced by compute_lesion_auc.py,
named "<subject_id>_lesion_auc_smooth<n>.csv" (or "<subject_id>_lesion_auc.csv", read as n=1).
Every subject must be present at every window. Generate them with, e.g.:

    for n in 1 5 10 20; do
      for csv in /path/to/*_csa_with_lesions.csv; do
        sub=$(basename "$csv" _csa_with_lesions.csv)
        python compute_lesion_auc.py -i "$csv" -o /path/to/auc/${sub}_lesion_auc_smooth${n}.csv -s $n
      done
    done

The longitudinal metric reported here is the AUC ratio to the BASELINE timepoint (each lesion
area's first session), recomputed from the per-session AUC values, rather than the ratio to the
previous timepoint that compute_lesion_auc.py writes. Deviations are reported against the
REFERENCE window (-r, default 10) -- the window whose results the paper reports -- so the table
answers "would the reported conclusion change had a different window been chosen?".

An "observation" is one post-baseline (subject, lesion area, session) triplet. Baseline sessions
are excluded: their ratio is 1.0 at every window by construction and would dilute the summary.

Outputs (written to -o, default the input folder):
  - sensitivity_long.csv        : one row per observation per window, with baseline ratios,
                                  lesion-area width and the window as a % of that width
  - sensitivity_table.csv/.tex  : THE supplementary table -- one row per window
  - sensitivity_by_lesion_area.csv : per lesion area, its width and worst deviation, sorted
                                  narrowest first (narrow areas are the sensitive ones)
  - sensitivity_trajectories.png: small multiple, one panel per lesion area, ratio to baseline
                                  vs. days since baseline, one line per window
  - sensitivity_deviations.png  : pooled strip plot of deviation from the reference window
Report one of the two figures, not both.

Author: Pierre-Louis Benveniste
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUC_COLUMNS = ["AUC", "AUC_left", "AUC_right"]
RATIO_COLUMNS = ["AUC_ratio_to_baseline", "AUC_left_ratio_to_baseline", "AUC_right_ratio_to_baseline"]
AREA_KEYS = ["subject_id", "lesion_area_id"]
OBSERVATION_KEYS = AREA_KEYS + ["session_id"]
FILENAME_PATTERN = re.compile(r"^(?P<subject>.+?)_lesion_auc(?:_smooth(?P<window>\d+))?\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-subject sensitivity of the CSA AUC to the smoothing window.")
    parser.add_argument("-i", "--input_dir", type=str, default=".",
                        help="Folder holding the per-subject, per-window AUC csvs. Default: current folder.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Folder for the outputs. Default: the input folder.")
    parser.add_argument("-r", "--reference_window", type=int, default=10,
                        help="Window the paper reports; all deviations are measured against it. Default 10.")
    parser.add_argument("-t", "--atrophy_threshold", type=float, default=0.05,
                        help="Atrophy threshold defining a 'conclusion', as a fraction of baseline AUC: an "
                             "observation is called atrophic when its ratio to baseline is below 1 - t. The "
                             "table counts observations whose call flips relative to the reference window. "
                             "Default 0.05 (5%%). Set to match your primary endpoint.")
    return parser.parse_args()


def load_auc_csvs(input_dir):
    """
    Reads every "<subject>_lesion_auc[_smooth<n>].csv" in input_dir into one long DataFrame with a
    smooth_window column. A file with no _smooth suffix is taken as the unsmoothed n=1 run.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*_lesion_auc*.csv"))):
        match = FILENAME_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        df = pd.read_csv(path)
        if df.empty:
            # Subject with no lesion at any timepoint: compute_lesion_auc.py writes a header-only csv.
            continue
        df["smooth_window"] = int(match.group("window") or 1)
        if "subject_id" not in df.columns:
            df["subject_id"] = match.group("subject")
        frames.append(df)

    if not frames:
        raise SystemExit(f"No '*_lesion_auc*.csv' files found in {input_dir}")

    return pd.concat(frames, ignore_index=True)


def add_baseline_ratios(df):
    """
    Adds, per (smooth_window, subject, lesion area), each timepoint's AUC divided by the AUC at that
    lesion area's baseline (chronologically first) session, the days elapsed since it, the lesion
    area's width in slices, and the smoothing window as a percentage of that width.
    Replaces the ratio-to-previous-timepoint columns written by compute_lesion_auc.py.
    """
    df = df.sort_values(["smooth_window"] + OBSERVATION_KEYS).copy()
    group = df.groupby(["smooth_window"] + AREA_KEYS, sort=False)

    for auc_column, ratio_column in zip(AUC_COLUMNS, RATIO_COLUMNS):
        df[ratio_column] = df[auc_column] / group[auc_column].transform("first")

    session_date = pd.to_datetime(df["session_id"].str.extract(r"(\d{8})")[0], format="%Y%m%d")
    baseline_date = session_date.groupby([df["smooth_window"], df["subject_id"], df["lesion_area_id"]]).transform("first")
    df["days_since_baseline"] = (session_date - baseline_date).dt.days

    df["lesion_area_width"] = df["end_pam50_slice"] - df["start_pam50_slice"] + 1
    df["window_pct_of_width"] = 100 * df["smooth_window"] / df["lesion_area_width"]

    stale = [c for c in df.columns if c.endswith("_ratio_to_previous_timepoint")] + ["days_since_previous_timepoint"]
    return df.drop(columns=[c for c in stale if c in df.columns])


def get_deviations(df_long, reference_window, ratio_column, quiet=False):
    """
    Wide table of one ratio column: one row per post-baseline observation, one column per window,
    plus "dev_n<window>" columns giving that window's deviation from the reference window, in
    percentage points of the ratio (0.7997 vs 0.8084 -> 0.87 pp).
    """
    df_post = df_long[df_long["days_since_baseline"] > 0]
    wide = df_post.pivot_table(index=OBSERVATION_KEYS + ["lesion_area_width"],
                               columns="smooth_window", values=ratio_column).reset_index()

    if reference_window not in wide.columns:
        raise SystemExit(f"Reference window n={reference_window} not found among the loaded windows "
                         f"{sorted(c for c in wide.columns if isinstance(c, (int, np.integer)))}")

    windows = sorted(c for c in wide.columns if isinstance(c, (int, np.integer)))
    incomplete = wide[windows].isna().any(axis=1).sum()
    if incomplete:
        if not quiet:
            print(f"WARNING: {incomplete} observation(s) are missing at least one window and are dropped.")
        wide = wide.dropna(subset=windows)

    for window in windows:
        wide[f"dev_n{window}"] = 100 * (wide[window] - wide[reference_window])

    return wide, windows


def spearman(a, b):
    """Spearman correlation, via Pearson on ranks (no scipy needed)."""
    return a.rank().corr(b.rank())


def build_table(df_long, wide, windows, reference_window, atrophy_threshold):
    """
    The supplementary table: one row per window, giving how far that window's baseline-referenced
    ratios sit from the reference window's, and how often the atrophy call changes.
    """
    reference_ratio = wide[reference_window]
    is_atrophic_reference = reference_ratio < (1 - atrophy_threshold)
    median_width_by_window = df_long.groupby("smooth_window")["lesion_area_width"].median()

    rows = []
    for window in windows:
        deviation = wide[f"dev_n{window}"].abs()
        is_atrophic = wide[window] < (1 - atrophy_threshold)
        rows.append({
            "smooth_window": window,
            "window_pct_of_median_width": 100 * window / median_width_by_window.loc[window],
            "median_abs_dev_pp": deviation.median(),
            "p95_abs_dev_pp": deviation.quantile(0.95),
            "max_abs_dev_pp": deviation.max(),
            "n_atrophy_call_changed": int((is_atrophic != is_atrophic_reference).sum()),
            "n_direction_changed": int(((wide[window] < 1) != (reference_ratio < 1)).sum()),
            "spearman_rho_vs_reference": spearman(wide[window], reference_ratio),
        })

    df_table = pd.DataFrame(rows)
    # The reference window is trivially identical to itself; blank it out rather than printing zeros.
    is_reference = df_table["smooth_window"] == reference_window
    df_table.loc[is_reference, ["median_abs_dev_pp", "p95_abs_dev_pp", "max_abs_dev_pp"]] = np.nan
    return df_table


LATEX_HEADERS = {
    "smooth_window": r"Window $n$",
    "window_pct_of_median_width": r"\% of median area width",
    "median_abs_dev_pp": r"Median $|\Delta|$ (pp)",
    "p95_abs_dev_pp": r"95th pct (pp)",
    "max_abs_dev_pp": r"Max (pp)",
    "n_atrophy_call_changed": "Atrophy call changed",
    "n_direction_changed": "Direction changed",
    "spearman_rho_vs_reference": r"Spearman $\rho$",
}


def write_latex_table(df_table, n_observations, output_tex):
    """
    Paper-ready LaTeX body: readable headers (the csv's snake_case would not compile), and the
    two count columns rendered as "k/N" so the denominator travels with the number.
    """
    df_tex = df_table.copy()
    for column in ["n_atrophy_call_changed", "n_direction_changed"]:
        df_tex[column] = df_tex[column].map(lambda k: f"{k}/{n_observations}")
    df_tex = df_tex.rename(columns=LATEX_HEADERS)

    with open(output_tex, "w") as f:
        f.write(df_tex.to_latex(index=False, escape=False, float_format="%.2f", na_rep="--"))


def build_by_lesion_area(wide, windows, reference_window):
    """Per lesion area: its width and its worst deviation at each non-reference window, narrowest first."""
    deviation_columns = [f"dev_n{w}" for w in windows if w != reference_window]
    df_area = (wide.groupby(AREA_KEYS + ["lesion_area_width"])[deviation_columns]
                   .apply(lambda g: g.abs().max()).reset_index())
    df_area["worst_abs_dev_pp"] = df_area[deviation_columns].max(axis=1)
    return df_area.sort_values("lesion_area_width").rename(
        columns={c: c.replace("dev_n", "max_abs_dev_n") + "_pp" for c in deviation_columns})


def plot_trajectories(df_long, windows, output_png):
    """Small multiple: one panel per lesion area, ratio to baseline vs. days since baseline."""
    areas = list(df_long.groupby(AREA_KEYS, sort=True).groups.keys())
    ncols = min(3, len(areas))
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             sharey=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(windows)))

    for ax, (subject_id, area_id) in zip(axes.ravel(), areas):
        df_area = df_long[(df_long["subject_id"] == subject_id) & (df_long["lesion_area_id"] == area_id)]
        for window, color in zip(windows, colors):
            df_window = df_area[df_area["smooth_window"] == window].sort_values("days_since_baseline")
            ax.plot(df_window["days_since_baseline"], df_window["AUC_ratio_to_baseline"],
                    marker="o", ms=4, color=color, label=f"n={window}")
        width = int(df_area["lesion_area_width"].iloc[0])
        ax.set_title(f"{subject_id} · area {area_id} ({width} slices)", fontsize=9)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.3)

    for ax in axes.ravel()[len(areas):]:
        ax.set_visible(False)
    # The last row can be partly empty, so label the bottom-most *visible* panel of each column.
    for column in range(ncols):
        visible = [ax for ax in axes[:, column] if ax.get_visible()]
        if visible:
            visible[-1].set_xlabel("days since baseline")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC ratio to baseline")
    axes[0, 0].legend(fontsize=8, title="window", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_png}")


def plot_deviations(wide, windows, reference_window, output_png):
    """Pooled strip plot: deviation from the reference window, one dot per observation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)

    for x, window in enumerate(windows):
        deviation = wide[f"dev_n{window}"]
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(deviation)), deviation,
                   s=18, alpha=0.6, color="tab:grey" if window == reference_window else "tab:blue")
        ax.hlines(deviation.median(), x - 0.25, x + 0.25, color="tab:red", linewidth=2, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"n={w}" + ("\n(reference)" if w == reference_window else "") for w in windows])
    ax.set_ylabel(f"deviation from n={reference_window} (pp of ratio to baseline)")
    ax.set_title("Sensitivity of baseline-referenced CSA AUC to smoothing window")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_png}")


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_long = add_baseline_ratios(load_auc_csvs(input_dir))
    wide, windows = get_deviations(df_long, args.reference_window, "AUC_ratio_to_baseline")

    df_table = build_table(df_long, wide, windows, args.reference_window, args.atrophy_threshold)
    df_area = build_by_lesion_area(wide, windows, args.reference_window)

    # Signal-to-sensitivity: the observed atrophy the study reports, against the largest shift any
    # alternative window could have produced. This is the number that decides whether the
    # sensitivity analysis is a footnote or a caveat.
    mean_atrophy_pp = 100 * (1 - wide[args.reference_window]).abs().mean()
    max_deviation_pp = df_table["max_abs_dev_pp"].max()

    # Hemicord metrics get a caption clause, not their own table.
    hemicord_max_dev = {}
    for ratio_column in ["AUC_left_ratio_to_baseline", "AUC_right_ratio_to_baseline"]:
        wide_hemi, _ = get_deviations(df_long, args.reference_window, ratio_column, quiet=True)
        hemicord_max_dev[ratio_column] = max(wide_hemi[f"dev_n{w}"].abs().max() for w in windows)

    df_long.to_csv(os.path.join(output_dir, "sensitivity_long.csv"), index=False)
    df_table.to_csv(os.path.join(output_dir, "sensitivity_table.csv"), index=False)
    df_area.to_csv(os.path.join(output_dir, "sensitivity_by_lesion_area.csv"), index=False)
    write_latex_table(df_table, len(wide), os.path.join(output_dir, "sensitivity_table.tex"))

    plot_trajectories(df_long, windows, os.path.join(output_dir, "sensitivity_trajectories.png"))
    plot_deviations(wide, windows, args.reference_window, os.path.join(output_dir, "sensitivity_deviations.png"))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\nSubjects: {df_long['subject_id'].nunique()}   "
          f"Lesion areas: {len(df_long.groupby(AREA_KEYS))}   "
          f"Post-baseline observations: {len(wide)}   "
          f"Windows: {windows}   Reference: n={args.reference_window}")
    print(f"\nSupplementary table (deviations in percentage points of the ratio to baseline, "
          f"atrophy threshold {100 * args.atrophy_threshold:.0f}%):")
    print(df_table.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="--"))
    print(f"\nPer lesion area, narrowest first:")
    print(df_area.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nMean observed atrophy at n={args.reference_window}: {mean_atrophy_pp:.2f} pp")
    print(f"Largest deviation from any alternative window: {max_deviation_pp:.2f} pp")
    print(f"Signal-to-sensitivity ratio: {mean_atrophy_pp / max_deviation_pp:.1f}x")
    print("Hemicord max deviation: " +
          ", ".join(f"{k.split('_')[1]} {v:.2f} pp" for k, v in hemicord_max_dev.items()))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
