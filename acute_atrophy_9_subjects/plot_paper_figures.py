"""
This script plots the CSA figures of selected subjects for the paper, in the same format as
plot_subject_csa.py (one line per session, lesion shading, vertebral level boundaries/labels),
with two paper-specific differences:
    - the PAM50 slice range shown can be set manually, per subject
    - sessions are renamed by time since the baseline (oldest) session: the oldest is "M0" and
      every subsequent one is "M<n>", n being the number of months since baseline (computed from
      the YYYYMMDD date embedded in session_id and rounded to the nearest month)

Smoothing (if requested) is applied on the full slice range before cropping, so slices at the
edges of the displayed range are still smoothed with their real neighbors.

Input:
    -i / --input_dir: folder holding the per-subject csvs (output of compute_csa_on_include.py,
        one "<subject_id>_csa_with_lesions.csv" per subject)
    -o / --output_dir: folder where the figures are saved ("<subject_id>_csa_paper.png")
    --subjects: subjects to plot (default: sub-003 sub-006)
    --slice_range: "<subject_id> <start_slice> <end_slice>", PAM50 slice range to display for that
        subject (both bounds inclusive). Can be given once per subject; a subject without one is
        plotted over its full slice range.
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA (default 1, no smoothing)

Example:
    python plot_paper_figures.py -i /path/to/csvs -o /path/to/figures \
        --slice_range sub-003 760 900 --slice_range sub-006 780 950 -s 10

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from plot_subject_csa import plot_csa_panel, smooth_csa_columns
from compute_lesion_auc import parse_session_date

DAYS_PER_MONTH = 30.4375


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the CSA figures of selected subjects for the paper.")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Folder holding the per-subject '<subject_id>_csa_with_lesions.csv' files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Folder where the figures are saved.")
    parser.add_argument("--subjects", type=str, nargs="+", default=["sub-003", "sub-006"],
                        help="Subjects to plot. Default: sub-003 sub-006.")
    parser.add_argument("--slice_range", type=str, nargs=3, action="append", default=[],
                        metavar=("SUBJECT", "START_SLICE", "END_SLICE"),
                        help="PAM50 slice range to display for a subject (inclusive). Repeat for each subject.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                        help="Window size (in slices) for moving-average smoothing of CSA. Default 1 (no smoothing).")
    return parser.parse_args()


def rename_sessions_by_month(df):
    """
    Replaces session_id by "M0" (oldest session) and "M<n>" (n = months since the oldest session).
    Returns the relabeled DataFrame and the new labels in chronological order.
    """
    dates = {session_id: parse_session_date(session_id) for session_id in df["session_id"].unique()}
    if any(date is None for date in dates.values()):
        raise SystemExit(f"Could not parse a YYYYMMDD date from every session_id: {sorted(dates)}")

    sessions = sorted(dates, key=dates.get)
    baseline_date = dates[sessions[0]]
    labels = {session_id: f"M{round((dates[session_id] - baseline_date).days / DAYS_PER_MONTH)}"
              for session_id in sessions}

    df = df.copy()
    df["session_id"] = df["session_id"].map(labels)
    return df, [labels[session_id] for session_id in sessions]


def plot_paper_figure(subject_id, input_csv, output_png, slice_range=None, smooth_window=1):
    df = pd.read_csv(input_csv)
    df = smooth_csa_columns(df, smooth_window, ["CSA_mm2"])
    df, sessions = rename_sessions_by_month(df)

    if slice_range is not None:
        start_slice, end_slice = slice_range
        df = df[(df["pam50_axial_slice"] >= start_slice) & (df["pam50_axial_slice"] <= end_slice)]
        if df.empty:
            raise SystemExit(f"No slices of {subject_id} in the range {start_slice}-{end_slice}.")

    n_sessions = len(sessions)
    palette = [cm.viridis(1 - i / max(n_sessions - 1, 1)) for i in range(n_sessions)]

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_csa_panel(ax, df, sessions, palette, "CSA_mm2",
                   f'CSA across PAM50 axial slices and vertebral levels for {subject_id}')

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {output_png}')


def main():
    args = parse_args()
    slice_ranges = {subject_id: (int(start), int(end)) for subject_id, start, end in args.slice_range}

    unknown = set(slice_ranges) - set(args.subjects)
    if unknown:
        raise SystemExit(f"--slice_range given for subject(s) not in --subjects: {sorted(unknown)}")

    for subject_id in args.subjects:
        input_csv = os.path.join(args.input_dir, f"{subject_id}_csa_with_lesions.csv")
        output_png = os.path.join(args.output_dir, f"{subject_id}_csa_paper.png")
        plot_paper_figure(subject_id, input_csv, output_png, slice_ranges.get(subject_id), args.smooth_window)


if __name__ == "__main__":
    main()
