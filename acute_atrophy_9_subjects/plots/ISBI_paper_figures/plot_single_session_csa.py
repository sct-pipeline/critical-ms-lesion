"""
This script plots the cross-sectional area (CSA) across PAM50 axial slices and vertebral levels
for a SINGLE session, using the per-scan csv produced by compute_csa_on_include.py
(csa_with_lesions.csv, i.e. the same csv as the per-subject one but without the
subject_id/session_id columns).

It reuses plot_csa_panel() from acute_atrophy_9_subjects/plots/plot_subject_csa.py, so the output has
exactly the same format as the multi-session subject plots (lesion shading, vertebral level
boundaries/labels, inverted x-axis, same fonts/legend). The only difference is that there is a
single line instead of one line per timepoint, drawn in blue.

Input:
    -i / --input_csv: path to the single-session csv (e.g. csa_with_lesions.csv)
    -o / --output: path to the output png file
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA (default 1, no smoothing)
    --subject_id: subject id used in the figure title (default sub-003)
    --session_id: session id used as the legend label (default M17)
    --hemi: plot left/right hemicord CSA as two stacked panels instead of the total CSA

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

LINE_COLOR = 'tab:blue'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_subject_csa import plot_csa_panel, smooth_csa_columns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSA across PAM50 axial slices and vertebral levels for a single session.")
    parser.add_argument("-i", "--input_csv", type=str, default="csa_with_lesions.csv",
                        help="Path to the single-session csv (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output", type=str, default="sub-003_csa_plot.png",
                        help="Path to the output png file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                        help="Window size (in slices) for moving-average smoothing of CSA before plotting. "
                             "Default 1 (no smoothing).")
    parser.add_argument("--subject_id", type=str, default="sub-003", help="Subject id shown in the figure title.")
    parser.add_argument("--session_id", type=str, default="M17", help="Session id shown in the legend.")
    parser.add_argument("--hemi", action="store_true",
                        help="Plot left/right hemicord CSA as two stacked panels instead of the total CSA.")
    return parser.parse_args()


def plot_single_session_csa(input_csv, output_png, smooth_window=1, subject_id="sub-003",
                            session_id="M17", hemi=False):
    df = pd.read_csv(input_csv)

    # plot_csa_panel()/smooth_csa_columns() expect the per-subject csv layout, which carries a
    # session_id column; a single-session csv doesn't have one, so we add it here.
    if "session_id" not in df.columns:
        df["session_id"] = session_id
    sessions = sorted(df["session_id"].unique())

    # Single session, so there is no time gradient to follow: plot it in blue rather than in the
    # viridis color a lone timepoint would get (bright yellow, barely visible on white).
    palette = [LINE_COLOR for _ in sessions]

    if hemi:
        df = smooth_csa_columns(df, smooth_window, ["CSA_left_mm2", "CSA_right_mm2"])
        fig, (ax_left, ax_right) = plt.subplots(2, 1, figsize=(16, 16))
        plot_csa_panel(ax_left, df, sessions, palette, "CSA_left_mm2",
                       f'Left hemicord CSA across PAM50 axial slices and vertebral levels for {subject_id}')
        plot_csa_panel(ax_right, df, sessions, palette, "CSA_right_mm2",
                       f'Right hemicord CSA across PAM50 axial slices and vertebral levels for {subject_id}')
        fig.tight_layout()
    else:
        df = smooth_csa_columns(df, smooth_window, ["CSA_mm2"])
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_csa_panel(ax, df, sessions, palette, "CSA_mm2",
                       f'CSA across PAM50 axial slices and vertebral levels for {subject_id}')

    output_dir = os.path.dirname(os.path.abspath(output_png))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {output_png}')

    return output_png


if __name__ == "__main__":
    args = parse_args()
    plot_single_session_csa(args.input_csv, args.output, args.smooth_window,
                            args.subject_id, args.session_id, args.hemi)
