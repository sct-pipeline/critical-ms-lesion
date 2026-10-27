"""
This script plots the cross-sectional area (CSA) of sub-002 at session 2015 (M62) across PAM50
axial slices and vertebral levels, before and after disc level correction, with a smoothing
window of 10 slices. It uses the same plotting format as plot_subject_csa.py (called in
compute_csa_on_include.py with smooth_window=10), with one line for the CSA before correction
and one line after correction.

Input:
    --before_csv: path to the per-scan csv before correction (default: csa_with_lesions_before_correction.csv next to this script)
    --after_csv: path to the per-scan csv after correction (default: csa_with_lesions_after_correction.csv next to this script)
    -o / --output: path to the output png file (default: sub-002_ses-2015_csa_plot_before_after_disc_label_correction_smooth10.png next to this script)
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA (default 10)

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from plot_subject_csa import smooth_csa_columns, plot_csa_panel

SUBJECT_ID = "sub-002"
SESSION = "2015 (M62)"
BEFORE_LABEL = "Before disc level correction"
AFTER_LABEL = "After disc level correction"
Y_MIN, Y_MAX = 30, 45


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSA of sub-002 (session 2015, M62) across PAM50 axial slices and vertebral levels, before and after disc level correction.")
    parser.add_argument("--before_csv", type=str, default=os.path.join(SCRIPT_DIR, "csa_with_lesions_before_correction.csv"),
                        help="Path to the per-scan csv before correction.")
    parser.add_argument("--after_csv", type=str, default=os.path.join(SCRIPT_DIR, "csa_with_lesions_after_correction.csv"),
                        help="Path to the per-scan csv after correction.")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(SCRIPT_DIR, f"{SUBJECT_ID}_ses-2015_csa_plot_before_after_disc_label_correction_smooth10.png"),
                        help="Path to the output png file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=10,
                        help="Window size (in slices) for moving-average smoothing of CSA before plotting (default 10).")
    return parser.parse_args()


def plot_csa_before_after_correction(before_csv, after_csv, output_png, smooth_window=10):
    # Load both csvs and tag each with a label, used as the "session_id" by plot_csa_panel
    # so that each one is plotted as its own line
    df_before = pd.read_csv(before_csv)
    df_before["session_id"] = BEFORE_LABEL
    df_after = pd.read_csv(after_csv)
    df_after["session_id"] = AFTER_LABEL
    df = pd.concat([df_before, df_after], ignore_index=True)

    sessions = [BEFORE_LABEL, AFTER_LABEL]
    n_sessions = len(sessions)
    palette = [cm.viridis(1 - i / max(n_sessions - 1, 1)) for i in range(n_sessions)]

    # Smooth the CSA (per line) and plot
    df = smooth_csa_columns(df, smooth_window, ["CSA_mm2"])
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_csa_panel(ax, df, sessions, palette, "CSA_mm2",
                   f'CSA across PAM50 axial slices and vertebral levels for {SUBJECT_ID} ses-{SESSION} (before vs after disc level correction)')

    # Fix the y limits, and move the vertebral level labels (placed at the previous ymin by
    # plot_csa_panel) down to the new ymin
    ax.set_ylim(Y_MIN, Y_MAX)
    for text in ax.texts:
        text.set_y(Y_MIN)

    output_dir = os.path.dirname(os.path.abspath(output_png))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {output_png}')

    return output_png


if __name__ == "__main__":
    args = parse_args()
    plot_csa_before_after_correction(args.before_csv, args.after_csv, args.output, args.smooth_window)
