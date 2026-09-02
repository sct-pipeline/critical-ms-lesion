"""
This script plots the cross-sectional area (CSA) across PAM50 axial slices and vertebral
levels for a single subject, using the per-subject csv produced by compute_csa_on_include.py
(<output_folder>/<subject_id>_csa_with_lesions.csv). Each session (timepoint) is plotted as
a separate colored line, following the same vertebral-level boundary/label convention used
in generate_csa_plot.py.

Input:
    -i / --input_csv: path to a subject's csv file (output of compute_csa_on_include.py)
    -o / --output: path to the output png file
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA (default 1, no smoothing)

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from generate_csa_plot import get_vert_indices, LABELS_FONT_SIZE, TICKS_FONT_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSA across PAM50 axial slices and vertebral levels for a single subject, one line per session.")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to a subject's csv file (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output png file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                         help="Window size (in slices) for moving-average smoothing of CSA before plotting. "
                              "Default 1 (no smoothing).")
    return parser.parse_args()


def plot_subject_csa(input_csv, output_png, smooth_window=1):
    df = pd.read_csv(input_csv)

    subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns and not df.empty else os.path.basename(input_csv).replace("_csa_with_lesions.csv", "")

    fig, ax = plt.subplots(figsize=(16, 8))

    # Aggregate lesion background: shade a PAM50 slice once if ANY session has a lesion there,
    # instead of shading per session (which produced overlapping/stacked shading across timepoints).
    # lesion_label is empty/NaN for slices with no lesion (pandas reads the empty strings written
    # by compute_csa_on_include.py back as NaN).
    LESION_COLOR = 'red'
    has_lesion_per_row = df["lesion_label"].apply(lambda v: pd.notna(v) and str(v).strip() != "")
    lesion_slices_agg = df.assign(has_lesion=has_lesion_per_row).groupby("pam50_axial_slice")["has_lesion"].any().sort_index()
    if lesion_slices_agg.any():
        ax.fill_between(lesion_slices_agg.index, 0, 1, where=lesion_slices_agg.values, color=LESION_COLOR, alpha=0.15,
                         transform=ax.get_xaxis_transform(), label='Lesion')

    # One line per session, sorted chronologically (session_id is e.g. "ses-YYYYMMDD", so a
    # lexical sort is also a chronological sort) so the color gradient tracks time.
    sessions = sorted(df["session_id"].unique())
    n_sessions = len(sessions)
    palette = [cm.viridis(1 - i / max(n_sessions - 1, 1)) for i in range(n_sessions)]

    for session_idx, session_id in enumerate(sessions):
        df_session = df[df["session_id"] == session_id].sort_values("pam50_axial_slice").copy()
        if smooth_window > 1:
            raw_csa = df_session["CSA_mm2"]
            smoothed_csa = raw_csa.rolling(window=smooth_window, center=True, min_periods=smooth_window).mean()
            # Keep the raw value at edge slices where a full window isn't available, instead
            # of averaging over a partial (smaller) window there.
            df_session["CSA_mm2"] = smoothed_csa.fillna(raw_csa)
        color = palette[session_idx]
        sns.lineplot(ax=ax, x="pam50_axial_slice", y="CSA_mm2", data=df_session, linewidth=2,
                     color=color, label=session_id)

    ymin, ymax = ax.get_ylim()

    # Vertebral level boundaries/labels: since PAM50 normalization maps every session onto the
    # same template slice numbering, the slice->VertLevel mapping should be consistent across
    # sessions, so we deduplicate across all sessions to get one boundary set for the x-axis
    df_vert = df.drop_duplicates(subset="pam50_axial_slice").sort_values("pam50_axial_slice").reset_index(drop=True)
    df_vert = df_vert.rename(columns={"pam50_axial_slice": "Slice (I->S)"})
    vert, ind_vert, ind_vert_mid = get_vert_indices(df_vert, single_subject=True)
    vert = [int(v) for v in vert]

    # Insert a vertical line for each intervertebral disc
    for x in ind_vert[1:-1]:
        ax.axvline(df_vert.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

    # Insert a text label for each vertebral level
    for idx, x in enumerate(ind_vert_mid):
        if vert[x] > 19:
            level = 'L' + str(vert[x] - 19)
        elif vert[x] > 7:
            level = 'T' + str(vert[x] - 7)
        else:
            level = 'C' + str(vert[x])
        ax.text(df_vert.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

    ax.set_title(f'CSA across PAM50 axial slices and vertebral levels for {subject_id}',
                 fontweight='bold', fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel('PAM50 Axial Slice #', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('CSA (mm2)', fontsize=LABELS_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
    ax.legend(loc='upper right', fontsize=TICKS_FONT_SIZE)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # Invert x-axis and add only horizontal grid lines, pushed behind the plotted lines
    ax.invert_xaxis()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # ax.set_ylim(40, 100)
    # ax.set_xlim(df_vert["Slice (I->S)"].max(), df_vert["Slice (I->S)"].min())

    output_dir = os.path.dirname(os.path.abspath(output_png))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {output_png}')

    return output_png


if __name__ == "__main__":
    args = parse_args()
    plot_subject_csa(args.input_csv, args.output, args.smooth_window)
