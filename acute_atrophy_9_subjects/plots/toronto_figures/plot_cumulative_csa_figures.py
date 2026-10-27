"""
This script plots the same CSA figures as plot_paper_figures.py (one line per session, lesion shading,
vertebral level boundaries/labels, sessions renamed M0/M<n> by months since baseline), but split into
cumulative frames for a presentation: one png per time point, where frame k shows the curves of the
first k sessions (M0 alone, then M0 + the next session, etc.).

Differences with plot_paper_figures.py:
    - curves are colored in yellow-green tones (M0 in yellow, later sessions progressively greener)
    - curves are slightly thicker
    - axes limits, vertebral level labels and colors are computed from ALL sessions and kept fixed
      across frames, so the frames can be flipped through without the plot jumping around

Input:
    -i / --input_dir: folder holding the per-subject csvs (output of compute_csa_on_include.py,
        one "<subject_id>_csa_with_lesions.csv" per subject)
    -o / --output_dir: folder where the figures are saved
        ("<subject_id>_csa_cumulative_<k>_<session>.png", e.g. sub-003_csa_cumulative_1_M0.png)
    --subjects: subjects to plot (default: sub-003 sub-006)
    --slice_range: "<subject_id> <start_slice> <end_slice>", PAM50 slice range to display for that
        subject (both bounds inclusive). Can be given once per subject; a subject without one is
        plotted over its full slice range.
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA (default 1, no smoothing)
    --linewidth: width of the CSA curves (default 3.5)
    --labels_font_size / --ticks_font_size: font sizes of the title and axis labels (default 22) and of
        the tick labels, vertebral level labels and legend (default 15)

Example:
    python plots/toronto_figures/plot_cumulative_csa_figures.py -i /path/to/csvs -o /path/to/figures \
        --slice_range sub-003 760 900 --slice_range sub-006 780 950 -s 10

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_subject_csa import plot_csa_panel, smooth_csa_columns
from plot_paper_figures import rename_sessions_by_month

# Portion of viridis used for the session colors: from yellow (1.0) to green (0.45)
COLORMAP_START = 1.0
COLORMAP_END = 0.45


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cumulative CSA figures (one png per time point).")
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
    parser.add_argument("--linewidth", type=float, default=3.5,
                        help="Width of the CSA curves. Default 3.5.")
    parser.add_argument("--labels_font_size", type=int, default=22,
                        help="Font size of the title and axis labels. Default 22.")
    parser.add_argument("--ticks_font_size", type=int, default=15,
                        help="Font size of the tick labels, vertebral level labels and legend. Default 15.")
    return parser.parse_args()


def yellow_green_palette(n_sessions):
    """Returns n_sessions colors going from yellow (first session) to green (last session)."""
    return [cm.viridis(COLORMAP_START - (COLORMAP_START - COLORMAP_END) * i / max(n_sessions - 1, 1))
            for i in range(n_sessions)]


def plot_cumulative_figures(subject_id, input_csv, output_dir, slice_range=None, smooth_window=1,
                            linewidth=3.5, labels_font_size=22, ticks_font_size=15):
    df = pd.read_csv(input_csv)
    df = smooth_csa_columns(df, smooth_window, ["CSA_mm2"])
    df, sessions = rename_sessions_by_month(df)

    if slice_range is not None:
        start_slice, end_slice = slice_range
        df = df[(df["pam50_axial_slice"] >= start_slice) & (df["pam50_axial_slice"] <= end_slice)]
        if df.empty:
            raise SystemExit(f"No slices of {subject_id} in the range {start_slice}-{end_slice}.")

    palette = yellow_green_palette(len(sessions))
    os.makedirs(output_dir, exist_ok=True)

    for n_shown in range(1, len(sessions) + 1):
        # The full panel (all sessions) is drawn on every frame so that axes limits and vertebral level
        # labels are identical across frames; the sessions not yet shown are then hidden.
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_csa_panel(ax, df, sessions, palette, "CSA_mm2", "",
                       labels_font_size=labels_font_size, ticks_font_size=ticks_font_size)

        shown_sessions = sessions[:n_shown]
        for line in ax.lines:
            if line.get_label() in sessions:
                line.set_linewidth(linewidth)
                line.set_visible(line.get_label() in shown_sessions)

        # Rebuild the legend with only the visible curves (and the lesion shading)
        handles, labels = ax.get_legend_handles_labels()
        kept = [(h, l) for h, l in zip(handles, labels) if l not in sessions or l in shown_sessions]
        legend = ax.legend([h for h, _ in kept], [l for _, l in kept], loc='upper right', fontsize=ticks_font_size)
        for legend_line in legend.get_lines():
            legend_line.set_linewidth(linewidth)

        output_png = os.path.join(output_dir, f"{subject_id}_csa_cumulative_{n_shown}_{sessions[n_shown - 1]}.png")
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
        plot_cumulative_figures(subject_id, input_csv, args.output_dir, slice_ranges.get(subject_id),
                                args.smooth_window, args.linewidth, args.labels_font_size, args.ticks_font_size)


if __name__ == "__main__":
    main()
