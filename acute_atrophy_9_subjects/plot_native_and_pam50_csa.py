"""
This script plots the cross-sectional area (CSA) of a single scan both in the PAM50-normalized
template space and in the scan's native slice space, stacked on the same figure (PAM50 on top,
native below), with vertebral level boundaries/labels on each panel.

Input:
    --csa_native: path to the csv with native-slice CSA (output of sct_process_segmentation
        without -normalize-PAM50, with columns "Slice (I->S)", "VertLevel", "MEAN(area)")
    --csa_pam50: path to the csv with PAM50-slice CSA (output of sct_process_segmentation
        with -normalize-PAM50 1, same columns)
    -o / --output: path to the output png file

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from generate_csa_plot import get_vert_indices, LABELS_FONT_SIZE, TICKS_FONT_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CSA in both native and PAM50 slice space for a single scan.")
    parser.add_argument("--csa_native", type=str, required=True, help="Path to the csv with native-slice CSA.")
    parser.add_argument("--csa_pam50", type=str, required=True, help="Path to the csv with PAM50-slice CSA.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output png file.")
    return parser.parse_args()


def plot_csa_panel(ax, df, title, xlabel):
    """
    Plots a single CSA-vs-slice panel with vertebral level boundaries/labels.
    """
    df = df.sort_values("Slice (I->S)").reset_index(drop=True)

    ax.plot(df["Slice (I->S)"], df["MEAN(area)"], linewidth=2, color="tab:blue")

    ymin, ymax = ax.get_ylim()

    # Slices outside the vertebral-labeled range (e.g. edges of the native FOV) have a NaN
    # VertLevel; drop them just for the boundary/label computation, not from the plotted line.
    df_vert = df.dropna(subset=["VertLevel"]).reset_index(drop=True)

    vert, ind_vert, ind_vert_mid = get_vert_indices(df_vert, single_subject=True)
    vert = [int(v) for v in vert]

    for x in ind_vert[1:-1]:
        ax.axvline(df_vert.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

    for idx, x in enumerate(ind_vert_mid):
        if vert[x] > 19:
            level = 'L' + str(vert[x] - 19)
        elif vert[x] > 7:
            level = 'T' + str(vert[x] - 7)
        else:
            level = 'C' + str(vert[x])
        ax.text(df_vert.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

    ax.set_title(title, fontweight='bold', fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('CSA (mm2)', fontsize=LABELS_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    ax.invert_xaxis()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def plot_native_and_pam50_csa(csa_native_csv, csa_pam50_csv, output_png):
    df_native = pd.read_csv(csa_native_csv)
    df_pam50 = pd.read_csv(csa_pam50_csv)

    fig, (ax_pam50, ax_native) = plt.subplots(2, 1, figsize=(16, 12))

    plot_csa_panel(ax_pam50, df_pam50, "CSA in PAM50 template space", "PAM50 Axial Slice #")
    plot_csa_panel(ax_native, df_native, "CSA in native space", "Native Axial Slice #")

    fig.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(output_png))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {output_png}')

    return output_png


if __name__ == "__main__":
    args = parse_args()
    plot_native_and_pam50_csa(args.csa_native, args.csa_pam50, args.output)
