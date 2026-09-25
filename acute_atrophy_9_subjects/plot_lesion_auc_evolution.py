"""
Plots the evolution of the CSA AUC at lesion level for several subjects, one line per subject,
using the lesion-area AUC csvs of compute_lesion_auc.py. For subjects with several lesion areas,
the lesion_area_id to use is chosen per subject.

Two figures are saved in the output folder: lesion_auc.png (CSA AUC) and lesion_auc_percent_change.png
(percent change of the AUC since baseline), both vs. months since the subject's baseline scan.

Example:
    python plot_lesion_auc_evolution.py -o figures \
        --subject sub-003 sub-003_lesion_auc_smooth10.csv 1 \
        --subject sub-005 sub-005_lesion_auc_smooth10.csv 2

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from compute_lesion_auc import parse_session_date

parser = argparse.ArgumentParser(description="Plot the CSA AUC at lesion level for several subjects.")
parser.add_argument("--subject", nargs=3, action="append", required=True, metavar=("SUBJECT", "AUC_CSV", "LESION_ID"),
                    help="Subject id, its lesion-area AUC csv and the lesion_area_id to plot. Repeat per subject.")
parser.add_argument("-o", "--output_dir", required=True, help="Folder where the figures are saved.")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

series = []
for subject_id, csv_path, lesion_id in args.subject:
    df = pd.read_csv(csv_path)
    df = df[df["lesion_area_id"] == int(lesion_id)].copy()
    df["date"] = pd.to_datetime(df["session_id"].map(parse_session_date))
    df = df.sort_values("date")
    df["months"] = (df["date"] - df["date"].iloc[0]).dt.days / 30.4375
    df["percent_change"] = 100 * (df["AUC"] / df["AUC"].iloc[0] - 1)
    series.append((subject_id, df))

for column, ylabel, filename in [("AUC", "CSA AUC (mm² × slices)", "lesion_auc.png"),
                                 ("percent_change", "CSA AUC change since baseline (%)", "lesion_auc_percent_change.png")]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for subject_id, df in series:
        ax.plot(df["months"], df[column], marker="o", linewidth=2, label=subject_id)
    ax.set_xlabel("Months since baseline scan")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.yaxis.grid(True)
    fig.savefig(os.path.join(args.output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
