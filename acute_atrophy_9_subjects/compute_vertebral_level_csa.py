"""
This script computes, for a single subject, the mean CSA over the C1-C2 and C2-C3 vertebral
levels (VertLevel 1-2 and 2-3, as in sct_process_segmentation -vert) at each timepoint, and
indicates whether a lesion is present at each of these levels.

As in compute_lesion_level_csa.py, the data is first truncated to the PAM50 slices that have a CSA in
every session, so all timepoints are averaged over exactly the same slices (n_slices is
therefore identical across sessions for a given level).

Input:
    -i / --input_csv: path to a subject's csv file (output of compute_csa_on_include.py)
    -o / --output: path to the output csv file
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA_mm2, applied
        per session on the full slice range before averaging (default 1, no smoothing)

Output columns (one row per session), with each of the following repeated for C1_C2 and C2_C3:
    - subject_id, session_id
    - timepoint: "M0" for the oldest (baseline) session, "M<n>" for the sessions n months later
    - n_slices_<level>: number of PAM50 slices used for the mean
    - CSA_<level>_mean_mm2: mean CSA_mm2 over those slices
    - lesion_present_<level>: True if the session has a lesion on at least one of those slices
    - CSA_<level>_change_since_baseline_pct: percent change of the mean CSA relative to the
      baseline (oldest) session, 100 * (CSA / baseline CSA - 1); 0 at baseline

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd

from compute_lesion_level_csa import get_common_slices, get_month_labels, CSA_COLUMNS
from plot_subject_csa import smooth_csa_columns

LEVELS = {"C1-C2": (1, 2), "C2-C3": (2, 3)}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the C1-C2 and C2-C3 mean CSA, and lesion presence, per timepoint for a single subject.")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to a subject's csv file (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output csv file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                         help="Window size (in slices) for moving-average smoothing of CSA_mm2, applied "
                              "per session before averaging. Default 1 (no smoothing).")
    return parser.parse_args()


def compute_vertebral_level_csa(input_csv, output_csv, smooth_window=1):
    df = pd.read_csv(input_csv)
    subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns and not df.empty else os.path.basename(input_csv).replace("_csa_with_lesions.csv", "")

    df = smooth_csa_columns(df, smooth_window, CSA_COLUMNS)
    sessions = sorted(df["session_id"].unique())
    month_labels = get_month_labels(sessions)
    df = df[df["pam50_axial_slice"].isin(get_common_slices(df))].copy()
    if df.empty:
        print(f"WARNING: {subject_id} has no PAM50 slice with a CSA in every session; "
              f"{os.path.basename(output_csv)} will contain NaN CSA values.")
    df["has_lesion"] = df["lesion_label"].apply(lambda v: pd.notna(v) and str(v).strip() != "")

    rows = []
    for session_id in sessions:
        df_session = df[df["session_id"] == session_id]
        row = {"subject_id": subject_id, "session_id": session_id, "timepoint": month_labels[session_id]}
        for level, (first_vert, last_vert) in LEVELS.items():
            name = level.replace("-", "_")
            df_level = df_session[df_session["VertLevel"].between(first_vert, last_vert)]
            row[f"n_slices_{name}"] = len(df_level)
            row[f"CSA_{name}_mean_mm2"] = df_level["CSA_mm2"].mean()
            row[f"lesion_present_{name}"] = bool(df_level["has_lesion"].any())
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # Percent change of each mean CSA relative to the baseline (oldest = first) session
    for level in LEVELS:
        name = level.replace("-", "_")
        csa = df_out[f"CSA_{name}_mean_mm2"]
        df_out[f"CSA_{name}_change_since_baseline_pct"] = 100 * (csa / csa.iloc[0] - 1)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Vertebral level CSA report saved to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    args = parse_args()
    compute_vertebral_level_csa(args.input_csv, args.output, args.smooth_window)
