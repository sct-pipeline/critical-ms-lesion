"""
This script computes the area under the CSA curve (AUC) of a subject over a fixed healthy (lesion-free)
PAM50 axial slice range, at each timepoint (session). It produces a table in
the same format as compute_lesion_auc.py, with the lesion area replaced by the given healthy region.

Input:
    -i / --input_csv: path to the subject's csv file (output of compute_csa_on_include.py)
    -o / --output_folder: path to the output folder (default: folder of this script)
    --start_slice / --end_slice: PAM50 slice range of the healthy region (default: 889 to 920)

Output columns: same as compute_lesion_auc.py (lesion_area_id is 1 for the healthy region)

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import pandas as pd

from plot_subject_csa import smooth_csa_columns
from compute_lesion_auc import CSA_COLUMNS, get_common_slices, compute_auc, parse_session_date


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-timepoint AUC of the CSA curve over a fixed healthy PAM50 slice range for a subject.")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to the subject's csv file (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--start_slice", type=int, required=True, help="First PAM50 axial slice of the healthy region.")
    parser.add_argument("--end_slice", type=int, required=True, help="Last PAM50 axial slice of the healthy region.")
    return parser.parse_args()


def compute_healthy_region_auc(input_csv, output_dir, start_slice, end_slice, smooth_window=1):
    df = pd.read_csv(input_csv)

    subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns and not df.empty else os.path.basename(input_csv).replace("_csa_with_lesions.csv", "")

    output_csv = os.path.join(output_dir, f"{subject_id}_healthy_region_{start_slice}_{end_slice}_auc_smooth{smooth_window}.csv")

    df = smooth_csa_columns(df, smooth_window, CSA_COLUMNS)

    # Truncate to the PAM50 slices common to every session, so all timepoints are compared
    # over the exact same slice range
    common_slices = get_common_slices(df)
    df_common = df[df["pam50_axial_slice"].isin(common_slices)].copy()

    # Warn if the healthy region is not fully covered by every session, or overlaps a lesion
    region_slices = set(range(start_slice, end_slice + 1))
    missing_slices = sorted(region_slices - common_slices)
    if missing_slices:
        print(f"Warning: {len(missing_slices)} slice(s) of the region [{start_slice}, {end_slice}] are not common to all sessions: {missing_slices}")
    df_region = df_common[df_common["pam50_axial_slice"].isin(region_slices)]
    if df_region["lesion_label"].apply(lambda v: pd.notna(v) and str(v).strip() != "").any():
        print(f"Warning: the region [{start_slice}, {end_slice}] contains lesion slices in at least one session")

    sessions = sorted(df_common["session_id"].unique())

    rows = []
    for session_id in sessions:
        df_session = df_common[df_common["session_id"] == session_id]
        rows.append({
            "subject_id": subject_id,
            "lesion_area_id": 1,
            "session_id": session_id,
            "start_pam50_slice": start_slice,
            "end_pam50_slice": end_slice,
            "AUC": compute_auc(df_session, start_slice, end_slice, "CSA_mm2"),
            "AUC_left": compute_auc(df_session, start_slice, end_slice, "CSA_left_mm2"),
            "AUC_right": compute_auc(df_session, start_slice, end_slice, "CSA_right_mm2"),
        })
    df_out = pd.DataFrame(rows)

    # Ratio of this timepoint's AUC to the previous timepoint's (rows are ordered chronologically);
    # NaN for the first timepoint
    for auc_column, ratio_column in [("AUC", "AUC_ratio_to_previous_timepoint"),
                                      ("AUC_left", "AUC_left_ratio_to_previous_timepoint"),
                                      ("AUC_right", "AUC_right_ratio_to_previous_timepoint")]:
        df_out[ratio_column] = df_out[auc_column] / df_out[auc_column].shift(1)

    # Days since the previous timepoint, parsed from the YYYYMMDD date embedded in session_id
    df_out["days_since_previous_timepoint"] = df_out["session_id"].apply(parse_session_date).diff().dt.days

    output_columns = ["subject_id", "lesion_area_id", "session_id", "start_pam50_slice", "end_pam50_slice",
                      "AUC", "AUC_ratio_to_previous_timepoint",
                      "AUC_left", "AUC_left_ratio_to_previous_timepoint",
                      "AUC_right", "AUC_right_ratio_to_previous_timepoint",
                      "days_since_previous_timepoint"]
    df_out = df_out[output_columns]

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"AUC report saved to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    args = parse_args()
    for smooth_window, suffix in [(1, ""), (10, "_smooth10")]:
        compute_healthy_region_auc(args.input_csv, args.output_folder, args.start_slice, args.end_slice, smooth_window=smooth_window)
