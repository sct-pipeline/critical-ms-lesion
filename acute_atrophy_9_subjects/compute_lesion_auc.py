"""
This script computes the area under the CSA curve (AUC) for each continuous lesion area of a
subject, at each timepoint (session).

A "continuous lesion area" is a contiguous run of PAM50 axial slices where at least one of the
subject's sessions has a lesion present (i.e. the union of lesion extents across all timepoints,
following the same per-slice aggregation used for lesion shading in plot_subject_csa.py). A
subject can have several such areas if lesions occur at separate, non-adjacent PAM50 slice ranges.

Before computing lesion areas and AUCs, the data is truncated to only the PAM50 axial slices
present in every session, so that all timepoints are compared over the exact same slice range.

For each continuous lesion area, for each timepoint, the AUC of CSA_mm2 vs. pam50_axial_slice is
computed (trapezoidal integration) over that area's slice range.

Input:
    -i / --input_csv: path to a subject's csv file (output of compute_csa_on_include.py)
    -o / --output: path to the output csv file
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA_mm2,
        applied per session before computing the AUC (default 1, no smoothing)

Output columns:
    - lesion_area_id: 1-indexed id of the continuous lesion area
    - session_id: timepoint
    - start_pam50_slice / end_pam50_slice: PAM50 slice range of the lesion area (same for every
      timepoint, since the area is defined by the union of lesion extents across timepoints)
    - AUC: area under the CSA_mm2 curve over that slice range, for that timepoint
    - AUC_ratio_to_previous_timepoint: AUC at this timepoint divided by AUC at the previous
      timepoint (within the same lesion area); NaN for each lesion area's first timepoint
    - days_since_previous_timepoint: number of days between this session's date and the previous
      timepoint's (within the same lesion area), parsed from the YYYYMMDD date embedded in
      session_id (e.g. "ses-20111227"); NaN for each lesion area's first timepoint, or if a
      session_id has no parseable date

Author: Pierre-Louis Benveniste
"""
import os
import re
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-lesion-area, per-timepoint AUC of the CSA curve for a single subject.")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to a subject's csv file (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output csv file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                         help="Window size (in slices) for moving-average smoothing of CSA_mm2, applied "
                              "per session before computing the AUC. Default 1 (no smoothing).")
    return parser.parse_args()


def smooth_csa(df, smooth_window):
    """
    Applies a centered moving-average to CSA_mm2, per session, before truncation/AUC computation.
    Edge slices that don't have a full window available keep their raw (unsmoothed) value,
    instead of being averaged over a partial window.
    """
    if smooth_window <= 1:
        return df

    smoothed_sessions = []
    for session_id, df_session in df.groupby("session_id", sort=False):
        df_session = df_session.sort_values("pam50_axial_slice").copy()
        raw_csa = df_session["CSA_mm2"]
        smoothed_csa = raw_csa.rolling(window=smooth_window, center=True, min_periods=smooth_window).mean()
        df_session["CSA_mm2"] = smoothed_csa.fillna(raw_csa)
        smoothed_sessions.append(df_session)

    return pd.concat(smoothed_sessions, ignore_index=True)


def get_common_slices(df):
    """
    Returns the set of pam50_axial_slice values that have an actual (non-NaN) CSA_mm2 in every
    session of df, so that all timepoints can be compared/truncated to the exact same PAM50
    slice range. A session's scan may not physically extend far enough to cover every PAM50
    slice its vertebral labeling implies, leaving a row present but CSA_mm2 as NaN there; such
    slices must not count as "common".
    """
    df_with_csa = df.dropna(subset=["CSA_mm2"])
    common_slices = None
    for session_id in df["session_id"].unique():
        session_slices = set(df_with_csa.loc[df_with_csa["session_id"] == session_id, "pam50_axial_slice"])
        common_slices = session_slices if common_slices is None else common_slices & session_slices
    return common_slices


def get_lesion_areas(df):
    """
    Groups PAM50 slices where any session has a lesion into contiguous (start, end) ranges.
    Input:
        df: DataFrame (already truncated to the common slices), with "pam50_axial_slice" and
            "lesion_label" columns
    Output:
        List of (start_pam50_slice, end_pam50_slice) tuples, one per continuous lesion area,
        sorted by start slice.
    """
    has_lesion_per_row = df["lesion_label"].apply(lambda v: pd.notna(v) and str(v).strip() != "")
    lesion_by_slice = df.assign(has_lesion=has_lesion_per_row).groupby("pam50_axial_slice")["has_lesion"].any().sort_index()
    lesion_slices = sorted(slice_id for slice_id, has_lesion in lesion_by_slice.items() if has_lesion)

    lesion_areas = []
    if not lesion_slices:
        return lesion_areas

    start = prev = lesion_slices[0]
    for slice_id in lesion_slices[1:]:
        if slice_id == prev + 1:
            prev = slice_id
            continue
        lesion_areas.append((start, prev))
        start = prev = slice_id
    lesion_areas.append((start, prev))

    return lesion_areas


def compute_auc(df_session, start_slice, end_slice):
    """
    Trapezoidal AUC of CSA_mm2 vs. pam50_axial_slice, restricted to [start_slice, end_slice].
    """
    df_range = df_session[(df_session["pam50_axial_slice"] >= start_slice) &
                           (df_session["pam50_axial_slice"] <= end_slice)].sort_values("pam50_axial_slice")
    if len(df_range) < 2:
        return np.nan
    return np.trapz(df_range["CSA_mm2"], df_range["pam50_axial_slice"])


def parse_session_date(session_id):
    """
    Extracts an 8-digit YYYYMMDD date from a session_id (e.g. "ses-20111227" -> 2011-12-27).
    Returns None if no 8-digit date could be found.
    """
    match = re.search(r"(\d{8})", str(session_id))
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def compute_lesion_auc(input_csv, output_csv, smooth_window=1):
    df = pd.read_csv(input_csv)

    subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns and not df.empty else os.path.basename(input_csv).replace("_csa_with_lesions.csv", "")

    df = smooth_csa(df, smooth_window)

    # Truncate to the PAM50 slices common to every session, so all timepoints are compared
    # over the exact same slice range
    common_slices = get_common_slices(df)
    df_common = df[df["pam50_axial_slice"].isin(common_slices)].copy()

    lesion_areas = get_lesion_areas(df_common)
    sessions = sorted(df_common["session_id"].unique())

    rows = []
    for lesion_area_id, (start_slice, end_slice) in enumerate(lesion_areas, start=1):
        for session_id in sessions:
            df_session = df_common[df_common["session_id"] == session_id]
            auc = compute_auc(df_session, start_slice, end_slice)
            rows.append({
                "subject_id": subject_id,
                "lesion_area_id": lesion_area_id,
                "session_id": session_id,
                "start_pam50_slice": start_slice,
                "end_pam50_slice": end_slice,
                "AUC": auc,
            })

    output_columns = ["subject_id", "lesion_area_id", "session_id", "start_pam50_slice", "end_pam50_slice",
                       "AUC", "AUC_ratio_to_previous_timepoint", "days_since_previous_timepoint"]

    if not rows:
        # No lesion at any timepoint for this subject: nothing to group/compute, just write
        # an empty csv with the expected columns.
        df_out = pd.DataFrame(columns=output_columns)
    else:
        df_out = pd.DataFrame(rows)

        # Ratio of this timepoint's AUC to the previous timepoint's, within each lesion area
        # (rows are already ordered chronologically by session_id within each lesion_area_id group).
        # NaN for each lesion area's first timepoint, since there is no previous one to compare to.
        df_out["AUC_ratio_to_previous_timepoint"] = df_out.groupby("lesion_area_id")["AUC"].apply(
            lambda auc: auc / auc.shift(1)).reset_index(drop=True)

        # Days since the previous timepoint, within each lesion area (same chronological ordering
        # as above), parsed from the YYYYMMDD date embedded in session_id.
        session_date = df_out["session_id"].apply(parse_session_date)
        df_out["days_since_previous_timepoint"] = session_date.groupby(df_out["lesion_area_id"]).diff().dt.days

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"AUC report saved to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    args = parse_args()
    compute_lesion_auc(args.input_csv, args.output, args.smooth_window)
