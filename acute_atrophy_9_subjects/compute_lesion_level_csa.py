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
computed (trapezoidal integration) over that area's slice range. The same is also computed for
the left and right hemicord (CSA_left_mm2 / CSA_right_mm2).

Input:
    -i / --input_csv: path to a subject's csv file (output of compute_csa_on_include.py)
    -o / --output: path to the output csv file
    -s / --smooth_window: window size (in slices) for moving-average smoothing of CSA_mm2,
        applied per session before computing the AUC (default 1, no smoothing)

Output columns:
    - lesion_area_id: 1-indexed id of the continuous lesion area
    - session_id: session of the scan
    - timepoint: "M0" for the oldest (baseline) session, "M<n>" for the sessions n months later
    - start_pam50_slice / end_pam50_slice: PAM50 slice range of the lesion area (same for every
      timepoint, since the area is defined by the union of lesion extents across timepoints)
    - CSA_AUC / CSA_AUC_left / CSA_AUC_right: area under the CSA_mm2 / CSA_left_mm2 /
      CSA_right_mm2 curve over that slice range, for that timepoint
    - CSA_AUC_ratio_to_previous_timepoint / CSA_AUC_left_ratio_to_previous_timepoint /
      CSA_AUC_right_ratio_to_previous_timepoint: CSA AUC at this timepoint divided by CSA AUC at
      the previous timepoint (within the same lesion area); NaN for each lesion area's first timepoint
    - CSA_mean_mm2 / CSA_median_mm2 (and the _left / _right equivalents): mean and median of
      CSA_mm2 (CSA_left_mm2 / CSA_right_mm2) over that slice range, for that timepoint (after
      smoothing, if -s is set)
    - CSA_above_mean_mm2 / CSA_above_median_mm2 / n_slices_above (and _below): mean and median
      CSA_mm2 over the lesion-free slices just above (superior) / below (inferior) the lesion area,
      each side spanning half the lesion's width (rounded up), and the number of slices actually
      used. The slices are identical for every timepoint; if the half-width isn't available on a
      side (edge of the common slice range, or another lesion in the way), as many slices as
      possible are used (n_slices_* tells how many; NaN stats when 0)
    - <column>_change_since_baseline_pct, for each of the AUC and mean/median CSA columns above
      (full cord, left, right, above and below): percent change relative to the lesion area's
      baseline (first) session, 100 * (value / baseline value - 1); 0 at baseline
    - days_since_previous_timepoint: number of days between this session's date and the previous
      timepoint's (within the same lesion area), parsed from the YYYYMMDD date embedded in
      session_id (e.g. "ses-20111227"); NaN for each lesion area's first timepoint, or if a
      session_id has no parseable date

Author: Pierre-Louis Benveniste
"""
import os
import re
import math
import argparse
import numpy as np
import pandas as pd

from plot_subject_csa import smooth_csa_columns


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-lesion-area, per-timepoint AUC of the CSA curve for a single subject.")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to a subject's csv file (output of compute_csa_on_include.py).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output csv file.")
    parser.add_argument("-s", "--smooth_window", type=int, default=1,
                         help="Window size (in slices) for moving-average smoothing of CSA_mm2, applied "
                              "per session before computing the AUC. Default 1 (no smoothing).")
    return parser.parse_args()


CSA_COLUMNS = ["CSA_mm2", "CSA_left_mm2", "CSA_right_mm2"]


def get_common_slices(df):
    """
    Returns the set of pam50_axial_slice values that have an actual (non-NaN) CSA_mm2,
    CSA_left_mm2 and CSA_right_mm2 in every session of df, so that all timepoints can be
    compared/truncated to the exact same PAM50 slice range. A session's scan may not physically
    extend far enough to cover every PAM50 slice its vertebral labeling implies, leaving a row
    present but CSA as NaN there; such slices must not count as "common".
    """
    df_with_csa = df.dropna(subset=CSA_COLUMNS)
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


def get_flank_slices(df_common, lesion_areas, start_slice, end_slice):
    """
    Slices above (superior, higher PAM50 slice index) and below (inferior) a lesion area, each
    spanning half the lesion's width (rounded up). Only slices common to every session are used
    (df_common), so every session gets exactly the same slices, and slices belonging to any
    lesion area are excluded so the flanks stay lesion-free. If fewer slices are available, as
    many as possible are kept.
    Output: {"above": [slices], "below": [slices]}
    """
    half_span = math.ceil((end_slice - start_slice + 1) / 2)
    lesion_slices = {s for start, end in lesion_areas for s in range(start, end + 1)}
    available = sorted(set(df_common["pam50_axial_slice"]) - lesion_slices)
    return {
        "above": [s for s in available if end_slice < s <= end_slice + half_span],
        "below": [s for s in available if start_slice - half_span <= s < start_slice],
    }


def compute_flank_stats(df_session, flank_slices):
    """
    Mean and median CSA_mm2 over the slices above and below a lesion area, and how many slices
    were used on each side.
    """
    stats = {}
    for side, slices in flank_slices.items():
        csa = df_session.loc[df_session["pam50_axial_slice"].isin(slices), "CSA_mm2"]
        stats[f"CSA_{side}_mean_mm2"] = csa.mean()
        stats[f"CSA_{side}_median_mm2"] = csa.median()
        stats[f"n_slices_{side}"] = len(slices)
    return stats


def compute_csa_stats(df_session, start_slice, end_slice):
    """
    Trapezoidal AUC (vs. pam50_axial_slice), mean and median of CSA_mm2, CSA_left_mm2 and
    CSA_right_mm2 over [start_slice, end_slice]. The AUC is NaN with fewer than 2 slices.
    """
    in_range = df_session[(df_session["pam50_axial_slice"] >= start_slice) &
                          (df_session["pam50_axial_slice"] <= end_slice)].sort_values("pam50_axial_slice")
    stats = {}
    for name, auc_name, csa_column in [("CSA", "CSA_AUC", "CSA_mm2"),
                                       ("CSA_left", "CSA_AUC_left", "CSA_left_mm2"),
                                       ("CSA_right", "CSA_AUC_right", "CSA_right_mm2")]:
        stats[auc_name] = (np.trapezoid(in_range[csa_column], in_range["pam50_axial_slice"])
                           if len(in_range) >= 2 else np.nan)
        stats[f"{name}_mean_mm2"] = in_range[csa_column].mean()
        stats[f"{name}_median_mm2"] = in_range[csa_column].median()
    return stats


def parse_session_date(session_id):
    """
    Extracts an 8-digit YYYYMMDD date from a session_id (e.g. "ses-20111227" -> 2011-12-27).
    Returns None if no 8-digit date could be found.
    """
    match = re.search(r"(\d{8})", str(session_id))
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def get_month_labels(session_ids):
    """
    Maps each session_id to "M0" (oldest session) or "M<n>", n being the number of months since
    the oldest session (rounded to the nearest month).
    """
    dates = {session_id: parse_session_date(session_id) for session_id in session_ids}
    baseline_date = min(dates.values())
    return {session_id: f"M{round((date - baseline_date).days / 30.4375)}" for session_id, date in dates.items()}


def compute_lesion_level_csa(input_csv, output_csv, smooth_window=1):
    df = pd.read_csv(input_csv)

    subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns and not df.empty else os.path.basename(input_csv).replace("_csa_with_lesions.csv", "")

    df = smooth_csa_columns(df, smooth_window, CSA_COLUMNS)

    # Truncate to the PAM50 slices common to every session, so all timepoints are compared
    # over the exact same slice range
    common_slices = get_common_slices(df)
    df_common = df[df["pam50_axial_slice"].isin(common_slices)].copy()

    lesion_areas = get_lesion_areas(df_common)
    sessions = sorted(df_common["session_id"].unique())
    month_labels = get_month_labels(sessions) if sessions else {}

    rows = []
    for lesion_area_id, (start_slice, end_slice) in enumerate(lesion_areas, start=1):
        flank_slices = get_flank_slices(df_common, lesion_areas, start_slice, end_slice)
        for session_id in sessions:
            df_session = df_common[df_common["session_id"] == session_id]
            rows.append({
                "subject_id": subject_id,
                "lesion_area_id": lesion_area_id,
                "session_id": session_id,
                "timepoint": month_labels[session_id],
                "start_pam50_slice": start_slice,
                "end_pam50_slice": end_slice,
                **compute_csa_stats(df_session, start_slice, end_slice),
                **compute_flank_stats(df_session, flank_slices),
            })

    output_columns = ["subject_id", "lesion_area_id", "session_id", "timepoint", "start_pam50_slice", "end_pam50_slice",
                       "CSA_AUC", "CSA_AUC_ratio_to_previous_timepoint",
                       "CSA_AUC_left", "CSA_AUC_left_ratio_to_previous_timepoint",
                       "CSA_AUC_right", "CSA_AUC_right_ratio_to_previous_timepoint",
                       "CSA_mean_mm2", "CSA_median_mm2",
                       "CSA_left_mean_mm2", "CSA_left_median_mm2",
                       "CSA_right_mean_mm2", "CSA_right_median_mm2",
                       "CSA_above_mean_mm2", "CSA_above_median_mm2", "n_slices_above",
                       "CSA_below_mean_mm2", "CSA_below_median_mm2", "n_slices_below",
                       "days_since_previous_timepoint"]
    baseline_change_sources = ["CSA_AUC", "CSA_AUC_left", "CSA_AUC_right",
                               "CSA_mean_mm2", "CSA_median_mm2",
                               "CSA_left_mean_mm2", "CSA_left_median_mm2",
                               "CSA_right_mean_mm2", "CSA_right_median_mm2",
                               "CSA_above_mean_mm2", "CSA_above_median_mm2",
                               "CSA_below_mean_mm2", "CSA_below_median_mm2"]
    output_columns += [f"{column}_change_since_baseline_pct" for column in baseline_change_sources]

    if not rows:
        # No lesion at any timepoint for this subject: nothing to group/compute, just write
        # an empty csv with the expected columns.
        df_out = pd.DataFrame(columns=output_columns)
    else:
        df_out = pd.DataFrame(rows)

        # Ratio of this timepoint's AUC to the previous timepoint's, within each lesion area
        # (rows are already ordered chronologically by session_id within each lesion_area_id group).
        # NaN for each lesion area's first timepoint, since there is no previous one to compare to.
        for auc_column, ratio_column in [("CSA_AUC", "CSA_AUC_ratio_to_previous_timepoint"),
                                          ("CSA_AUC_left", "CSA_AUC_left_ratio_to_previous_timepoint"),
                                          ("CSA_AUC_right", "CSA_AUC_right_ratio_to_previous_timepoint")]:
            df_out[ratio_column] = df_out.groupby("lesion_area_id")[auc_column].apply(
                lambda auc: auc / auc.shift(1)).reset_index(drop=True)

        # Days since the previous timepoint, within each lesion area (same chronological ordering
        # as above), parsed from the YYYYMMDD date embedded in session_id.
        session_date = df_out["session_id"].apply(parse_session_date)
        df_out["days_since_previous_timepoint"] = session_date.groupby(df_out["lesion_area_id"]).diff().dt.days

        # Percent change since the baseline (first) session of each lesion area, for the AUC and
        # CSA statistics: 100 * (value / baseline value - 1); 0 at baseline.
        for column in baseline_change_sources:
            df_out[f"{column}_change_since_baseline_pct"] = df_out.groupby("lesion_area_id")[column].transform(
                lambda values: 100 * (values / values.iloc[0] - 1))

        df_out = df_out[output_columns]

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Lesion-level CSA report saved to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    args = parse_args()
    compute_lesion_level_csa(args.input_csv, args.output, args.smooth_window)
