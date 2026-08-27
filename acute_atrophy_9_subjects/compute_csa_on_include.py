"""
This script segments the spinal cord, lesions and vertebrae for every scan listed in an
include json file (i.e. the output of create_include_json.py), then computes the
cross-sectional area (CSA) at each axial slice across the PAM50 template.

Lesions smaller than --min-lesion-size (mm3) are filtered out as spurious detections
(same threshold/logic as detect_critical_lesion.py's get_lesion_stats).

For each scan, a per-scan csv is written with one row per axial slice:
    - scan_axial_slice: native axial slice index (in the input scan's own space)
    - pam50_axial_slice: corresponding axial slice index in the PAM50 template
    - CSA_mm2: cross-sectional area at that slice
    - lesion_label: comma-separated label(s) of the lesion(s) present at that slice (empty if none)
    - lesion_volume_mm3: comma-separated volume(s) (mm3) of the lesion(s) present at that slice (empty if none)

Per-scan csvs are then aggregated into a single csv across all scans (with subject_id,
session_id and scan_file columns added).

Input:
    -i / --include_json: path to the include json file (output of create_include_json.py)
    --min-lesion-size: minimum lesion size (in mm3) to keep a lesion; smaller lesions are
        treated as spurious detections and discarded (default: 15.0)
    -o / --output_folder: path to the output folder where results will be saved

Author: Pierre-Louis Benveniste
"""
import os
import sys
import json
import argparse
import traceback
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from detect_critical_lesion import run_sc_segmentation, run_vert_labeling, run_lesion_segmentation, get_lesion_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Segment SC, lesions and vertebrae, then compute CSA across PAM50 axial slices (with lesion annotations) for all scans in an include json file.")
    parser.add_argument("-i", "--include_json", type=str, required=True, help="Path to the include json file (output of create_include_json.py).")
    parser.add_argument("--min-lesion-size", type=float, default=15.0, help="Minimum lesion size (in mm3) to keep a lesion; smaller lesions are treated as spurious detections and discarded (default: 15.0)")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the CSA results will be saved.")
    return parser.parse_args()


def compute_csa_native_and_pam50(sc_mask, vert_levels, output_path, qc_folder):
    """
    Computes the spinal cord CSA per axial slice, both in the scan's native slice numbering
    and in the PAM50-normalized slice numbering. The two commands process the same slices in
    the same order, so the resulting rows correspond 1:1 (only the reported "Slice (I->S)"
    numbering, and its resulting sorting, differs between them).
    Input:
        sc_mask: Path to the SC segmentation mask (NIfTI format)
        vert_levels: Path to the vertebral labeling (NIfTI format)
        output_path: Path to the output folder
        qc_folder: Path to the quality control folder
    Output:
        output_csv_native: Path to the csv file with native slice numbering
        output_csv_pam50: Path to the csv file with PAM50 slice numbering
    """
    output_csv_native = os.path.join(output_path, "csa_native.csv")
    output_csv_pam50 = os.path.join(output_path, "csa_pam50.csv")

    if not os.path.exists(output_csv_native):
        assert os.system(f"sct_process_segmentation -i {sc_mask} -discfile {vert_levels} -perslice 1 -o {output_csv_native} -qc {qc_folder}") == 0, "Error running the sct_process_segmentation command (native slices)"

    if not os.path.exists(output_csv_pam50):
        assert os.system(f"sct_process_segmentation -i {sc_mask} -discfile {vert_levels} -perslice 1 -normalize-PAM50 1 -o {output_csv_pam50} -qc {qc_folder}") == 0, "Error running the sct_process_segmentation command (PAM50 slices)"

    return output_csv_native, output_csv_pam50


def add_lesion_columns(df, lesion_statistics):
    """
    Annotates each row (native axial slice) of df with the label(s) and volume(s) of any
    lesion(s) present at that slice. A slice with no lesion gets empty strings; a slice
    covered by multiple lesions gets comma-separated values for both columns.
    Input:
        df: DataFrame with a "scan_axial_slice" column
        lesion_statistics: list of dicts (output of get_lesion_stats), each with keys
            "label", "size" (mm3) and "slices" (native axial slice indices)
    Output:
        df with two additional columns: "lesion_label" and "lesion_volume_mm3"
    """
    lesion_labels = []
    lesion_volumes = []
    for scan_slice in df["scan_axial_slice"]:
        matching_lesions = [lesion for lesion in lesion_statistics if scan_slice in lesion["slices"]]
        lesion_labels.append(",".join(str(lesion["label"]) for lesion in matching_lesions))
        lesion_volumes.append(",".join(f"{lesion['size']:.3f}" for lesion in matching_lesions))
    df["lesion_label"] = lesion_labels
    df["lesion_volume_mm3"] = lesion_volumes
    return df


def compute_csa_with_lesions(input_scan, output_path, min_lesion_size_mm3=15.0, lesion_mask_input=None):
    """
    Segments the SC, lesions and vertebrae for one scan, then builds a per-slice CSA csv
    annotated with lesion label(s)/volume(s).
    Input:
        input_scan: Path to the MRI scan (NIfTI format)
        output_path: Path to the parent output folder (a per-scan subfolder is created inside it)
        min_lesion_size_mm3: Minimum lesion size (in mm3) to keep a lesion
        lesion_mask_input: Path to a pre-existing lesion segmentation mask (NIfTI format), if any
    Output:
        Path to the per-scan csv file
    """
    # Build the output folder
    image_name = input_scan.split("/")[-1].replace(".nii.gz", "")
    output_path = os.path.join(output_path, image_name)
    os.makedirs(output_path, exist_ok=True)
    qc_folder = os.path.join(output_path, "qc")
    os.makedirs(qc_folder, exist_ok=True)

    output_csv_path = os.path.join(output_path, "csa_with_lesions.csv")
    if os.path.exists(output_csv_path):
        return output_csv_path

    print(f"Computing CSA with lesions for scan: {input_scan}")
    # Copy img to output folder
    assert os.system(f"cp {input_scan} {output_path}") == 0, "Error copying the input scan to the output folder"

    # SC segmentation
    sc_mask = run_sc_segmentation(input_scan, output_path, qc_folder)
    # Vert labeling
    vert_levels = run_vert_labeling(input_scan, output_path, qc_folder)
    # Lesion segmentation
    lesion_mask = run_lesion_segmentation(input_scan, sc_mask, lesion_mask_input, output_path, qc_folder)

    # Lesion statistics (size-filtered; includes native slice indices and volume per lesion)
    lesion_statistics = get_lesion_stats(lesion_mask, sc_mask, input_scan, vert_levels, output_path, qc_folder, min_lesion_size_mm3=min_lesion_size_mm3)

    # Compute CSA per native slice and per PAM50 slice
    csv_native, csv_pam50 = compute_csa_native_and_pam50(sc_mask, vert_levels, output_path, qc_folder)
    df_native = pd.read_csv(csv_native)
    df_pam50 = pd.read_csv(csv_pam50)

    df = pd.DataFrame({
        "scan_axial_slice": df_native["Slice (I->S)"],
        "pam50_axial_slice": df_pam50["Slice (I->S)"],
        "CSA_mm2": df_native["MEAN(area)"],
    })

    # Annotate each slice with any lesion(s) present at that slice
    df = add_lesion_columns(df, lesion_statistics)

    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def main():
    args = parse_args()
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the include json file
    with open(args.include_json, "r") as f:
        include_data = json.load(f)

    # Initialize a dataframe to store the per-slice results for all scans and a list to track failures
    df_all = pd.DataFrame()
    failed_scans = []

    # Iterate over all subjects, sessions and scans listed in the include json file
    for subject_id, sessions in include_data.items():
        for session_id, scans in sessions.items():
            for scan_name, scan_path in scans.items():
                print(f"Processing {scan_name} ({subject_id}/{session_id})")
                try:
                    scan_csv = compute_csa_with_lesions(scan_path, output_folder, min_lesion_size_mm3=args.min_lesion_size)
                    df_scan = pd.read_csv(scan_csv)
                    df_scan["subject_id"] = subject_id
                    df_scan["session_id"] = session_id
                    df_scan["scan_file"] = scan_path
                    df_all = pd.concat([df_all, df_scan], ignore_index=True)
                except Exception as e:
                    print(f"Error processing {scan_path}: {e}")
                    traceback.print_exc()
                    failed_scans.append({"subject_id": subject_id, "session_id": session_id, "scan_file": scan_path, "error": str(e)})

    # Save the aggregated per-slice CSA report for all scans
    final_csv_path = os.path.join(output_folder, "final_csa_with_lesions.csv")
    df_all.to_csv(final_csv_path, index=False)
    print(f"Final CSA report saved to: {final_csv_path}")

    # Save the list of scans that failed processing, if any
    if failed_scans:
        failed_scans_csv_path = os.path.join(output_folder, "failed_scans.csv")
        pd.DataFrame(failed_scans).to_csv(failed_scans_csv_path, index=False)
        print(f"{len(failed_scans)} scan(s) failed processing. See: {failed_scans_csv_path}")

    return None


if __name__ == "__main__":
    main()
