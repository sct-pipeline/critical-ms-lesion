"""
This script runs spinal cord segmentation (sct_deepseg spinalcord) on every scan
listed in an include json file (i.e. the output of create_include_json.py) and
generates a single aggregated QC report for visual inspection of the segmentations.

Input:
    -i / --include_json: path to the include json file (output of create_include_json.py)
    -o / --output_folder: path to the output folder where the SC segmentations will be saved

Author: Pierre-Louis Benveniste
"""
import os
import sys
import json
import argparse
import traceback
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from detect_critical_lesion import run_sc_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a QC report of spinal cord segmentation for all scans listed in an include json file.")
    parser.add_argument("-i", "--include_json", type=str, required=True, help="Path to the include json file (output of create_include_json.py).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the SC segmentations will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = args.output_folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(qc_folder, exist_ok=True)

    # Load the include json file
    with open(args.include_json, "r") as f:
        include_data = json.load(f)

    failed_scans = []

    # Iterate over all subjects, sessions and scans listed in the include json file
    for subject_id, sessions in include_data.items():
        for session_id, scans in sessions.items():
            for scan_name, scan_path in scans.items():
                print(f"Running SC segmentation for {scan_name} ({subject_id}/{session_id})")
                image_name = scan_name.replace(".nii.gz", "")
                scan_output_path = os.path.join(output_folder, image_name)
                os.makedirs(scan_output_path, exist_ok=True)
                try:
                    run_sc_segmentation(scan_path, scan_output_path, qc_folder)
                except Exception as e:
                    print(f"Error processing {scan_path}: {e}")
                    traceback.print_exc()
                    failed_scans.append({"subject_id": subject_id, "session_id": session_id, "scan_file": scan_path, "error": str(e)})

    print(f"QC report saved to: {qc_folder}")

    # Save the list of scans that failed processing, if any
    if failed_scans:
        failed_scans_csv_path = os.path.join(output_folder, "failed_scans.csv")
        pd.DataFrame(failed_scans).to_csv(failed_scans_csv_path, index=False)
        print(f"{len(failed_scans)} scan(s) failed processing. See: {failed_scans_csv_path}")

    return None


if __name__ == "__main__":
    main()
