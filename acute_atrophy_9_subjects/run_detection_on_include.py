"""
This script runs the critical lesion detection pipeline on every scan listed in
an include json file (i.e. the output of create_include_json.py)

For each scan, the subject's sex and date of birth are looked up in the
dataset's participants.tsv file and `detect_critical_lesions` is called
(lesion segmentation is run automatically since no lesion mask is provided).
Per-subject reports are aggregated into a single csv file.

Input:
    -i / --include-json: path to the include json file (output of create_include_json.py)
    -d / --dataset_path: path to the dataset (BIDS format), used to locate participants.tsv
    --hc-data: path to the healthy control data folder (used for atrophy detection)
    --min-lesion-size: minimum lesion size (in mm3) to keep a lesion; smaller lesions are treated as spurious detections and discarded (default: 15.0)
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
from detect_critical_lesion import detect_critical_lesions


def parse_args():
    parser = argparse.ArgumentParser(description="Run the critical lesion detection pipeline on all scans listed in an include json file.")
    parser.add_argument("-i", "--include_json", type=str, required=True, help="Path to the include json file (output of create_include_json.py).")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format), used to locate participants.tsv.")
    parser.add_argument("--hc-data", type=str, required=True, help="Path to the folder containing the healthy control data (used for atrophy detection).")
    parser.add_argument("--min-lesion-size", type=float, default=15.0, help="Minimum lesion size (in mm3) to keep a lesion; smaller lesions are treated as spurious detections and discarded (default: 15.0)")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the detection results will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = args.output_folder
    path_hc_data = args.hc_data
    os.makedirs(output_folder, exist_ok=True)

    # Load the include json file
    with open(args.include_json, "r") as f:
        include_data = json.load(f)

    # Load the participants tsv file
    participants_tsv_path = os.path.join(args.dataset_path, "participants.tsv")
    participants_df = pd.read_csv(participants_tsv_path, sep="\t")

    # Initialize a dataframe to store the reports for all scans and a list to track failures
    df_reports = pd.DataFrame()
    failed_scans = []

    # Iterate over all subjects, sessions and scans listed in the include json file
    for subject_id, sessions in include_data.items():
        sex = participants_df[participants_df["participant_id"] == subject_id]["sex"].values[0]
        date_of_birth = str(participants_df[participants_df["participant_id"] == subject_id]["date_of_birth"].values[0])
        date_of_birth = date_of_birth.replace("-", "")

        for session_id, scans in sessions.items():
            for scan_name, scan_path in scans.items():
                print(f"Running detection for {scan_name} ({subject_id}/{session_id})")
                try:
                    sub_report_csv = detect_critical_lesions(scan_path, sex, date_of_birth, output_folder, path_hc_data, min_lesion_size_mm3=args.min_lesion_size)
                    if sub_report_csv is None:
                        print(f"No lesions detected for {scan_name} ({subject_id}/{session_id}). Skipping report aggregation.")
                        continue
                    df_report_sub = pd.read_csv(sub_report_csv)
                    df_report_sub["subject_id"] = subject_id
                    df_report_sub["session_id"] = session_id
                    df_report_sub["scan_file"] = scan_path
                    df_reports = pd.concat([df_reports, df_report_sub], ignore_index=True)
                except Exception as e:
                    print(f"Error processing {scan_path}: {e}")
                    traceback.print_exc()
                    failed_scans.append({"subject_id": subject_id, "session_id": session_id, "scan_file": scan_path, "error": str(e)})

    # Save the aggregated report for all scans
    final_report_csv_path = os.path.join(output_folder, "final_report.csv")
    df_reports.to_csv(final_report_csv_path, index=False)
    print(f"Final report saved to: {final_report_csv_path}")

    # Save the list of scans that failed processing, if any
    if failed_scans:
        failed_scans_csv_path = os.path.join(output_folder, "failed_scans.csv")
        pd.DataFrame(failed_scans).to_csv(failed_scans_csv_path, index=False)
        print(f"{len(failed_scans)} scan(s) failed processing. See: {failed_scans_csv_path}")

    return None


if __name__ == "__main__":
    main()
