"""
This file runs the detection of critical lesions on the labeled lesion segmentation.

Input:
    -d: path to the dataset (BIDS format)
    --csv: path to the csv file containing the lesion labels (i.e. the output of generate_qc_for_critical_lesion_identification.py)
    --path-hc-data: path to the folder containing the healthy control data (used for atrophy detection)
    -o: path to the output folder where the lesion segmentation and QC files will be saved. (It can be the derivatives folder)

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from detect_critical_lesion import detect_critical_lesions
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run the detection of critical lesions on the labeled lesion segmentation.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("--csv", type=str, required=True, help="Path to the csv file containing the lesion labels (i.e. the output of generate_qc_for_critical_lesion_identification.py).")
    parser.add_argument("--path-hc-data", type=str, required=True, help="Path to the folder containing the healthy control data (used for atrophy detection).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the lesion segmentation and QC files will be saved. (It can be the derivatives folder)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    csv_file = args.csv
    output_folder = args.output_folder
    path_hc_data = args.path_hc_data

    # Load the csv file:
    with open(csv_file, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    data = [line.strip().split(",") for line in lines[1:]]

    # Convert data to a df
    df = pd.DataFrame(data, columns=header)

    # Load the participants tsv file
    participants_tsv_path = os.path.join(dataset_path, "participants.tsv")
    participants_df = pd.read_csv(participants_tsv_path, sep="\t")

    # Initialize a dataframe to store the reports for each lesion
    df_reports = pd.DataFrame()

    # For each lesion mask, run the detection of critical lesions:
    for index, row in df.iterrows():
        lesion_mask_path = row["lesion_mask_file"]
        mri_scan_path = row["original_scan_file"]
        label = row["critical_lesion"]

        # We exclude the following lesion segs to avoid pipeline errors:
        to_exclude = ["sub-015_ses-20061013_acq-axCerv_T2w_label-lesion_seg_lesion-3_mask.nii.gz",
                      "sub-019_ses-19990611_acq-axCerv_T2w_label-lesion_seg_lesion-2_mask.nii.gz",
                      "sub-032_ses-20000418_acq-axCerv_T2w_label-lesion_seg_lesion-2_mask.nii.gz",
                      "sub-055_ses-20180820_acq-axCerv_T2w_label-lesion_seg_lesion-1_mask.nii.gz"]
        if lesion_mask_path.split("/")[-1] in to_exclude:
            continue

        # Subject id 
        subject_id = os.path.basename(mri_scan_path).split("_")[0]
        sex = participants_df[participants_df["participant_id"] == subject_id]["sex"].values[0]
        date_birth = participants_df[participants_df["participant_id"] == subject_id]["date_of_birth"].values[0]
        date_birth = date_birth.replace("-", "")
        lesion_id = lesion_mask_path.split("/")[-1].split("_")[-2]

        # Build output path for the subject:
        subject_lesion_output_path = os.path.join(output_folder, subject_id + "_" + lesion_id)
        os.makedirs(subject_lesion_output_path, exist_ok=True)

        # Run the detection of critical lesions:
        sub_report_csv = detect_critical_lesions(mri_scan_path, sex, date_birth, subject_lesion_output_path, path_hc_data, lesion_mask_input=lesion_mask_path)

        # Store the laterality report for this lesion in the dictionary
        df_report_sub = pd.read_csv(sub_report_csv)
        df_report_sub["scan_file"] = mri_scan_path
        df_report_sub["lesion_mask_file"] = lesion_mask_path
        df_report_sub["critical_lesion_label"] = label
        df_reports = pd.concat([df_reports, df_report_sub], ignore_index=True)

    # Save the final report for all lesions
    final_report_csv_path = os.path.join(output_folder, "final_report.csv")
    df_reports.to_csv(final_report_csv_path, index=False)

    return None


if __name__ == "__main__":
    main()