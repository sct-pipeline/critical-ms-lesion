"""
This code goes through all labeled lesion segmentation in a folder. It generates one lesion mask file per lesion. 
Then it generates a QC file for each lesion mask file. The QC file is generated using the sct_qc command.

Input:
    -d: path to the BIDS dataset
    -p: path the predicted segmentation folder
    -o: path to the output folder where the QC files will be saved

Author: Pierre-Louis Benveniste
"""
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import nibabel as nib
import numpy as np
from scipy.ndimage import label
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QC files for critical lesion identification.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the BIDS dataset.")
    parser.add_argument("-p", "--predicted_segmentation_folder", type=str, required=True, help="Path to the predicted segmentation folder.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the QC files will be saved.")
    return parser.parse_args()


def main(dataset_path, predicted_segmentation_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Get all lesion segmentation files in the predicted segmentation folder
    lesion_segmentation_files = list(Path(predicted_segmentation_folder).rglob("*_label-lesion_seg.nii.gz"))
    lesion_segmentation_files = [str(file) for file in lesion_segmentation_files]
    lesion_segmentation_files = sorted(lesion_segmentation_files)  # Sort the files to ensure consistent order

    # Build qc path
    qc_output_folder = os.path.join(output_folder, "QC")
    os.makedirs(qc_output_folder, exist_ok=True)

    # Initialize a df which has columns lesion_mask_file, original_scan_file and critical_lesion (boolean)
    df_lesion_qc = pd.DataFrame(columns=["lesion_mask_file", "original_scan_file", "critical_lesion"])
   
    # For each lesion segmentation file, we generate a QC file for each lesion
    for lesion_segmentation_file in tqdm(lesion_segmentation_files):
        # We need to find the corresponding SC mask
        sc_mask_file = lesion_segmentation_file.replace("_label-lesion_seg.nii.gz", "_label-SC_seg.nii.gz")
        # We need to find the corresponding original scan
        original_scan_file = lesion_segmentation_file.replace(predicted_segmentation_folder, dataset_path).replace("_label-lesion_seg.nii.gz", ".nii.gz")
        # We load the lesion segmentation file
        lesion_segmentation_nii = nib.load(lesion_segmentation_file)
        lesion_segmentation_data = lesion_segmentation_nii.get_fdata()
        # We label each connected component with a different label
        s = np.ones((3, 3, 3))  # Define the structure for connectivity (26-connectivity)
        labeled_lesions, num_lesions = label(lesion_segmentation_data, structure=s)
        print(f"Found {num_lesions} lesions in file {lesion_segmentation_file}")
        if num_lesions == 0:
            continue
        # For each lesion, we generate a QC file
        for lesion_label in range(1, num_lesions + 1):
            # We create a binary mask for the current lesion
            lesion_mask = (labeled_lesions == lesion_label).astype(np.uint8)
            # We save the binary mask as a nifti file
            lesion_mask_nii = nib.Nifti1Image(lesion_mask, lesion_segmentation_nii.affine, lesion_segmentation_nii.header)
            relative_path = Path(os.path.relpath(lesion_segmentation_file, predicted_segmentation_folder)).parent
            lesion_mask_file = os.path.join(output_folder, relative_path, os.path.basename(lesion_segmentation_file).replace(".nii.gz", f"_lesion-{lesion_label}_mask.nii.gz"))
            # build output folder
            os.makedirs(os.path.dirname(lesion_mask_file), exist_ok=True)
            nib.save(lesion_mask_nii, lesion_mask_file)
            # We generate the QC file for the current lesion mask
            assert os.system(f"sct_qc -i {original_scan_file} -p sct_deepseg_lesion -s {sc_mask_file} -d {lesion_mask_file} -plane axial -qc {qc_output_folder}") == 0, "Error running the sct_qc command"
            # We add a line to the df
            new_row = {"lesion_mask_file": lesion_mask_file, "original_scan_file": original_scan_file, "critical_lesion": None}
            df_lesion_qc = pd.concat([df_lesion_qc, pd.DataFrame([new_row])], ignore_index=True)

    # We save the df as a csv file
    df_lesion_qc.to_csv(os.path.join(output_folder, "lesion_labels.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_path, args.predicted_segmentation_folder, args.output_folder)