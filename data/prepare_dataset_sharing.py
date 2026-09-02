"""
This script copies the selected files to a new location and segments the spinal cord and lesions for those images.

Input:
    -d: path to the dataset (BIDS format)
    --include: path to the yml file containing the selected images
    -o: path to the output folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import shutil
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Copy the selected files to a new location and segment the spinal cord and lesions for those images.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("--include", type=str, required=True, help="Path to the yml file containing the selected images.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder.")

    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_args()
    input_path = args.dataset_path
    include_file = args.include
    output_folder = args.output_folder

    # Build the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # We build a QC path in the output folder
    qc_folder = os.path.join(output_folder, "qc_lesion_seg")
    os.makedirs(qc_folder, exist_ok=True)

    # Copy everything from the input path to the output folder, except derivatives, sourcedata folder and sub-XX folders
    for item in os.listdir(input_path):
        if item not in ["derivatives", "sourcedata"] and not item.startswith("sub-"):
            # Copy even if not a folder (e.g. participants.tsv)
            if os.path.isdir(os.path.join(input_path, item)):
                shutil.copytree(os.path.join(input_path, item), os.path.join(output_folder, item))
            else:
                shutil.copy2(os.path.join(input_path, item), os.path.join(output_folder, item))

    # Load the include.yml file
    with open(include_file, "r") as f:
        include_dict = yaml.safe_load(f)
    # Get the list of scans to include
    include_scans = include_dict["FILES_SEG"]

    # List all files in the input path
    list_images = list(Path(input_path).rglob("*.nii.gz"))
    list_images = [str(image) for image in list_images]

    # For each file in the include list
    for file in include_scans:
        print(file)
        # We find the corresponding file in the list images
        matching_files = [image for image in list_images if file in image]
        if len(matching_files) == 0:
            print(f"File {file} not found in the input path.")
            break
        elif len(matching_files) > 1:
            print(f"Multiple files found for {file}: {matching_files}. Please check the include file.")
            break
        else:
            input_scan = matching_files[0]
            # We copy the file to the output folder, keeping the same relative path
            relative_path = os.path.relpath(input_scan, input_path)
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_scan, output_path)
            # We build the derivatives output path
            derivatives_output_path = os.path.join(output_folder, "derivatives", "labels", os.path.dirname(relative_path))
            lesion_seg_path = os.path.join(derivatives_output_path, file.replace(".nii.gz", "_label-lesion_seg.nii.gz"))
            sc_seg_path = os.path.join(derivatives_output_path, file.replace(".nii.gz", "_label-SC_seg.nii.gz"))
            # We segment the spinal cord and lesions for the file and save the segmentations in the derivatives folder
            assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {input_scan} -o {sc_seg_path} -qc {qc_folder}") == 0, "Error running the SC segmentation model"
            assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_scan} -test-time-aug -o {lesion_seg_path} -qc {qc_folder} -qc-seg {sc_seg_path} -qc-plane Axial" ) == 0, "Error running the lesion segmentation model"

    return None


if __name__ == "__main__":
    main()