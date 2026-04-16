"""
This script is used to perform lesion segmentation on all T2w axial scans of the dataset which will be used for critical lesion detection.
Two steps:
- segmentation of the SC
- segmentation of the lesions and QC creation with the SC segmentation and the lesion segmentation

Input:
    -d: path to the dataset (BIDS format)
    --include: yml file of the files to include in the segmentation
    -o: path to the output folder where the lesion segmentation will be saved
    -cerv: flag to only perform the segmentation on cervical scans (i.e. those that contain "cerv" in their name)
    -label: flag to label each lesion with a different label

Author: Pierre-Louis Benveniste
"""
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import nibabel as nib
import numpy as np
from scipy.ndimage import label
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Perform lesion segmentation on all T2w axial scans of the dataset which will be used for critical lesion detection.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("--include", type=str, help="YML file of the files to include in the segmentation.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the lesion segmentation will be saved. (It can be the derivatives folder)")
    parser.add_argument("-cerv", "--cervical", action="store_true", help="Flag to only perform the segmentation on cervical scans.")
    parser.add_argument("-label", "--label_lesions", action="store_true", help="Flag to label each lesion with a different label. By default, all lesions will have the same label (i.e. 1).")
    return parser.parse_args()


def get_t2w_ax_scans(dataset_path, cervical=False):
    # Get all T2w axial scans in the dataset
    t2w_axial_scans = list(Path(dataset_path).rglob("*T2w.nii.gz"))
    t2w_axial_scans = [str(scan) for scan in t2w_axial_scans]
    # Keep only axial scans
    t2w_axial_scans = [scan for scan in t2w_axial_scans if "acq-ax" in scan.split("/")[-1].lower()]
    keywords = ["spine", "cerv", "thor", "lumb"]
    if cervical:
        keywords = ["cerv"]
    filtered_scans = [scan for scan in t2w_axial_scans if any(keyword in scan.split("/")[-1].lower() for keyword in keywords)]
    # Remove scans that contain myelo in their name
    filtered_scans = [scan for scan in filtered_scans if "myelo" not in scan.split("/")[-1].lower()]
    # Remove localizer scans (i.e. those that contain "loc" in their name)
    filtered_scans = [scan for scan in filtered_scans if "loc" not in scan.split("/")[-1].lower()]
    # Remove scans with "ce-gad"
    filtered_scans = [scan for scan in filtered_scans if "ce-gad" not in scan.split("/")[-1].lower()]
    filtered_scans_wuthout_run_above_1 = []
    for scan in filtered_scans:
        scan_name = scan.split("/")[-1]
        if "run-" in scan_name:
            run_value = scan_name.split("run-")[1][0:2]
            if run_value == "01":
                filtered_scans_wuthout_run_above_1.append(scan)
        else:
            filtered_scans_wuthout_run_above_1.append(scan)
    filtered_scans = filtered_scans_wuthout_run_above_1

    # Sort by alphanumeric order
    filtered_scans = sorted(filtered_scans)

    return filtered_scans


def main(dataset_path, output_folder, include_yml, cervical=False):

    # Build output folder 
    os.makedirs(output_folder, exist_ok=True)
    
    # Build a QC folder
    qc_folder = os.path.join(output_folder, "QC")
    os.makedirs(qc_folder, exist_ok=True)
    
    # Get all T2w axial scans in the dataset
    t2w_axial_scans = get_t2w_ax_scans(dataset_path, cervical=cervical)
    print(len(t2w_axial_scans), "T2w axial scans found in the dataset.")

    # Load the yml file
    if include_yml:
        with open(include_yml, "r") as f:
            include_dict = yaml.safe_load(f)
        # Get the list of scans to include
        include_scans = include_dict["FILES_SEG"]
        # Keep only the scans that are in the include list
        t2w_axial_scans = [scan for scan in t2w_axial_scans if any(included_scan in scan for included_scan in include_scans)]
        print(len(t2w_axial_scans), "T2w axial scans found in the dataset after applying the include filter.")

    # For each scan:
    for scan in tqdm(t2w_axial_scans):
        sub_name = scan.split("/")[-1].split("_")[0]
        ses_name = scan.split("/")[-1].split("_")[1]
        # Copy the relatie path of the scan in the output folder
        relative_path = Path(os.path.relpath(scan, dataset_path)).parent
        output_pred_folder = os.path.join(output_folder, relative_path)
        # Build an output folder:
        os.makedirs(output_pred_folder, exist_ok=True)
        # Build output path:
        sc_mask = os.path.join(output_pred_folder, scan.split("/")[-1].replace(".nii.gz", "_label-SC_seg.nii.gz"))
        lesion_mask = os.path.join(output_pred_folder, scan.split("/")[-1].replace(".nii.gz", "_label-lesion_seg.nii.gz"))
        # Run the SC seg command
        assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {scan} -o {sc_mask}") == 0, "Error running the SC segmentation model"
        # Run the lesion seg command
        assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {scan} -o {lesion_mask} -test-time-aug -qc {qc_folder} -qc-seg {sc_mask} -qc-plane Axial" ) == 0, "Error running the lesion segmentation model"
        # If the label_lesions flag is set, we will label each lesion with a different label (i.e. 1, 2, 3, etc.)
        if args.label_lesions:
            # We load the lesion mask and we label each connected component with a different label
            lesion_mask_nii = nib.load(lesion_mask)
            lesion_mask_data = lesion_mask_nii.get_fdata()
            # We label each connected component with a different label
            labeled_lesions, num_lesions = label(lesion_mask_data)
            # We save the new lesion mask
            labeled_lesions_nii = nib.Nifti1Image(labeled_lesions, lesion_mask_nii.affine, lesion_mask_nii.header)
            # Output path for labeled seg
            labeled_lesion_mask = os.path.join(output_pred_folder, scan.split("/")[-1].replace(".nii.gz", "_label-lesion_seg-labeled.nii.gz"))
            nib.save(labeled_lesions_nii, labeled_lesion_mask)


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_path, args.output_folder, args.include, args.cervical)