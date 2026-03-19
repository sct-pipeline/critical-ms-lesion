"""
This script is used to perform lesion segmentation on all T2w axial scans of the dataset which will be used for critical lesion detection.
Two steps:
- segmentation of the SC
- segmentation of the lesions and QC creation with the SC segmentation and the lesion segmentation

Input:
    -d: path to the dataset (BIDS format)
    -o: path to the output folder where the lesion segmentation will be saved
    -cerv: flag to only perform the segmentation on cervical scans (i.e. those that contain "cerv" in their name)

Author: Pierre-Louis Benveniste
"""
import os
from pathlib import Path
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Perform lesion segmentation on all T2w axial scans of the dataset which will be used for critical lesion detection.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the lesion segmentation will be saved.")
    parser.add_argument("-cerv", "--cervical", action="store_true", help="Flag to only perform the segmentation on cervical scans.")
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


def main(dataset_path, output_folder, cervical=False):

    # Build output folder 
    os.makedirs(output_folder, exist_ok=True)
    
    # Build a QC folder
    qc_folder = os.path.join(output_folder, "QC")
    os.makedirs(qc_folder, exist_ok=True)
    
    # Get all T2w axial scans in the dataset
    t2w_axial_scans = get_t2w_ax_scans(dataset_path, cervical=cervical)
    print(len(t2w_axial_scans), "T2w axial scans found in the dataset.")

    # For each scan:
    for scan in tqdm(t2w_axial_scans):
        sub_name = scan.split("/")[-1].split("_")[0]
        ses_name = scan.split("/")[-1].split("_")[1]
        # Build an output folder:
        output_folder_scan = os.path.join(output_folder, sub_name, ses_name)
        os.makedirs(output_folder_scan, exist_ok=True)
        # Build output path:
        sc_mask = os.path.join(output_folder_scan, scan.split("/")[-1].replace(".nii.gz", "_label-SC_seg.nii.gz"))
        lesion_mask = os.path.join(output_folder_scan, scan.split("/")[-1].replace(".nii.gz", "_label-lesion_seg.nii.gz"))
        # Run the SC seg command
        assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {scan} -o {sc_mask}") == 0, "Error running the SC segmentation model"
        # Run the lesion seg command
        assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {scan} -o {lesion_mask} -qc {qc_folder} -qc-seg {sc_mask} -qc-plane Axial" ) == 0, "Error running the lesion segmentation model"


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_path, args.output_folder, args.cervical)