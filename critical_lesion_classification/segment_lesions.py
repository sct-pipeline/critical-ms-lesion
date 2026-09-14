"""
Step 1 of the critical lesion classification study: segment the spinal cord and the MS lesions of
every scan listed in the include yml file, with the SCT lesion_ms model.

The predicted segmentations mirror the relative path of each scan inside the dataset:
    <output_folder>/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w_label-SC_seg.nii.gz
    <output_folder>/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w_label-lesion_seg.nii.gz

A QC report of the lesion segmentation overlaid on the spinal cord segmentation is generated in
<output_folder>/QC, and a summary csv listing every scan with its segmentations is written to
<output_folder>/predicted_segmentations.csv.

Input:
    -i / --include: path to the include yml file
    -o: path to the output folder where the predicted segmentations will be saved
    --overwrite: recompute the segmentations even if they already exist

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from include_io import load_include, get_pred_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Segment the spinal cord and the MS lesions of every scan of the include yml file with the SCT lesion_ms model.")
    parser.add_argument("-i", "--include", type=str, required=True, help="Path to the include yml file.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the predicted segmentations will be saved.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute the segmentations even if they already exist.")
    return parser.parse_args()


def segment_scan(image, sc_seg, lesion_seg, qc_folder, overwrite=False):
    """
    Segment the spinal cord and the MS lesions of one scan.
    Input:
        image: path to the MRI scan (NIfTI format)
        sc_seg: path where the spinal cord segmentation will be saved
        lesion_seg: path where the lesion segmentation will be saved
        qc_folder: path to the QC folder
        overwrite: whether to recompute the segmentations even if they already exist
    Output:
        None
    """
    os.makedirs(os.path.dirname(sc_seg), exist_ok=True)

    if overwrite or not os.path.exists(sc_seg):
        assert os.system(
            f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {image} -o {sc_seg}"
        ) == 0, "Error running the SC segmentation model"

    if overwrite or not os.path.exists(lesion_seg):
        assert os.system(
            f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {lesion_seg} -test-time-aug -qc {qc_folder} -qc-seg {sc_seg} -qc-plane Axial"
        ) == 0, "Error running the lesion segmentation model"


def main():
    args = parse_args()
    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    qc_folder = os.path.join(output_folder, "QC")
    os.makedirs(qc_folder, exist_ok=True)

    # Load the include file and convert to entries:
    entry_list = load_include(args.include)

    rows = []
    failed_scans = []
    for entry in tqdm(entry_list, desc="Segmenting lesions"):
        image = entry["image"]
        sc_seg, lesion_seg = get_pred_paths(entry, output_folder)
        try:
            segment_scan(entry["image"], sc_seg, lesion_seg, qc_folder, overwrite=args.overwrite)
        except Exception as error:
            print(f"Error segmenting {entry['image']}: {error}")
            traceback.print_exc()
            failed_scans.append({**{key: entry[key] for key in ("subject", "session", "scan_id", "image")}, "error": str(error)})
            continue

        rows.append({
            "subject": entry["subject"],
            "session": entry["session"],
            "scan_id": entry["scan_id"],
            "scan_file": entry["image"],
            "manual_seg_file": entry["label"],
            "pred_sc_seg_file": sc_seg,
            "pred_seg_file": lesion_seg,
        })
        break

    # Save the summary csv listing every scan with its segmentations
    summary_csv = os.path.join(output_folder, "predicted_segmentations.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"{len(rows)} scans segmented. Summary saved to: {summary_csv}")
    print(f"QC report saved to: {qc_folder}")

    if failed_scans:
        failed_csv = os.path.join(output_folder, "failed_scans.csv")
        pd.DataFrame(failed_scans).to_csv(failed_csv, index=False)
        print(f"{len(failed_scans)} scan(s) failed. See: {failed_csv}")


if __name__ == "__main__":
    main()
