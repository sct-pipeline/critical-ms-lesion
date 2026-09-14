"""
This script scans a BIDS dataset and builds the include yml file which drives the critical lesion
classification study: every scan which has a manual lesion segmentation in the derivatives folder
is listed, together with the path of that segmentation.

The manual lesion segmentations are expected to have value 1 for non-critical lesions and value 2
for critical lesions. Any other value will be considered unexpected and the scan will be skipped.

Input:
    -d: path to the dataset (BIDS format)
    -o: path to the output include yml file

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Values used in the manual lesion segmentations
NON_CRITICAL_VALUE = 1
CRITICAL_VALUE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Build the include yml file pairing every scan of the study with its manual lesion segmentation.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output include yml file.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset_path)
    derivatives_path = os.path.join(dataset_path, "derivatives/labels")

    # List every manual lesion segmentation of the derivatives folder
    manual_segs = sorted(Path(derivatives_path).rglob(f"*_label-lesion_seg.nii.gz"))
    manual_segs = [str(seg) for seg in manual_segs]
    print(f"Found {len(manual_segs)} manual lesion segmentations in {derivatives_path}")

    entries = []
    skipped = []
    scans_with_critical = 0
    scans_without_critical = 0
    scans_with_no_lesions = 0
    for manual_seg in tqdm(manual_segs):
        # The scan is at the same relative path in the dataset, without the label suffix
        image_path = manual_seg.replace(f"_label-lesion_seg.nii.gz", ".nii.gz").replace("derivatives/labels/", "")

        if not os.path.exists(image_path):
            skipped.append((image_name, "scan not found in the dataset"))
            continue

        # Check the values of the manual segmentation and count its lesions
        seg_data = nib.load(manual_seg).get_fdata()
        values = sorted(int(value) for value in np.unique(seg_data) if value != 0)
        unexpected_values = [value for value in values if value not in (NON_CRITICAL_VALUE, CRITICAL_VALUE)]
        if unexpected_values:
            skipped.append((image_name, f"unexpected values in the manual segmentation: {unexpected_values}"))
            continue

        entries.append({
            "image": image_path,
            "label": manual_seg
        })

        # Count the number of scans with critical lesions, scans without critical lesions, and scans with no lesions at all
        if CRITICAL_VALUE in values:
            scans_with_critical += 1
        elif NON_CRITICAL_VALUE in values:
            scans_without_critical += 1
        else:
            scans_with_no_lesions += 1

    # Write the include file
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump({"FILES_SEG": entries}, f, sort_keys=False, default_flow_style=False)

    print(f"{len(entries)} scans written to {output_path}")
    if skipped:
        print(f"{len(skipped)} scans skipped:")
        for image_name, reason in skipped:
            print(f"  - {image_name}: {reason}")

    print(f"Total scans processed: {len(entries)}")
    print(f"Scans with critical lesions: {scans_with_critical}")
    print(f"Scans without critical lesions: {scans_without_critical}")
    print(f"Scans with no lesions: {scans_with_no_lesions}")


if __name__ == "__main__":
    main()