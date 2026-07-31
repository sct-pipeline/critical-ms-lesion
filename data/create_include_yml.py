"""
This script scans a BIDS dataset and creates an include.yml file listing all
axial scans of the spinal cord (cervical, thoracic, lumbar), excluding axial
brain scans.

An image is considered "axial spinal cord" if its filename contains an
`acq-ax*` entity that is not `acq-axBrain` (e.g. axCerv, axThor, axThorUpper,
axThorLower, axLumb, ...).

Input:
    -d: path to the dataset (BIDS format)
    -o: path to the output include.yml file

Author: Pierre-Louis Benveniste
"""
import re
import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create an include.yml file listing all axial spinal cord scans in a BIDS dataset.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output include.yml file.")
    return parser.parse_args()


def is_axial_spinal_cord_scan(filename):
    """
    Return True if the filename has an acq-ax* entity that does not refer to the brain.
    """
    match = re.search(r"acq-(ax[A-Za-z]*)", filename)
    if match is None:
        return False
    acq_value = match.group(1)
    return "brain" not in acq_value.lower()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    # List all nii.gz files in the anat folders of the dataset (skip derivatives/sourcedata)
    all_scans = sorted(dataset_path.rglob("anat/*.nii.gz"))
    all_scans = [f for f in all_scans if "derivatives" not in f.parts and "sourcedata" not in f.parts]

    # Keep only axial spinal cord scans (exclude axial brain scans)
    included_scans = [f.name for f in all_scans if is_axial_spinal_cord_scan(f.name)]

    # Write the include.yml file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump({"FILES_SEG": included_scans}, f, default_flow_style=False, sort_keys=False)

    print(f"Found {len(included_scans)} axial spinal cord scans out of {len(all_scans)} anat scans.")
    print(f"Include file written to: {output_path}")


if __name__ == "__main__":
    main()
