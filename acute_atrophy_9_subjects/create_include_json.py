"""
This script scans a BIDS dataset and creates a JSON file listing all axial
scans of the spinal cord (cervical, thoracic, lumbar), excluding axial brain
scans. The output is structured as:

{
    "sub-XXX": {
        "ses-YYY": {
            "<scan_1>.nii.gz": "<path/to/scan_1.nii.gz>",
            "<scan_2>.nii.gz": "<path/to/scan_2.nii.gz>",
            ...
        },
        ...
    },
    ...
}

An image is considered "axial spinal cord" if its filename contains an
`acq-ax*` entity that is not `acq-axBrain` (e.g. axCerv, axThor, axThorUpper,
axThorLower, axLumb, ...).

Input:
    -d: path to the dataset (BIDS format)
    -o: path to the output json file
    --contrast: if provided, the script will only process scans that contain the selected contrast in their filename (e.g. T1w, T2w, ...)


Author: Pierre-Louis Benveniste
"""
import re
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a json file listing all axial spinal cord scans in a BIDS dataset, grouped by subject and session.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset (BIDS format).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output json file.")
    parser.add_argument("--contrast", type=str, help="If provided, the script will only process scans that contain the selected contrast in their filename (e.g. T1w, T2w, ...)")
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


scans_to_remove = [
    "sub-007_ses-20250318_acq-axLumb_run-02_T1w.nii.gz",
    "sub-007_ses-20250318_acq-axLumb_run-02_T2w.nii.gz"
]


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    # List all nii.gz files in the anat folders of the dataset (skip derivatives/sourcedata)
    all_scans = sorted(dataset_path.rglob("anat/*.nii.gz"))
    all_scans = [f for f in all_scans if "derivatives" not in f.parts and "sourcedata" not in f.parts]

    # Remove all scans where Gad was used
    all_scans = [f for f in all_scans if "ce-Gad" not in f.name]

    # Keep only axial spinal cord scans (exclude axial brain scans)
    included_scans = [f for f in all_scans if is_axial_spinal_cord_scan(f.name)]
    # Remove scans that are in the scans_to_remove list
    included_scans = [f for f in included_scans if f.name not in scans_to_remove]
    # If a contrast is provided, keep only scans that contain the selected contrast in their filename
    if args.contrast:
        included_scans = [f for f in included_scans if args.contrast in f.name]

    # Group the included scans by subject and session, mapping each filename to its path
    scans_by_subject = {}
    for f in included_scans:
        subject = next(part for part in f.parts if part.startswith("sub-"))
        session = next(part for part in f.parts if part.startswith("ses-"))
        scans_by_subject.setdefault(subject, {}).setdefault(session, {})[f.name] = str(f)

    # Write the output json file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scans_by_subject, f, indent=4, sort_keys=True)

    print(f"Found {len(included_scans)} axial spinal cord scans out of {len(all_scans)} anat scans.")
    print(f"Include file written to: {output_path}")


if __name__ == "__main__":
    main()
