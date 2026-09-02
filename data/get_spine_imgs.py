"""
This script takes a subject folder as input and returns the available spine scans in the subject folder. 
Format of the folder (BIDS)

Input:
    -i: subject folder (e.g. sub-001)

Output:
    - A list of available spine scans in the subject folder.

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Get available spine scans in a subject folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to the subject folder (e.g. sub-001).")
    return parser.parse_args()


def get_spine_scans(input_folder):
    # Initialize output dictionnary
    spine_scans = {}

    # Get sessions folder
    session_folders = [f for f in os.listdir(input_folder) if f.startswith("ses-") and os.path.isdir(os.path.join(input_folder, f))]

    # Iterate over sessions
    for session in session_folders:
        session_path = os.path.join(input_folder, session)
        # Get anat folder
        anat_folder = os.path.join(session_path, "anat")
        # Get all .nii.gz files in anat folder
        scans = list(Path(anat_folder).rglob("*.nii.gz"))
        scans = [ str(scan) for scan in scans] 
        # Only scans that contain either spine, cerv, thor or lumb are kept
        keywords = ["spine", "cerv", "thor", "lumb"]
        filtered_scans = [scan for scan in scans if any(keyword in scan.lower() for keyword in keywords)]
        # Remove scans that contain myelo in their name
        filtered_scans = [scan for scan in filtered_scans if "myelo" not in scan.lower()]
        # Remove localizer scans (i.e. those that contain "loc" in their name)
        filtered_scans = [scan for scan in filtered_scans if "loc" not in scan.split("/")[-1].lower()]

        # Add to output dictionnary
        spine_scans[session] = filtered_scans

    return spine_scans


if __name__ == "__main__":
    args = parse_args()
    spine_scans = get_spine_scans(args.input_folder)
    for session, scans in spine_scans.items():
        print(f"Session: {session}")
        for scan in scans:
            print(f"\t{scan}")