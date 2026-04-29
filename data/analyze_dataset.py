"""
This code takes as input the include yml file and analyzes the resolution of the images.

Input:
    --include-yml: path to the include yml file containing the list of lesion masks to analyze
    -d: path to the dataset (BIDS format)
    -o: path to the output folder where the analysis results will be saved.

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from loguru import logger
import yaml
import pandas as pd
from pathlib import Path
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the resolution of the images in the dataset.")
    parser.add_argument("--include-yml", type=str, required=True, help="Path to the include yml file containing the list of lesion masks to analyze")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset (BIDS format)")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the analysis results will be saved.")
    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_args()
    include_file = args.include_yml
    dataset_path = args.dataset
    output_folder = args.output_folder

    # Build the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a logger in the output folder
    if os.path.exists(os.path.join(output_folder, "analyze_dataset.log")):
        os.remove(os.path.join(output_folder, "analyze_dataset.log"))
    logger.add(os.path.join(output_folder, "analyze_dataset.log"),level="INFO")

    # Load the include.yml file
    with open(include_file, "r") as f:
        include_dict = yaml.safe_load(f)
    # Get the list of scans to include
    include_scans = include_dict["FILES_SEG"]

    # List all files in the input path
    list_images = list(Path(dataset_path).rglob("*.nii.gz"))
    list_images = [str(image) for image in list_images]

    # Count the number of each contrasts in the dataset
    contrasts = {}
    for image in include_scans:
        contrast = image.split("/")[-1].split("_")[-1].split(".nii.gz")[0]
        if contrast not in contrasts:
            contrasts[contrast] = 0
        contrasts[contrast] += 1
    logger.info("Contrasts in the dataset:")
    for contrast, count in contrasts.items():
        logger.info(f"{contrast}: {count}")

    # Analyze the resolution of the images in the dataset:
    resolutions = []
    for scan in include_scans:
        # We find the corresponding file in the list images
        matching_files = [image for image in list_images if scan in image]
        matching_file = matching_files[0] if len(matching_files) == 1 else None
        # Load the image and get its resolution
        img = nib.load(matching_file)
        # Set orientation to RAS
        img = nib.as_closest_canonical(img)
        # Print the orientation
        orientation = nib.aff2axcodes(img.affine)
        # logger.info(f"File: {matching_file.split('/')[-1]}, Orientation: {orientation}")
        # Get the resolution
        resolution = img.header.get_zooms()
        resolutions.append((matching_file, resolution))
        # logger.info(f"Resolution: {resolution}")

    # Log the mean +- std resolution for each dimension
    resolutions_array = pd.DataFrame(resolutions, columns=["file", "resolution"])
    resolutions_array[["x", "y", "z"]] = pd.DataFrame(resolutions_array["resolution"].tolist(), index=resolutions_array.index)
    mean_resolution = resolutions_array[["x", "y", "z"]].mean()
    std_resolution = resolutions_array[["x", "y", "z"]].std()
    # Log the mean +- std resolution for each dimension
    logger.info("Mean +- std resolution for each dimension in RAS orientation:")
    for dim in ["x", "y", "z"]:
        logger.info(f"Mean resolution for {dim}: {mean_resolution[dim]:.2f} +- {std_resolution[dim]:.2f}")


if __name__ == "__main__":
    main()