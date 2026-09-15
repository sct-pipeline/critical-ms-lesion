"""
Step 4 of the critical lesion classification study: extract, for every lesion, the MRI features
used to discriminate critical from non-critical lesions (lesion size, focal spinal cord atrophy
compared to a healthy control group, and spinal cord tract involvement).

The feature pipeline of detection/detect_critical_lesion.py is run twice per scan:
    - on the manual lesion segmentation (binarized), which gives the reference lesions and their
      ground-truth label (critical lesions have value 2, non-critical lesions have value 1)
    - on the lesion segmentation predicted by the lesion_ms model, which is what is available at
      inference time, and whose lesions are labelled critical when at least --overlap-ratio of
      their voxels fall inside a critical manual lesion (the rule of evaluate_segmentation.py)

Everything that only depends on the image (spinal cord segmentation, vertebral labelling, PAM50
CSA, registration to the template and warped atlas) is computed once and shared between the two
runs, so only the lesion-dependent steps are computed twice.

Outputs, in the output folder:
    features_manual.csv   one row per manual lesion, with its features and its ground-truth label
    features_pred.csv     one row per predicted lesion, with its features and its inherited label
    failed_scans.csv      scans whose feature extraction failed, with the error
    manual/ and pred/     the per-scan working folders of the feature pipeline (plots, QC, csvs)

Input:
    -d: path to the dataset (BIDS format), only used to read participants.tsv, which must have the
        columns participant_id, sex and date_of_birth
    -i / --include: path to the include yml file
    -p / --pred-folder: folder holding the predicted segmentations (output of segment_lesions.py)
    --path-hc-data: path to the healthy control data folder (spine-generic), used for the
        normalization of the atrophy measures
    -o: path to the output folder where the features will be saved
    --min-lesion-size: minimum lesion size (mm3) to keep a lesion (default: 15.0)
    --overlap-ratio: minimum fraction of a predicted lesion that must fall inside a critical manual
        lesion for it to be labelled critical, as in evaluate_segmentation.py (default: 0.1)
    --mask-source: which mask(s) to process: manual, pred or both (default: both)
    --no-age-normalization: compare each subject to all the healthy controls of the same sex, instead
        of only those of the same 10-year age group (default: age normalization applied)

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
import traceback

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from include_io import load_include, check_entries_exist, get_pred_paths, CRITICAL_VALUE, MIN_LESION_SIZE_MM3
from detect_critical_lesion import detect_critical_lesions


# 26-connectivity, same structure as detect_critical_lesion.get_lesion_stats, so that the component
# labels below are the lesion_label of the per-lesion reports that pipeline writes
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 3)


# Image-level products of the feature pipeline: they only depend on the scan and its spinal cord
# segmentation, so they are computed on the manual run and reused by the predicted run
SHARED_FILES = ["_label-SC_seg.nii.gz", "_label-discs_dlabel.nii.gz", "_label-discs_dlabel.json"]
SHARED_DIRECTORIES = ["laterality_outputs"]
# csa_pam50.csv is image-level too, but the pipeline appends lesion-dependent columns to it, so it
# is copied rather than symlinked
SHARED_COPIES = ["csa_pam50.csv"]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract the MRI features of every lesion, from the manual and from the predicted lesion segmentations.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset (BIDS format), containing participants.tsv.")
    parser.add_argument("-i", "--include", type=str, required=True, help="Path to the include yml file.")
    parser.add_argument("-p", "--pred-folder", type=str, required=True, help="Folder holding the predicted segmentations (output of segment_lesions.py).")
    parser.add_argument("--path-hc-data", type=str, required=True, help="Path to the healthy control data folder (spine-generic), used for the atrophy measures.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the features will be saved.")
    parser.add_argument("--min-lesion-size", type=float, default=MIN_LESION_SIZE_MM3, help=f"Minimum lesion size (mm3) to keep a lesion (default: {MIN_LESION_SIZE_MM3}).")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Minimum fraction of a predicted lesion that must fall inside a critical manual lesion for it to be labelled critical, as in evaluate_segmentation.py (default: 0.1).")
    parser.add_argument("--mask-source", type=str, default="both", choices=["manual", "pred", "both"], help="Which mask(s) to process (default: both).")
    parser.add_argument("--no-age-normalization", action="store_true", help="Compare each subject to all the healthy controls of the same sex, instead of only those of the same 10-year age group. The healthy-control comparison is cached per scan, so use a different output folder than a run with age normalization.")
    return parser.parse_args()


def get_subject_demographics(participants_df, subject):
    """
    Get the sex and the date of birth of a subject, both needed to compare its spinal cord measures
    with the matching healthy controls.
    Output:
        (sex, date_of_birth) with the date of birth formatted as YYYYMMDD
    """
    rows = participants_df[participants_df["participant_id"] == subject]
    return rows["sex"].values[0], str(rows["date_of_birth"].values[0]).replace("-", "")


def binarize_manual_mask(manual_seg_path, output_path):
    """
    Write a binarized copy of a manual lesion segmentation. The feature pipeline labels connected
    components on any non-zero voxel and sct_analyze_lesion expects a binary mask, so the 1/2
    distinction of the manual segmentation must not reach them (two lesions of different classes
    touching each other would otherwise also be handled inconsistently).
    Output:
        Path to the binarized mask
    """
    if os.path.exists(output_path):
        return output_path
    nii = nib.load(manual_seg_path)
    binary_data = (nii.get_fdata() > 0).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(binary_data, nii.affine, nii.header), output_path)
    return output_path


def share_image_level_outputs(source_folder, target_folder, scan_id):
    """
    Make the image-level products computed by a first run of the feature pipeline available to a
    second run on the same scan, so that the spinal cord segmentation, the vertebral labelling, the
    PAM50 CSA and above all the registration to the template are not recomputed.
    Input:
        source_folder: per-scan folder of the run that already completed
        target_folder: per-scan folder of the run about to start
        scan_id: filename of the scan without its extension
    Output:
        None
    """
    os.makedirs(target_folder, exist_ok=True)

    for suffix in SHARED_FILES:
        source = os.path.join(source_folder, scan_id + suffix)
        target = os.path.join(target_folder, scan_id + suffix)
        if os.path.exists(source) and not os.path.exists(target):
            os.symlink(source, target)

    for directory in SHARED_DIRECTORIES:
        source = os.path.join(source_folder, directory)
        target = os.path.join(target_folder, directory)
        if os.path.isdir(source) and not os.path.exists(target):
            os.symlink(source, target)

    for filename in SHARED_COPIES:
        source = os.path.join(source_folder, filename)
        target = os.path.join(target_folder, filename)
        if os.path.exists(source) and not os.path.exists(target):
            # Copied and not symlinked: the pipeline appends lesion-dependent columns to this csv
            pd.read_csv(source).to_csv(target, index=False)


def run_feature_pipeline(entry, lesion_mask, mask_source, sex, date_birth, output_folder, path_hc_data, min_lesion_size, age_normalization=True):
    """
    Run the feature pipeline of detect_critical_lesion.py on one scan with one lesion mask.
    Input:
        entry: one entry of the include file (output of load_include)
        lesion_mask: path to the lesion segmentation to extract the features from
        mask_source: "manual" or "pred", used to pick the working folder
        sex / date_birth: demographics of the subject
        output_folder: root output folder of the study
        path_hc_data: path to the healthy control data folder
        min_lesion_size: minimum lesion size (mm3) to keep a lesion
        age_normalization: whether to compare the subject only to the healthy controls of its
            10-year age group (if False, all the healthy controls of the same sex are used)
    Output:
        (df_features, scan_folder): the per-lesion report of the scan (None if it has no lesion
        above the size threshold) and the per-scan working folder of the pipeline
    """
    mask_source_folder = os.path.join(output_folder, mask_source)
    os.makedirs(mask_source_folder, exist_ok=True)
    # detect_critical_lesions appends the scan name to the output folder it is given
    scan_folder = os.path.join(mask_source_folder, entry["scan_id"])

    report_csv = detect_critical_lesions(
        entry["image"], sex, date_birth, mask_source_folder, path_hc_data,
        lesion_mask_input=lesion_mask, min_lesion_size_mm3=min_lesion_size,
        age_normalization=age_normalization,
    )
    if report_csv is None:
        return None, scan_folder

    return pd.read_csv(report_csv), scan_folder


def get_components(mask_data, voxel_volume, min_size_mm3):
    """
    Label the connected components of a lesion mask, discarding those smaller than min_size_mm3.
    The labels of the kept components are left untouched (i.e. never renumbered), so that they stay
    aligned with the lesion labels used by detect_critical_lesion.get_lesion_stats.
    Output:
        (labeled, labels) where labeled is the labelled array (small components zeroed out)
    """
    labeled, num_components = ndimage.label(mask_data > 0, structure=CONNECTIVITY_STRUCTURE)

    labels = []
    for label in range(1, num_components + 1):
        component_mask = labeled == label
        if int(np.sum(component_mask)) * voxel_volume < min_size_mm3:
            labeled[component_mask] = 0
            continue
        labels.append(label)

    return labeled, labels


def label_lesions(manual_seg_path, pred_seg_path, min_size_mm3, overlap_ratio):
    """
    Decide, for every manual and every predicted lesion of a scan, whether it is critical.
    A manual lesion is critical when its connected component holds at least one voxel of value 2.
    A predicted lesion is critical when at least `overlap_ratio` of its voxels fall inside a
    critical manual lesion, which is the same rule as evaluate_segmentation.py.
    Input:
        manual_seg_path: path to the manual lesion segmentation (1 = non-critical, 2 = critical)
        pred_seg_path: path to the predicted lesion segmentation
        min_size_mm3: minimum lesion size (mm3) to keep a connected component
        overlap_ratio: minimum fraction of a predicted lesion that must fall inside a critical
            manual lesion for it to be labelled critical
    Output:
        (manual_labels, pred_labels), two dicts lesion_label -> label information
    """
    manual_nii = nib.load(manual_seg_path)
    manual_data = manual_nii.get_fdata()
    pred_data = nib.load(pred_seg_path).get_fdata()
    voxel_volume = float(np.prod(manual_nii.header.get_zooms()[:3]))

    labeled_manual, manual_lesion_labels = get_components(manual_data, voxel_volume, min_size_mm3)
    labeled_pred, pred_lesion_labels = get_components(pred_data, voxel_volume, min_size_mm3)
    critical_mask = manual_data == CRITICAL_VALUE

    manual_labels = {}
    for label in manual_lesion_labels:
        values = manual_data[labeled_manual == label]
        n_critical = int(np.sum(values == CRITICAL_VALUE))
        manual_labels[label] = {
            "is_critical": bool(n_critical > 0),
        }

    pred_labels = {}
    for label in pred_lesion_labels:
        component_mask = labeled_pred == label
        n_voxels = int(np.sum(component_mask))
        overlap_fraction = float(np.sum(np.logical_and(component_mask, critical_mask)) / n_voxels)
        pred_labels[label] = {
            "is_critical": bool(overlap_fraction >= overlap_ratio),
            "overlap_fraction": overlap_fraction,
            # Whether the predicted lesion overlaps any manual lesion at all: the ones that do not
            # are the false positives of the segmentation model
            "matched": bool(np.any(np.logical_and(component_mask, labeled_manual > 0))),
        }

    return manual_labels, pred_labels


def add_manual_labels(df_features, entry, manual_labels):
    """
    Add the ground-truth label of every manual lesion to its feature row: a lesion is critical when
    its connected component contains at least one voxel of value 2 in the manual segmentation.
    """
    df_features = df_features.copy()
    df_features.insert(0, "mask_source", "manual")
    df_features.insert(0, "scan_id", entry["scan_id"])
    df_features.insert(0, "session", entry["session"])
    df_features.insert(0, "subject", entry["subject"])
    df_features["scan_file"] = entry["image"]
    df_features["manual_seg_file"] = entry["label"]

    df_features["critical_lesion_label"] = df_features["lesion_label"].map(
        lambda label: int(manual_labels.get(label, {}).get("is_critical", False))
    )
    df_features["critical_voxel_fraction"] = df_features["lesion_label"].map(
        lambda label: manual_labels.get(label, {}).get("critical_voxel_fraction", np.nan)
    )
    return df_features


def add_pred_labels(df_features, entry, pred_seg_path, pred_labels):
    """
    Add the inherited label of every predicted lesion to its feature row: a predicted lesion is
    critical when at least overlap_ratio of its voxels fall inside a critical manual lesion.
    Predicted lesions overlapping no manual lesion at all are the false positives of the
    segmentation model; they are kept (flagged with matched=False and labelled non-critical) so
    that the end-to-end performance of the pipeline can also be measured.
    """
    df_features = df_features.copy()
    df_features.insert(0, "mask_source", "pred")
    df_features.insert(0, "scan_id", entry["scan_id"])
    df_features.insert(0, "session", entry["session"])
    df_features.insert(0, "subject", entry["subject"])
    df_features["scan_file"] = entry["image"]
    df_features["manual_seg_file"] = entry["label"]
    df_features["pred_seg_file"] = pred_seg_path

    df_features["critical_lesion_label"] = df_features["lesion_label"].map(
        lambda label: int(pred_labels.get(label, {}).get("is_critical", False))
    )
    df_features["overlap_fraction"] = df_features["lesion_label"].map(
        lambda label: pred_labels.get(label, {}).get("overlap_fraction", np.nan)
    )
    df_features["matched"] = df_features["lesion_label"].map(
        lambda label: bool(pred_labels.get(label, {}).get("matched", False))
    )
    return df_features


def main():
    args = parse_args()
    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    masks_folder = os.path.join(output_folder, "binarized_manual_masks")
    os.makedirs(masks_folder, exist_ok=True)

    entries = load_include(args.include)

    participants_df = pd.read_csv(os.path.join(os.path.abspath(args.dataset_path), "participants.tsv"), sep="\t")

    df_manual_features = pd.DataFrame()
    df_pred_features = pd.DataFrame()
    failed_scans = []

    for entry in tqdm(entries, desc="Extracting features"):
        _, pred_seg_path = get_pred_paths(entry, args.pred_folder)
        try:
            sex, date_birth = get_subject_demographics(participants_df, entry["subject"])

            # Decide once per scan which lesions are critical, on both mask sources
            manual_labels, pred_labels = label_lesions(
                entry["label"], pred_seg_path, args.min_lesion_size, args.overlap_ratio
            )

            manual_scan_folder, pred_scan_folder = None, None

            if args.mask_source in ("manual", "both"):
                binarized_manual = binarize_manual_mask(
                    entry["label"],
                    os.path.join(masks_folder, entry["scan_id"] + "_label-lesion_seg_bin.nii.gz"),
                )
                df_scan, manual_scan_folder = run_feature_pipeline(
                    entry, binarized_manual, "manual", sex, date_birth,
                    output_folder, args.path_hc_data, args.min_lesion_size,
                    age_normalization=not args.no_age_normalization,
                )
                if df_scan is not None:
                    df_manual_features = pd.concat(
                        [df_manual_features, add_manual_labels(df_scan, entry, manual_labels)], ignore_index=True
                    )
                else:
                    print(f"No manual lesion above {args.min_lesion_size} mm3 in {entry['scan_id']}")

            if args.mask_source in ("pred", "both"):
                pred_scan_folder = os.path.join(output_folder, "pred", entry["scan_id"])
                # Reuse the image-level products of the manual run (above all the registration)
                if manual_scan_folder is not None and os.path.isdir(manual_scan_folder):
                    share_image_level_outputs(manual_scan_folder, pred_scan_folder, entry["scan_id"])
                df_scan, pred_scan_folder = run_feature_pipeline(
                    entry, pred_seg_path, "pred", sex, date_birth,
                    output_folder, args.path_hc_data, args.min_lesion_size,
                    age_normalization=not args.no_age_normalization,
                )
                if df_scan is not None:
                    df_pred_features = pd.concat(
                        [df_pred_features, add_pred_labels(df_scan, entry, pred_seg_path, pred_labels)], ignore_index=True
                    )
                else:
                    print(f"No predicted lesion above {args.min_lesion_size} mm3 in {entry['scan_id']}")

        except Exception as error:
            print(f"Error extracting the features of {entry['scan_id']}: {error}")
            traceback.print_exc()
            failed_scans.append({
                "subject": entry["subject"],
                "session": entry["session"],
                "scan_id": entry["scan_id"],
                "scan_file": entry["image"],
                "error": str(error),
            })

        break

    # Save the feature tables
    if not df_manual_features.empty:
        manual_csv = os.path.join(output_folder, "features_manual.csv")
        df_manual_features.to_csv(manual_csv, index=False)
        n_critical = int(df_manual_features["critical_lesion_label"].sum())
        print(
            f"{len(df_manual_features)} manual lesions ({n_critical} critical) from "
            f"{df_manual_features['subject'].nunique()} subjects saved to: {manual_csv}"
        )
    if not df_pred_features.empty:
        pred_csv = os.path.join(output_folder, "features_pred.csv")
        df_pred_features.to_csv(pred_csv, index=False)
        n_matched = int(df_pred_features["matched"].sum())
        n_critical = int(df_pred_features["critical_lesion_label"].sum())
        print(
            f"{len(df_pred_features)} predicted lesions ({n_matched} matched to a manual lesion, "
            f"{n_critical} critical) from {df_pred_features['subject'].nunique()} subjects saved to: {pred_csv}"
        )

    if failed_scans:
        failed_csv = os.path.join(output_folder, "failed_scans.csv")
        pd.DataFrame(failed_scans).to_csv(failed_csv, index=False)
        print(f"{len(failed_scans)} scan(s) failed. See: {failed_csv}")


if __name__ == "__main__":
    main()
