"""
Steps 2 and 3 of the critical lesion classification study: evaluate the lesion segmentation
predicted by the SCT lesion_ms model against the manual lesion segmentation.

Two evaluations are reported:
    1. All lesions, critical and non-critical together: the manual segmentation is binarized and
       compared to the prediction with the scores of include_io.py, per scan: voxel-wise Dice, and
       lesion-wise F1-score, PPV and sensitivity.
    2. Critical lesions only: the manual segmentation is restricted to its critical voxels (value
       2). For each connected component of that mask, the predicted components of which at least
       --overlap-ratio of the voxels fall inside the lesion are kept and merged; the lesion counts
       as detected when the IoU between it and those predicted components is above --overlap-ratio,
       and the Dice of that pair is reported.

Outputs, in the output folder:
    scanwise_metrics.csv          one row per scan (evaluation 1)
    critical_lesions.csv          one row per critical lesion (evaluation 2)
    missed_critical_lesions/      one mask per critical lesion that was not detected, holding that
                                  lesion alone, to review why it was missed
    evaluate_segmentation.log     full report

Input:
    -i / --include: path to the include yml file
    -p / --pred-folder: folder holding the predicted segmentations (output of segment_lesions.py)
    -o: path to the output folder where the evaluation results will be saved
    --overlap-ratio: minimum fraction of a predicted component that must fall inside a critical
        lesion for that component to be attributed to it, and minimum IoU for that lesion to count
        as detected (default: 0.1)

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
import prettytable
from scipy import ndimage
from loguru import logger
from tqdm import tqdm

from include_io import (load_include, check_entries_exist, get_pred_paths, CRITICAL_VALUE,
                        dice_score, lesion_wise_tp_fp_fn, lesion_f1_score, lesion_ppv, lesion_sensitivity)


# 26-connectivity, same structure as detection/detect_critical_lesion.py get_lesion_stats
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the predicted lesion segmentations against the manual ones, overall and for critical lesions.")
    parser.add_argument("-i", "--include", type=str, required=True, help="Path to the include yml file.")
    parser.add_argument("-p", "--pred-folder", type=str, required=True, help="Folder holding the predicted segmentations (output of segment_lesions.py).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the evaluation results will be saved.")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Minimum fraction of a predicted component falling inside a critical lesion to attribute it to that lesion, and minimum IoU for that lesion to count as detected (default: 0.1).")
    return parser.parse_args()


def load_mask(mask_path):
    """
    Load a segmentation mask.
    Output:
        (nii, data, voxel_volume_mm3)
    """
    nii = nib.load(mask_path)
    return nii, nii.get_fdata(), float(np.prod(nii.header.get_zooms()[:3]))


def get_overlapping_components(labeled, lesion_mask, overlap_ratio):
    """
    Find the connected components of a labelled array that belong to a given lesion, i.e. those of
    which at least `overlap_ratio` of the voxels fall inside that lesion.
    Input:
        labeled: labelled array of the predicted segmentation
        lesion_mask: boolean array of the lesion to attribute the components to
        overlap_ratio: minimum fraction of a component's voxels that must fall inside lesion_mask
    Output:
        The list of labels of the components attributed to that lesion
    """
    overlapping_labels = [int(label) for label in np.unique(labeled[lesion_mask]) if label > 0]

    kept = []
    for label in overlapping_labels:
        component_mask = labeled == label
        n_overlap = int(np.sum(np.logical_and(component_mask, lesion_mask)))
        if n_overlap / int(np.sum(component_mask)) >= overlap_ratio:
            kept.append(label)
    return kept


def intersection_over_union(mask_a, mask_b):
    """Intersection over union of two binary masks (0 when both are empty)."""
    union = int(np.sum(np.logical_or(mask_a, mask_b)))
    if union == 0:
        return 0.0
    return float(np.sum(np.logical_and(mask_a, mask_b)) / union)


def evaluate_scan(entry, pred_seg_path, overlap_ratio, missed_folder):
    """
    Compare the manual and the predicted lesion segmentations of one scan.
    Input:
        entry: one entry of the include file (output of load_include)
        pred_seg_path: path to the predicted lesion segmentation
        overlap_ratio: minimum fraction of a predicted component falling inside a critical lesion to
            attribute it to that lesion, and minimum IoU for that lesion to count as detected
        missed_folder: folder where every critical lesion that was not detected is saved, alone in
            a mask of its own
    Output:
        (scan_row, critical_rows): the per-scan scores of evaluation 1, and one row per critical
        lesion for evaluation 2
    """
    manual_nii, manual_data, voxel_volume = load_mask(entry["label"])
    _, pred_data, _ = load_mask(pred_seg_path)
    if manual_data.shape != pred_data.shape:
        raise ValueError(
            f"Manual and predicted segmentations do not have the same shape: "
            f"{manual_data.shape} ({entry['label']}) vs {pred_data.shape} ({pred_seg_path})"
        )

    manual_binary = (manual_data > 0).astype(np.uint8)
    pred_binary = (pred_data > 0).astype(np.uint8)
    labeled_pred, n_pred_lesions = ndimage.label(pred_binary, structure=CONNECTIVITY_STRUCTURE)
    _, n_manual_lesions = ndimage.label(manual_binary, structure=CONNECTIVITY_STRUCTURE)

    # ------------------------------------------- evaluation 1: critical and non-critical together
    # Lesions found, missed and predicted spuriously, to be summed over the cohort
    true_positives, false_positives, false_negatives = lesion_wise_tp_fp_fn(manual_binary, pred_binary, overlap_ratio)

    scan_row = {
        "subject": entry["subject"],
        "session": entry["session"],
        "scan_id": entry["scan_id"],
        "contrast": entry["contrast"],
        "scan_file": entry["image"],
        "manual_seg_file": entry["label"],
        "pred_seg_file": pred_seg_path,
        "dice": dice_score(pred_binary, manual_binary),
        "lesion_f1": lesion_f1_score(manual_binary, pred_binary, overlap_ratio),
        "lesion_ppv": lesion_ppv(manual_binary, pred_binary, overlap_ratio),
        "lesion_sensitivity": lesion_sensitivity(manual_binary, pred_binary, overlap_ratio),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "manual_volume_mm3": float(np.sum(manual_binary) * voxel_volume),
        "pred_volume_mm3": float(np.sum(pred_binary) * voxel_volume),
        "n_manual_lesions": int(n_manual_lesions),
        "n_pred_lesions": int(n_pred_lesions),
    }

    # ------------------------------------------------------------ evaluation 2: critical lesions
    # Keep only the critical voxels of the manual segmentation, and label its lesions
    critical_mask = manual_data == CRITICAL_VALUE
    labeled_critical, n_critical_lesions = ndimage.label(critical_mask, structure=CONNECTIVITY_STRUCTURE)
    scan_row["n_critical_lesions"] = int(n_critical_lesions)

    critical_rows = []
    for label in range(1, n_critical_lesions + 1):
        lesion_mask = labeled_critical == label
        # Predicted components of which at least overlap_ratio of the voxels fall inside the lesion
        kept_labels = get_overlapping_components(labeled_pred, lesion_mask, overlap_ratio)
        pred_lesion_mask = np.isin(labeled_pred, kept_labels)

        lesion_iou = intersection_over_union(lesion_mask, pred_lesion_mask)
        detected = lesion_iou > overlap_ratio

        # A lesion the model missed is saved alone in a mask, to review it against the scan
        missed_mask_file = ""
        if not detected:
            missed_mask_file = os.path.join(missed_folder, f"{entry['scan_id']}_critical-lesion-{label}.nii.gz")
            nib.save(nib.Nifti1Image(lesion_mask.astype(np.uint8), manual_nii.affine, manual_nii.header), missed_mask_file)

        critical_rows.append({
            "subject": entry["subject"],
            "session": entry["session"],
            "scan_id": entry["scan_id"],
            "lesion_label": label,
            "volume_mm3": float(np.sum(lesion_mask) * voxel_volume),
            "n_pred_components": len(kept_labels),
            "pred_volume_mm3": float(np.sum(pred_lesion_mask) * voxel_volume),
            "iou": lesion_iou,
            "detected": detected,
            # Cast to uint8 here too: both masks are boolean, and they are added inside dice_score
            "dice": dice_score(pred_lesion_mask.astype(np.uint8), lesion_mask.astype(np.uint8)),
            "missed_mask_file": missed_mask_file,
        })

    return scan_row, critical_rows


def format_mean_std(values):
    """Format a series of values as 'mean ± std (median)'."""
    values = pd.Series(values).dropna()
    if values.empty:
        return "n/a"
    return f"{values.mean():.3f} ± {values.std():.3f} (median {values.median():.3f})"


def main():
    args = parse_args()
    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    missed_folder = os.path.join(output_folder, "missed_critical_lesions")
    os.makedirs(missed_folder, exist_ok=True)

    # Initialize a logger in the output folder
    path_logger = os.path.join(output_folder, "evaluate_segmentation.log")
    if os.path.exists(path_logger):
        os.remove(path_logger)
    logger.add(path_logger, level="INFO")

    # Load the include file and convert to entries
    entries = load_include(args.include)

    scan_rows, critical_rows, missing_predictions = [], [], []
    for entry in tqdm(entries, desc="Evaluating"):
        _, pred_seg_path = get_pred_paths(entry, args.pred_folder)
        if not os.path.exists(pred_seg_path):
            missing_predictions.append(entry["scan_id"])
            continue
        scan_row, scan_critical_rows = evaluate_scan(entry, pred_seg_path, args.overlap_ratio, missed_folder)
        scan_rows.append(scan_row)
        critical_rows.extend(scan_critical_rows)

    if missing_predictions:
        logger.warning(f"{len(missing_predictions)} scan(s) have no predicted segmentation and were skipped: {missing_predictions}")
    if not scan_rows:
        raise RuntimeError("No scan could be evaluated: check that segment_lesions.py ran on this include file.")

    df_scans = pd.DataFrame(scan_rows)
    df_critical = pd.DataFrame(critical_rows)

    logger.info(f"Scans evaluated: {len(df_scans)} from {df_scans['subject'].nunique()} subjects")
    logger.info(
        f"Manual lesions: {int(df_scans['n_manual_lesions'].sum())} "
        f"({int(df_scans['n_critical_lesions'].sum())} critical) | "
        f"Predicted lesions: {int(df_scans['n_pred_lesions'].sum())}"
    )
    logger.info(f"Overlap ratio: {args.overlap_ratio}")

    # ------------------------------------------- evaluation 1: critical and non-critical together
    segmentation_table = prettytable.PrettyTable(["Score", "Value (per scan)"])
    for column, label in [("dice", "Dice"), ("lesion_f1", "Lesion-wise F1"),
                          ("lesion_ppv", "Lesion-wise PPV"), ("lesion_sensitivity", "Lesion-wise sensitivity")]:
        segmentation_table.add_row([label, format_mean_std(df_scans[column])])
    segmentation_table.add_row(["Total manual volume per scan (mm3)", format_mean_std(df_scans["manual_volume_mm3"])])
    segmentation_table.add_row(["Total predicted volume per scan (mm3)", format_mean_std(df_scans["pred_volume_mm3"])])
    logger.info(f"All lesions (critical and non-critical), averaged over the {len(df_scans)} scans:\n{segmentation_table}")

    # Same detection, but counted over the whole cohort, where every lesion weighs the same (the
    # per-scan scores above give a scan with one lesion as much weight as a scan with four)
    n_tp, n_fp, n_fn = (int(df_scans[column].sum()) for column in ("tp", "fp", "fn"))
    f1_denominator = n_tp + (n_fp + n_fn) / 2
    pooled_table = prettytable.PrettyTable(["Metric", "Value"])
    pooled_table.add_row(["Lesions found (TP)", n_tp])
    pooled_table.add_row(["Lesions missed (FN)", n_fn])
    pooled_table.add_row(["Spurious lesions (FP)", n_fp])
    pooled_table.add_row(["Sensitivity", f"{n_tp / (n_tp + n_fn):.3f}" if (n_tp + n_fn) else "n/a"])
    pooled_table.add_row(["PPV", f"{n_tp / (n_tp + n_fp):.3f}" if (n_tp + n_fp) else "n/a"])
    pooled_table.add_row(["F1", f"{n_tp / f1_denominator:.3f}" if f1_denominator else "n/a"])
    logger.info(f"All lesions, pooled over the {len(df_scans)} scans:\n{pooled_table}")

    # ------------------------------------------------------------ evaluation 2: critical lesions
    if df_critical.empty:
        logger.warning("No critical lesion (value 2) found in the manual segmentations.")
    else:
        n_detected = int(df_critical["detected"].sum())
        critical_table = prettytable.PrettyTable(["Metric", "Value"])
        critical_table.add_row(["Critical lesions", len(df_critical)])
        critical_table.add_row(["Detected (IoU > %.0f%%)" % (args.overlap_ratio * 100), f"{n_detected} ({n_detected / len(df_critical):.1%})"])
        critical_table.add_row(["IoU, all lesions", format_mean_std(df_critical["iou"])])
        critical_table.add_row(["Dice, all lesions", format_mean_std(df_critical["dice"])])
        critical_table.add_row(["Dice, detected lesions", format_mean_std(df_critical.loc[df_critical["detected"], "dice"])])
        critical_table.add_row(["Manual volume per lesion (mm3)", format_mean_std(df_critical["volume_mm3"])])
        critical_table.add_row(["Predicted volume per lesion (mm3)", format_mean_std(df_critical["pred_volume_mm3"])])
        logger.info(f"Critical lesions only:\n{critical_table}")
        logger.info(f"Masks of the {len(df_critical) - n_detected} missed critical lesions saved to: {missed_folder}")

    df_scans.to_csv(os.path.join(output_folder, "scanwise_metrics.csv"), index=False)
    df_critical.to_csv(os.path.join(output_folder, "critical_lesions.csv"), index=False)
    logger.info(f"Evaluation complete. Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
