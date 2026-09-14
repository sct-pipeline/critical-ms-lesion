"""
Shared conventions, I/O helpers and segmentation scores for the critical lesion classification study.

The study is driven by an include yml file which lists every scan of the study together with
its manual lesion segmentation. In the manual segmentation, critical lesions have value 2 and
non-critical lesions have value 1. Paths are absolute:

    FILES_SEG:
      - image: <path_to_dataset>/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w.nii.gz
        label: <path_to_dataset>/derivatives/labels/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w_label-lesion_seg.nii.gz
      - image: <path_to_dataset>/sub-002/ses-20180820/anat/sub-002_ses-20180820_acq-axCerv_T2w.nii.gz
        label: <path_to_dataset>/derivatives/labels/sub-002/ses-20180820/anat/sub-002_ses-20180820_acq-axCerv_T2w_label-lesion_seg.nii.gz

This module holds three things:
    - the conventions shared by every script of the study (segmentation values, minimum lesion
      size, columns of the feature csvs that are metadata rather than features)
    - the readers of the include file and the naming convention of the predicted segmentations
    - the segmentation scores used to evaluate a predicted lesion segmentation against a manual
      one: voxel-wise Dice, and the lesion-wise F1-score, PPV and sensitivity computed by 3D
      connected-component analysis (adapted from the ATLAS v2.0 grand challenge)

Note that the lesion-wise scores below label connected components with the default
scipy.ndimage.label structure (6-connectivity), whereas evaluate_segmentation.py and
extract_features.py use 26-connectivity to stay aligned with the lesion labels of
detection/detect_critical_lesion.py. The two can therefore count lesions slightly differently.

Author: Pierre-Louis Benveniste
"""
import os
import re
import yaml
from scipy import ndimage
import numpy as np


# Values used in the manual lesion segmentations
NON_CRITICAL_VALUE = 1
CRITICAL_VALUE = 2

# Minimum lesion size (mm3) kept everywhere in the study, to stay consistent with
# detect_critical_lesion.get_lesion_stats (boundary from 10.1016/j.nicl.2018.01.011)
MIN_LESION_SIZE_MM3 = 15.0

# Columns of the feature csvs that describe a lesion but must never be fed to the classifier
METADATA_COLUMNS = [
    "subject",
    "session",
    "scan_id",
    "scan_file",
    "manual_seg_file",
    "pred_seg_file",
    "mask_source",
    "lesion_label",
    "critical_lesion_label",
    "critical_voxel_fraction",
    "overlap_fraction",
    "matched",
]


def parse_subject_session(image_path):
    """
    Extract the BIDS subject and session entities of a scan, from its filename first and from the
    rest of its path otherwise.
    Output:
        (subject, session), session being None if the dataset is not longitudinal
    """
    parts = os.path.normpath(image_path).split(os.sep)
    filename = parts[-1]

    subject_match = re.search(r"(sub-[A-Za-z0-9]+)", filename)
    session_match = re.search(r"(ses-[A-Za-z0-9]+)", filename)

    subject = subject_match.group(1) if subject_match else None
    session = session_match.group(1) if session_match else None

    # Fall back on the folder names when the entities are not in the filename
    if subject is None:
        subject = next((part for part in parts if part.startswith("sub-")), None)
    if session is None:
        session = next((part for part in parts if part.startswith("ses-")), None)

    if subject is None:
        raise ValueError(f"Could not find a sub-* entity for scan {image_path}")

    return subject, session


def get_scan_id(image_path):
    """Return the filename of a scan without its .nii.gz extension (used to name output folders)."""
    return os.path.basename(image_path).replace(".nii.gz", "")


def load_include(include_yml):
    """
    Load the include yml file listing the scans of the study and their manual lesion segmentation.
    Input:
        include_yml: path to the include yml file, whose FILES_SEG entries hold the absolute path
            of every scan ("image") and of its manual lesion segmentation ("label")
    Output:
        A list of dicts, one per scan, with keys:
            image: absolute path to the MRI scan
            label: absolute path to the manual lesion segmentation (1 = non-critical, 2 = critical)
            image_rel: sub-XXX/ses-YYYYMMDD/anat folder of the scan, used to mirror the BIDS tree
                in the output folders
            scan_id: filename of the scan without its extension
            subject / session: BIDS entities of the scan
            contrast: last entity of the filename (e.g. T2w, T2star)
    """
    with open(include_yml, "r") as f:
        include_dict = yaml.safe_load(f)

    if not isinstance(include_dict, dict) or "FILES_SEG" not in include_dict:
        raise KeyError(f"{include_yml} must contain a top-level FILES_SEG key")

    entries = []
    for item in include_dict["FILES_SEG"]:
        if not isinstance(item, dict) or "image" not in item or "label" not in item:
            raise ValueError(
                f"Every FILES_SEG entry must be a dict with an 'image' and a 'label' key, got: {item}"
            )
        image = item["image"]
        label = item["label"]
        subject, session = parse_subject_session(image)
        entries.append({
            "image": image,
            "label": label,
            "image_rel": os.path.join(*os.path.dirname(image).split(os.sep)[-3:]),
            "scan_id": get_scan_id(image),
            "subject": subject,
            "session": session,
            "contrast": get_scan_id(image).split("_")[-1]
        })

    # Guard against duplicated scans, which would silently duplicate lesions in the analyses
    scan_ids = [entry["scan_id"] for entry in entries]
    duplicates = sorted({scan_id for scan_id in scan_ids if scan_ids.count(scan_id) > 1})
    if duplicates:
        raise ValueError(f"The include file lists the same scan several times: {duplicates}")

    return entries


def check_entries_exist(entries):
    """
    Check that every scan and manual segmentation listed in the include file exists on disk.
    Output:
        The list of missing files (empty if everything is there)
    """
    missing = []
    for entry in entries:
        for key in ("image", "label"):
            if not os.path.exists(entry[key]):
                missing.append(entry[key])
    return missing


def get_pred_paths(entry, pred_folder):
    """
    Build the paths of the SC and lesion segmentations predicted for a scan by segment_lesions.py.
    The predicted segmentations mirror the sub-XXX/ses-YYYYMMDD/anat folder of the scan inside the
    prediction folder, using the same naming convention as detection/predict_lesion_seg.py:
        <pred_folder>/sub-001/ses-20150604/anat/<scan_id>_label-SC_seg.nii.gz
        <pred_folder>/sub-001/ses-20150604/anat/<scan_id>_label-lesion_seg.nii.gz
    Input:
        entry: one entry of the include file (output of load_include)
        pred_folder: folder where segment_lesions.py wrote its predictions
    Output:
        (sc_seg_path, lesion_seg_path)
    """
    output_dir = os.path.join(os.path.abspath(pred_folder), entry["image_rel"])
    sc_seg = os.path.join(output_dir, entry["scan_id"] + "_label-SC_seg.nii.gz")
    lesion_seg = os.path.join(output_dir, entry["scan_id"] + "_label-lesion_seg.nii.gz")
    return sc_seg, lesion_seg


# ---------------------------------------------------------------------------------------------
# Segmentation scores: voxel-wise Dice and lesion-wise F1 / PPV / sensitivity, used by
# evaluate_segmentation.py to report the quality of the predicted lesion segmentations.
# ---------------------------------------------------------------------------------------------


def dice_score(prediction, groundtruth, smooth=1.):
    """
    Computes the voxel-wise Dice score between two binary masks. The smoothing term keeps the score
    defined when both masks are empty (in which case it returns 1) and softens it on very small
    lesions, so a per-scan Dice never divides by zero.

    Parameters
    ----------
    prediction : array-like, bool
        3D array of the predicted segmentation. If not boolean, will be converted.
    groundtruth : array-like, bool
        3D array with a shape matching 'prediction'. If not boolean, will be converted.
    smooth : scalar, float
        Optional. Smoothing term added to the numerator and the denominator. Default: 1.

    Returns
    -------
    dice (float): Dice score, between 0 and 1.
    """
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()

    dice = (2 * numer + smooth) / (denor + smooth)
    print(dice)
    return dice


def lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio=0.1):
    """
    Computes the true positives, false positives, and false negatives two masks. Masks are considered true positives
    if there is at least `overlap_ratio` overlap between the truth and the prediction.
    i.e. if overlap_ratio = 0.1, then at least 10% of the lesion voxels should overlap between the truth and 
    the prediction to be considered as true positive.
    Adapted from: https://github.com/npnl/atlas2_grand_challenge/blob/main/isles/scoring.py#L341

    Parameters
    ----------
    truth : array-like, bool
        3D array. If not boolean, will be converted.
    prediction : array-like, bool
        3D array with a shape matching 'truth'. If not boolean, will be converted.
    overlap_ratio : scalar, float
        Optional. Fraction of the voxels of a ground-truth lesion that the prediction must cover
        for that lesion to count as detected. Default: 0.1.

    Returns
    -------
    tp (int): 3D connected-component from the ground-truth image of which at least `overlap_ratio` of the voxels are covered by the prediction image.
    fp (int): 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn (int): 3D connected-component from the ground-truth image of which less than `overlap_ratio` of the voxels are covered by the prediction image.

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    tp, fp, fn = 0, 0, 0

    # For each true lesion, check if at least a threshold overlap_ratio of the lesion voxels overlap with the prediction.
    # This determines true positives and false negatives (unpredicted lesions)
    labeled_ground_truth, num_truth_lesions = ndimage.label(truth.astype(bool))
    for idx_lesion in range(1, num_truth_lesions + 1):
        lesion = labeled_ground_truth == idx_lesion
        num_truth_lesion_voxels = np.sum(lesion)  # Total number of voxels in the GT lesion
        overlapping_voxels = np.sum(lesion * prediction)  # Number of GT voxels that overlap with the prediction
        # Check if at least 10% of the lesion voxels overlap with the prediction
        if overlapping_voxels / num_truth_lesion_voxels >= overlap_ratio:
            tp += 1
        else:
            fn += 1

    # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
    labeled_prediction, num_pred_lesions = ndimage.label(prediction.astype(bool))
    for idx_lesion in range(1, num_pred_lesions+1):
        lesion = labeled_prediction == idx_lesion
        lesion_pred_sum = lesion + truth
        if(np.max(lesion_pred_sum) <= 1):  # No overlap
            fp += 1

    return tp, fp, fn


def lesion_f1_score(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise F1-score between two masks by defining true positive lesions (tp), false positive lesions (fp)
    and false negative lesions (fn) using 3D connected-component-analysis.

    A ground-truth lesion counts as a true positive when at least `overlap_ratio` of its voxels are covered by the prediction.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the ref is empty and prediction is empty --> it's false positive
    elif not np.any(truth) and np.any(prediction):
        return 0.0
    # if both are not empty, it's true positive
    else:
        tp, fp, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        f1_score = empty_value

        # Compute f1_score
        denom = tp + (fp + fn)/2
        if(denom != 0):
            f1_score = tp / denom
        return f1_score


def lesion_ppv(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise positive predictive value (PPV) between two masks
    Returns
    -------
    ppv (float): Lesion-wise positive predictive value as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the predction is not empty and ref is empty, it's false positive
    elif not np.any(truth) and np.any(prediction):
        return 0.0
    # if both are not empty, it's true positive
    else:
        tp, fp, _ = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        ppv = 1.0

        # Compute ppv
        denom = tp + fp
        # denom should ideally not be zero inside this else as it should be caught by the empty checks above
        if(denom != 0):
            ppv = tp / denom
        return ppv


def lesion_sensitivity(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise sensitivity between two masks
    Returns
    -------
    sensitivity (float): Lesion-wise sensitivity as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    # if the predction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:

        tp, _, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        sensitivity = empty_value

        # Compute sensitivity
        denom = tp + fn
        if(denom != 0):
            sensitivity = tp / denom
        return sensitivity