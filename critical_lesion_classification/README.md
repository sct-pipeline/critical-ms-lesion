# Critical lesion classification

Fully automatic classification of critical vs. non-critical spinal cord demyelinating lesions, from
axial cervical MRI.

This is the next step of the critical lesion detection study (see [`detection/`](../detection),
whose feature pipeline is reused here). The previous study segmented lesions automatically and had
them manually labelled critical or non-critical, so segmentation quality was never measured and the
classifier was trained and tested on the same (automatic) mask source. Here, **manual lesion
segmentations are the ground truth** (value 1 = non-critical, value 2 = critical), the SCT
`lesion_ms` model is evaluated as a segmentation and detection tool, and the classifier is trained
on manual-mask features but **tested on predicted-mask features** — the situation at inference time.

## Pipeline

```
include.yml ──► segment_lesions.py ──► evaluate_segmentation.py
     │                   │
     │                   ▼
     └──────────► extract_features.py ──► train_xgboost.py
```

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `create_include_yml.py` | Builds the include yml pairing every scan with its manual lesion segmentation |
| 1 | `segment_lesions.py` | Segments the spinal cord and the lesions of every scan with SCT (`sct_deepseg spinalcord`, `sct_deepseg lesion_ms`) |
| 2–3 | `evaluate_segmentation.py` | Segmentation and detection quality of the predictions vs. the manual segmentations, overall and for critical lesions only |
| 4 | `extract_features.py` | Runs the `detection/` feature pipeline on the manual masks and on the predicted masks, and transfers the critical/non-critical labels onto the predicted lesions |
| 5 | `train_xgboost.py` | Subject-wise nested cross-validation of an XGBoost classifier, with SHAP feature importance |

Shared module: `include_io.py` (include file, naming conventions, metadata columns, and the
segmentation scores: voxel-wise Dice and lesion-wise F1/PPV/sensitivity).

## The include file

```yaml
FILES_SEG:
  - image: <path_to_dataset>/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w.nii.gz
    label: <path_to_dataset>/derivatives/labels/sub-001/ses-20150604/anat/sub-001_ses-20150604_acq-axCerv_T2w_label-lesion_seg.nii.gz
  - image: <path_to_dataset>/sub-002/ses-20180820/anat/sub-002_ses-20180820_acq-axCerv_T2w.nii.gz
    label: <path_to_dataset>/derivatives/labels/sub-002/ses-20180820/anat/sub-002_ses-20180820_acq-axCerv_T2w_label-lesion_seg.nii.gz
```

Paths are absolute. In the manual segmentations, **critical lesions have value 2 and non-critical
lesions have value 1**. Scans must follow the `sub-XXX_ses-YYYYMMDD_...` convention, since the age
of the subject at the time of the scan is derived from the session entity, and `participants.tsv`
must have the `participant_id`, `sex` and `date_of_birth` columns.

## Evaluation design

The classifier is trained on the features of **all** manual lesions (both classes, including the
lesions the segmentation model missed) and evaluated with a subject-wise nested cross-validation:

- outer loop: `GroupKFold(5)` on subjects, which estimates the performance
- inner loop: `GroupKFold(3)` + `BayesSearchCV` (20 iterations, F1), which picks the hyperparameters
- no subject ever appears in both a training and a test fold

Each outer fold is scored on three test sets built from the same held-out subjects:

| Arm | Test lesions | Question it answers |
| --- | --- | --- |
| `pred_matched` | Predicted lesions matched to a manual lesion | How well does the classifier work on automatically segmented lesions? |
| `pred_all` | Every predicted lesion, unmatched ones counted as non-critical | How well does the **whole automatic pipeline** work, false positives included? |
| `manual` | Manual lesions | Reference: what the classifier would achieve on perfect segmentations |

The gap between `manual` and `pred_matched` is the cost of replacing manual masks by automatic
ones; the gap between `pred_matched` and `pred_all` is the cost of the model's false positives.

Missing values are left as such and handled natively by XGBoost — a feature can legitimately be
undefined (e.g. the standard deviation of a measure across the slices of a single-slice lesion),
and that missingness is informative. Infinite values, produced by the ratio-based asymmetry
measures when a quadrant area is zero, are turned into missing values.

## Usage

```bash
# 0. Build the include file (skip if you already have one)
python create_include_yml.py -d $DATASET -o include.yml

# 1. Segment the spinal cord and the lesions
python segment_lesions.py -i include.yml -o $OUT/predictions

# 2-3. Evaluate the segmentation and the detection, overall and on critical lesions
python evaluate_segmentation.py -i include.yml -p $OUT/predictions -o $OUT/segmentation_eval

# 4. Extract the features from the manual and from the predicted masks
python extract_features.py -d $DATASET -i include.yml -p $OUT/predictions \
    --path-hc-data $SPINE_GENERIC -o $OUT/features

# 5. Train and analyse the classifier
python train_xgboost.py -f $OUT/features -o $OUT/model
```

`extract_features.py` is by far the longest step (SC segmentation, vertebral labelling,
registration to the PAM50 template and atlas warping per scan). Everything that only depends on the
image is computed once on the manual run and symlinked into the predicted run, so only the
lesion-dependent steps run twice. Every step skips the scans it has already processed, so an
interrupted run can simply be relaunched.

## Requirements

SCT ≥ 7.2 (`sct_deepseg`, `sct_process_segmentation`, `sct_register_to_template`,
`sct_analyze_lesion`), and `xgboost`, `scikit-learn`, `scikit-optimize`, `shap`, `nibabel`,
`pandas`, `seaborn`, `prettytable`, `loguru`, `pyyaml`, `tqdm`.
