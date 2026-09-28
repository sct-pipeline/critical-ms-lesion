# Critical lesion classification

Automatic classification of critical vs. non-critical spinal cord demyelinating lesions on axial
cervical MRI. The manual segmentations (1 = non-critical, 2 = critical) are the ground truth: the
SCT `lesion_ms` model is first scored against them, then feeds the classifier. The feature pipeline
of [`detection/`](../detection) is reused.

## Usage

```bash
# 0. Build the include file listing every scan and its manual segmentation
python create_include_yml.py -d $DATASET -o include.yml

# 1. Segment the cord, the discs and the lesions of every scan
python segment_lesions.py -i include.yml -o $OUT/predictions

# 2. Score the predicted segmentations, overall and on critical lesions
python evaluate_segmentation.py -i include.yml -p $OUT/predictions -o $OUT/segmentation_eval

# 3. Extract the features of every lesion, from the manual and from the predicted masks
python extract_features.py -d $DATASET -i include.yml -p $OUT/predictions \
    --path-hc-data $SPINE_GENERIC -o $OUT/features

# 4. Train and analyse the classifier
python train_xgboost.py -f $OUT/features -o $OUT/model
```

Steps 1 and 3 skip the scans they have already processed, so an interrupted run can be relaunched.
Step 3 is by far the longest, because of the registration to PAM50 of every scan. `include_io.py`
holds the shared conventions, the include file reader and the segmentation scores.