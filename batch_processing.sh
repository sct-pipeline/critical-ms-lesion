#!/bin/bash
# NB: This file is best displayed with 120 col.
# 
# Processing of dataset of MS patients. Data organized with BIDS. This script is designed to be run 
# across multiple subjects in parallel using 'sct_run_batch', but it can also be used to run processing on a single 
# subject. The input data is assumed to be in BIDS format.
# 
# IMPORTANT: This script MUST be run from the root folder of the repository, because it relies on Python scripts located 
#  in the root folder.
#
# Usage:
#   ./batch_processing.sh <SUBJECT>
#
# Example:
#   ./batch_processing.sh sub-03
#
# Author: Julien Cohen-Adad

# Parameters
# TODO: remove following parameter if not needed
# vertebral_levels="1:3"  # Vertebral levels to extract metrics from. "2:12" means from C2 to T5 (included)
# List of tracts to extract:
# TODO: deal with that later
tracts=(
  "32,33"\
  "51"\
  "52"\
  "53"\
  "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29"\
  "30,31"\
  "34,35"\
  "4,5"\
  "4,5,8,9,10,11,16,17,18,19,20,21,22,23,24,25,26,27"\
  "0,1,2,3,6,7,12,13,14,15"\
)
# The following global variables are retrieved from the caller sct_run_batch but could be overwritten by 
# uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Save script path
PATH_SCRIPT=$PWD


# CONVENIENCE FUNCTIONS
# =====================================================================================================================

label_vertebrae_if_does_not_exist() {
  # This function checks if a disc manual label file already exists, then:
  #   - If it does, copy it locally.
  #   - If it doesn't, perform automatic labeling.
  # This allows you to add manual labels on a subject-by-subject basis without disrupting the pipeline.

  local file="${1}"
  local file_seg="${2}"
  # Update global variable with segmentation file name
  # TODO: modify the name below to _label_disc when Nathan's TotalSpineSeg is ready (the current method outputs a file called FILE_seg_labeled_discs.nii.gz)
  FILELABEL="${file}"_seg_labeled_discs
  FILELABELMANUAL="${PATH_DATA}"/derivatives/labels/"${SUBJECT}"/anat/"${FILELABEL}".nii.gz
  echo "Looking for manual label: ${FILELABELMANUAL}"
  if [[ -e "${FILELABELMANUAL}" ]]; then
    echo "Found! Copying manual labels."
    rsync -avzh "${FILELABELMANUAL}" "${FILELABEL}".nii.gz
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    # TODO: replace with Nathan's TotalSpineSeg
    sct_label_vertebrae -i "${file}".nii.gz -s "${file_seg}".nii.gz -c t2 -qc "${PATH_QC}" -qc-subject "${SUBJECT}"
  fi
  # Generate labeled segmentation based on disc labels
  sct_label_vertebrae -i "${file}".nii.gz -s "${file_seg}".nii.gz -discfile "${FILELABEL}".nii.gz -c t2 -qc "${PATH_QC}" -qc-subject "${SUBJECT}"
  FILELABELVERTEBRAE="${file}"_seg_labeled
}

segment_sc_if_does_not_exist() {
  # This function checks if a manual spinal cord segmentation file already exists, then:
  #   - If it does, copy it locally.
  #   - If it doesn't, perform automatic spinal cord segmentation.
  # This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.

  local file="${1}"
  # Update global variable with segmentation file name
  FILESEG="${file}"_seg
  FILESEGMANUAL="${PATH_DATA}"/derivatives/labels/"${SUBJECT}"/anat/"${FILESEG}".nii.gz
  echo
  echo "Looking for manual segmentation: ${FILESEGMANUAL}"
  if [[ -e "${FILESEGMANUAL}" ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh "${FILESEGMANUAL}" "${FILESEG}".nii.gz
    sct_qc -i "${file}".nii.gz -s "${FILESEG}".nii.gz -p sct_deepseg_sc -qc "${PATH_QC}" -qc-subject "${SUBJECT}"
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg -i "${file}".nii.gz -task seg_sc_contrast_agnostic -thr 0 -qc "${PATH_QC}" -qc-subject "${SUBJECT}"
  fi
}

segment_lesion_if_does_not_exist() {
  # This function checks if a manual MS lesion segmentation file already exists, then:
  #   - If it does, copy it locally.
  #   - If it doesn't, perform automatic MS lesion segmentation.
  # This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.

  local file="${1}"
  local file_seg="${2}"
  # Update global variable with segmentation file name
  FILELESION="${file}"_lesion-seg
  FILELESIONMANUAL="${PATH_DATA}"/derivatives/labels/"${SUBJECT}"/anat/"${FILELESION}".nii.gz
  echo
  echo "Looking for manual segmentations: ${FILESEGMANUAL}"
  if [[ -e "${FILELESIONMANUAL}" ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh "${FILELESIONMANUAL}" "${FILELESION}".nii.gz
    sct_qc -i "${file}".nii.gz -p sct_deepseg_lesion -d "${FILELESION}".nii.gz -s "${file_seg}".nii.gz -qc "${PATH_QC}" -plane sagittal -qc-subject "${SUBJECT}"
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg -i "${file}".nii.gz -task seg_ms_lesion -o "${FILELESION}".nii.gz
    sct_qc -i "${file}".nii.gz -p sct_deepseg_lesion -d "${FILELESION}".nii.gz -s "${file_seg}".nii.gz -qc "${PATH_QC}" -plane sagittal -qc-subject "${SUBJECT}"
  fi
  # Compute lesion analysis
  sct_analyze_lesion -m "${FILELESION}".nii.gz -s "${file_seg}".nii.gz -i "${file}".nii.gz
}

# SCRIPT STARTS HERE
# =====================================================================================================================

# Retrieve input params
SUBJECT="${1}"

# get starting time:
start="$(date +%s)"

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd "${PATH_DATA_PROCESSED}"
# Copy source images
rsync -avzh "${PATH_DATA}"/"${SUBJECT}" .


# T2w
# =====================================================================================================================
cd "${SUBJECT}"/anat/
file_t2="${SUBJECT}"_acq-sagCervCube_T2w
echo "👉 Processing: ${file_t2}"
# Segment spinal cord (only if it does not exist)
# Note: we output the soft segmentation for better CSA precision
segment_sc_if_does_not_exist "${file_t2}"
file_t2_seg="${FILESEG}"
# Create labels in the cord at mid-vertebral levels
label_vertebrae_if_does_not_exist "${file_t2}" "${file_t2_seg}"
file_label_vert="${FILELABELVERTEBRAE}"
# Compute average CSA as defined by variable 'vertebral_levels'
sct_process_segmentation -i "${file_t2_seg}".nii.gz -vertfile "${file_label_vert}".nii.gz \
                         -perslice 1 -o "${PATH_RESULTS}"/"${SUBJECT}"_CSA.csv -append 1 -qc "${PATH_QC}"
# Segment lesions (only if it does not exist)
segment_lesion_if_does_not_exist "${file_t2}" "${file_t2_seg}"
file_lesion="${FILELESION}"

# Go back to parent folder
cd ..


# Verify presence of output files and write log file if error
# ======================================================================================================================
FILES_TO_CHECK=(
  "anat/${file_t2_seg}".nii.gz
)
for file in "${FILES_TO_CHECK[@]}"; do
  if [ ! -e "${file}" ]; then
    echo "${SUBJECT}/${file} does not exist" >> "${PATH_LOG}/error.log"
  fi
done

# Display useful info for the log
end="$(date +%s)"
runtime="$((end-start))"
echo
echo "~~~"
echo "SCT version: $(sct_version)"
echo "Ran on:      $(uname -nsr)"
echo "Duration:    $((runtime / 3600))hrs $(( (runtime / 60) % 60))min $((runtime % 60))sec"
echo "~~~"
