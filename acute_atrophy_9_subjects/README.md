# Acute Atrophy

Pipeline for measuring spinal cord cross-sectional area (CSA) and lesion-related atrophy,
in PAM50 template space, across a subject's timepoints.

<img src="https://github.com/user-attachments/assets/d217f8dc-cd41-4570-97e3-c6a390617308" width="800"/>

Example plot of a subject's CSA across 4 timepoints.

## Pipeline

1. **`create_include_json.py`** — scans a BIDS dataset and builds a JSON file listing all
   axial spinal cord scans (cervical/thoracic/lumbar), grouped by subject and session.
2. **`generate_qc_sc_segmentation.py`** *(optional)* — runs SC segmentation on every scan in
   the include json and produces an aggregated QC report, for visually checking segmentation
   quality before running the full pipeline.
3. **`compute_csa_on_include.py`** — the main pipeline. For every scan in the include json it
   segments the spinal cord, vertebrae and MS lesions, then computes per-slice CSA in PAM50
   space (with lesion label/volume annotations). Per-scan results are gathered into a
   per-subject csv (`<subject_id>_csa_with_lesions.csv`) and an all-subjects csv
   (`final_csa_with_lesions.csv`). For each subject it also generates, without and with
   `smooth_window=10` smoothing:
   - a CSA plot (`<subject_id>_csa_plot.png` / `_csa_plot_smooth10.png`)
   - a per-lesion-area AUC csv (`<subject_id>_lesion_auc.csv` / `_lesion_auc_smooth10.csv`)

## Standalone scripts

These are called internally by `compute_csa_on_include.py`, but can also be run on their own:

- **`plot_subject_csa.py`** — plots CSA vs. PAM50 axial slice for a subject, one colored line
  per session (color runs along a viridis scale keyed to session date), with lesion shading
  and vertebral level boundaries. Supports moving-average smoothing (`-s/--smooth_window`).
- **`compute_lesion_auc.py`** — for a subject, finds each continuous lesion area (union of
  lesion extent across all sessions), truncates all sessions to the PAM50 slice range they
  actually share (excluding slices where any session lacks real CSA coverage), and computes
  the AUC of the CSA curve over that range per timepoint, along with the AUC ratio and days
  elapsed since the previous timepoint. Also supports `-s/--smooth_window`.
- **`plot_native_and_pam50_csa.py`** — plots CSA vs. slice for a single scan in both PAM50 and
  native slice space, stacked in one figure, from a pair of `sct_process_segmentation` csvs
  (`--csa_native` / `--csa_pam50`).
