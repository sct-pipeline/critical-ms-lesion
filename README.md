# critical-ms-lesion

Detection of critical MS lesion in the spinal cord.

The pipeline does the following:
- cord segmentation on sag image + axial images
- vertebral labeling on sag image
- bring labeling to axial images
- registration axial images to PAM50
- segment lesion on axial images (and possibly on sagittal-- to be decided)
- compute lesion morphometry
- compute lesion overlap with CST
- compute focal atrophy using normalized CSA recent method

Collaboration between NeuroPoly Lab and Mayo Clinic (Mark Keegan and colleagues).
