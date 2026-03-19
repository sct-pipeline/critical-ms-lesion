#
# Plot a single subject morphometric metrics (two sessions) together with normative values computed from
# normative database (spine-generic dataset in PAM50 space) per slice and vertebral levels
#
# Example usage:
#       python generate_figures.py
#       -path-HC $SCT_DIR/data/PAM50_normalized_metrics
#       -participant-file $SCT_DIR/data/PAM50_normalized_metrics/participants.tsv
#       -ses1 sub-001_ses-01_T2w_metrics_perslice_PAM50.csv
#       -ses2 sub-001_ses-02_T2w_metrics_perslice_PAM50.csv
#       -single-subject-sex M
#
# Author: Jan Valosek
# 
# Source: https://github.com/valosekj/dcm-brno/blob/main/03_plotting_scripts/02b_generate_figures_PAM50_two_sessions.py

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


METRICS = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(compression_ratio)', 'MEAN(eccentricity)',
           'MEAN(solidity)']

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64'
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 8

AGE_DECADES = ['10-20', '21-30', '31-40', '41-50', '51-60']

COLORS_SEX = {
    'M': 'blue',
    'F': 'red'
    }

SEX_TO_LEGEND = {
    'M': 'males',
    'F': 'females'
    }


def load_normative_data(path_HC, path_participants, min_slice=None, max_slice=None):
    """
    Load normative data from spine-generic dataset in PAM50 space
    :param path_HC:
    :param path_participants:
    :param min_slice:
    :param max_slice:
    :return:
    """
    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file), dtype=METRICS_DTYPE)

            # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
            df_subject['MEAN(compression_ratio)'] = df_subject['MEAN(diameter_AP)'] / df_subject['MEAN(diameter_RL)']

            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)
    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])

    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if path_participants:
        df_participants = pd.read_csv(path_participants, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id"]],
                      on='participant_id')
        # Recode age into age bins by 10 years (decades)
        df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=AGE_DECADES)

    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    # if min_slice and max_slice are provided, filter df to keep only rows with Slice (I->S) between min_slice and max_slice
    if min_slice and max_slice:
        df = df[(df['Slice (I->S)'] >= min_slice) & (df['Slice (I->S)'] <= max_slice)].reset_index(drop=True)

    return df


def load_single_subject_data(path_single_subject):
    """
    Load single subject data
    Input:
        path_single_subject: path to single subject CSV file
    Output:
        df_single_subject: pandas DataFrame with single subject data
        single_subject_min: minimum slice number from the single subject data
        single_subject_max: maximum slice number from the single subject data
    """
    df_single_subject = pd.read_csv(path_single_subject, dtype=METRICS_DTYPE)
    # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
    df_single_subject['MEAN(compression_ratio)'] = df_single_subject['MEAN(diameter_AP)'] / \
                                                   df_single_subject['MEAN(diameter_RL)']
    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df_single_subject['MEAN(solidity)'] = df_single_subject['MEAN(solidity)'] * 100

    # Get the min and max slice number from the single subject data
    single_subject_min = df_single_subject['Slice (I->S)'].min()
    single_subject_max = df_single_subject['Slice (I->S)'].max()

    return df_single_subject, single_subject_min, single_subject_max


def get_vert_indices(df, single_subject=False):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    if not single_subject:
        # Get unique participant IDs
        subjects = df['participant_id'].unique()
        # Get vert levels for one certain subject
        vert = df[df['participant_id'] == subjects[0]]['VertLevel']
    else:
        vert = df['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return subjectID, sessionID


def create_lineplot(df, df_ses1, subID, number_of_subjects, path_out_png, sex=None):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are plotting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with spine-generic normative values
        df_ses1 (pd.dataFrame): dataframe with single subject values from session 1
        subID (str): subject ID
        number_of_subjects (int): number of subjects in spine-generic dataset
        path_out_png (str): path of the output png save
        sex (str): sex to filter spine-generic subjects on; possible options: 'M', 'F'
    """

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):
        # Note: we are plotting slices not levels to avoid averaging across levels
        if sex:
            # Plot spine-generic multi-subject data for a given sex
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df[df['sex'] == sex], errorbar='sd',
                         linewidth=2, color=COLORS_SEX[sex],
                         label=f'spine-generic {SEX_TO_LEGEND[sex]} (N = {number_of_subjects})')
        else:
            # Plot spine-generic multi-subject data
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', linewidth=2, color='black',
                         label=f'spine-generic all subjects (N = {number_of_subjects})')

        # Plot single subject data session 1
        sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_ses1, linewidth=2, color='green',
                     label=f'{subID}')

        ymin, ymax = axs[index].get_ylim()

        # Add legend
        if index == 1:
            axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
        else:
            axs[index].get_legend().remove()

        # Add master title
        plt.suptitle(f'Morphometric measures for {subID} in PAM50 template space',
                     fontweight='bold', fontsize=LABELS_FONT_SIZE, y=0.92)

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('PAM50 Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with labels
            if vert[x] > 19:
                level = 'L' + str(vert[x] - 19)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    plt.savefig(path_out_png, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_out_png)


def create_lineplot_asymetry(df_sub, subID, path_out_png, lesion_statistics):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are plotting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with the subject values
        path_out (str): path to output directory
        lesion_statistics (list of dicts): list of dictionaries containing lesion statistics, where each dictionary has the following keys: 'label', 'size', 'CoM' and 'slices'
    """
    METRICS_ASYMMETRY = ['MEAN(area_quadrant_anterior_left)', "DIFF(area_quadrant_anterior_left-right)", "NORM_DIFF(area_quadrant_anterior_left-right)", "RATIO(area_quadrant_anterior_left/right)",
                         'MEAN(area_quadrant_posterior_left)', "DIFF(area_quadrant_posterior_left-right)", "NORM_DIFF(area_quadrant_posterior_left-right)", "RATIO(area_quadrant_posterior_left/right)",
                         'MEAN(symmetry_dice_RL)', 'MEAN(symmetry_hausdorff_RL)', 'MEAN(symmetry_difference_RL)']
    
    METRIC_TO_AXIS_ASYMETRY = {
    'MEAN(area_quadrant_anterior_left)': 'Anterior Quadrant Area [mm²]',
    'RATIO(area_quadrant_anterior_left/right)': 'Anterior Quadrant Area Ratio (L/R)',
    'DIFF(area_quadrant_anterior_left-right)': 'Anterior Quadrant Area Difference (L-R) [mm²]',
    'NORM_DIFF(area_quadrant_anterior_left-right)': 'Anterior Quadrant Area Normalized Diff (L-R)',
    'MEAN(area_quadrant_posterior_left)': 'Posterior Quadrant Area [mm²]',
    'RATIO(area_quadrant_posterior_left/right)': 'Posterior Quadrant Area Ratio (L/R)',
    'DIFF(area_quadrant_posterior_left-right)': 'Posterior Quadrant Area Difference (L-R) [mm²]',
    'NORM_DIFF(area_quadrant_posterior_left-right)': 'Posterior Quadrant Area Normalized Diff (L-R)',
    'MEAN(symmetry_dice_RL)': 'Symmetry Dice',
    'MEAN(symmetry_hausdorff_RL)': 'Symmetry Hausdorff',
    'MEAN(symmetry_difference_RL)': 'Symmetry Difference',
    }

    fig, axes = plt.subplots(3, 4, figsize=(30, 20))
    axs = axes.ravel()

    # Remove rows with NaN values
    df_sub = df_sub.dropna(subset=METRICS_ASYMMETRY).reset_index(drop=True)
    df_sub = df_sub.dropna(subset=['VertLevel']).reset_index(drop=True)

    # Loop across metrics
    for index, metric in enumerate(METRICS_ASYMMETRY):
        # We color the brackground colons where the lesions are
        colors = ['maroon', 'goldenrod', 'deeppink', 'darkorchid', 'olive']
        for lesion_idx, lesion in enumerate(lesion_statistics):
            lesion_slices = sorted(list(lesion['slices']))
            start, end = lesion_slices[0], lesion_slices[-1]
            # Create the x-range for this specific lesion
            x_range = np.arange(start, end + 1) # +1 to include the last slice
            # Fill the background from y=0 to y=1
            axs[index].fill_between(x_range, 0, 1, color=colors[lesion_idx % len(colors)], alpha=0.2, label=f'Lesion {lesion_idx + 1}', transform=axs[index].get_xaxis_transform())
            
        if metric in ["MEAN(area_quadrant_anterior_left)", "MEAN(area_quadrant_posterior_left)"]:
            # Plot the first metric in blue (corresponding to the left side)
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_sub, linewidth=2, color='blue',
                         label=f'Left') 
            # Now we plot the right side
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric.replace('_left', '_right'),
                         data=df_sub, linewidth=2, color='red', label=f'Right')
        elif metric in ['MEAN(symmetry_dice_RL)', 'MEAN(symmetry_hausdorff_RL)', 'MEAN(symmetry_difference_RL)']:
            # Plot the first metric in purple (corresponding to the right-left symmetry)
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_sub, linewidth=2, color='purple',
                         label=f'Right-Left')
            # Plot the AP symmetry metrics in orange
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric.replace('_RL', '_AP'), data=df_sub, linewidth=2, color='orange',
                         label=f'Anterior-Posterior')
        else:
            # Plot single subject data session 1
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_sub, linewidth=2, color='green',
                        label=f'{subID}')

        ymin, ymax = axs[index].get_ylim()

        # Add legend
        if metric in ["MEAN(area_quadrant_anterior_left)", "MEAN(area_quadrant_posterior_left)", 'MEAN(symmetry_dice_RL)', 'MEAN(symmetry_hausdorff_RL)', 'MEAN(symmetry_difference_RL)']:
            axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
        else:
            axs[index].get_legend().remove()

        # Add master title
        plt.suptitle(f'Asymetry plots for {subID} across axial slices and vertebral levels',
                     fontweight='bold', fontsize=LABELS_FONT_SIZE, y=0.92)

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS_ASYMETRY[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_sub, single_subject=True)
        vert = [int(v) for v in vert]  # Convert vert to integer values to avoid issues with string labels when plotting vertebral levels
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df_sub.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with labels
            if vert[x] > 19:
                level = 'L' + str(vert[x] - 19)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    plt.savefig(path_out_png, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_out_png)


def load_normative_data_asymmetry(path_HC, path_participants, min_slice=None, max_slice=None):
    """
    Load normative data from spine-generic dataset in PAM50 space
    :param path_HC:
    :param path_participants:
    :param min_slice:
    :param max_slice:
    :return:
    """
    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file))

            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)
    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first word before _ in the last part (after slash) of the filename
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[-1].str.split('_').str[0])

    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if path_participants:
        df_participants = pd.read_csv(path_participants, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id", "pathology"]],
                      on='participant_id')
        # Recode age into age bins by 10 years (decades)
        df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=AGE_DECADES)

    # Remove the subjects which are not HC
    df = df[df['pathology'] == 'HC'].reset_index(drop=True)

    # if min_slice and max_slice are provided, filter df to keep only rows with Slice (I->S) between min_slice and max_slice
    if min_slice and max_slice:
        df = df[(df['Slice (I->S)'] >= min_slice) & (df['Slice (I->S)'] <= max_slice)].reset_index(drop=True)

    return df


def create_lineplot_asymetry_with_hc(df_sub, df_hc, subID, path_out_png, lesion_statistics):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are plotting slices not levels to avoid averaging across levels.
    Args:
        df_sub (pd.dataFrame): dataframe with the subject values
        df_hc (pd.dataFrame): dataframe with the healthy control values
        subID (str): subject ID
        path_out_png (str): path to output PNG file
        lesion_statistics (list of dicts): list of dictionaries containing lesion statistics, where each dictionary has the following keys: 'label', 'size', 'CoM' and 'slices'
    """
    fig, axes = plt.subplots(3, 3, figsize=(30, 20))
    axs = axes.ravel()
    METRICS_ASYMMETRY = ['MEAN(symmetry_dice_RL)', 'MEAN(symmetry_hausdorff_RL)', 'MEAN(symmetry_difference_RL)',
                         'MEAN(symmetry_dice_AP)', 'MEAN(symmetry_hausdorff_AP)', 'MEAN(symmetry_difference_AP)',
                         'NORM_DIFF(area_quadrant_anterior_left-right)', 'NORM_DIFF(area_quadrant_posterior_left-right)',
                         'NORM_DIFF(area_left-right)']
    
    METRIC_TO_AXIS_ASYMMETRY = {
    'MEAN(symmetry_dice_RL)': 'Symmetry Dice RL',
    'MEAN(symmetry_hausdorff_RL)': 'Symmetry Hausdorff RL',
    'MEAN(symmetry_difference_RL)': 'Symmetry Difference RL',
    'MEAN(symmetry_dice_AP)': 'Symmetry Dice AP',
    'MEAN(symmetry_hausdorff_AP)': 'Symmetry Hausdorff AP',
    'MEAN(symmetry_difference_AP)': 'Symmetry Difference AP',
    'NORM_DIFF(area_quadrant_anterior_left-right)': 'Anterior Quadrant Area Normalized Diff (L-R)',
    'NORM_DIFF(area_quadrant_posterior_left-right)': 'Posterior Quadrant Area Normalized Diff (L-R)',
    'NORM_DIFF(area_left-right)': 'Left-Right Area Normalized Diff (L-R)'
    }

    # Remove rows with NaN values
    df_sub = df_sub.dropna(subset=METRICS_ASYMMETRY).reset_index(drop=True)
    df_sub = df_sub.dropna(subset=['VertLevel']).reset_index(drop=True)

    number_of_subjects = df_hc['participant_id'].nunique()

    # Loop across metrics
    for index, metric in enumerate(METRICS_ASYMMETRY):

        # We color the brackground colons where the lesions are
        colors = ['maroon', 'goldenrod', 'deeppink', 'darkorchid', 'olive']
        for lesion_idx, lesion in enumerate(lesion_statistics):
            lesion_slices = sorted(list(lesion['slices_pam50']))
            start, end = lesion_slices[0], lesion_slices[-1]
            # Create the x-range for this specific lesion
            x_range = np.arange(start, end + 1) # +1 to include the last slice
            # Fill the background from y=0 to y=1
            axs[index].fill_between(x_range, 0, 1, color=colors[lesion_idx % len(colors)], alpha=0.2, label=f'Lesion {lesion_idx + 1}', transform=axs[index].get_xaxis_transform())
        
        # Plot spine-generic multi-subject data
        sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_hc, errorbar='sd', linewidth=2, color='black',
            label=f'spine-generic all subjects (N = {number_of_subjects})')
        # Plot the first metric in purple (corresponding to the right-left symmetry)
        sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_sub, linewidth=2, color='green',
                        label=f'{subID}')
        
        ymin, ymax = axs[index].get_ylim()

        # Add legend
        if index == 0:
            axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
        else:
            axs[index].get_legend().remove()


        # Add master title
        plt.suptitle(f'Asymetry plots for {subID} across axial slices and vertebral levels',
                     fontweight='bold', fontsize=LABELS_FONT_SIZE, y=0.92)

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS_ASYMMETRY[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_sub, single_subject=True)
        vert = [int(v) for v in vert]  # Convert vert to integer values to avoid issues with string labels when plotting vertebral levels
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df_sub.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with labels
            if vert[x] > 19:
                level = 'L' + str(vert[x] - 19)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    plt.savefig(path_out_png, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_out_png)

    return None


def create_lineplot_laterality(df_sub, subID, path_out_png): 
    """
    Create lineplot for laterality of lesions across vertebral levels.
    Note: we are plotting slices not levels to avoid averaging across levels.
    Args:
        df_sub (pd.dataFrame): dataframe with the subject values
        subID (str): subject ID
        path_out_png (str): path to output PNG file
    Output:
        A PNG file with the lineplot for laterality of lesions across vertebral levels.
    """

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    axs = axes.ravel()
    METRICS_LATERALITY = ["white matter", "gray matter", "dorsal columns", "lateral funiculi", "ventral funiculi", "total % (all tracts)"]

    # Remove rows with NaN values
    df_sub = df_sub.dropna(subset=METRICS_LATERALITY).reset_index(drop=True)
    df_sub = df_sub.dropna(subset=['VertLevel']).reset_index(drop=True)

    # Loop across metrics
    for index, metric in enumerate(METRICS_LATERALITY):
        # One line per lesion (lesion_label column) in the subject, colored by lesion label
        colors = ['maroon', 'goldenrod', 'deeppink', 'darkorchid', 'olive']
        for lesion_idx, lesion_label in enumerate(df_sub['lesion_label'].unique()):
            df_sub_lesion = df_sub[df_sub['lesion_label'] == lesion_label]
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df_sub_lesion, linewidth=2, color=colors[lesion_idx % len(colors)], 
                        label=f'Lesion {lesion_label}')
        
        ymin, ymax = axs[index].get_ylim()

        # Add legend
        if index == 0:
            axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
        else:
            axs[index].get_legend().remove()


        # Add master title
        plt.suptitle(f'Lesion laterality plots for {subID} across axial slices',
                     fontweight='bold', fontsize=LABELS_FONT_SIZE, y=0.92)

        # Add labels
        axs[index].set_ylabel(metric, fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_sub, single_subject=True)
        vert = [int(v) for v in vert]  # Convert vert to integer values to avoid issues with string labels when plotting vertebral levels
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df_sub.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with labels
            if vert[x] > 19:
                level = 'L' + str(vert[x] - 19)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df_sub.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    plt.savefig(path_out_png, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_out_png)