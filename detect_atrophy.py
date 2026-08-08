"""
This script detects atrophy in the spinal cord using the output of sct_process_segmentation with PAM-50 normalized CSA. 
It considers that there is atrophy in the spinal cord if the CSA, if locally the CSA is less than 70% of the average CSA
of the above and below vertebral level.

Input: 
    -csa-file: csv file with the CSA values of the spinal cord of the subject.

Output:
    -vertebral-levels: a list containing the vertebral levels where atrophy was detected. It is empty if no atrophy was detected.

Example of usage:
    python detect_atrophy.py -csa-file csa_values.csv

Author: Pierre-Louis Benveniste
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect atrophy in the spinal cord from CSA values')
    parser.add_argument('-csv-file', type=str, required=True, help='CSV file with CSA values of the spinal cord')
    return parser.parse_args()


def main(): 
    # Parse arguments
    args = parse_arguments()
    csv_file = args.csv_file

    # Read csv file
    data = pd.read_csv(csv_file)
    # Keep only some columns
    data = data[['Slice (I->S)', 'VertLevel', 'MEAN(area)', 'MEAN(diameter_RL)']]
    # Convert VertLevel to string
    data['VertLevel'] = data['VertLevel'].astype(str)

    # We first plot the mean CSA and mean diameter across the spinal cord
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data['Slice (I->S)'], data['MEAN(area)'], 'g-')
    ax2.plot(data['Slice (I->S)'], data['MEAN(diameter_RL)'], 'b-')
    ax1.set_xlabel('Slice (I->S)')
    ax1.set_ylabel('CSA (mm^2)', color='g')
    ax2.set_ylabel('Diameter (mm)', color='b')
    plt.show()

    # We look at the same plot but with the vertebral levels
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data['VertLevel'], data['MEAN(area)'], 'g-')
    ax2.plot(data['VertLevel'], data['MEAN(diameter_RL)'], 'b-')
    ax1.set_xlabel('VertLevel')
    ax1.set_ylabel('CSA (mm^2)', color='g')
    ax2.set_ylabel('Diameter (mm)', color='b')
    plt.show()


if __name__ == "__main__":
    main()    