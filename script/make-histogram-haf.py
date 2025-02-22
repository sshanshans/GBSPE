import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def extract_numbers_from_logs_optimized(folder_path):
    mc_list = []
    gbs_list = []
    gt_list = []

    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('log'):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('mc:'):
                        # Extract and convert the 'mc' value directly
                        mc_value = float(line.split()[1])
                        mc_list.append(mc_value)
                    elif line.startswith('gbs:'):
                        # Extract and convert the 'gbs' value directly
                        gbs_value = float(line.split()[1])
                        gbs_list.append(gbs_value)
                    elif line.startswith('gt:'):
                        # Extract and convert the 'gbs' value directly
                        gt_value = float(line.split()[1])
                        gt_list.append(gt_value)

    return mc_list, gbs_list, gt_list


def plot_combined(mc_values, gbs_values, gt_values, filename):
    # Convert inputs to numpy arrays once
    mc_values = np.array(mc_values)
    gbs_values = np.array(gbs_values)
    gt_values = np.array(gt_values)

    # Compute entrywise ratios and other required data
    ratio_values = mc_values / gbs_values

    # Filter ratios and calculate results
    count_geq_1 = np.sum(ratio_values >= 1)
    count_leq_1 = np.sum(ratio_values <= 1)

    ratio_filtered1 = ratio_values[ratio_values >= 1]
    ratio_filtered2 = 1 / ratio_values[ratio_values <= 1]
    
    # Setup the figure for 3 subplots (side-by-side boxplot and two histograms)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 2: Histogram of MC/GBS (≥ 1)
    axes[0].hist(ratio_filtered1, edgecolor='black', alpha=0.75)
    axes[0].set_xlabel(r"${V^{\text{MC}}_{\text{Haf}}}/{V^{\text{GBS-I}}_{\text{Haf}}}$", fontsize=20)
    axes[0].set_ylabel('Frequency', fontsize=20)
    axes[0].set_title(f'When GBS-P is more efficient', fontsize=20)

    # Plot 3: Histogram of GBS/MC (≤ 1, inverted to ≥ 1)
    axes[1].hist(ratio_filtered2, edgecolor='black', alpha=0.75)
    axes[1].set_xlabel(r"${V^{\text{GBS-P}}_{\text{Haf}}}/{V^{\text{MC}}_{\text{Haf}}}$", fontsize=20)
    axes[1].set_ylabel('Frequency', fontsize=20)
    axes[1].set_title(f'When MC is more efficient ', fontsize=20)

    # Plot 1: Side-by-side boxplot of the ratio (log 10 base transformed)
    sns.boxplot(data=[ratio_filtered1, ratio_filtered2], ax=axes[2], showfliers=False)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['GBS-P gains', 'MC gains'])
    axes[2].set_ylabel('Values', fontsize=20)
    axes[2].set_title(f'Box plot of the ratio', fontsize=20)

    # Set tick text size for all subplots
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=15)  # Major ticks
        ax.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)

def track_results_in_table():
    # Define the range of N and K values
    N_values = [3, 4, 5, 6]
    K_values = [2, 3, 4, 5, 6, 7, 8]

    # Loop through each N and K
    for i, N in enumerate(N_values):
        print(N)
        for j, K in enumerate(K_values):
            print(K)
            # Prepare experiment ID and folder path
            phi = 'haf'
            exp_id = f'{phi}-N_{N}-K_{K}'
            path_to_folder = f'/work/GBSPE/exp/haf/{exp_id}'

            try:
                # Extract mc_values, gbs_values and vandermonde
                mc_values, gbs_values, gt_values = extract_numbers_from_logs_optimized(path_to_folder)
                print('done')

                # Plot combined graphs and get the result
                filename = f'/work/GBSPE/exp/haf/fig2/{exp_id}.png'
                plot_combined(mc_values, gbs_values, gt_values, filename)

            except Exception as e:
                # If something fails, log or handle the error (e.g., print a warning)
                print(f"Failed for N={N}, K={K}: {e}")

def main():
    track_results_in_table()

if __name__ == "__main__":
    main()
