import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# this plot helps summarized results from expv2.py

def extract_numbers_from_logs_optimized(folder_path):
    eff_list = []
    vdet_list = []

    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('log'):
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('gbs_eff: '):
                        # Extract and convert the 'gbs efficient' value directly
                        eff_value = float(line.split()[1])
                        eff_list.append(eff_value)
                    elif line.startswith('vandermonde:'):
                        # Extract and convert the 'gbs' value directlyc
                        next_line = next(file, '').strip()
                        vdet_value = float(next_line) 
                        vdet_list.append(vdet_value)

    return eff_list, vdet_list

def process_list(eff_list, vdet_list, filename):
    # Convert inputs to numpy arrays once
    eff_binary_values = np.array(eff_list)
    vdet_values = np.array(vdet_list)

    # Compute vondermont reweighted
    eff_weighted_values = eff_binary_values * vdet_values  # Elementwise multiplication

    # Compute total
    total_count = len(eff_binary_values)
    total_vdet_weight = np.sum(vdet_values)

    # Compute ratios
    count_geq_1 = np.sum(eff_binary_values)
    ratio_count = count_geq_1 / total_count

    weighted_geq_1 = np.sum(eff_weighted_values)
    ratio_weighted = weighted_geq_1 / total_vdet_weight

    # Cumulative sum for convergence plots
    cumulative_count = np.cumsum(eff_binary_values)
    cumulative_weighted = np.cumsum(eff_weighted_values)
    cumulative_vdet = np.cumsum(vdet_values)
    
    n_values = np.arange(1, total_count + 1)
    cumulative_nval = np.cumsum(n_values)

    cumulative_cc = cumulative_count / n_values
    cumulative_ww = cumulative_weighted / cumulative_vdet
    
    # Setup the figure for subplots
    plt.figure(figsize=(10, 8))
    plt.plot(n_values, cumulative_ww, color='green')
    plt.xlabel('n', fontsize=20)
    plt.ylabel('Fraction of GBS-P advantage', fontsize=20)
    plt.ylim(-0.05, 1.1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # Set tick size
    plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust major tick size
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Adjust minor tick size

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()

    return ratio_count, ratio_weighted, total_count

def track_results_in_table():
    # Define the range of N and K values
    N_values = [3, 4, 5, 6]
    K_values = [2, 3, 4, 5, 6, 7, 8]

    # Initialize a table (NumPy array) to store the results
    results_table = np.zeros((len(N_values), len(K_values)))
    weights_table = np.zeros((len(N_values), len(K_values))) 
    counts_table = np.zeros((len(N_values), len(K_values)))

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
                eff_list, vdet_list = extract_numbers_from_logs_optimized(path_to_folder)
                print('done')

                # Plot combined graphs and get the result
                filename = f'/work/GBSPE/exp/haf/fig/{exp_id}.png'
                ratio_count, ratio_weighted, total_count = process_list(eff_list, vdet_list, filename)

                # Store the result in the table
                results_table[i, j] = ratio_count
                weights_table[i, j] = ratio_weighted
                counts_table[i, j] = total_count

            except Exception as e:
                # If something fails, log or handle the error (e.g., print a warning)
                print(f"Failed for N={N}, K={K}: {e}")
                results_table[i, j] = np.nan  # Ensure failure results in NaN in the table

    # Create a DataFrame to clearly label N and K
    df0 = pd.DataFrame(weights_table, index=[f'N={N}' for N in N_values], columns=[f'K={K}' for K in K_values])
    df1 = pd.DataFrame(results_table, index=[f'N={N}' for N in N_values], columns=[f'K={K}' for K in K_values])
    df2 = pd.DataFrame(counts_table, index=[f'N={N}' for N in N_values], columns=[f'K={K}' for K in K_values])

    # Display the table
    print(df0)
    print(df1)
    print(df2)

    # Optionally return the DataFrame if needed
    return df0, df1, df2

def main():
    df0, df1, df2 = track_results_in_table()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df1, annot=True, cmap='YlGnBu', cbar=True, fmt=".2f", linewidths=.5)
    plt.title('Heatmap of Proportion of GBS more efficient than MC')
    plt.xlabel('K Values')
    plt.ylabel('N Values')
    plt.tight_layout()
    plt.savefig('/work/GBSPE/exp/haf/fig/ratio_count.png')

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df0 * 100, annot=True, cmap='YlGnBu', cbar=True, fmt=".2f", linewidths=.5)
    # Set tick size
    plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust major tick size
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Adjust minor tick size
    plt.tight_layout()
    plt.savefig('/work/GBSPE/exp/haf/fig/ratio_weight.png')

if __name__ == "__main__":
    main()
