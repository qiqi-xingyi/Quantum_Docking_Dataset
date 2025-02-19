# --*-- conding:utf-8 --*--
# @Time : 1/14/25 9:22 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : get_avg.py

import os
import re
import numpy as np

# Define the folder containing the docking logs
docking_folder = "docking_output_low_energy_6mu3"

# Output file for the summarized results
output_file = os.path.join(docking_folder, "summary_results.txt")

# Regex pattern to extract rows from the table in the log file
data_pattern = re.compile(r"\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")

# Initialize a list to store the results
results = []

if __name__ == '__main__':

    # Loop through all top folders
    for top_folder in sorted(os.listdir(docking_folder)):
        top_path = os.path.join(docking_folder, top_folder)

        # Ensure it is a folder and contains a docking log
        if os.path.isdir(top_path):
            log_file = os.path.join(top_path, f"docking_log_{top_folder}.txt")
            if os.path.exists(log_file):
                with open(log_file, "r") as file:
                    lines = file.readlines()

                # Extract table data from the log file
                affinities = []
                rmsd_lb = []
                rmsd_ub = []

                for line in lines:
                    match = data_pattern.match(line)
                    if match:
                        affinities.append(float(match.group(2)))
                        rmsd_lb.append(float(match.group(3)))
                        rmsd_ub.append(float(match.group(4)))

                # Compute averages for the current top folder
                if affinities and rmsd_lb and rmsd_ub:
                    avg_affinity = np.mean(affinities)
                    avg_rmsd_lb = np.mean(rmsd_lb)
                    avg_rmsd_ub = np.mean(rmsd_ub)
                    results.append((top_folder, avg_affinity, avg_rmsd_lb, avg_rmsd_ub))

    # Write the results to the summary file
    with open(output_file, "w") as out_file:
        out_file.write("Top_Folder\tAvg_Affinity\tAvg_RMSD_LB\tAvg_RMSD_UB\n")
        for result in results:
            out_file.write(f"{result[0]}\t{result[1]:.4f}\t{result[2]:.4f}\t{result[3]:.4f}\n")

    print(f"Summary results saved to {output_file}")