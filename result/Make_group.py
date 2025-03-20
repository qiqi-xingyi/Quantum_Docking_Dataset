# --*-- conding:utf-8 --*--
# @Time : 3/20/25 7:24 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Make_group.py

import os
import shutil

if __name__ == '__main__':

    # Define paths
    result_path = 'result'  # Path to the result folder
    grouped_path = 'grouped_result'  # Path for the new grouped folders

    # Create the grouped_by_chain_length folder if it doesn’t exist
    if not os.path.exists(grouped_path):
        os.makedirs(grouped_path)

    # Traverse all subfolders in the result folder
    for subfolder in os.listdir(result_path):
        subfolder_path = os.path.join(result_path, subfolder)

        # Check if it’s a directory
        if os.path.isdir(subfolder_path):
            # Construct the path to the .xyz file (e.g., result/1bai/1bai.xyz)
            xyz_file = os.path.join(subfolder_path, subfolder + '.xyz')

            # Check if the .xyz file exists
            if os.path.exists(xyz_file):
                # Read the first line to get the chain length
                with open(xyz_file, 'r') as f:
                    first_line = f.readline().strip()
                    try:
                        chain_length = int(first_line)
                    except ValueError:
                        print(f"错误：{xyz_file} 中的链长不是整数")
                        continue

                # Define the chain length folder (e.g., grouped_by_chain_length/chain_11)
                chain_folder = os.path.join(grouped_path, f'chain_{chain_length}')

                # Create the chain length folder if it doesn’t exist
                if not os.path.exists(chain_folder):
                    os.makedirs(chain_folder)

                # Define the destination path (e.g., grouped_by_chain_length/chain_11/1bai)
                destination = os.path.join(chain_folder, subfolder)

                # Copy the subfolder if the destination doesn’t already exist
                if not os.path.exists(destination):
                    shutil.copytree(subfolder_path, destination)
                else:
                    print(f"Destination {destination} already exists, skipping copy")
            else:
                print(f".xyz file not found in {subfolder_path}")