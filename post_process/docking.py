# --*-- conding:utf-8 --*--
# @Time : 3/20/25 8:43 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : docking.py

from docking import DockingPipeline
import os
import random


def docking_test(chain_dir, protein_id, xyz_files, ligand_file, num_trials=20):
    """
    Perform docking tests for all .xyz files under the specified protein ID.

    Parameters:
    - chain_dir: Chain folder name (e.g., chain-5)
    - protein_id: Protein ID (e.g., 1fkf)
    - xyz_files: List of paths to all .xyz files under the specified protein ID
    - ligand_file: Path to the ligand file
    - num_trials: Number of trials, default is 20
    """
    print(f'Processing protein ID: {protein_id} in {chain_dir}')

    # Define output path using chain_dir and protein_id to organize results
    docking_output_path = f'docking_result/{chain_dir}/{protein_id}'
    seed_log_file = f"./{docking_output_path}/seed_log.txt"
    os.makedirs(os.path.dirname(seed_log_file), exist_ok=True)

    with open(seed_log_file, "w") as seed_file:
        seed_file.write("Trial\tSeed\n")

    for trial in range(1, num_trials + 1):
        seed = random.randint(1, 100000)
        print(f"Trial {trial} - Using seed: {seed}")
        with open(seed_log_file, "a") as seed_file:
            seed_file.write(f"{trial}\t{seed}\n")

        for xyz_file in xyz_files:
            print(f'Processing XYZ file: {xyz_file}')
            xyz_basename = os.path.splitext(os.path.basename(xyz_file))[0]
            quantum_output_dir = f"./{docking_output_path}/quantum_trial_{trial}/{xyz_basename}"
            os.makedirs(quantum_output_dir, exist_ok=True)
            pipeline = DockingPipeline(quantum_output_dir)
            pipeline.dock_quantum(
                xyz_file,
                ligand_file,
                quantum_output_dir,
                f"docking_log_trial_{trial}_{os.path.basename(xyz_file)}.txt",
                seed
            )
            print(f"Trial {trial} docking for {xyz_file} completed.")


if __name__ == "__main__":

    # Define root directory
    grouped_result_dir = "./data/grouped_result"
    selected_dir = "./data/selected"

    # Iterate through all chain-* folders
    for chain_dir in os.listdir(grouped_result_dir):
        if chain_dir.startswith("chain-") or chain_dir.startswith("chain_"):
            chain_path = os.path.join(grouped_result_dir, chain_dir)
            if os.path.isdir(chain_path):
                # Iterate through all protein ID folders in the chain folder
                for protein_id in os.listdir(chain_path):
                    protein_path = os.path.join(chain_path, protein_id)
                    if os.path.isdir(protein_path):
                        # Collect all .xyz files in the protein ID folder
                        xyz_files = [os.path.join(protein_path, f) for f in os.listdir(protein_path) if
                                     f.endswith('.xyz')]
                        if not xyz_files:
                            print(f"No .xyz files found in {chain_dir}/{protein_id}")
                            continue

                        # Define ligand file path (adjusted to _ligand.mol2 according to image)
                        ligand_file = os.path.join(selected_dir, protein_id, f"{protein_id}_ligand.mol2")
                        if not os.path.exists(ligand_file):
                            print(f"Ligand file not found: {ligand_file}")
                            continue

                        # Perform docking test for the specified protein ID
                        docking_test(chain_dir, protein_id, xyz_files, ligand_file, num_trials=20)
