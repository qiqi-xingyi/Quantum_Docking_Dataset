# --*-- conding:utf-8 --*--
# @Time : 1/17/25 3:58 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : find_data.py

import os

def parse_pocket_pdb(pdb_file_path):
    """
    Parse the pocket PDB file and return a dictionary:
    chain_residues = {
        chain_id: [(resnum1, resname1), (resnum2, resname2), ...],
        ...
    }
    Each element is sorted in ascending order by residue number.
    """
    chain_residues = {}

    with open(pdb_file_path, 'r') as f:
        for line in f:
            # Only parse ATOM lines
            if line.startswith("ATOM"):
                # In PDB format, chain ID is usually at column 22 (index 21)
                chain_id = line[21].strip()

                # Residue number is usually at columns 23-26 (index 22:26)
                resnum_str = line[22:26].strip()
                try:
                    resnum = int(resnum_str)
                except ValueError:
                    continue

                # Three-letter amino acid name is at columns 18-20 (index 17:20)
                resname = line[17:20].strip()

                if chain_id not in chain_residues:
                    chain_residues[chain_id] = []

                # Prevent duplicate addition of the same residue (same resnum)
                if not chain_residues[chain_id] or chain_residues[chain_id][-1][0] != resnum:
                    chain_residues[chain_id].append((resnum, resname))

    # Sort by residue number
    for c_id in chain_residues:
        chain_residues[c_id].sort(key=lambda x: x[0])

    return chain_residues


def split_consecutive_segments(chain_id, residues, folder_name, pdb_file, out_f):
    """
    Split the list of residues in the same chain (already sorted by number) based on continuity and output to a file.
    Only keep continuous segments with a length of 6 to 9.
    """
    if not residues:
        return

    start_idx = 0
    n = len(residues)

    for i in range(1, n):
        # Check continuity (whether res_num is consecutive)
        if residues[i][0] != residues[i - 1][0] + 1:
            # [start_idx, i-1] is a continuous segment
            segment = residues[start_idx:i]
            # Write the result_10_11
            _write_segment_if_in_range(segment, chain_id, folder_name, pdb_file, out_f)
            start_idx = i

    # Process the last segment
    if start_idx < n:
        segment = residues[start_idx:n]
        _write_segment_if_in_range(segment, chain_id, folder_name, pdb_file, out_f)


def _write_segment_if_in_range(segment, chain_id, folder_name, pdb_file, out_f):
    """
    Write to a file if the segment length is between 6 and 9; otherwise, skip it.
    """
    length = len(segment)
    if length < 12 or length > 14:
        return  # Skip segments that are out of range

    # If the length requirement is met, write to file
    resnums = [str(item[0]) for item in segment]
    resnames = [item[1] for item in segment]

    # Start and end residue numbers
    start_num = segment[0][0]
    end_num = segment[-1][0]

    # Format the output line
    line = (
        f"{folder_name}\t"
        f"{pdb_file}\t"
        f"Chain {chain_id}\t"
        f"Residues {start_num}-{end_num}\t"
        f"{'-'.join(resnames)}"
    )
    out_f.write(line + "\n")


def main():
    # Main directory of the dataset
    root_dir = "Data/refined-set"
    # Output file
    output_file = "Data/consecutive_12_14.txt"

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Traverse all subfolders under the main dataset directory (each subfolder corresponds to a sub-protein)
        for folder_name in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(subfolder_path):
                continue

            # Only process PDB files ending with "_pocket.pdb"
            pdb_files = [
                f for f in os.listdir(subfolder_path)
                if f.endswith("_pocket.pdb")  # Can be modified for stricter matching if needed
            ]

            for pdb_file in pdb_files:
                pdb_path = os.path.join(subfolder_path, pdb_file)

                # Parse pocket PDB
                chain_residues = parse_pocket_pdb(pdb_path)

                # Split consecutive segments for each chain and write the output
                for chain_id, residues in chain_residues.items():
                    split_consecutive_segments(chain_id, residues, folder_name, pdb_file, out_f)

    print(f"Processing complete, results written to {output_file}")


if __name__ == "__main__":
    main()
