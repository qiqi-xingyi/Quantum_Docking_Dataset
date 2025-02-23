# --*-- conding:utf-8 --*--
# @Time : 2/21/25 6:05 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : bench_work.py

import os
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from Qiskit_VQE import VQE
from Qiskit_VQE import StateCalculator


def read_config(file_path):
    """
    Reads a config file with lines like:
      INSTANCE=abc/def/ghi
      TOKEN=your_token_here
    Returns a dict containing the parsed key-value pairs.
    """
    config = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    except Exception as e:
        print(f"Failed to read config file: {e}")
        return None
    return config


def parse_txt_file(txt_file_path):
    """
    Reads the specified TXT file. Each line is expected to look like:
        4b5d    4b5d_pocket.pdb    Chain A    Residues 192-203    VAL-VAL-TYR-PRO-...
    Returns a list of dictionaries with keys:
        {
          'pdb_id': <str>,
          'pocket_file': <str>,
          'chain': <str>,
          'residues_range': <str>,
          'sequence': <str>   # single-letter sequence
        }
    """

    # Dictionary mapping three-letter codes to single-letter codes
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    fragments = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split on tabs. Adjust if your file uses different delimiters/spaces.
            parts = line.split('\t')
            if len(parts) < 5:
                continue

            pdb_id = parts[0]
            pocket_file = parts[1]
            chain_info = parts[2]  # e.g. "Chain A"
            residue_info = parts[3]  # e.g. "Residues 192-203"
            seq_info_3letter = parts[4]  # e.g. "VAL-VAL-TYR-PRO-..."

            # Extract chain from something like "Chain A"
            chain = chain_info.replace("Chain", "").strip()
            # Extract residue range from something like "Residues 192-203"
            residues_range = residue_info.replace("Residues", "").strip()

            # Convert from three-letter code to single-letter code
            # Split by '-'
            three_letter_codes = seq_info_3letter.split('-')
            single_letter_codes = []
            for code_3letter in three_letter_codes:
                code_3letter = code_3letter.strip().upper()
                if code_3letter in three_to_one:
                    single_letter_codes.append(three_to_one[code_3letter])
                else:
                    # Handle non-standard codes or unknown residues as needed
                    raise ValueError(f"Unknown three-letter code: {code_3letter}")

            seq_info_1letter = "".join(single_letter_codes)

            fragments.append({
                'pdb_id': pdb_id,
                'pocket_file': pocket_file,
                'chain': chain,
                'residues_range': residues_range,
                # Store the single-letter sequence needed by your quantum pipeline
                'sequence': seq_info_1letter
            })
    return fragments


def pick_unique_fragments(fragments, max_count=10):
    """
    Keeps only the first occurrence of each PDB ID
    and returns up to max_count fragments.
    """
    selected = []
    used_pdb_ids = set()
    for frag in fragments:
        if frag['pdb_id'] not in used_pdb_ids:
            selected.append(frag)
            used_pdb_ids.add(frag['pdb_id'])
        if len(selected) >= max_count:
            break
    return selected


def run_vqe_for_fragment(frag, service):
    """
    Performs the quantum/VQE process for a single fragment.

    frag: dict with keys:
      'pdb_id' : str
      'sequence': str (single-letter codes)
      ...
    service: QiskitRuntimeService object
    """

    main_chain_residue_seq = frag['sequence']
    # Create empty side chains (one for each main-chain residue)
    side_chain_residue_sequences = ['' for _ in range(len(main_chain_residue_seq))]

    protein_name = frag['pdb_id']
    print(f"\n=== Processing protein {protein_name} ===")
    print(f"Residue sequence: {main_chain_residue_seq}")
    print(f"Sequence length: {len(main_chain_residue_seq)}")

    # 1. Create the peptide object
    peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

    # 2. Define the interaction
    mj_interaction = MiyazawaJerniganInteraction()

    # 3. Define penalty parameters
    penalty_terms = PenaltyParameters(10, 10, 10)

    # 4. Create the protein folding problem
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    # 5. Build the Hamiltonian
    hamiltonian = protein_folding_problem.qubit_op()

    # 6. Create and run VQE
    qubits_num = hamiltonian.num_qubits + 2
    print(f"Number of qubits: {qubits_num}")

    vqe_instance = VQE(service=service, hamiltonian=hamiltonian,
                       min_qubit_num=qubits_num, maxiter=150)
    energy_list, res, ansatz = vqe_instance.run_vqe()

    # 7. Save energy list
    # energy_list_file = f'./QC_Status_Analysis/System_Energy/energy_list_{protein_name}.txt'
    # os.makedirs(os.path.dirname(energy_list_file), exist_ok=True)
    # with open(energy_list_file, 'w') as file:
    #     for item in energy_list:
    #         file.write(str(item) + '\n')
    # print(f"Energy list saved to: {energy_list_file}")

    # 8. Probability distribution
    state_calculator = StateCalculator(service, qubits_num, ansatz)
    prob_distribution = state_calculator.get_probability_distribution(res)

    # prob_dist_file = f'./QC_Status_Analysis/Prob_distribution/prob_distribution_{protein_name}.txt'
    # os.makedirs(os.path.dirname(prob_dist_file), exist_ok=True)
    # with open(prob_dist_file, 'w') as file:
    #     for key, value in prob_distribution.items():
    #         file.write(f'{key}: {value}\n')
    # print(f"Probability distribution saved to: {prob_dist_file}")

    # 9. Interpret and save XYZ
    protein_result = protein_folding_problem.interpret(prob_distribution)
    output_dir = f"Post_Processing/process_data/{protein_name}"
    os.makedirs(output_dir, exist_ok=True)

    protein_result.save_xyz_file(name=protein_name, path=output_dir)
    print(f"XYZ file saved at: {output_dir}/{protein_name}.xyz")


if __name__ == "__main__":

    # User-adjustable parameters
    txt_file_path = "consecutive_12_14.txt"  # Path to your fragment text file
    config_path = "config.txt"  # Path to your config file (INSTANCE,TOKEN)
    max_fragments = 10  # Take the first 10 unique PDB IDs

    # 1) Read the config (INSTANCE and TOKEN)
    config = read_config(config_path)
    if not config or "INSTANCE" not in config or "TOKEN" not in config:
        print("Could not read INSTANCE or TOKEN from config. Please check your config file.")
        exit(1)

    # 2) Initialize QiskitRuntimeService
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=config["INSTANCE"],
        token=config["TOKEN"]
    )

    # 3) Parse the TXT file
    all_fragments = parse_txt_file(txt_file_path)

    # 4) Select up to 10 unique fragments (one per PDB ID)
    selected_fragments = pick_unique_fragments(all_fragments, max_fragments)

    # 5) Process each selected fragment in a loop
    for idx, fragment in enumerate(selected_fragments, start=1):
        print(f"\n--- Fragment {idx}/{len(selected_fragments)} ---")
        run_vqe_for_fragment(fragment, service)

    print("\nAll processing is complete.")
