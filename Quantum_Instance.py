# --*-- conding:utf-8 --*--
# @Time : 11/8/24 1:06 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Quantum_Instance.py

import os
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from qiskit_ibm_runtime import QiskitRuntimeService
from Qiskit_VQE import VQE
from Qiskit_VQE import StateCalculator

main_chain_residue_seq = "YFASGQPYRYER"
side_chain_residue_sequences = ['' for _ in range(len(main_chain_residue_seq))]
protein_name = '4zb8'


def read_config(file_path):

    config = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                config[key.strip()] = value.strip()
    except Exception as e:
        print(f"Fail to read token file: {e}")
        return None
    return config

if __name__ == '__main__':

    char_count = len(main_chain_residue_seq)
    print(f'Num of Acid:{char_count}')

    side_site = len(side_chain_residue_sequences)
    print(side_chain_residue_sequences)
    print(f'Num of Side cite:{side_site}')

    # create Peptide
    peptide = Peptide(main_chain_residue_seq , side_chain_residue_sequences)

    # Interaction definition (e.g. Miyazawa-Jernigan)
    mj_interaction = MiyazawaJerniganInteraction()

    # Penalty Parameters Definition
    penalty_terms = PenaltyParameters(10, 10, 10)

    # Create Protein Folding case
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    # create quantum Op
    hamiltonian = protein_folding_problem.qubit_op()

    # print('Operator',hamiltonian)

    ########## Create a Quantum Service ############

    config_path = "config.txt"

    config = read_config(config_path)

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=config["INSTANCE"],
        token=config["TOKEN"]
    )

    # ansatz = EfficientSU2(hamiltonian.num_qubits)
    qubits_num = hamiltonian.num_qubits + 2
    print(f'Num of qubits:{qubits_num}')
    vqe_instance = VQE(service=service, hamiltonian=hamiltonian, min_qubit_num=qubits_num, maxiter=150)

    # Run the VQE algorithm
    energy_list, res, ansatz = vqe_instance.run_vqe()

    with open('./QC_Status_Analysis/System_Enegry/energy_list.txt', 'w') as file:
        for item in energy_list:
            file.write(str(item) + '\n')

    state_calculator = StateCalculator(service,qubits_num,ansatz)
    prob_distribution = state_calculator.get_probability_distribution(res)

    print(f'VQE_result:{prob_distribution}')

    with open('./QC_Status_Analysis/Prob_distribution/prob_distribution.txt', 'w') as file:
        for key, value in prob_distribution.items():
            file.write(f'{key}: {value}\n')

    protein_result = protein_folding_problem.interpret(prob_distribution)

    # save to .xyz
    output_dir = f"Post_Processing/process_data/{protein_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    protein_result.save_xyz_file(name=protein_name, path=output_dir)
    print("Protein structure saved as .xyz file")
