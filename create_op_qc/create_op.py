# --*-- conding:utf-8 --*--
# @Time : 2/13/25 11:08 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : create_op.py

import os
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
import json
from qiskit.quantum_info import SparsePauliOp
import numpy as np



main_chain_residue_seq = "GAVEDGATMTFF"
side_chain_residue_sequences = ['' for _ in range(len(main_chain_residue_seq))]
protein_name = '2xxx'


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

    print(type(hamiltonian))

    # 将 SparsePauliOp 转换为 JSON 兼容格式，并手动转换复数
    hamiltonian_list = [
        (pauli, (coeff.real, coeff.imag))  # 将复数拆分成 (实部, 虚部)
        for pauli, coeff in hamiltonian.to_list()
    ]

    # 保存 JSON
    with open("operator.json", "w") as f:
        json.dump(hamiltonian_list, f)

