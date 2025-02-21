# --*-- conding:utf-8 --*--
# @Time : 2/11/25 1:42 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : create_qc.py

# !/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit import qasm3
import qiskit

def main():

    num_qubits = 77  # num of qubit
    reps = 10  # EfficientSU2

    ansatz = EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement='full')

    decomposed_circuit = ansatz.decompose()

    # qasm_str = qiskit.qasm2.dumps(decomposed_circuit)
    qasm_str = qasm3.dumps(decomposed_circuit, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1)

    output_file = '../my_circuit.qasm'
    with open(output_file, 'w') as f:
        f.write(qasm_str)

    print(f"QASM file: {output_file}")

if __name__ == "__main__":
    main()
