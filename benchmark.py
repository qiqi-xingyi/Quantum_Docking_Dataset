# --*-- conding:utf-8 --*--
# @Time : 2/21/25 6:05 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : benchmark.py

import os
import shutil
import time

# 根据你的工程结构，确保可以正确导入如下模块
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

# 这里替换成你实际的 VQE5 类所在的模块
from Qiskit_VQE import VQE5
from Qiskit_VQE import StateCalculator

# 如果你需要在这个脚本中初始化 QiskitRuntimeService
# from qiskit_ibm_runtime import QiskitRuntimeService


def read_config(file_path):
    """
    读取 config 文件 (INSTANCE=xxx / TOKEN=xxx)
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
    读取指定 TXT 文件的每一行，解析得到数据结构：
    [
      {
        'pdb_id': <str>,
        'pocket_file': <str>,
        'chain': <str>,
        'residues_range': <str>,
        'sequence': <str>   # single-letter sequence
      },
      ...
    ]
    """
    # 三字母转一字母的映射
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

            # 假设以 TAB 分隔，若你的数据是空格或其他分隔符，请自行调整
            parts = line.split('\t')
            if len(parts) < 5:
                continue

            pdb_id = parts[0]
            pocket_file = parts[1]
            chain_info = parts[2]          # e.g. "Chain A"
            residue_info = parts[3]        # e.g. "Residues 192-203"
            seq_info_3letter = parts[4]    # e.g. "VAL-VAL-TYR-PRO-..."

            # 解析 chain
            chain = chain_info.replace("Chain", "").strip()
            # 解析残基编号区间
            residues_range = residue_info.replace("Residues", "").strip()

            # 三字母 -> 一字母
            three_letter_codes = seq_info_3letter.split('-')
            single_letter_codes = []
            for code_3letter in three_letter_codes:
                code_3letter = code_3letter.strip().upper()
                if code_3letter in three_to_one:
                    single_letter_codes.append(three_to_one[code_3letter])
                else:
                    raise ValueError(f"Unknown residue code: {code_3letter}")

            seq_info_1letter = "".join(single_letter_codes)

            fragments.append({
                'pdb_id': pdb_id,
                'pocket_file': pocket_file,
                'chain': chain,
                'residues_range': residues_range,
                'sequence': seq_info_1letter
            })
    return fragments


def pick_unique_fragments(fragments, max_count=25):
    """
    保留前 max_count 个不同 pdb_id 的片段
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


def run_vqe_for_fragment(frag, service, max_iter=150):
    """
    使用VQE5来预测该fragment对应的蛋白结构，保存5组最优结构。
    """
    main_chain_sequence = frag['sequence']
    protein_id = frag['pdb_id']

    print(f"\n=== Processing protein {protein_id} ===")
    print(f"Residue sequence: {main_chain_sequence}")
    print(f"Sequence length: {len(main_chain_sequence)}")

    # 1. 构建蛋白对象（此处只用主链，不处理侧链）
    side_chain_seq = ['' for _ in range(len(main_chain_sequence))]
    peptide = Peptide(main_chain_sequence, side_chain_seq)

    # 2. 定义相互作用 & 罚函数参数
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(10, 10, 10)

    # 3. 建立“蛋白质折叠”问题并构建哈密顿量
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    hamiltonian = protein_folding_problem.qubit_op()

    # 这里根据你的需求，把额外加的 qubits 数改为 5（代替原先 +2 的做法）
    qubits_num = hamiltonian.num_qubits + 5
    print(f"Number of qubits: {qubits_num}")

    # 4. 调用 VQE5
    vqe_instance = VQE5(
        service=service,
        hamiltonian=hamiltonian,
        min_qubit_num=qubits_num,
        maxiter=max_iter
    )

    # run_vqe() 返回 (energy_list, best_solution, ansatz, top_results)
    # 其中 top_results 通常是 [(能量值, 参数), ...] 共5组
    energy_list, final_solution, ansatz, top_results = vqe_instance.run_vqe()

    # 5. 根据结果做后处理和保存
    # 设置输出目录：每个 pdb_id 一个文件夹
    output_dir = f"result/{protein_id}"
    os.makedirs(output_dir, exist_ok=True)

    # (a) 保存全部迭代过程的能量列表
    energy_list_file = os.path.join(output_dir, f"energy_list_{protein_id}.txt")
    with open(energy_list_file, 'w') as file:
        for item in energy_list:
            file.write(str(item) + '\n')
    print(f"Energy list saved to: {energy_list_file}")

    # (b) 计算 final_solution 的概率分布并解释成 3D 坐标
    state_calculator = StateCalculator(service, qubits_num, ansatz)
    final_prob_dist = state_calculator.get_probability_distribution(final_solution)
    protein_result = protein_folding_problem.interpret(final_prob_dist)

    # (c) 保存最终结构 .xyz
    protein_result.save_xyz_file(name=protein_id, path=output_dir)
    print(f"Protein structure saved at: {output_dir}/{protein_id}.xyz")

    # (d) 保存 top 5 结果各自的能量
    top_energies_file = os.path.join(output_dir, f"top_5_energies_{protein_id}.txt")
    with open(top_energies_file, 'w') as f_top:
        for rank, (energy_val, best_params) in enumerate(top_results, start=1):
            f_top.write(f"Rank {rank}: {energy_val}\n")

    # (e) 对每个 top result 再做一次概率分布、解析并存 .xyz
    for rank, (energy_val, best_params) in enumerate(top_results, start=1):
        prob_dist_best = state_calculator.get_probability_distribution(best_params)
        protein_result_best = protein_folding_problem.interpret(prob_dist_best)
        xyz_file_name = f"{protein_id}_top_{rank}"
        protein_result_best.save_xyz_file(name=xyz_file_name, path=output_dir)
        print(f"Top {rank} best energy = {energy_val}, xyz saved: {xyz_file_name}.xyz")

    print(f"Finished processing: {protein_id}\n")


def main():
    # ====================
    # 0) 用户可修改的参数
    # ====================
    txt_file_path = "Data/5_7.txt"       # 需要预测的片段TXT文件
    config_path = "config.txt"      # IBM Quantum config 文件
    max_fragments = 25              # 最多选多少个片段
    max_iter = 10                  # VQE 最大迭代次数

    # 1) 读取配置
    config = read_config(config_path)
    if not config or "INSTANCE" not in config or "TOKEN" not in config:
        print("Could not read INSTANCE or TOKEN from config. Please check your config file.")
        return

    # 2) 初始化量子服务
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=config["INSTANCE"],
        token=config["TOKEN"]
    )

    # 3) 从TXT中读入所有片段
    all_fragments = parse_txt_file(txt_file_path)

    # 4) 选取去重后的前 max_fragments 个
    selected_fragments = pick_unique_fragments(all_fragments, max_fragments)

    # 5) 对每个fragment运行量子预测
    log_file_path = "execution_time_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write("Protein_ID\tSequence\tExecution_Time(s)\n")

        for idx, fragment in enumerate(selected_fragments, start=1):
            protein_id = fragment['pdb_id']
            seq = fragment['sequence']
            print(f"\n--- Fragment {idx}/{len(selected_fragments)}: PDB={protein_id} ---")

            start_time = time.time()
            run_vqe_for_fragment(fragment, service, max_iter=max_iter)
            end_time = time.time()

            elapsed = end_time - start_time
            # 记录日志
            log_file.write(f"{protein_id}\t{seq}\t{elapsed:.2f}\n")

    print("\nAll processing is complete. Log saved to:", log_file_path)


if __name__ == "__main__":
    main()

