# --*-- conding:utf-8 --*--
# @Time : 3/20/25 8:43 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : docking.py

import os
import glob
import subprocess
import numpy as np
import shutil

# BioPython 解析 PDB/CIF
from Bio.PDB import PDBParser, MMCIFParser, PDBIO

# Modeller 用于从 Cα 轨迹生成全原子模型
from modeller import environ, log
from modeller.automodel import automodel, assess
from modeller import alignment, model

class UnifiedDockingPipeline:
    """
    不依赖原先的类，整合所有功能：
      1. 解析 & 缩放 XYZ
      2. 调用 Modeller 生成全原子蛋白
      3. 处理 AF3 的 CIF 文件（可选）
      4. 处理并平移 ligand (MOL2)
      5. 转换至 PDBQT
      6. 调用 AutoDock Vina 进行 docking
    """

    def __init__(self,
                 grouped_result_dir,   # e.g. "data/grouped_result"
                 selected_dir,         # e.g. "data/selected"
                 output_dir,           # e.g. "docking_output"
                 seed=42):
        """
        初始化
        :param grouped_result_dir: 存放按 chain_5, chain_6 等分组后的预测结果 (XYZ/可能还有 CIF)
        :param selected_dir:       存放 PDBbind 数据集(含 .mol2 配体)
        :param output_dir:         对接结果输出目录
        :param seed:               docking 时的随机数种子
        """
        self.grouped_result_dir = grouped_result_dir
        self.selected_dir = selected_dir
        self.output_dir = output_dir
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)

    ############################
    #   Step 1: 处理 XYZ 文件   #
    ############################

    def parse_and_scale_xyz(self, xyz_file, target_distance=3.8):
        """
        读取 XYZ 文件，提取氨基酸单字母和坐标，并缩放至期望的平均 Cα-Cα 距离 (默认为3.8Å).
        返回序列(列表)和缩放后的坐标(列表).
        假设每行形如:  A  0.123  1.234  2.345
        """
        sequence = []
        coords = []
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            # 前两行常常是注释或原子数等信息，这里直接跳过
            # 你可以根据实际 XYZ 格式自行调整
            lines = lines[2:]

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4:
                aa, x, y, z = parts[0], parts[1], parts[2], parts[3]
                sequence.append(aa)
                coords.append([float(x), float(y), float(z)])
            else:
                print(f"[WARNING] 无法解析的 XYZ 行: {line.strip()}")

        if len(sequence) < 2:
            raise ValueError("XYZ 文件中氨基酸数 < 2，无法计算平均距离。")

        # 计算当前平均 Cα-Cα 距离
        distances = []
        for i in range(len(coords) - 1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i + 1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            distances.append(dist)
        current_avg_dist = np.mean(distances)

        scale_factor = target_distance / current_avg_dist
        scaled_coords = [[c[0] * scale_factor, c[1] * scale_factor, c[2] * scale_factor]
                         for c in coords]

        print(f"[INFO] XYZ 缩放因子: {scale_factor:.4f}")
        return sequence, scaled_coords

    def write_ca_pdb(self, sequence, coords, pdb_out, chain_id="A"):
        """
        根据 Cα 座标信息写出简单的 PDB (只包含 CA).
        :param sequence: 氨基酸单字母列表
        :param coords:   与之对应的缩放后坐标
        :param pdb_out:  输出的 CA-only PDB 文件名
        """
        # 单字母 -> 三字母
        aa_map = {
            'A': 'ALA','R': 'ARG','N': 'ASN','D': 'ASP','C': 'CYS','Q': 'GLN',
            'E': 'GLU','G': 'GLY','H': 'HIS','I': 'ILE','L': 'LEU','K': 'LYS',
            'M': 'MET','F': 'PHE','P': 'PRO','S': 'SER','T': 'THR','W': 'TRP',
            'Y': 'TYR','V': 'VAL'
        }

        with open(pdb_out, 'w') as f:
            for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords), start=1):
                aa3 = aa_map.get(aa.upper(), "UNK")
                f.write(
                    f"ATOM  {i:5d}  CA  {aa3} {chain_id}{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")

        print(f"[INFO] 已写出 CA-only PDB 文件: {pdb_out}")

    def build_full_model_with_modeller(self, ca_pdb, out_pdb="full_model.pdb", chain_id="A"):
        """
        使用 Modeller 从 CA-only PDB 生成全原子模型.
        """
        # 写出对齐文件
        # 这里简单写一个 "protein.ali" 对齐
        sequence = self.extract_sequence_from_ca_pdb(ca_pdb, chain_id=chain_id)
        ali_file = os.path.join(os.path.dirname(ca_pdb), "protein.ali")

        # Modeller 需要一个对齐序列, 以 '>' 开头, 并以 '*' 结尾
        seq_str = "".join(sequence) + "*"

        start_res = 1
        end_res = len(sequence)

        with open(ali_file, "w") as f:
            f.write(f">P1;ca_model\n")
            f.write(f"structureX:ca_model:{start_res}:{chain_id}:{end_res}:{chain_id}::::\n")
            f.write(seq_str + "\n\n")
            f.write(f">P1;full_model\n")
            f.write(f"sequence:full_model:{start_res}:{chain_id}:{end_res}:{chain_id}::::\n")
            f.write(seq_str + "\n")

        # 运行 Modeller
        log.none()  # 关闭 Modeller 日志 (或用 log.verbose() 查看详细输出)
        env = environ()
        env.io.atom_files_directory = [os.path.dirname(ca_pdb)]

        # 读入对齐
        aln = alignment(env)
        mdl = model(env, file=ca_pdb, model_segment=(f"FIRST:{chain_id}", f"LAST:{chain_id}"))
        aln.append_model(mdl, align_codes='ca_model', atom_files=ca_pdb)
        aln.append(file=ali_file, align_codes='full_model')
        aln.align2d()

        class MyModel(automodel):
            def special_patches(self, aln_):
                # 可在这里自定义对链的命名或其他操作
                self.rename_segments(segment_ids=[chain_id])

        a = MyModel(env,
                    alnfile=ali_file,
                    knowns='ca_model',
                    sequence='full_model',
                    assess_methods=(assess.DOPE, assess.GA341))
        a.starting_model = 1
        a.ending_model = 1
        a.make()

        # Modeller 默认输出 "full_model.B99990001.pdb" 之类的文件
        # 找到它并重命名
        generated = f"full_model.B99990001.pdb"
        if os.path.exists(generated):
            shutil.move(generated, out_pdb)
            print(f"[INFO] 已生成全原子模型: {out_pdb}")
        else:
            raise FileNotFoundError("Modeller 输出文件未找到，可能出错。")

    def extract_sequence_from_ca_pdb(self, pdb_file, chain_id="A"):
        """
        从 CA-only PDB 提取氨基酸序列(三字母 -> 单字母).
        注意这里假设 PDB 中只有 CA 原子，每条记录都是同一个链 ID.
        """
        mapping_3to1 = {
            'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
            'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
            'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
        }
        seq = []
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("ca_model", pdb_file)

        for model in structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    resname = residue.resname.strip()
                    seq.append(mapping_3to1.get(resname, 'X'))  # 不识别则记为 X

        return seq

    ################################
    #   Step 2: 处理 AF3 CIF 文件   #
    ################################

    def convert_cif_to_pdbqt(self, cif_file, out_pdbqt):
        """
        将 AF3 生成的 CIF 文件转为 PDBQT:
          1) 用 BioPython 解析 CIF
          2) 写出临时 PDB
          3) 用 Open Babel 转换为 PDBQT（加氢、分配电荷）
        """
        parser = MMCIFParser(QUIET=True)
        structure_id = os.path.splitext(os.path.basename(cif_file))[0]
        structure = parser.get_structure(structure_id, cif_file)

        # 临时写出 pdb
        tmp_pdb = cif_file.replace(".cif", "_temp.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_pdb)

        # 用 Open Babel 转换 pdb -> pdbqt
        self.run_obabel_convert(tmp_pdb, out_pdbqt)
        print(f"[INFO] CIF -> PDBQT 转换完成: {out_pdbqt}")

        # 清理临时文件
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)

    ############################
    #   Step 3: 处理 MOL2 配体   #
    ############################

    def translate_mol2_to_origin(self, mol2_in, mol2_out):
        """
        读取 MOL2 ATOM 坐标，计算几何中心并平移至原点，写出新的 MOL2。
        """
        atoms = []
        with open(mol2_in, 'r') as f:
            lines = f.readlines()

        atom_section = False
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                continue
            elif line.startswith("@<TRIPOS>") and atom_section:
                atom_section = False
                break
            if atom_section:
                parts = line.split()
                if len(parts) < 6:
                    continue
                # parts: [atom_id, atom_name, x, y, z, atom_type, ...]
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atoms.append((x, y, z))
                except ValueError:
                    pass

        if not atoms:
            raise ValueError("未在 MOL2 中找到原子坐标，无法平移。")

        center_x = sum(a[0] for a in atoms) / len(atoms)
        center_y = sum(a[1] for a in atoms) / len(atoms)
        center_z = sum(a[2] for a in atoms) / len(atoms)
        print(f"[INFO] 配体几何中心: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")

        # 将 lines 中的坐标减去中心
        atom_section = False
        new_lines = []
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                new_lines.append(line)
                continue
            elif line.startswith("@<TRIPOS>") and atom_section:
                atom_section = False
                new_lines.append(line)
                continue

            if atom_section:
                parts = line.split()
                if len(parts) < 6:
                    new_lines.append(line)
                    continue
                # 平移
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    x -= center_x
                    y -= center_y
                    z -= center_z
                    # 重构行
                    # 例如: atom_id, atom_name, x, y, z, atom_type, ...
                    # 保持其余字段不变
                    new_line = (
                        f"{parts[0]:>7} {parts[1]:<9} {x:>10.4f} {y:>10.4f} {z:>10.4f} {parts[5]}"
                    )
                    if len(parts) > 6:
                        extra = " ".join(parts[6:])
                        new_line += f" {extra}"
                    new_line += "\n"
                    new_lines.append(new_line)
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        with open(mol2_out, 'w') as f:
            f.writelines(new_lines)
        print(f"[INFO] 已平移 MOL2 至原点: {mol2_out}")

    ############################
    #   Step 4: 转换为 PDBQT    #
    ############################

    def run_obabel_convert(self, input_file, output_file, add_h=True):
        """
        用 Open Babel 将 input_file 转换为 output_file，并加氢/分配电荷.
        这里以 pdb/pdbqt/mol2 为例.
        """
        if not self.check_tool("obabel"):
            raise EnvironmentError("未检测到 obabel，请确认已安装并在 PATH 中。")

        cmd = ["obabel", input_file, "-O", output_file,
               "--partialcharge", "gasteiger"]
        if add_h:
            cmd.append("-h")
        # -xr: 移除现有氢 ？可根据需要添加/移除
        # 例如: cmd += ["-xr"]

        subprocess.run(cmd, check=True)
        print(f"[INFO] Open Babel 转换完成: {input_file} -> {output_file}")

    def check_tool(self, tool_name):
        """
        检查系统中是否能通过 `type tool_name` 找到指定可执行文件。
        """
        return subprocess.call(f"type {tool_name}",
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE) == 0

    ############################
    #   Step 5: 调用 AutoDock Vina 进行对接  #
    ############################

    def run_autodock_vina(self, receptor_pdbqt, ligand_pdbqt, out_pdbqt, log_file,
                          center=None, box_size=18, exhaustiveness=8):
        """
        用 AutoDock Vina 进行对接:
        :param center: (x, y, z)，如不指定则以 0,0,0
        :param box_size: 盒子大小(单边长度)
        :param exhaustiveness: 计算穷尽度
        """
        if not self.check_tool("vina"):
            raise EnvironmentError("未检测到 vina，请确认已安装并在 PATH 中。")

        if center is None:
            center = (0.0, 0.0, 0.0)

        cx, cy, cz = center
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--out", out_pdbqt,
            "--log", log_file,
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--center_z", str(cz),
            "--size_x", str(box_size),
            "--size_y", str(box_size),
            "--size_z", str(box_size),
            "--exhaustiveness", str(exhaustiveness),
            "--seed", str(self.seed)
        ]

        subprocess.run(cmd, check=True)
        print(f"[INFO] 对接完成，结果输出: {out_pdbqt}, 日志: {log_file}")

    ############################
    #   Step 6: 整体流程入口     #
    ############################

    def run_pipeline(self):
        """
        主入口：遍历 grouped_result_dir 下的所有 chain_* 文件夹，
        并在每个文件夹里寻找多个子目录(如 1fkf/、3ckz/ 等)。
        在每个子目录中：
          - 找到 .xyz 文件(可能多个)
          - 找到可能的 .cif 文件(AI/AF3结果)
          - 找到对应的配体 .mol2 (在 selected_dir/<pdb_id>/xxx_ligand.mol2)
          - 执行全流程
        """
        chain_folders = sorted(glob.glob(os.path.join(self.grouped_result_dir, "chain_*")))
        for chain_folder in chain_folders:
            pdb_folders = [f for f in os.listdir(chain_folder)
                           if os.path.isdir(os.path.join(chain_folder, f))]
            for pdb_id in pdb_folders:
                pdb_id_path = os.path.join(chain_folder, pdb_id)
                # 找 xyz
                xyz_files = glob.glob(os.path.join(pdb_id_path, "*.xyz"))
                if not xyz_files:
                    print(f"[INFO] 跳过 {pdb_id_path}，未发现 xyz 文件。")
                    continue

                # 找配体
                ligand_mol2 = os.path.join(self.selected_dir, pdb_id, f"{pdb_id}_ligand.mol2")
                if not os.path.exists(ligand_mol2):
                    print(f"[WARNING] 未找到配体 {ligand_mol2}，跳过对接。")
                    continue

                # 先将配体平移至原点 (可选)
                ligand_mol2_trans = ligand_mol2.replace(".mol2", "_trans.mol2")
                self.translate_mol2_to_origin(ligand_mol2, ligand_mol2_trans)
                # 转成 pdbqt
                ligand_pdbqt = ligand_mol2_trans.replace(".mol2", ".pdbqt")
                self.run_obabel_convert(ligand_mol2_trans, ligand_pdbqt)

                # 看看有没有 AF3 的 .cif
                cif_files = glob.glob(os.path.join(pdb_id_path, "*.cif"))
                if cif_files:
                    af3_cif = cif_files[0]  # 如果有多个，简单取第一个
                else:
                    af3_cif = None

                for xyz_file in xyz_files:
                    try:
                        print(f"\n[INFO] 处理 XYZ: {xyz_file}")
                        # 1) xyz -> CA pdb
                        seq, scaled_coords = self.parse_and_scale_xyz(xyz_file)
                        ca_pdb = xyz_file.replace(".xyz", "_ca.pdb")
                        self.write_ca_pdb(seq, scaled_coords, ca_pdb)

                        # 2) CA pdb -> full model
                        full_model_pdb = xyz_file.replace(".xyz", "_full.pdb")
                        self.build_full_model_with_modeller(ca_pdb, out_pdb=full_model_pdb)

                        # 3) full_model -> pdbqt
                        full_model_pdbqt = full_model_pdb.replace(".pdb", ".pdbqt")
                        self.run_obabel_convert(full_model_pdb, full_model_pdbqt)

                        # 4) docking
                        # 4.1) 量子模型 docking
                        out_prefix = f"{pdb_id}_{os.path.splitext(os.path.basename(xyz_file))[0]}"
                        out_dir = os.path.join(self.output_dir, out_prefix + "_quantum")
                        os.makedirs(out_dir, exist_ok=True)

                        quantum_out = os.path.join(out_dir, "docking_output.pdbqt")
                        quantum_log = os.path.join(out_dir, "docking_log.txt")
                        # 这里 center 默认为(0,0,0)，如果需要可自行计算蛋白或配体的中心
                        self.run_autodock_vina(
                            receptor_pdbqt=full_model_pdbqt,
                            ligand_pdbqt=ligand_pdbqt,
                            out_pdbqt=quantum_out,
                            log_file=quantum_log,
                            center=(0,0,0),
                            box_size=18,
                            exhaustiveness=8
                        )

                        # 4.2) AF3 cif docking (可选)
                        if af3_cif:
                            af3_pdbqt = af3_cif.replace(".cif", ".pdbqt")
                            self.convert_cif_to_pdbqt(af3_cif, af3_pdbqt)

                            out_dir_af3 = os.path.join(self.output_dir, out_prefix + "_af3")
                            os.makedirs(out_dir_af3, exist_ok=True)

                            af3_out = os.path.join(out_dir_af3, "docking_output.pdbqt")
                            af3_log = os.path.join(out_dir_af3, "docking_log.txt")

                            self.run_autodock_vina(
                                receptor_pdbqt=af3_pdbqt,
                                ligand_pdbqt=ligand_pdbqt,
                                out_pdbqt=af3_out,
                                log_file=af3_log,
                                center=(0,0,0),
                                box_size=18,
                                exhaustiveness=8
                            )
                            print(f"[INFO] AF3 docking 完成: {af3_cif}")
                        else:
                            print("[INFO] 未检测到 AF3 cif 文件，跳过 AF3 docking。")

                    except Exception as e:
                        print(f"[ERROR] 处理 {xyz_file} 时出错: {e}")


def main():
    """
    简单演示如何调用 UnifiedDockingPipeline 完成对新预测集的 docking 流程。
    根据实际情况修改参数、目录等。
    """
    grouped_result_dir = "data/grouped_result"  # 存放 chain_5, chain_6, ...
    selected_dir = "data/selected"             # 存放配体 .mol2
    output_dir = "docking_output"              # docking 结果输出

    pipeline = UnifiedDockingPipeline(
        grouped_result_dir=grouped_result_dir,
        selected_dir=selected_dir,
        output_dir=output_dir,
        seed=42
    )
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

