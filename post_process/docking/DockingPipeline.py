# --*-- conding:utf-8 --*--
# @Time : 3/20/25 8:39 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : DockingPipeline.py

import os
import subprocess
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from modeller import environ, log, alignment, model
from modeller.automodel import automodel, assess
import random

class DockingPipeline:

    def __init__(self, output_dir):
        """Initialize the docking pipeline, requiring only the output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.protein_pdbqt = os.path.join(output_dir, 'protein.pdbqt')
        self.receptor_pdbqt = os.path.join(output_dir, 'receptor.pdbqt')
        self.ligand_pdbqt = os.path.join(output_dir, 'ligand.pdbqt')

    ### Process XYZ file and generate protein PDBQT
    def process_xyz_to_pdbqt(self, xyz_file):
        """Generate protein PDBQT file from XYZ file"""
        sequence, coords = self._read_xyz(xyz_file)
        scaled_coords = self._adjust_scale(coords, target_distance=3.8)
        ca_pdb_content = self._generate_ca_pdb_content(sequence, scaled_coords)
        full_pdb_content = self._generate_full_model(ca_pdb_content)
        self._convert_to_pdbqt_with_obabel(full_pdb_content, self.protein_pdbqt, is_content=True)
        print(f"Protein PDBQT generated: {self.protein_pdbqt}")

    ### Process CIF file and generate receptor PDBQT
    def process_cif_to_pdbqt(self, cif_file):
        """Generate receptor PDBQT file from CIF file"""
        translated_pdb_content = self._translate_cif_to_pdb(cif_file)
        self._convert_to_pdbqt_with_obabel(translated_pdb_content, self.receptor_pdbqt, is_content=True)
        print(f"Receptor PDBQT generated: {self.receptor_pdbqt}")

    ### Process MOL2 file and generate ligand PDBQT
    def process_mol2_to_pdbqt(self, mol2_file):
        """Generate ligand PDBQT file from MOL2 file"""
        translated_mol2_content = self._translate_mol2(mol2_file)
        self._convert_to_pdbqt_with_obabel(translated_mol2_content, self.ligand_pdbqt, is_content=True,
                                           input_format='mol2')
        print(f"Ligand PDBQT generated: {self.ligand_pdbqt}")

    ### Quantum docking
    def dock_quantum(self, xyz_file, mol2_file, output_dir, log_file_name, seed):
        """Perform docking for quantum prediction results"""
        docking_output_dir = os.path.join(output_dir, 'docking_results')
        os.makedirs(docking_output_dir, exist_ok=True)
        log_file = os.path.join(docking_output_dir, log_file_name)

        # Process input files
        self.process_xyz_to_pdbqt(xyz_file)
        self.process_mol2_to_pdbqt(mol2_file)

        # Perform docking
        center = self._calculate_center_of_mass(self.ligand_pdbqt)
        output_file = os.path.join(docking_output_dir, 'docking_output.pdbqt')
        self._run_vina(self.protein_pdbqt, self.ligand_pdbqt, output_file, log_file, center, seed)
        scores = self._parse_docking_results(log_file)
        print(f"Quantum docking completed with seed {seed}.")
        return scores

    ### AF3 docking
    def dock_af3(self, cif_file, mol2_file, output_dir, log_file_name, seed):
        """Perform docking for AF3 prediction results"""
        docking_output_dir = os.path.join(output_dir, 'docking_results')
        os.makedirs(docking_output_dir, exist_ok=True)
        log_file = os.path.join(docking_output_dir, log_file_name)

        # Process input files
        self.process_cif_to_pdbqt(cif_file)
        self.process_mol2_to_pdbqt(mol2_file)

        # Perform docking
        center = self._calculate_center_of_mass(self.ligand_pdbqt)
        output_file = os.path.join(docking_output_dir, 'docking_output.pdbqt')
        self._run_vina(self.receptor_pdbqt, self.ligand_pdbqt, output_file, log_file, center, seed)
        scores = self._parse_docking_results(log_file)
        print(f"AF3 docking completed with seed {seed}.")
        return scores

    def _read_xyz(self, xyz_file):
        sequence = []
        coordinates = []
        with open(xyz_file, 'r') as f:
            lines = f.readlines()[2:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 4:
                    sequence.append(parts[0])
                    coordinates.append(tuple(map(float, parts[1:])))
        return sequence, coordinates

    def _adjust_scale(self, coords, target_distance=3.8):
        distances = [np.sqrt(sum((c2[i] - c1[i]) ** 2 for i in range(3)))
                     for c1, c2 in zip(coords[:-1], coords[1:])]
        scale_factor = target_distance / np.mean(distances)
        return [(x * scale_factor, y * scale_factor, z * scale_factor) for x, y, z in coords]

    def _generate_ca_pdb_content(self, sequence, coords):
        res_map = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                   'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                   'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                   'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
        lines = []
        for i, (res, (x, y, z)) in enumerate(zip(sequence, coords), 1):
            res_name = res_map.get(res.upper(), 'UNK')
            line = f"ATOM  {i:5d}  CA  {res_name} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            lines.append(line)
        lines.append("END\n")
        return ''.join(lines)

    def _generate_full_model(self, ca_pdb_content):
        env = environ()
        env.io.atom_files_directory = [self.output_dir]
        log.none()
        temp_ca_pdb = os.path.join(self.output_dir, 'temp_ca.pdb')
        with open(temp_ca_pdb, 'w') as f:
            f.write(ca_pdb_content)

        # 从每一行提取残基名
        lines = ca_pdb_content.splitlines()[:-1]  # 排除 "END" 行
        sequence = ''
        res_map_reverse = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        for line in lines:
            res_name = line[17:20].strip()  # 提取第 18-20 列的残基名
            sequence += res_map_reverse.get(res_name, 'X')  # 转换为单字母代码，未知残基用 'X'

        # 生成对齐文件
        ali_content = f">P1;ca_model\nstructureX:temp_ca.pdb:1:A:{len(sequence)}:A::::\n{sequence}*\n\n>P1;protein_full\nsequence:protein_full:1:A:{len(sequence)}:A::::\n{sequence}*\n"
        temp_ali = os.path.join(self.output_dir, 'temp.ali')
        with open(temp_ali, 'w') as f:
            f.write(ali_content)

        # 以下保持不变
        aln = alignment(env)
        mdl = model(env, file=temp_ca_pdb, model_segment=('FIRST:A', 'LAST:A'))
        aln.append_model(mdl, align_codes='ca_model', atom_files=temp_ca_pdb)
        aln.append(file=temp_ali, align_codes='protein_full')
        aln.align2d()

        class MyModel(automodel):
            def special_patches(self, aln):
                self.rename_segments(segment_ids=['A'])

        a = MyModel(env, alnfile=temp_ali, knowns='ca_model', sequence='protein_full', assess_methods=(assess.DOPE,))
        a.starting_model = 1
        a.ending_model = 1
        a.make()
        output_model = os.path.join(self.output_dir, 'protein_full.B99990001.pdb')
        with open(output_model, 'r') as f:
            full_pdb_content = f.read()
        os.remove(temp_ca_pdb)
        os.remove(temp_ali)
        os.remove(output_model)
        return full_pdb_content

    def _translate_cif_to_pdb(self, cif_file):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file)
        center = np.mean([atom.coord for atom in structure.get_atoms()], axis=0)
        for atom in structure.get_atoms():
            atom.coord -= center
        from io import StringIO
        from Bio.PDB.PDBIO import PDBIO
        io = PDBIO()
        io.set_structure(structure)
        output = StringIO()
        io.save(output)
        return output.getvalue()

    def _translate_mol2(self, mol2_file):
        atoms = []
        with open(mol2_file, 'r') as f:
            lines = f.readlines()
            atom_section = False
            for line in lines:
                if line.startswith('@<TRIPOS>ATOM'):
                    atom_section = True
                    continue
                elif line.startswith('@<TRIPOS>') and atom_section:
                    break
                if atom_section and len(line.split()) >= 6:
                    parts = line.split()
                    atoms.append({'x': float(parts[2]), 'y': float(parts[3]), 'z': float(parts[4]), 'line': line})
        center = np.mean([[a['x'], a['y'], a['z']] for a in atoms], axis=0)
        translated_lines = []
        for atom in atoms:
            x, y, z = atom['x'] - center[0], atom['y'] - center[1], atom['z'] - center[2]
            parts = atom['line'].split()
            translated_lines.append(
                f"{parts[0]:>6} {parts[1]:<10} {x:>8.3f} {y:>8.3f} {z:>8.3f} {' '.join(parts[5:])}\n")
        with open(mol2_file, 'r') as f:
            original = f.read().split('@<TRIPOS>ATOM')[0] + '@<TRIPOS>ATOM\n' + ''.join(translated_lines) + '@<TRIPOS>'
        return original

    def _convert_to_pdbqt_with_obabel(self, content, output_file, is_content=True, input_format='pdb'):
        if not self._is_tool_available('obabel'):
            raise EnvironmentError("Open Babel is not installed.")
        cmd = ['obabel', f'-i{input_format}', '-opdbqt', output_file, '--partialcharge', 'gasteiger', '-h', '-xr']
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        process.communicate(input=content if is_content else None)

    def _is_tool_available(self, tool):
        return subprocess.call(f"type {tool}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

    def _calculate_center_of_mass(self, pdbqt_file):
        coords = []
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        center = tuple(sum(c) / len(coords) for c in zip(*coords))
        return center

    def _run_vina(self, receptor, ligand, output_file, log_file, center, seed):
        cmd = [
            'vina', '--receptor', receptor, '--ligand', ligand,
            '--out', output_file, '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z',
            str(center[2]),
            '--size_x', '18', '--size_y', '18', '--size_z', '18', '--exhaustiveness', '16', '--seed', str(seed)
        ]
        with open(log_file, 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=log, check=True)

    def _parse_docking_results(self, log_file):
        scores = []
        with open(log_file, 'r') as f:
            for line in f:
                if 'REMARK VINA RESULT:' in line:
                    scores.append(float(line.split()[3]))
        return scores





