from Bio.PDB import PDBParser
import os
import numpy as np

# https://biopython.org/docs/1.75/api/Bio.PDB.Atom.html

def feature_extraction():
    return feature_atom_residue()
    # return feature_residue()
    # return feature_atom()

def feature_atom_residue():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_residue_names = set()
    file_names = []

    # 第一步：遍历一次以收集所有不同的残基名称
    for file in os.listdir("./data/SCOP40mini"):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        file_names.append(file)
        for atom, residue in zip(structure.get_atoms(), structure.get_residues()):
            all_residue_names.add(atom.get_name()+" "+residue.get_resname())
        # for stru in structure:
        #     for chain in stru:
        #         for residue in chain:
        #             all_residue_names.add(chain.get_name()+" "+residue.get_resname())
            # all_residue_names.add(residue.get_name()+" "+residue.get_resname())
    
    # 创建一个列表，按照排序的残基名称
    residue_names_sorted = sorted(list(all_residue_names))

    print("Residue names:", residue_names_sorted)

    # 初始化一个空的NumPy矩阵
    residue_matrix = np.zeros((len(file_names), len(residue_names_sorted)))

    # 第二步：再次遍历每个文件，更新矩阵中相应的残基数量
    for file_index, file in enumerate(os.listdir("./data/SCOP40mini")):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        residue_counts = dict.fromkeys(residue_names_sorted, 0)
        
        # for residue in structure.get_residues():
        #     residue_counts[residue.get_name()+" "+residue.get_resname()] += 1
        for atom, residue in zip(structure.get_atoms(), structure.get_residues()):
            residue_counts[atom.get_name()+" "+residue.get_resname()] += 1
        
        # 更新矩阵
        for residue_index, residue_name in enumerate(residue_names_sorted):
            residue_matrix[file_index, residue_index] = residue_counts[residue_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {len(file_names)}")

    print("Feature extraction completed.")  # 打印函数完成消息

    return residue_matrix, residue_names_sorted

def feature_residue():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_residue_names = set()
    file_names = []

    # 第一步：遍历一次以收集所有不同的残基名称
    for file in os.listdir("./data/SCOP40mini"):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        file_names.append(file)
        for residue in structure.get_residues():
            all_residue_names.add(residue.get_resname())
    
    # 创建一个列表，按照排序的残基名称
    residue_names_sorted = sorted(list(all_residue_names))

    print("Residue names:", residue_names_sorted)

    # 初始化一个空的NumPy矩阵
    residue_matrix = np.zeros((len(file_names), len(residue_names_sorted)))

    # 第二步：再次遍历每个文件，更新矩阵中相应的残基数量
    for file_index, file in enumerate(os.listdir("./data/SCOP40mini")):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        residue_counts = dict.fromkeys(residue_names_sorted, 0)
        
        for residue in structure.get_residues():
            residue_counts[residue.get_resname()] += 1
        
        # 更新矩阵
        for residue_index, residue_name in enumerate(residue_names_sorted):
            residue_matrix[file_index, residue_index] = residue_counts[residue_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {len(file_names)}")

    print("Feature extraction completed.")  # 打印函数完成消息

    return residue_matrix, residue_names_sorted

def feature_atom():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_atom_names = set()
    file_names = []

    # 第一步：遍历一次以收集所有不同的原子名称
    for file in os.listdir("./data/SCOP40mini"):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        file_names.append(file)
        for atom in structure.get_atoms():
            all_atom_names.add(atom.get_name())

    # 创建一个列表，按照排序的原子名称
    atom_names_sorted = sorted(list(all_atom_names))

    print("Atom names:", atom_names_sorted)
    
    # 初始化一个空的NumPy矩阵
    atom_matrix = np.zeros((len(file_names), len(atom_names_sorted)))

    # 第二步：再次遍历每个文件，更新矩阵中相应的原子数量
    for file_index, file in enumerate(os.listdir("./data/SCOP40mini")):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        atom_counts = dict.fromkeys(atom_names_sorted, 0)
        
        for atom in structure.get_atoms():
            atom_counts[atom.get_name()] += 1
        
        # 更新矩阵
        for atom_index, atom_name in enumerate(atom_names_sorted):
            atom_matrix[file_index, atom_index] = atom_counts[atom_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {len(file_names)}")
        
    print("Feature extraction completed.")  # 打印函数完成消息
 
    return atom_matrix, atom_names_sorted

if __name__ == "__main__":
    # 调用函数并获取返回的矩阵和原子名称列表
    atom_matrix, atom_names_sorted = feature_extraction()

    # 打印结果以验证
    print("Atom names:", atom_names_sorted)
    print("Matrix shape:", atom_matrix.shape)

    # 打印矩阵的一小部分或特定行列以查看具体值
    
# '1H', '1HA', '1HB', '1HD', '1HD1', '1HD2', '1HE', '1HE2', '1HG', '1HG1', '1HG2', '1HH1', '1HH2', '1HN', '1HZ', '2H', '2HA', '2HB', '2HD', '2HD1', '2HD2', '2HE', '2HE2', '2HG', '2HG1', 
# '2HG2', '2HH1', '2HH2', '2HN', '2HZ', '3H', '3HB', '3HD1', '3HD2', '3HE', '3HG1', '3HG2', '3HZ', 'AD1', 'AD2', 'AE1', 'AE2', 'AS', 'C', 'C1', 'C2', 'C2A', 'C3', 'C4', 'C4A', 'C5', 'C5A', 'C6', 'CA', 'CA1', 'CB', 'CB1', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CF', 'CG', 'CG1', 'CG2', 'CH1', 'CH2', 'CI', 'CJ', 'CM', 'CN', 'CX', 'CZ', 'CZ2', 'CZ3', 'H', 'H1', 'H2', 'H3', 'H4A', 'HA', 'HB', 'HB1', 'HB2', 'HD1', 'HD2', 'HE', 'HE1', 'HE2', 'HE3', 'HG', 'HG1', 'HG2', 'HH', 'HH2', 'HN', 'HN1', 'HNZ', 'HO', 'HZ', 'HZ1', 'HZ2', 'HZ3', 'N', 'N1', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 
# 'NH2', 'NI', 'NX', 'NZ', 'O', 'O1', 'O1P', 'O2', 'O2P', 'O3', 'O3P', 'O4P', 'OD', 'OD1', 'OD2', 'OD3', 'OE', 'OE1', 'OE2', 'OF', 'OG', 'OG1', 'OH', 'OJ1', 'OJ2', 'OX1', 'OX2', 'OXT', 'P', 'QA', 'QB', 'QD', 'QD1', 'QD2', 'QE', 'QE2', 'QG', 'QG1', 'QG2', 'QH1', 'QH2', 'QQD', 'QQG', 'QR', 'QZ', 'SD', 'SE', 'SEG', 'SG'

# '143', '5HP', 'ALA', 'ARG', 'ASN', 'ASP', 'CAS', 'CEA', 'CME', 'CSD', 'CSO', 'CSS', 'CSW', 'CYG', 'CYS', 'CZZ', 'FME', 'GLN', 'GLU', 'GLY', 'HIC', 'HIS', 'ILE', 'KCX', 'LEU', 'LLP', 'LYS', 'MET', 'MLY', 'MLZ', 'MSE', 'OCS', 'PCA', 'PHE', 'PRO', 'SEC', 'SER', 'THR', 'TRP', 'TYR', 'UNK', 'VAL'