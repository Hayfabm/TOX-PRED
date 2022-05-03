from collections import defaultdict
from itertools import count
import os
import pickle
import numpy as np
from rdkit import Chem
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# "Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)"


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def dump_dictionary(dictionary, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict(dictionary), f)


radius = 2
train_seq_label = pd.read_csv("train.csv")
train_sequence = train_seq_label["sequence"].values  # 4098
train_label = train_seq_label["label"].values
# 4098

print(len(train_sequence))
print(len(train_label))
"""
val_seq_label = pd.read_csv("Data/val_set.csv")
val_sequence = val_seq_label["sequence"].values  # 878
val_label = val_seq_label["label"].values  # 878


test_seq_label = pd.read_csv("Data/test_set.csv")
test_sequence = test_seq_label["sequence"]  # 878
test_label = test_seq_label["label"].values  # 878
"""

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

compounds_train, adjacencies_train, labels_train = [], [], []
for i in range(len(train_sequence)):

    sequence = train_sequence[i]
    label = train_label[i]

    mol = Chem.rdmolfiles.MolFromSequence(sequence)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    compounds_train.append(fingerprints)
    # print(compounds_train)
    adjacency = create_adjacency(mol)
    adjacencies_train.append(adjacency)
    # print(adjacencies_train)

    labels_train.append(np.array([float(label)]))
    # print(labels_train)

print(len(compounds_train))
print(len(adjacencies_train))
print(len(labels_train))


"""
compounds_val, adjacencies_val, labels_val = [], [], []
for i in range(len(val_sequence)):

    sequence = val_sequence[i]
    label = val_label[i]

    mol = Chem.rdmolfiles.MolFromSequence(sequence)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    compounds_val.append(fingerprints)

    adjacency = create_adjacency(mol)
    adjacencies_val.append(adjacency)

    labels_val.append(np.array([float(label)]))

print(len(compounds_val))
print(len(adjacencies_val))
print(len(labels_val))
"""

"""
compounds_test, adjacencies_test, labels_test = [], [], []
for i in range(len(test_sequence)):

    sequence = test_sequence[i]
    label = test_label[i]

    mol = Chem.rdmolfiles.MolFromSequence(sequence)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    compounds_test.append(fingerprints)

    adjacency = create_adjacency(mol)
    adjacencies_test.append(adjacency)

    labels_test.append(np.array([float(label)]))

print(len(compounds_test))
print(len(adjacencies_test))
print(len(labels_test))

dir_input = "Data/rdkit/radius2/"
os.makedirs(dir_input, exist_ok=True)

np.save(dir_input + "compounds", compounds)
np.save(dir_input + "adjacencies", adjacencies)
np.save(dir_input + "labels", labels)
dump_dictionary(fingerprint_dict, dir_input + "fingerprint_dict.pickle")

print("The preprocess of dataset has finished!")

"""
