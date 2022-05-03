"""utils file"""
from typing import List, Tuple
import os
import numpy as np
import pandas as pd


def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequence"]), list(dataset["label"])


# generate PC6 table
def amino_encode_table_6():
    """
    path = os.path.join(os.path.dirname(__file__), "6-pc.csv")
    df = pd.read_csv(path, sep=" ", index_col=0)
    H1 = (df["H1"] - np.mean(df["H1"])) / (np.std(df["H1"], ddof=1))
    print(H1)
    V = (df["V"] - np.mean(df["V"])) / (np.std(df["V"], ddof=1))
    print(V)
    P1 = (df["P1"] - np.mean(df["P1"])) / (np.std(df["P1"], ddof=1))
    Pl = (df["Pl"] - np.mean(df["Pl"])) / (np.std(df["Pl"], ddof=1))
    PKa = (df["PKa"] - np.mean(df["PKa"])) / (np.std(df["PKa"], ddof=1))
    NCI = (df["NCI"] - np.mean(df["NCI"])) / (np.std(df["NCI"], ddof=1))
    c = np.array([H1, V, P1, Pl, PKa, NCI])
    print(c)
    """
    H1 = [
        6.20140363e-01,
        2.90065654e-01,
        -9.00203753e-01,
        -7.40167531e-01,
        1.19026941e00,
        4.80108668e-01,
        -4.00090557e-01,
        1.38031242e00,
        -1.50033959e00,
        1.06023998e00,
        6.40144891e-01,
        -7.80176586e-01,
        1.20027167e-01,
        -8.50192434e-01,
        -2.53057277e00,
        -1.80040751e-01,
        -5.00113196e-02,
        1.08024450e00,
        8.10183378e-01,
        2.60058862e-01,
    ]
    V = [
        -1.23870736e00,
        -7.68468803e-01,
        -8.94965725e-01,
        -2.89980445e-01,
        1.18123376e00,
        -1.99493896e00,
        1.77508181e-01,
        5.76248480e-01,
        7.54994131e-01,
        5.76248480e-01,
        5.92748078e-01,
        -3.80728237e-01,
        -8.42716997e-01,
        2.24257044e-01,
        8.92490786e-01,
        -1.18920857e00,
        -5.84223286e-01,
        -2.87368008e-02,
        2.00621369e00,
        1.23073256e00,
    ]
    P1 = [
        -8.36274310e-02,
        -1.04998886e00,
        1.73759218e00,
        1.47741795e00,
        -1.16149210e00,
        2.50882293e-01,
        7.71230752e-01,
        -1.16149210e00,
        1.10574048e00,
        -1.27299534e00,
        -9.75653362e-01,
        1.21724372e00,
        -1.20795178e-01,
        8.08398500e-01,
        8.08398500e-01,
        3.25217787e-01,
        1.02211305e-01,
        -9.01317867e-01,
        -1.08715660e00,
        -7.89814626e-01,
    ]
    Pl = [
        -1.38471134e-02,
        -5.39472234e-01,
        -1.83940533e00,
        -1.58507059e00,
        -3.07745030e-01,
        -3.08027624e-02,
        8.84802286e-01,
        -2.54334736e-03,
        2.09995713e00,
        -2.51508794e-02,
        -1.60796072e-01,
        -3.47308211e-01,
        1.55709377e-01,
        -2.11663019e-01,
        2.67644920e00,
        -1.94707370e-01,
        -2.39922434e-01,
        -3.64546455e-02,
        -7.60178266e-02,
        -2.06011136e-01,
    ]
    PKa = [
        7.00138152e-01,
        -9.32085761e-01,
        -1.27571185e00,
        5.58392391e-02,
        -1.49047815e00,
        7.00138152e-01,
        -1.53343141e00,
        7.86044674e-01,
        1.28859783e-02,
        7.86044674e-01,
        4.42418587e-01,
        -6.74366195e-01,
        -8.03225978e-01,
        -3.00672826e-02,
        -3.00672826e-02,
        1.41745761e-01,
        -3.73693369e-01,
        6.14231630e-01,
        2.80484793e00,
        9.87925000e-02,
    ]
    NCI = [
        -4.41192734e-01,
        -1.11480899e00,
        -9.18093420e-01,
        -4.47114195e-01,
        2.58337226e-02,
        2.20216317e00,
        -7.16148504e-01,
        -2.19037946e-01,
        -2.79375334e-01,
        2.43005254e-01,
        -5.10466146e-01,
        -4.68800588e-01,
        3.13235559e00,
        2.05154041e-01,
        1.18654558e-01,
        -4.80566609e-01,
        -5.00176645e-01,
        3.25013654e-01,
        3.23704012e-02,
        -1.88769279e-01,
    ]
    c = np.array([H1, V, P1, Pl, PKa, NCI])
    # print(c)
    amino = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    table = {}
    for index, key in enumerate(amino):
        table[key] = list(c[0:6, index])
    table["X"] = [0, 0, 0, 0, 0, 0]
    return table


# sequence padding (token:'X')
def padding_seq(seq, length=1000, pad_value="X"):
    data = []
    for key in range(len(seq)):
        sequence = seq[key]
        if len(sequence) < length:
            sequence = [sequence + pad_value * (length - len(sequence))]
            data.append(sequence)

    return data


# PC encoding
def PC_encoding(data):
    table = amino_encode_table_6()
    dat = []
    for key in data:
        integer_encoded = []
        for amino in key[0]:
            integer_encoded.append(table[amino])
            dat.append(integer_encoded)

    return dat


def encoding(data, method):
    # method = integer or onehot
    dat = {}
    for key in data.keys():
        integer_encoded = []
        for amino in list(data[key]):
            integer_encoded.append(table_dict[method][amino])
        dat[key] = integer_encoded
    return dat


# PC6 (input: fasta)
def PC_6(seq, length=1000):
    data = padding_seq(seq, length)
    dat = PC_encoding(data)
    return dat
