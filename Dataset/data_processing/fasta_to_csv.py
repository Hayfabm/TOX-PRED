import pandas as pd


from Bio import SeqIO
import sys

"""
sys.stdout = open("test_set.csv", "w")
for re in SeqIO.parse("test_set.fa", "fasta"):
    print(">{}\t{}".format(str(re.description), re.seq))


sys.stdout.close()

"""

df = pd.read_csv("train_set.csv", names=["sequence"])
print(df)
df[["pssm", "sequence"]] = df["sequence"].str.split("\t", expand=True)
# df["pssm"] = df["pssm"].removeprefix(">")
df = df[["pssm", "sequence"]]
df["pssm"] = df["pssm"].str.replace(">", "")
df["pssm"] = df["pssm"].astype(str) + ".txt"
df.loc[df.index[0:1352], "label"] = "1"
df.loc[df.index[1352:2704], "label"] = "0"
print(df)
df.to_csv("train.csv", index=True)

