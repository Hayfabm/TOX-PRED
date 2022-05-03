import pandas as pd
import math


df = pd.read_csv("test.csv", names=["sequence"])
print(df)
df[["drop", "sequence"]] = df["sequence"].str.split("\t", expand=True)
# df["pssm"] = df["pssm"].removeprefix(">")
df = df[["drop", "sequence"]]
df.pop("drop")
df.loc[df.index[0:799], "label"] = "0"
df.loc[df.index[799:1598], "label"] = "1"
print(df)
df.to_csv("test_set.csv", index=True)
