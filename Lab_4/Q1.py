import pandas as pd
from math import log2

from collections import Counter

df = pd.read_csv("titanic.csv")

columns = df.columns.to_list()
print(columns)

col = input("Enter the column to find entropy of: ")

if col not in columns:
    print("Invalid Column")
    quit()

data = df[col]
counts = Counter(data)
total = sum(counts.values())

entropy = 0.0
for c in counts.values():
    probablity = c / total
    entropy -= probablity * log2(probablity)

print(f"Entropy of column '{col}': {entropy}")
