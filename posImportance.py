import numpy as np
import pandas as pd

# Load immunogenicity dataset
df = pd.read_csv("functional_9mers.csv")

peptides = df["Peptide"].values
labels = df["Label"].values

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Initialize counts
pos_counts = np.zeros((9, 20))
neg_counts = np.zeros((9, 20))

aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

for pep, label in zip(peptides, labels):
    for i, aa in enumerate(pep):
        if aa in aa_to_index:
            if label == 1:
                pos_counts[i, aa_to_index[aa]] += 1
            else:
                neg_counts[i, aa_to_index[aa]] += 1

# Normalize
pos_freq = pos_counts / pos_counts.sum(axis=1, keepdims=True)
neg_freq = neg_counts / neg_counts.sum(axis=1, keepdims=True)

# Difference
diff = pos_freq - neg_freq

# Position importance = total absolute difference
position_importance = np.sum(np.abs(diff), axis=1)

for i, score in enumerate(position_importance):
    print(f"Position {i+1}: {score:.4f}")