import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("functional_9mers.csv")

peptides = df["Peptide"].values
labels = df["Label"].values

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

# Count frequencies
pos_counts = np.zeros((9, 20))
neg_counts = np.zeros((9, 20))

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

# Difference matrix
diff = pos_freq - neg_freq

# Plot heatmap
plt.figure(figsize=(12, 6))

im = plt.imshow(diff, aspect='auto')

# Axis labels
plt.xticks(range(20), list(amino_acids))
plt.yticks(range(9), [f"P{i+1}" for i in range(9)])

plt.xlabel("Amino Acid")
plt.ylabel("Peptide Position")
plt.title("Immunogenicity Enrichment Heatmap")

# Colorbar
plt.colorbar(im, label="Enrichment (Positive - Negative)")

plt.tight_layout()
plt.show()