import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def compute_diff_matrix(df):
    peptides = df["Peptide"].values
    labels = df["Label"].values

    pos_counts = np.zeros((9, 20))
    neg_counts = np.zeros((9, 20))

    for pep, label in zip(peptides, labels):
        for i, aa in enumerate(pep):
            if aa in aa_to_index:
                if label == 1:
                    pos_counts[i, aa_to_index[aa]] += 1
                else:
                    neg_counts[i, aa_to_index[aa]] += 1

    pos_freq = pos_counts / pos_counts.sum(axis=1, keepdims=True)
    neg_freq = neg_counts / neg_counts.sum(axis=1, keepdims=True)

    return pos_freq - neg_freq

# Load datasets
df_binding = pd.read_csv("binding_9mers.csv")
df_immuno = pd.read_csv("functional_9mers.csv")

# Compute matrices
diff_binding = compute_diff_matrix(df_binding)
diff_immuno = compute_diff_matrix(df_immuno)

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Same color scale
vmax = max(np.abs(diff_binding).max(), np.abs(diff_immuno).max())
vmin = -vmax

# Binding
im1 = axes[0].imshow(diff_binding, aspect='auto', vmin=vmin, vmax=vmax)
axes[0].set_title("MHC Binding Enrichment")
axes[0].set_xticks(range(20))
axes[0].set_xticklabels(list(amino_acids))
axes[0].set_yticks(range(9))
axes[0].set_yticklabels([f"P{i+1}" for i in range(9)])

# Immunogenicity
im2 = axes[1].imshow(diff_immuno, aspect='auto', vmin=vmin, vmax=vmax)
axes[1].set_title("Immunogenicity Enrichment")
axes[1].set_xticks(range(20))
axes[1].set_xticklabels(list(amino_acids))
axes[1].set_yticks(range(9))
axes[1].set_yticklabels([f"P{i+1}" for i in range(9)])

# ADD COLORBAR CODE
plt.subplots_adjust(right=0.88)

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Enrichment (Positive - Negative)")

plt.show()