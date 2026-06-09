import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# Load immunogenicity dataset

df_immuno = pd.read_csv("functional_9mers.csv")

# Keep only HLA-A*02:01
df_immuno = df_immuno[df_immuno["MHC"] == "HLA-A*02:01"].copy()

print("Total peptides:", len(df_immuno))
print("Class distribution:")
print(df_immuno["Label"].value_counts(normalize=True))


# 2️⃣ One-hot encoding function

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptide):
    encoding = np.zeros((9, 20))
    for pos, aa in enumerate(peptide):
        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1
    return encoding.flatten()


# Create X and y

X_immuno = np.array([
    one_hot_encode(p) for p in df_immuno["Peptide"]
])

y_immuno = df_immuno["Label"].values

print("X shape:", X_immuno.shape)
print("y shape:", y_immuno.shape)

# Define Neural Network

model_immuno = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)


# 5-Fold Cross-Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(
    model_immuno,
    X_immuno,
    y_immuno,
    cv=cv,
    scoring='roc_auc'
)

print("\nImmunogenicity Model AUC per fold:", auc_scores)
print("Mean AUC:", np.mean(auc_scores))
print("Std Dev:", np.std(auc_scores))


# Train Final Model on All Data

model_immuno.fit(X_immuno, y_immuno)