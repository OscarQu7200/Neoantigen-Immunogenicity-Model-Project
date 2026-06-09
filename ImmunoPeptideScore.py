import pandas as pd
import numpy as np

# Binding Model
df_9mers = pd.read_csv("binding_9mers.csv")

allele_name = "HLA-A*02:01"

df_allele = df_9mers[df_9mers["MHC"] == allele_name].copy()

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptide):
    encoding = np.zeros((9, 20))
    for pos, aa in enumerate(peptide):
        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1
    return encoding.flatten()

X = np.array([one_hot_encode(p) for p in df_allele["Peptide"]])
y = df_allele["Label"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring='roc_auc'
)

print("Mean CV AUC:", np.mean(auc_scores))

# fit final model on full dataset
model.fit(X, y)

# Load immunogenicity dataset
df_immuno = pd.read_csv("functional_9mers.csv")

# Keep only HLA-A*02:01
df_immuno = df_immuno[df_immuno["MHC"] == "HLA-A*02:01"].copy()

# print("Total immunogenic peptides:", len(df_immuno))
# print(df_immuno["Label"].value_counts(normalize=True))

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptide):
    encoding = np.zeros((9, 20))
    for pos, aa in enumerate(peptide):
        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1
    return encoding.flatten()

X_immuno = np.array([
    one_hot_encode(p) for p in df_immuno["Peptide"]
])

y_immuno = df_immuno["Label"].values

# print("X_immuno shape:", X_immuno.shape)
# print("y_immuno shape:", y_immuno.shape)

from sklearn.metrics import roc_auc_score

binding_scores = model.predict_proba(X_immuno)[:, 1]

binding_auc_on_immuno = roc_auc_score(y_immuno, binding_scores)

print("Binding model predicting immunogenicity AUC:", binding_auc_on_immuno)