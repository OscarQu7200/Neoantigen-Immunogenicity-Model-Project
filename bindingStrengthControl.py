import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier


# ONE-HOT ENCODING

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptide):
    encoding = np.zeros((9, 20))
    for pos, aa in enumerate(peptide):
        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1
    return encoding.flatten()

# TRAIN BINDING MODEL

print("\n--- Training Binding Model ---")

df_binding = pd.read_csv("binding_9mers.csv")
df_binding = df_binding[df_binding["MHC"] == "HLA-A*02:01"].copy()

X_binding = np.array([one_hot_encode(p) for p in df_binding["Peptide"]])
y_binding = df_binding["Label"].values

model_binding = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

binding_auc = cross_val_score(
    model_binding,
    X_binding,
    y_binding,
    cv=cv,
    scoring='roc_auc'
)

print("Binding CV AUC:", np.mean(binding_auc))

# Fit on FULL binding dataset
model_binding.fit(X_binding, y_binding)


# LOAD IMMUNOGENICITY DATA

print("\n--- Loading Immunogenicity Data ---")

df_immuno = pd.read_csv("functional_9mers.csv")
df_immuno = df_immuno[df_immuno["MHC"] == "HLA-A*02:01"].copy()

X_immuno = np.array([one_hot_encode(p) for p in df_immuno["Peptide"]])
y_immuno = df_immuno["Label"].values

print("Total immunogenic peptides:", len(df_immuno))


# PREDICT BINDING SCORES FOR IMMUNO PEPTIDES

binding_probs = model_binding.predict_proba(X_immuno)[:, 1]
df_immuno["BindingScore"] = binding_probs


# SELECT TOP 30% STRONG BINDERS

threshold = np.percentile(binding_probs, 70)

df_strong = df_immuno[df_immuno["BindingScore"] >= threshold].copy()

print("Strong binder subset size:", len(df_strong))
print("Class distribution in strong binders:")
print(df_strong["Label"].value_counts(normalize=True))

# RETRAIN IMMUNOGENICITY MODEL on binders

X_strong = np.array([one_hot_encode(p) for p in df_strong["Peptide"]])
y_strong = df_strong["Label"].values

model_immuno_strong = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

auc_scores_strong = cross_val_score(
    model_immuno_strong,
    X_strong,
    y_strong,
    cv=cv,
    scoring='roc_auc'
)

print("\nStrong Binder Immunogenicity AUC per fold:", auc_scores_strong)
print("Mean AUC (Strong Binders):", np.mean(auc_scores_strong))
print("Std Dev:", np.std(auc_scores_strong))