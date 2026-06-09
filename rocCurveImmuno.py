import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_curve,
    auc
)

# Load dataset

df_immuno = pd.read_csv("functional_9mers.csv")

df_immuno = df_immuno[
    df_immuno["MHC"] == "HLA-A*02:01"
].copy()

print("Total peptides:", len(df_immuno))

print("\nClass distribution:")
print(df_immuno["Label"].value_counts(normalize=True))

# One-hot encoding

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

aa_to_index = {
    aa: i
    for i, aa in enumerate(amino_acids)
}

def one_hot_encode(peptide):

    encoding = np.zeros((9, 20))

    for pos, aa in enumerate(peptide):

        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1

    return encoding.flatten()

# Create X and y

X_immuno = np.array([
    one_hot_encode(p)
    for p in df_immuno["Peptide"]
])

y_immuno = df_immuno["Label"].values

print("\nX shape:", X_immuno.shape)
print("y shape:", y_immuno.shape)

# neural network

model_immuno = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# 5-Fold Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

auc_scores = cross_val_score(
    model_immuno,
    X_immuno,
    y_immuno,
    cv=cv,
    scoring="roc_auc"
)

print("\nImmunogenicity AUC per fold:")
print(auc_scores)

print("\nMean AUC:", np.mean(auc_scores))
print("Std Dev:", np.std(auc_scores))

# ROC Curve

X_train, X_test, y_train, y_test = train_test_split(
    X_immuno,
    y_immuno,
    test_size=0.20,
    stratify=y_immuno,
    random_state=42
)

model_immuno.fit(X_train, y_train)

y_score = model_immuno.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_score
)

roc_auc = auc(
    fpr,
    tpr
)

print("\nTest ROC-AUC:", roc_auc)

# Plot ROC Curve

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve for HLA-A*02:01 Immunogenicity Prediction"
)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    "immunogenicity_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()