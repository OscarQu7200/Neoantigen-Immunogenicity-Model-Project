import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Load Data
df = pd.read_csv("functional_9mers.csv")

X_seq = df["Peptide"].values
y = df["Label"].values

print("Total samples:", len(df))
print("Positive ratio:", np.mean(y))


# One-Hot Encoding Function
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptides):
    n_samples = len(peptides)
    encoding = np.zeros((n_samples, 9, 20))
    
    for i, pep in enumerate(peptides):
        for j, aa in enumerate(pep):
            if aa in aa_to_index:
                encoding[i, j, aa_to_index[aa]] = 1
                
    return encoding.reshape(n_samples, -1)


X = one_hot_encode(X_seq)


# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nImmunogenicity Logistic Regression AUC per fold:")
print(auc_scores)
print("Mean AUC:", np.mean(auc_scores))
print("Std Dev:", np.std(auc_scores))