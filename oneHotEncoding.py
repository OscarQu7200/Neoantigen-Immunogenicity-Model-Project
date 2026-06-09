import numpy as np
import pandas as pd

df_9mers = pd.read_csv("binding_9mers.csv")

# Standard amino acids 
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Create mapping
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(peptide):
    encoding = np.zeros((9, 20))
    for pos, aa in enumerate(peptide):
        if aa in aa_to_index:
            encoding[pos, aa_to_index[aa]] = 1
    return encoding.flatten()

# Apply encoding
X = np.array([one_hot_encode(p) for p in df_9mers["Peptide"]])

# Labels
y = df_9mers["Label"].values

# print(X.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_probs))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))