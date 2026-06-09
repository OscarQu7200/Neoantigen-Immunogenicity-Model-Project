import pandas as pd
import numpy as np

df_9mers = pd.read_csv("binding_9mers.csv")

# print(df_9mers["MHC"].value_counts().head(20))

allele_name = "HLA-A*02:01"

df_allele = df_9mers[df_9mers["MHC"] == allele_name].copy()

# print("Number of samples:", len(df_allele))
# print("\nClass balance:")
# print(df_allele["Label"].value_counts(normalize=True) * 100)

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

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, roc_auc_score

# y_pred = model.predict(X_test)
# y_probs = model.predict_proba(X_test)[:, 1]

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_probs))

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))