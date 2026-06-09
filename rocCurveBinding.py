import pandas as pd
import numpy as np

df_9mers = pd.read_csv("binding_9mers.csv")

allele_name = "HLA-A*02:01"

df_allele = df_9mers[df_9mers["MHC"] == allele_name].copy()

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Create mapping
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

print("AUC scores for each fold:", auc_scores)
print("Mean AUC:", np.mean(auc_scores))
print("Std Dev:", np.std(auc_scores))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# train model
model.fit(X_train, y_train)

# probabilities
y_score = model.predict_proba(X_test)[:, 1]

# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, linewidth=2,
         label=f"Binding Model (AUC = {roc_auc:.3f})")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for HLA-A*02:01 Binding Prediction")
plt.legend()
plt.tight_layout()

plt.savefig("binding_roc.png", dpi=300)
plt.show()