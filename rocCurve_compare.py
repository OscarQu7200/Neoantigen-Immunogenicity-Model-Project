import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc

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

# FUNCTION TO COMPUTE MEAN ROC CURVE

def compute_mean_roc(X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        aucs.append(roc_auc)

        interp_tpr = np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)

    mean_tpr = np.mean(
        tprs,
        axis=0
    )

    mean_tpr[-1] = 1.0

    mean_auc = np.mean(
        aucs
    )

    std_auc = np.std(
        aucs
    )

    return mean_fpr, mean_tpr, mean_auc, std_auc

# BINDING DATA

print("Loading binding dataset...")

df_binding = pd.read_csv(
    "binding_9mers.csv"
)

df_binding = df_binding[
    df_binding["MHC"] == "HLA-A*02:01"
].copy()

X_binding = np.array([
    one_hot_encode(p)
    for p in df_binding["Peptide"]
])

y_binding = df_binding["Label"].values

print("Binding samples:", len(df_binding))

binding_fpr, binding_tpr, binding_auc, binding_std = compute_mean_roc(
    X_binding,
    y_binding
)

print(
    f"Binding Mean AUC = {binding_auc:.3f} ± {binding_std:.3f}"
)

# IMMUNOGENICITY DATA

print("\nLoading immunogenicity dataset...")

df_immuno = pd.read_csv(
    "functional_9mers.csv"
)

df_immuno = df_immuno[
    df_immuno["MHC"] == "HLA-A*02:01"
].copy()

X_immuno = np.array([
    one_hot_encode(p)
    for p in df_immuno["Peptide"]
])

y_immuno = df_immuno["Label"].values

print("Immunogenicity samples:", len(df_immuno))

immuno_fpr, immuno_tpr, immuno_auc, immuno_std = compute_mean_roc(
    X_immuno,
    y_immuno
)

print(
    f"Immunogenicity Mean AUC = {immuno_auc:.3f} ± {immuno_std:.3f}"
)

# FIGURE 3

plt.figure(figsize=(7, 7))

plt.plot(
    binding_fpr,
    binding_tpr,
    linewidth=3,
    label=f"Binding Model (AUC = {binding_auc:.3f})"
)

plt.plot(
    immuno_fpr,
    immuno_tpr,
    linewidth=3,
    label=f"Immunogenicity Model (AUC = {immuno_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.title(
    "Binding vs Immunogenicity Prediction\n(HLA-A*02:01, 5-Fold Cross-Validation)",
    fontsize=14
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "Figure3_Binding_vs_Immunogenicity_ROC.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()