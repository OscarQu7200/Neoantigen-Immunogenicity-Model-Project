import pandas as pd

# Load your dataset
df = pd.read_csv("functional_training_table.csv")

# Create a length column
df["peptide_length"] = df["Peptide"].astype(str).str.len()

# See distribution of lengths
length_counts = df["peptide_length"].value_counts().sort_index()

print("Peptide length distribution:")
print(length_counts)

df_9mers = df[df["peptide_length"] == 9].copy()
df_9mers.to_csv("functional_9mers.csv", index=False)

'''
df = pd.read_csv("binding_training_table.csv")
df["peptide_length"] = df["Peptide"].astype(str).str.len()
df_9mers = df[df["peptide_length"] == 9].copy()

print(df_9mers["Label"].value_counts())
print("\nClass balance (%):")
print(df_9mers["Label"].value_counts(normalize=True) * 100)
'''