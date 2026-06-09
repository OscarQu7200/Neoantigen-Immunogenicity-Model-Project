import pandas as pd

# Load datasets
binding_df = pd.read_csv("binding_data.csv")
functional_df = pd.read_csv("functional_immunogenicity_data.csv")

# Inspect unique qualitative values
# print("Binding qualitative values:")
# print(binding_df["Qualitative Measurement"].value_counts())

# print("\nFunctional qualitative values:")
# print(functional_df["Qualitative Measurement"].value_counts())


def label_from_qualitative(q):
    q = str(q).strip().lower()

    if q.startswith("positive"):
        return 1

    if q == "negative":
        return 0

    return None  # unexpected case


# Apply labeling
binding_df["Label"] = binding_df["Qualitative Measurement"].apply(label_from_qualitative)
functional_df["Label"] = functional_df["Qualitative Measurement"].apply(label_from_qualitative)


# Drop unresolved rows (should be zero)
binding_df = binding_df.dropna(subset=["Label"])
functional_df = functional_df.dropna(subset=["Label"])


# Convert to integers
binding_df["Label"] = binding_df["Label"].astype(int)
functional_df["Label"] = functional_df["Label"].astype(int)


# Print distributions
print("Binding label distribution:")
print(binding_df["Label"].value_counts(normalize=True))

print("\nFunctional label distribution:")
print(functional_df["Label"].value_counts(normalize=True))


# Save labeled files
binding_df.to_csv("binding_labeled.csv", index=False)
functional_df.to_csv("functional_labeled.csv", index=False)

print("\nLabeled datasets saved successfully.")