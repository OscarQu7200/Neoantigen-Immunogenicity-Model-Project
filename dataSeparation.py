import pandas as pd

# Load full labeled dataset
df = pd.read_csv("assay_data_with_types.csv")

# print(df["Assay_Type"].value_counts())

# New CSV that filters for MHC binding assay types
# binding_df = df[df["Assay_Type"] == "MHC_binding"].copy()
# print("Binding rows:", len(binding_df))

# binding_df.to_csv("binding_data.csv", index = False)

# New CSV that filters for functional assay types
functional_df = df[df["Assay_Type"].isin([
    "Cytokine_release",
    "Cytotoxicity"
])].copy()

print("Functional rows:", len(functional_df))

functional_df.to_csv("functional_immunogenicity_data.csv", index=False)

print("Files saved successfully.")