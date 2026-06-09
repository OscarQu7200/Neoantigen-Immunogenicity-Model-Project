import pandas as pd
df = pd.read_csv("iedb_assay_data.csv")

# Loading the CSV properly
'''
df = pd.read_csv("assay_database.csv")
print(df.shape)        # rows, columns
print(df.columns)      # see column names
df.head(5)
'''

# Function to classify assay types based on given criteria
def classify_assay(method, response):
    """
    Classify immunological assay type based on IEDB method and response fields.
    
    Parameters
    ----------
    method : str
        Experimental method used in the assay.
    response : str
        Biological response measured.
    
    Returns
    -------
    str
        Assay_Type category.
    """

    method = str(method).lower()
    response = str(response).lower()

    # 1️⃣ MHC / TCR binding assays
    if any(keyword in method for keyword in [
        "tetramer", "multimer", "pentamer", "dextramer",
        "binding", "affinity", "stability", "surface plasmon resonance", "spr"
    ]):
        return "MHC_binding"

    # 2️⃣ Cytokine release assays
    if any(keyword in response for keyword in [
        "ifn", "interferon", "il-", "tnf", "cytokine"
    ]) or any(keyword in method for keyword in [
        "elispot", "elisa", "luminex", "intracellular staining"
    ]):
        return "Cytokine_release"

    # 3️⃣ Cytotoxicity / killing assays
    if any(keyword in response for keyword in [
        "cytotoxic", "killing", "lysis", "cell death"
    ]) or any(keyword in method for keyword in [
        "51cr", "chromium", "killing assay"
    ]):
        return "Cytotoxicity"

    # 4️⃣ Proliferation assays
    if any(keyword in response for keyword in [
        "proliferation", "cell division", "expansion"
    ]) or any(keyword in method for keyword in [
        "cfse", "thymidine"
    ]):
        return "Proliferation"

    # 5️⃣ Structural biology
    if any(keyword in method for keyword in [
        "x-ray", "crystallography", "nmr", "cryo-em"
    ]):
        return "Structure"

    # 6️⃣ Fallback
    return "Other"

# Apply the classifier to the entire dataset
# df["Assay_Type"] = df.apply(
#     lambda row: classify_assay(row["Method"], row["Response measured"]),
#     axis = 1
# )

# df.to_csv("assay_data_with_types.csv", index = False)

df = pd.read_csv("assay_data_with_types.csv")
print(df["Assay_Type"].value_counts()) # Display counts of each assay type

# print(df.sample(10)[["Method", "Response measured", "Assay_Type"]])
