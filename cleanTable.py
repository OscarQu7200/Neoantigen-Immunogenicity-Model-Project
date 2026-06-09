# import pandas as pd

# binding_df = pd.read_csv("binding_labeled.csv")

# binding_clean = binding_df[[
#     "Epitope - Name",
#     "MHC Restriction - Name",
#     "Label"
# ]].copy()

# binding_clean.columns = ["Peptide", "MHC", "Label"]

# binding_clean.to_csv("binding_training_table.csv", index=False)

import pandas as pd

binding_df = pd.read_csv("functional_labeled.csv")

binding_clean = binding_df[[
    "Epitope - Name",
    "MHC Restriction - Name",
    "Label"
]].copy()

binding_clean.columns = ["Peptide", "MHC", "Label"]

binding_clean.to_csv("binding_training_table.csv", index=False)
