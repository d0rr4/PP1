import pandas as pd

# load the dataset
df = pd.read_csv("../data/3FTx/3FTx.csv")

def extract_uniprot(identifier):
    parts = str(identifier).split("|")
    if len(parts) >= 2 and parts[0] in ["SP", "TR"]:
        return parts[1]
    return None

# extract IDs
df["Entry"] = df["identifier"].apply(extract_uniprot)

# keep only valid UniProt IDs
uniprot_df = df[["Entry"]].dropna().drop_duplicates()

# write tsv
uniprot_df.to_csv("metadata/3FTx.tsv", sep="\t", index=False)