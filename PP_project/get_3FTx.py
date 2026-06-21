import requests
import h5py
import io
import pandas as pd

# Load the dataset
df = pd.read_csv("../data/3FTx/3FTx.csv")

def extract_uniprot(identifier):
    parts = str(identifier).split("|")
    if len(parts) >= 2 and parts[0] in ["SP", "TR"]:
        return parts[1]
    return None

def get_source(identifier):
    parts = str(identifier).split("|")
    if parts[0] in ["SP", "TR"]:
        return parts[0]
    elif parts[0] == "NCBI":
        return "NCBI"
    return "None"

# Print source breakdown
df["source"] = df["identifier"].apply(get_source)
source_counts = df["source"].value_counts()
print("Identifier source breakdown:")
for source in ["SP", "TR", "NCBI", "None"]:
    print(f"  {source}: {source_counts.get(source, 0)}")
print()

# Extract UniProt IDs, strip isoform suffixes (e.g. Q17RY6-2 -> Q17RY6)
df["Entry"] = df["identifier"].apply(extract_uniprot)
df["Entry"] = df["Entry"].str.split("-").str[0]
uniprot_ids = df["Entry"].dropna().drop_duplicates().tolist()

print(f"Total unique UniProt IDs extracted (SP + TR): {len(uniprot_ids)} "
      f"(from {source_counts.get('SP', 0)} SP + {source_counts.get('TR', 0)} TR, before deduplication)")

# Fetch embeddings via search endpoint (supports h5), in batches of 100
embeddings = {}
n_batches = (len(uniprot_ids) - 1) // 100 + 1

for i in range(0, len(uniprot_ids), 100):
    batch = uniprot_ids[i:i+100]
    query = " OR ".join([f"accession:{acc}" for acc in batch])
    response = requests.get(
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query": query,
            "format": "h5"
        }
    )
    response.raise_for_status()

    with h5py.File(io.BytesIO(response.content), "r") as f:
        for accession in f.keys():
            embeddings[accession] = f[accession][:]

    print(f"  Processed batch {i//100 + 1}/{n_batches} ...")

# Save to disk
with h5py.File("embeddings.h5", "w") as out:
    for accession, vector in embeddings.items():
        out.create_dataset(accession, data=vector)

# Summary
found = set(embeddings.keys())
missing = [uid for uid in uniprot_ids if uid not in found]

print(f"\nEmbeddings found and saved: {len(found)} / {len(uniprot_ids)}")
if missing:
    print(f"No embedding found for {len(missing)} IDs: {missing}")
print("Saved to embeddings.h5")