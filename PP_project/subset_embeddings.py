import h5py
import pandas as pd
from pathlib import Path

# --- Config ---
H5_SOURCE = "embeds/per-protein.h5"
METADATA  = "datasets/3FTx.tsv"
H5_OUT    = "embeds/3FTx.h5"

Path("embeds").mkdir(exist_ok=True)

# --- Load metadata ---
df = pd.read_csv(METADATA, sep="\t")
accessions = set(df["Entry"].dropna().astype(str).str.strip())
print(f"Accessions to extract: {len(accessions)}")

# --- Subset HDF5 ---
found, missing = [], []

with h5py.File(H5_SOURCE, "r") as src, h5py.File(H5_OUT, "w") as dst:
    all_keys = set(src.keys())
    for acc in accessions:
        if acc in all_keys:
            dst.create_dataset(acc, data=src[acc][:])
            found.append(acc)
        else:
            missing.append(acc)

print(f"Extracted : {len(found)}")
print(f"Not in H5 : {len(missing)}")
