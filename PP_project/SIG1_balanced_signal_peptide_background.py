import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION (Update file paths if needed) ---
SIGNAL_TSV = "datasets/metazoa_nontoxin_signalpeptide.tsv"
NO_SIGNAL_TSV = "datasets/metazoa_nontoxin_nonsignalpeptide.tsv"
BIG_H5_FILE = "embeddings/per-protein.h5"  # Path to your downloaded massive precomputed file
OUTPUT_H5_FILE = "embeddings/metazoa_nontoxin_signalpeptide5050.h5"
OUTPUT_TSV_FILE = "datasets/metazoa_nontoxin_signalpeptide5050.tsv"  # Path for the new metadata file
# --------------------------------------------------

print("Step 1: Loading UniProt query lists...")
df_signal = pd.read_csv(SIGNAL_TSV, sep="\t")
df_no_signal = pd.read_csv(NO_SIGNAL_TSV, sep="\t")

# Extract unique accessions from the 'Entry' column
signal_ids = set(df_signal["Entry"].dropna().unique())
no_signal_ids = set(df_no_signal["Entry"].dropna().unique())

print("Step 2: Scanning precomputed H5 structure & keys...")
if not os.path.exists(BIG_H5_FILE):
    raise FileNotFoundError(
        f"Could not find your large embedding file at: {BIG_H5_FILE}"
    )

with h5py.File(BIG_H5_FILE, "r") as f_in:
    # Handle both flat H5 structures and nested group structures (e.g., 'prot_t5')
    if "prot_t5" in f_in:
        src_group = f_in["prot_t5"]
        has_subgroup = True
    else:
        src_group = f_in
        has_subgroup = False

    # Extract available keys lazily without loading data arrays into memory
    available_h5_keys = set(src_group.keys())

# Cross-reference to avoid KeyErrors
valid_signal_ids = list(signal_ids.intersection(available_h5_keys))
valid_no_signal_ids = list(no_signal_ids.intersection(available_h5_keys))

# Determine our maximum 50/50 data ceiling
N = min(len(valid_signal_ids), len(valid_no_signal_ids))
print(
    f"-> Found {len(valid_signal_ids)} valid signal and {len(valid_no_signal_ids)} non-signal keys in H5."
)
print(f"-> Balancing dataset to {N} entries per group (Total: {N * 2}).")

# Randomly sample to enforce perfect balance
np.random.seed(42)
sampled_signal = np.random.choice(valid_signal_ids, size=N, replace=False)
sampled_no_signal = np.random.choice(valid_no_signal_ids, size=N, replace=False)

# Filter the original dataframes to retain all metadata columns for the sampled IDs
df_sampled_signal = df_signal[df_signal["Entry"].isin(sampled_signal)].copy()
df_sampled_no_signal = df_no_signal[
    df_no_signal["Entry"].isin(sampled_no_signal)
].copy()

# Inject the target classification labels explicitly
df_sampled_signal["Label"] = "Has_Signal"
df_sampled_no_signal["Label"] = "No_Signal"

# Merge both sub-populations into a single balanced DataFrame
df_5050 = pd.concat([df_sampled_signal, df_sampled_no_signal], ignore_index=True)

# Generate final extraction list from the synced DataFrame to preserve identical tracking order
final_ids = df_5050["Entry"].tolist()

print(
    f"Step 3: Exporting balanced metadata layout to {OUTPUT_TSV_FILE}..."
)
# Ensure target base directories exist
os.makedirs(os.path.dirname(OUTPUT_TSV_FILE), exist_ok=True)
df_5050.to_csv(OUTPUT_TSV_FILE, sep="\t", index=False)

print(f"Step 4: Extracting embeddings and generating lightweight flat H5...")
with h5py.File(BIG_H5_FILE, "r") as f_in:
    src_group = f_in["prot_t5"] if has_subgroup else f_in

    with h5py.File(OUTPUT_H5_FILE, "w") as f_out:
        # We now write directly to f_out instead of creating a subgroup
        for protein_id in tqdm(final_ids, desc="Subsetting arrays"):
            embedding_vector = src_group[protein_id][:]
            f_out.create_dataset(protein_id, data=embedding_vector)

print(
    "Finished! The resulting H5 file is flat, and a synchronized TSV metadata file has been generated."
)