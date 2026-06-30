import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_FASTA = "datasets/3FTx_mature_sequences.fasta"
NO_SIGNAL_TSV = "datasets/non3FTx_nonsignal.tsv"
SIGNAL_TSV = "datasets/non3FTx_signal.tsv"
BIG_H5_FILE = "embeddings/per-protein.h5"  # Source embeddings

# Outputs
OUTPUT_H5_FILE = "embeddings/non3FTx_5050_lengthmatched.h5"
OUTPUT_TSV_MERGED = "datasets/non3FTx_5050_lengthmatched_metadata.tsv"
# ---------------------

def get_fasta_lengths(fasta_path):
    """Parses a FASTA file and returns a list of sequence lengths."""
    lengths = []
    with open(fasta_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    lengths.append(len(seq))
                    seq = ""
            else:
                seq += line.strip()
        if seq:  # Catch the last sequence
            lengths.append(len(seq))
    return lengths

def load_clean_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]
    if "Entry" not in df.columns or "Length" not in df.columns:
        raise KeyError(f"{tsv_path} must contain 'Entry' and 'Length' columns.")
    return df

print("Step 1: Loading target FASTA to establish baseline lengths...")
target_lengths = get_fasta_lengths(TARGET_FASTA)
print(f"  -> Found {len(target_lengths)} target sequences.")

print("Step 2: Loading source TSVs...")
df_no_signal = load_clean_tsv(NO_SIGNAL_TSV)
df_signal = load_clean_tsv(SIGNAL_TSV)
print(f"  -> Loaded {len(df_no_signal)} non-signal sequences.")
print(f"  -> Loaded {len(df_signal)} signal sequences.")

print("\nStep 3: Matching length distributions (Bins of 15) for strict 50/50 split...")
max_len = max(max(target_lengths), df_no_signal["Length"].max(), df_signal["Length"].max())
bin_edges = np.arange(0, max_len + 15, 15)
target_counts, _ = np.histogram(target_lengths, bins=bin_edges)

sampled_no_signal_dfs = []
sampled_signal_dfs = []

# Print table header
header = f"{'Bin Range':<12} | {'Target (FASTA)':<15} | {'Avail -Signal':<15} | {'Avail +Signal':<15} | {'Sampled (EACH)'}"
print(header)
print("-" * len(header))

for i in range(len(bin_edges) - 1):
    bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
    target_count = target_counts[i]

    # Inclusive of the upper bound only for the final bin
    if i == len(bin_edges) - 2:
        cond_ns = (df_no_signal["Length"] >= bin_min) & (df_no_signal["Length"] <= bin_max)
        cond_s = (df_signal["Length"] >= bin_min) & (df_signal["Length"] <= bin_max)
    else:
        cond_ns = (df_no_signal["Length"] >= bin_min) & (df_no_signal["Length"] < bin_max)
        cond_s = (df_signal["Length"] >= bin_min) & (df_signal["Length"] < bin_max)

    candidates_no_signal = df_no_signal[cond_ns]
    candidates_signal = df_signal[cond_s]

    avail_ns = len(candidates_no_signal)
    avail_s = len(candidates_signal)

    # STRICT 50/50: Take the minimum across all three constraints
    take_5050 = min(target_count, avail_ns, avail_s)

    # Print bin statistics
    bin_label = f"[{bin_min:3d} - {bin_max:3d})" if i < len(bin_edges) - 2 else f"[{bin_min:3d} - {bin_max:3d}]"
    if target_count > 0 or avail_ns > 0 or avail_s > 0:
        print(f"{bin_label:<12} | {target_count:<15} | {avail_ns:<15} | {avail_s:<15} | {take_5050}")

    # Sample datasets symmetrically
    if take_5050 > 0:
        sampled_no_signal_dfs.append(candidates_no_signal.sample(n=take_5050, random_state=42))
        sampled_signal_dfs.append(candidates_signal.sample(n=take_5050, random_state=42))

print("-" * len(header))

# Combine all sampled bins and assign labels
df_final_no_signal = pd.concat(sampled_no_signal_dfs, ignore_index=True) if sampled_no_signal_dfs else pd.DataFrame()
if not df_final_no_signal.empty: df_final_no_signal['Label'] = 'No_Signal'

df_final_signal = pd.concat(sampled_signal_dfs, ignore_index=True) if sampled_signal_dfs else pd.DataFrame()
if not df_final_signal.empty: df_final_signal['Label'] = 'Has_Signal'

# Merge into one master dataframe
df_merged = pd.concat([df_final_no_signal, df_final_signal], ignore_index=True)
merged_entries = df_merged['Entry'].tolist()

print(f"\n  -> Merged dataset created with {len(df_merged)} total sequences ({len(df_final_signal)} Signal / {len(df_final_no_signal)} Non-Signal).")

print(f"\nStep 4: Filtering and Extracting H5 Embeddings from {BIG_H5_FILE}...")
os.makedirs(os.path.dirname(OUTPUT_H5_FILE), exist_ok=True)

# Diagnostic counters
found_count = 0
missing_count = 0
missing_entries_list = []

with h5py.File(BIG_H5_FILE, "r") as f_in:
    # Handle standard UniProt T5 structure vs flat structure
    src_group = f_in["prot_t5"] if "prot_t5" in f_in else f_in
    
    # Check which entries actually exist in the H5 file before trying to write
    available_keys = set(src_group.keys())
    valid_entries = [e for e in merged_entries if e in available_keys]
    
    missing_count = len(merged_entries) - len(valid_entries)
    if missing_count > 0:
        missing_entries_list = [e for e in merged_entries if e not in available_keys]
        print(f"  -> ⚠️ WARNING: {missing_count} entries were missing from the source H5 file!")
        print(f"  -> Example missing IDs: {missing_entries_list[:5]}")
    
    print(f"  -> Proceeding to write {len(valid_entries)} valid embeddings...")
    
    with h5py.File(OUTPUT_H5_FILE, "w") as f_out:
        for entry in tqdm(valid_entries, desc="Writing combined 50/50 H5 file"):
            data = src_group[entry][:]
            f_out.create_dataset(entry, data=data)
            found_count += 1

# Optional: Drop missing entries from the final TSV metadata so it perfectly matches the H5 file
if missing_count > 0:
    print("  -> Updating metadata TSV to exclude missing H5 entries...")
    df_merged = df_merged[df_merged['Entry'].isin(valid_entries)]

print("\nStep 5: Writing combined TSV metadata...")
df_merged.to_csv(OUTPUT_TSV_MERGED, sep="\t", index=False)

print("\n🎉 Finished!")
print(f"  -> 50/50 H5 Embeddings saved to: {OUTPUT_H5_FILE} ({found_count} datasets)")
print(f"  -> Metadata saved to:            {OUTPUT_TSV_MERGED}")