import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION (Update file paths if needed) ---
TOXIN_TSV = "datasets/toxins.tsv"
SIGNAL_TSV = "datasets/metazoa_nontoxin_signalpeptide.tsv"
NO_SIGNAL_TSV = "datasets/metazoa_nontoxin_nonsignalpeptide.tsv"

BIG_H5_FILE = "embeddings/per-protein.h5"  # Path to your downloaded massive precomputed file

# Output 1: Balanced 50/50 background embeddings and metadata
OUTPUT_H5_FILE = "embeddings/metazoa_nontoxin_signalpeptide5050_lengthmatched.h5"
OUTPUT_TSV_5050_PATH = "datasets/metazoa_nontoxin_signalpeptide5050_lengthmatched.tsv"

# Output 2: Standalone distribution-matched +signal background sequences and metadata
OUTPUT_SIGNAL_ONLY_FASTA_PATH = (
    "datasets/metazoa_nontoxin_signalpeptide_lengthmatched.fasta"
)
OUTPUT_TSV_SIGNAL_ONLY_PATH = (
    "datasets/metazoa_nontoxin_signalpeptide_lengthmatched.tsv"
)
# --------------------------------------------------


def load_and_clean_tsv(tsv_path, require_sequence=False):
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]

    if "Entry" not in df.columns or "Length" not in df.columns:
        raise KeyError(f"{tsv_path} must contain 'Entry' and 'Length' columns.")

    if require_sequence and "Sequence" not in df.columns:
        raise KeyError(
            f"The TSV file '{tsv_path}' must contain a 'Sequence' column to generate a FASTA file. "
            f"Please ensure you check the 'Sequence' option when downloading from UniProt."
        )
    return df


print("Step 1: Loading UniProt query lists...")
df_toxin = load_and_clean_tsv(TOXIN_TSV)
df_signal = load_and_clean_tsv(SIGNAL_TSV, require_sequence=True)
df_no_signal = load_and_clean_tsv(NO_SIGNAL_TSV)

# Restrict all to the 15-300 amino acid range to align with your pipeline
df_toxin = df_toxin[(df_toxin["Length"] >= 15) & (df_toxin["Length"] <= 300)].copy()
df_signal = df_signal[
    (df_signal["Length"] >= 15) & (df_signal["Length"] <= 300)
].copy()
df_no_signal = df_no_signal[
    (df_no_signal["Length"] >= 15) & (df_no_signal["Length"] <= 300)
].copy()

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

# Cross-reference to avoid KeyErrors during extraction
df_signal = df_signal[df_signal["Entry"].isin(available_h5_keys)].copy()
df_no_signal = df_no_signal[df_no_signal["Entry"].isin(available_h5_keys)].copy()

print(
    f"  -> Valid H5 keys found: {len(df_signal)} signal, {len(df_no_signal)} non-signal."
)

print("\nStep 3: Matching length distributions to Toxin baseline...")
# Setup bins: 15-30, 30-45, ..., 285-300
bin_edges = np.arange(15, 301, 15)
toxin_counts, _ = np.histogram(df_toxin["Length"], bins=bin_edges)

# Storage for 50/50 balanced sets
sampled_signal_dfs = []
sampled_no_signal_dfs = []

# Storage for the standalone +signal set
sampled_signal_only_dfs = []

# Print table header
header = f"{'Bin Range':<12} | {'Toxins (Target)':<15} | {'+Signal (Avail)':<15} | {'-Signal (Avail)':<15} | {'Sampled 50/50':<14} | {'Sampled +Sig Only'}"
print(header)
print("-" * len(header))

# Iterate through each length bin
for i in range(len(bin_edges) - 1):
    bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
    target_count = toxin_counts[i]

    # Inclusive of the upper bound only for the final bin
    if i == len(bin_edges) - 2:
        cond_s = (df_signal["Length"] >= bin_min) & (df_signal["Length"] <= bin_max)
        cond_ns = (df_no_signal["Length"] >= bin_min) & (
            df_no_signal["Length"] <= bin_max
        )
    else:
        cond_s = (df_signal["Length"] >= bin_min) & (df_signal["Length"] < bin_max)
        cond_ns = (df_no_signal["Length"] >= bin_min) & (
            df_no_signal["Length"] < bin_max
        )

    candidates_signal = df_signal[cond_s]
    candidates_no_signal = df_no_signal[cond_ns]

    avail_signal = len(candidates_signal)
    avail_nsignal = len(candidates_no_signal)

    # 1. Size calculation for the 50/50 balance split
    take_5050 = min(target_count, avail_signal, avail_nsignal)

    # 2. Size calculation for the standalone +signal split (unconstrained by -signal)
    take_sig_only = min(target_count, avail_signal)

    # Print bin statistics
    bin_label = (
        f"[{bin_min:3d} - {bin_max:3d})"
        if i < len(bin_edges) - 2
        else f"[{bin_min:3d} - {bin_max:3d}]"
    )
    print(
        f"{bin_label:<12} | {target_count:<15} | {avail_signal:<15} | {avail_nsignal:<15} | {take_5050:<14} | {take_sig_only}"
    )

    # Sample for 50/50 output
    if take_5050 > 0:
        sampled_signal_dfs.append(
            candidates_signal.sample(n=take_5050, random_state=42)
        )
        sampled_no_signal_dfs.append(
            candidates_no_signal.sample(n=take_5050, random_state=42)
        )

    # Sample for standalone +signal output
    if take_sig_only > 0:
        sampled_signal_only_dfs.append(
            candidates_signal.sample(n=take_sig_only, random_state=42)
        )

print("-" * len(header))

# Compile 50/50 parts and map associated label assignments
df_final_signal = pd.concat(sampled_signal_dfs).copy()
df_final_signal["Label"] = "Has_Signal"

df_final_no_signal = pd.concat(sampled_no_signal_dfs).copy()
df_final_no_signal["Label"] = "No_Signal"

# Merge everything into a complete tracking layout for the 50/50 background model
df_5050_all = pd.concat([df_final_signal, df_final_no_signal], ignore_index=True)
final_5050_ids = df_5050_all["Entry"].tolist()

# Process Dataframe for standalone +signal file
df_final_signal_only = pd.concat(sampled_signal_only_dfs).copy()
df_final_signal_only["Label"] = "Has_Signal"

print(f"\n  -> Summary:")
print(
    f"     * Balanced 50/50 Dataset: {len(df_final_signal)} signal + {len(df_final_no_signal)} non-signal (Total: {len(df_5050_all)} records)"
)
print(
    f"     * Standalone +Signal Dataset: Total {len(df_final_signal_only)} distribution-matched sequences"
)

print(
    "\nStep 4: Writing output files (H5 Embeddings, FASTA Sequences, & TSV Layouts)..."
)

# Make sure directory layers exist
os.makedirs(os.path.dirname(OUTPUT_TSV_5050_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_TSV_SIGNAL_ONLY_PATH), exist_ok=True)

# 1. Write the 50/50 metadata TSV file
print(f"  -> Generating 50/50 Metadata TSV: {OUTPUT_TSV_5050_PATH}")
df_5050_all.to_csv(OUTPUT_TSV_5050_PATH, sep="\t", index=False)

# 2. Write the standalone +signal metadata TSV file
print(f"  -> Generating +Signal Metadata TSV: {OUTPUT_TSV_SIGNAL_ONLY_PATH}")
df_final_signal_only.to_csv(OUTPUT_TSV_SIGNAL_ONLY_PATH, sep="\t", index=False)

# 3. Write the standalone +signal FASTA file
print(f"  -> Generating FASTA: {OUTPUT_SIGNAL_ONLY_FASTA_PATH}")
with open(OUTPUT_SIGNAL_ONLY_FASTA_PATH, "w") as f_fasta:
    for _, row in tqdm(
        df_final_signal_only.iterrows(),
        total=len(df_final_signal_only),
        desc="Writing FASTA entries",
    ):
        f_fasta.write(f">{row['Entry']}\n{row['Sequence']}\n")

# 4. Extract and write the 50/50 H5 file using the synchronized dataset array order
print(f"  -> Generating H5 Embeddings: {OUTPUT_H5_FILE}")
os.makedirs(os.path.dirname(OUTPUT_H5_FILE), exist_ok=True)
with h5py.File(BIG_H5_FILE, "r") as f_in:
    src_group = f_in["prot_t5"] if has_subgroup else f_in

    with h5py.File(OUTPUT_H5_FILE, "w") as f_out_5050:
        for protein_id in tqdm(final_5050_ids, desc="Writing H5 arrays"):
            embedding_vector = src_group[protein_id][:]
            f_out_5050.create_dataset(protein_id, data=embedding_vector)

print("\n🎉 Finished! Created background datasets with matching length profiles:")
print(f"  1. Balanced 50/50 Embeddings:  {OUTPUT_H5_FILE}")
print(f"  2. Balanced 50/50 Metadata:    {OUTPUT_TSV_5050_PATH}")
print(f"  3. Standalone +Signal FASTA:   {OUTPUT_SIGNAL_ONLY_FASTA_PATH}")
print(f"  4. Standalone +Signal Metadata: {OUTPUT_TSV_SIGNAL_ONLY_PATH}")