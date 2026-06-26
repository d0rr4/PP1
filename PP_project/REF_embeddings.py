import os
import pandas as pd
import h5py

# 1. Define paths (matching your previous script's output)
tsv_file = "datasets/uniref50_under_2k_subset.tsv"
master_h5_path = "embeddings/per-protein.h5"
output_h5_path = "embeddings/uniref50_under_2k_subset.h5"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

# 2. Load your sampled protein IDs
print(f"Reading target IDs from {tsv_file}...")
if not os.path.exists(tsv_file):
    print(f"❌ Error: {tsv_file} not found. Run your cluster sampling script first.")
    exit()

df = pd.read_csv(tsv_file, sep="\t")
# The 'From' column contains the raw UniProt Accession IDs
target_ids = set(df["From"].astype(str).tolist())
total_targets = len(target_ids)
print(f"Loaded {total_targets:,} target IDs for embedding extraction.")

# 3. Open files and perform a high-speed targeted copy
print(f"\nOpening master embedding file: {master_h5_path}...")
if not os.path.exists(master_h5_path):
    print(f"❌ Error: Master file '{master_h5_path}' not found. Check your directory structure.")
    exit()

extracted_count = 0
missing_count = 0

with h5py.File(master_h5_path, "r") as infile, h5py.File(output_h5_path, "w") as outfile:
    print("Extracting matching tensors...")
    
    for idx, uniprot_id in enumerate(target_ids, 1):
        if uniprot_id in infile:
            # Efficiently copies the dataset slice along with its metadata/attributes
            infile.copy(uniprot_id, outfile)
            extracted_count += 1
        else:
            missing_count += 1
            
        # Quick progress update every 2,000 items
        if idx % 2000 == 0 or idx == total_targets:
            print(f"  - Processed {idx:,}/{total_targets:,} IDs...")

# 4. Print final data validation summary
print("\n" + "=" * 50)
print("🎉 Embedding Subsetting Complete!")
print("=" * 50)
print(f"Successfully Extracted: {extracted_count:,} embeddings")
if missing_count > 0:
    print(f"⚠️ Warning: Missing IDs:  {missing_count:,} (IDs not found in the master .h5)")
print(f"Saved Subset Destination: {output_h5_path}")
print("=" * 50)