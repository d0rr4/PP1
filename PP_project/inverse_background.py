import time
from pathlib import Path
import h5py

# 1. Define paths (Wrapped in Path to ensure .exists() and .name work properly)
main_h5_path = Path("embeddings/Prot-T5.h5")
full_h5_path = Path("embeddings/per-protein.h5")
output_h5_path = Path("embeddings/inverse_Prot-T5.h5")

# Quick sanity check to ensure files exist
if not main_h5_path.exists():
    raise FileNotFoundError(f"Could not find the main data HDF5 file at {main_h5_path}")
if not full_h5_path.exists():
    raise FileNotFoundError(f"Could not find the full reference HDF5 file at {full_h5_path}")

# 2. Extract keys from both files
print("Reading protein IDs from datasets...")
with h5py.File(main_h5_path, 'r') as main_f:
    main_ids = set(main_f.keys())
print(f"✅ Loaded {len(main_ids):,} IDs from main dataset ({main_h5_path.name})")

with h5py.File(full_h5_path, 'r') as full_f:
    full_ids = set(full_f.keys())
print(f"✅ Loaded {len(full_ids):,} IDs from full Swiss-Prot dataset ({full_h5_path.name})")

# 3. Calculate the inverse set (Full Swiss-Prot MINUS Main Data)
# This instantly filters out any protein that exists in your main dataset
inverse_ids = full_ids - main_ids
print(f"\nCalculated inverse dataset size: {len(inverse_ids):,} strictly non-main proteins to write.")

# 4. Copy data to the new file
print(f"\nWriting inverse dataset to {output_h5_path.name}... (This may take a few minutes)")
start_time = time.time()

with h5py.File(full_h5_path, 'r') as full_f, h5py.File(output_h5_path, 'w') as out_f:
    total_to_write = len(inverse_ids)
    
    # Sort the IDs just to keep the HDF5 internal structure neat and organized
    for idx, prot_id in enumerate(sorted(inverse_ids), 1):
        # Slice the array from the full file and dump it directly into the output file
        embedding_vector = full_f[prot_id][:]
        out_f.create_dataset(prot_id, data=embedding_vector)
        
        # Print progress tracker
        if idx % 50000 == 0 or idx == total_to_write:
            print(f" ── Copied {idx:,} / {total_to_write:,} embeddings...")

elapsed_time = time.time() - start_time
print(f"\n🎉 Done! Inverse dataset safely generated.")
print(f"Saved at: {output_h5_path}")
print(f"Time elapsed: {elapsed_time:.1f} seconds.")