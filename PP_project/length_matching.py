import h5py
import pandas as pd
import numpy as np

# --- Configuration ---
input_h5_path = "embeddings/uniprotkb_NOT_go_exp_OR_go_ida_OR_go_ipi_OR_2026_06_17.h5" 
output_h5_path = "embeddings/uniprotkb_NOT_go_exp_OR_go_ida_OR_go_ipi_OR_2026_06_17_lengthmatched.h5"

# Now we need BOTH metadata files
background_tsv_path = "datasets/uniprotkb_reviewed_true_AND_NOT_go_exp_2026_06_17.tsv"
target_tsv_path = "datasets/uniprotkb_reviewed_true_AND_go_exp_OR_g_2026_06_18.tsv" # <-- ADD YOUR TARGET TSV HERE

target_size = 100000 # Keeping this low (e.g., 20k) to protect your 12GB WSL RAM

def load_and_clean_tsv(filepath):
    """Helper function to load TSVs and grab just ID and Length."""
    df = pd.read_csv(filepath, sep='\t')
    col_id = df.columns[0] 
    col_length = [c for c in df.columns if 'Length' in c][0]
    df = df[[col_id, col_length]].dropna()
    df.columns = ['ID', 'Length']
    return df

def create_target_matched_subset():
    print("1. Loading metadata...")
    bg_df = load_and_clean_tsv(background_tsv_path)
    target_df = load_and_clean_tsv(target_tsv_path)

    print("2. Matching Background IDs with your .h5 file...")
    with h5py.File(input_h5_path, 'r') as f:
        available_ids = set(f.keys())
    bg_df = bg_df[bg_df['ID'].isin(available_ids)]
    print(f"   Found {len(bg_df)} matching unannotated background entries.")

    print("3. Analyzing Target Length Distribution...")
    # Get 10 quantile bins from the TARGET data, and extract the exact bin edges
    target_df['Length_Bin'], bins = pd.qcut(target_df['Length'], q=10, retbins=True, labels=False, duplicates='drop')
    
    # Calculate how many sequences we need per bin to hit 'target_size' perfectly
    bin_proportions = target_df['Length_Bin'].value_counts(normalize=True)
    required_counts_per_bin = (bin_proportions * target_size).round().astype(int)

    print("4. Sampling Background to match Target Distribution...")
    # Widen the absolute outer limits just in case the background has slightly more extreme outliers
    bins[0] = -np.inf 
    bins[-1] = np.inf 

    # Apply the TARGET's bin edges to the BACKGROUND data
    bg_df['Length_Bin'] = pd.cut(bg_df['Length'], bins=bins, labels=False)

    sampled_dfs = []
    for bin_idx, required_count in required_counts_per_bin.items():
        # Get all background sequences that fall into this specific target bin
        available_in_bin = bg_df[bg_df['Length_Bin'] == bin_idx]
        
        if len(available_in_bin) >= required_count:
            # Sample exactly what we need
            sampled_dfs.append(available_in_bin.sample(required_count, random_state=42))
        else:
            # Edge case: If the background is unexpectedly sparse in this exact length range
            print(f"   Warning: Only found {len(available_in_bin)} background proteins for bin {bin_idx} (needed {required_count}). Taking all.")
            sampled_dfs.append(available_in_bin)

    sampled_bg_df = pd.concat(sampled_dfs)
    
    # Trim overflow if rounding math gave us slightly more than target_size
    if len(sampled_bg_df) > target_size:
        sampled_bg_df = sampled_bg_df.sample(target_size, random_state=42)

    sampled_ids = set(sampled_bg_df['ID'].tolist())
    print(f"   Successfully matched distributions! Selected {len(sampled_ids)} entries.")

    print("5. Extracting and downcasting embeddings to new file...")
    with h5py.File(input_h5_path, 'r') as f_in, h5py.File(output_h5_path, 'w') as f_out:
        for entry_id in sampled_ids:
            data = f_in[entry_id][:]
            f_out.create_dataset(entry_id, data=data.astype(np.float16))

    print(f"\nSuccess! Target-matched subset saved to: {output_h5_path}")

if __name__ == "__main__":
    create_target_matched_subset()