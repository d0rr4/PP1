import pandas as pd
import h5py
import numpy as np
import random
import os

# --- Configuration ---
TARGET_TSV = 'datasets/uniref50_under_2k_subset_seqs.tsv'
FULL_TSV = 'datasets/uniref50_under_2k_seqs.tsv'
EMBEDDINGS_H5 = 'embeddings/per-protein.h5'

# Outputs
OUT_H5_REAL = 'embeddings/uniref50_background_real.h5'
OUT_FASTA_REAL = 'datasets/uniref50_background_real.fasta'       # Added
OUT_FASTA_RANDOM = 'datasets/uniref50_background_random.fasta'
OUT_FASTA_POLY_A = 'datasets/uniref50_background_polyA.fasta'
OUT_FASTA_SHUFFLED = 'datasets/uniref50_background_shuffled.fasta'

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
SEED = 42  # Global random seed for deterministic execution

def generate_backgrounds():
    # --- Establish Absolute Seeding ---
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading metadata...")
    target_df = pd.read_csv(TARGET_TSV, sep='\t')
    full_df = pd.read_csv(FULL_TSV, sep='\t')

    target_lengths = target_df['Length'].tolist()
    target_clusters = set(target_df['Cluster ID'].dropna())
    
    # Check if we have sequences for backgrounds
    has_target_sequence = 'Sequence' in target_df.columns
    has_full_sequence = 'Sequence' in full_df.columns
    
    if not has_target_sequence:
        print("WARNING: No 'Sequence' column found in target TSV. Background 4 (shuffled) will be skipped or fail.")
    if not has_full_sequence:
        print("WARNING: No 'Sequence' column found in full TSV. Background 1 FASTA cannot be written.")

    # =====================================================================
    # Background 1: Real Proteins, Length-Matched, Non-overlapping Clusters
    # =====================================================================
    print("Generating Background 1 (Real Embeddings & FASTA)...")
    pool_df = full_df[~full_df['Cluster ID'].isin(target_clusters)].copy()
    
    selected_uniprots = []
    
    # Open the FASTA writer alongside your loop logic
    with open(OUT_FASTA_REAL, 'w', encoding='utf-8') as f_real:
        for L in target_lengths:
            matches = pool_df[pool_df['Length'] == L]
            
            if not matches.empty:
                # Switched to random.choice from standard library to cleanly progress 
                # the global random state linearly over every iteration step.
                idx = random.choice(matches.index.tolist())
            else:
                # Fallback: Find the closest length if exact match isn't available
                idx = (pool_df['Length'] - L).abs().idxmin()
                
            selected_uid = pool_df.loc[idx, 'From']
            selected_cluster = pool_df.loc[idx, 'Cluster ID']
            
            selected_uniprots.append(selected_uid)
            
            # Extract and save the real sequence to FASTA
            if has_full_sequence:
                selected_seq = pool_df.loc[idx, 'Sequence']
                f_real.write(f">{selected_uid}\n{selected_seq}\n")
            
            # Drop the ENTIRE cluster from the pool
            pool_df = pool_df[pool_df['Cluster ID'] != selected_cluster]
        
    print(f"Extracting {len(selected_uniprots)} embeddings from H5 file...")
    
    with h5py.File(EMBEDDINGS_H5, 'r') as h5_in, h5py.File(OUT_H5_REAL, 'w') as h5_out:
        missing_count = 0
        for uid in selected_uniprots:
            if uid in h5_in:
                h5_out.create_dataset(uid, data=h5_in[uid][:])
            else:
                missing_count += 1
        if missing_count > 0:
            print(f"WARNING: {missing_count} selected proteins were not found in {EMBEDDINGS_H5}.")

    # =====================================================================
    # Backgrounds 2, 3, 4: Synthetic FASTA Generation
    # =====================================================================
    print("Generating Synthetic FASTAs (Backgrounds 2, 3, 4)...")
    
    with open(OUT_FASTA_RANDOM, 'w', encoding='utf-8') as f_rand, \
         open(OUT_FASTA_POLY_A, 'w', encoding='utf-8') as f_poly, \
         open(OUT_FASTA_SHUFFLED, 'w', encoding='utf-8') as f_shuf:
        
        for idx, row in target_df.iterrows():
            uid = row['From']
            length = row['Length']
            
            # Background 2: Random AA
            random_seq = ''.join(random.choices(STANDARD_AAS, k=length))
            f_rand.write(f">{uid}_random\n{random_seq}\n")
            
            # Background 3: Poly-Alanine
            poly_a_seq = 'A' * length
            f_poly.write(f">{uid}_polyA\n{poly_a_seq}\n")
            
            # Background 4: Shuffled Target
            if has_target_sequence:
                original_seq = str(row['Sequence'])
                seq_list = list(original_seq)
                random.shuffle(seq_list)
                shuffled_seq = ''.join(seq_list)
                f_shuf.write(f">{uid}_shuffled\n{shuffled_seq}\n")

    print("Background generation complete!")

if __name__ == "__main__":
    generate_backgrounds()