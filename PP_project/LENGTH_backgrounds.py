import pandas as pd
import h5py
import numpy as np
import random
import os

# --- Configuration ---
TARGET_TSV = 'datasets/uniref50_under_2k_subset_seqs.tsv'
FULL_TSV = 'datasets/uniref50_under_2k.tsv'
EMBEDDINGS_H5 = 'embeddings/per-protein.h5'

OUT_H5_REAL = 'embeddings/LENGTH_real.h5'
OUT_FASTA_RANDOM = 'datasets/LENGTH_random_aa.fasta'
OUT_FASTA_POLY_A = 'datasets/LENGTH_poly_alanine.fasta'
OUT_FASTA_SHUFFLED = 'datasets/LENGTH_shuffled.fasta'

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_backgrounds():
    print("Loading metadata...")
    target_df = pd.read_csv(TARGET_TSV, sep='\t')
    full_df = pd.read_csv(FULL_TSV, sep='\t')

    target_lengths = target_df['Length'].tolist()
    target_clusters = set(target_df['Cluster ID'].dropna())
    
    # Check if we have sequences for background 4
    has_sequence = 'Sequence' in target_df.columns
    if not has_sequence:
        print("WARNING: No 'Sequence' column found in target TSV. Background 4 (shuffled) will be skipped or fail.")

    # =====================================================================
    # Background 1: Real Proteins, Length-Matched, Non-overlapping Clusters
    # =====================================================================
    print("Generating Background 1 (Real Embeddings)...")
    # Filter out any rows in the full dataset that belong to target clusters
    pool_df = full_df[~full_df['Cluster ID'].isin(target_clusters)].copy()
    
    selected_uniprots = []
    
    for L in target_lengths:
        # Try to find exact length match
        matches = pool_df[pool_df['Length'] == L]
        
        if not matches.empty:
            # Randomly select one exact match
            idx = matches.sample(1).index[0]
        else:
            # Fallback: Find the closest length if exact match isn't available
            idx = (pool_df['Length'] - L).abs().idxmin()
            
        selected_uniprots.append(pool_df.loc[idx, 'From'])
        # Drop to sample without replacement
        pool_df = pool_df.drop(idx)
        
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
    
    with open(OUT_FASTA_RANDOM, 'w') as f_rand, \
         open(OUT_FASTA_POLY_A, 'w') as f_poly, \
         open(OUT_FASTA_SHUFFLED, 'w') as f_shuf:
        
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
            if has_sequence:
                original_seq = str(row['Sequence'])
                # Ensure we only shuffle the actual length, though length col should match seq length
                seq_list = list(original_seq)
                random.shuffle(seq_list)
                shuffled_seq = ''.join(seq_list)
                f_shuf.write(f">{uid}_shuffled\n{shuffled_seq}\n")

    print("Background generation complete!")

if __name__ == "__main__":
    generate_backgrounds()