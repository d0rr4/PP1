import os
import re
import h5py
import pandas as pd

def extract_uniprot_id(key):
    """
    Robustly extracts a standard UniProt Accession from decorated strings.
    Examples handled:
      - 'sp|P12345|TOX_ANEMN' -> 'P12345'
      - 'P12345_prot_t5'     -> 'P12345'
      - '>P12345'            -> 'P12345'
    """
    # 1. Strip FASTA character artifacts
    key = str(key).lstrip('>')
    
    # 2. Handle pipe-separated database formats (e.g., sp|P12345|NAME)
    if '|' in key:
        parts = key.split('|')
        if len(parts) > 1:
            return parts[1].strip()
            
    # 3. Handle pipeline suffixes (e.g., P12345_prot_t5 or P12345_1)
    # Splits by underscore and grabs the primary accession head
    return key.split('_')[0].strip()


def merge_and_filter_embeddings_robust(file1_path, file2_path, tsv_path, output_path, id_col="Entry"):
    """
    Filters and merges two HDF5 embedding files by normalizing their keys 
    to match the UniProt IDs extracted from a length-filtered TSV file.
    """
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        raise FileNotFoundError("One or both of the input .h5 files could not be found.")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found at {tsv_path}.")

    # 1. Load TSV and clean reference IDs
    print(f"Loading length data from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    
    if id_col not in df.columns:
        id_col = df.columns[0]
        print(f"Warning: Column '{id_col}' not found. Defaulting to first column: '{id_col}'")

    # Filter for lengths between 15 and 300 (inclusive)
    valid_df = df[(df["Length"] >= 15) & (df["Length"] <= 300)]
    valid_ids = {extract_uniprot_id(pid) for pid in valid_df[id_col].dropna()}
    print(f"Found {len(valid_ids)} target valid UniProt IDs within the 15-300 aa range.")

    print(f"Creating standardized merged file: {output_path}")
    
    with h5py.File(output_path, 'w') as f_out:
        
        # --- Process File 1 ---
        print(f"\nReading and filtering File 1: {file1_path}...")
        added_count_1 = 0
        debug_printed_1 = 0
        
        with h5py.File(file1_path, 'r') as f1:
            for original_key in f1.keys():
                clean_id = extract_uniprot_id(original_key)
                
                if clean_id in valid_ids:
                    # Print first 3 matches as visual validation for you
                    if debug_printed_1 < 3:
                        print(f"   [Match Found] Original Key: '{original_key}' -> Saved As: '{clean_id}'")
                        debug_printed_1 += 1
                        
                    # Standardize the output key to the clean ID
                    out_key = clean_id
                    if out_key in f_out:
                        out_key = f"{clean_id}_2"
                        
                    f1.copy(original_key, f_out, name=out_key)
                    added_count_1 += 1
                    
        print(f"  -> Successfully added {added_count_1} standard embeddings from File 1.")
        
        # --- Process File 2 ---
        print(f"\nReading and filtering File 2: {file2_path}...")
        added_count_2 = 0
        debug_printed_2 = 0
        
        with h5py.File(file2_path, 'r') as f2:
            for original_key in f2.keys():
                clean_id = extract_uniprot_id(original_key)
                
                if clean_id in valid_ids:
                    if debug_printed_2 < 3:
                        print(f"   [Match Found] Original Key: '{original_key}' -> Cross-referenced as: '{clean_id}'")
                        debug_printed_2 += 1
                        
                    # Determine key format & handle duplicate resolution
                    out_key = clean_id
                    if out_key in f_out:
                        out_key = f"{clean_id}_2"
                        
                    f2.copy(original_key, f_out, name=out_key)
                    added_count_2 += 1
                    
        print(f"  -> Successfully added {added_count_2} standard embeddings from File 2.")
                
    # Final confirmation
    with h5py.File(output_path, 'r') as f_check:
        total_datasets = len(f_check.keys())
    print(f"\n🎉 Process Complete! Total standardized datasets stored: {total_datasets}")


if __name__ == "__main__":
    file_a = "embeddings/toxins_signalpeptide.h5"
    file_b = "embeddings/toxins_signalpeptide_signalp6_removed_prot_t5.h5"
    tsv_dataset = "datasets/toxins.tsv"
    combined_file = "embeddings/toxins_signalpeptide_signalp6_merged_filtered.h5"
    
    merge_and_filter_embeddings_robust(
        file1_path=file_a, 
        file2_path=file_b, 
        tsv_path=tsv_dataset, 
        output_path=combined_file,
        id_col="Entry"
    )