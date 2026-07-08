import h5py
import os

def concatenate_uniprot_embeddings_keep_all(file1_path, file2_path, output_path):
    """
    Concatenates two UniProt HDF5 embedding files into a new one.
    If a protein ID exists in both files, the one from the second file 
    is saved with a '_2' suffix so both are kept.
    """
    # Check if input files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        raise FileNotFoundError("One or both of the input .h5 files could not be found.")

    print(f"Creating merged file: {output_path}")
    
    with h5py.File(output_path, 'w') as f_out:
        # 1. Copy everything from the first file
        print(f"Reading from {file1_path}...")
        with h5py.File(file1_path, 'r') as f1:
            for key in f1.keys():
                f1.copy(key, f_out)
        
        # 2. Copy from the second file, handling duplicates with a suffix
        print(f"Reading from {file2_path}...")
        with h5py.File(file2_path, 'r') as f2:
            for key in f2.keys():
                if key in f_out:
                    new_key = f"{key}_2"
                    print(f"🔗 Duplicate found! Saving second file's '{key}' as '{new_key}'")
                    f2.copy(key, f_out, name=new_key)
                else:
                    f2.copy(key, f_out)
                
    print(f"\n🎉 Successfully combined files! Total datasets stored: {len(h5py.File(output_path, 'r').keys())}")

# --- Example Usage ---
if __name__ == "__main__":
    file_a = "embeddings/toxins_signalpeptide.h5"
    file_b = "embeddings/toxins_signalpeptide_signalp6_removed_prot_t5.h5"

    combined_file = "embeddings/toxins_signalpeptide_signalp6_merged.h5"
    
    concatenate_uniprot_embeddings_keep_all(file_a, file_b, combined_file)

