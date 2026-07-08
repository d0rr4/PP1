import pandas as pd
import os

def generate_fasta_from_excel(input_excel, output_fasta_all, output_fasta_full):
    # 1. Load the Excel data
    print("Loading data...")
    df = pd.read_excel(input_excel)

    valid_sequences = []
    full_seq_only = []

    # 2. Extract and format sequences into memory
    for idx, row in df.iterrows():
        full_seq = row.get('full_seq')
        mature_seq = row.get('mature_seq')
        identifier = row.get('identifier')
        id_new = row.get('id_new')
        
        # Determine sequence priority and flag if it's a full sequence
        seq = None
        is_full_seq = False
        
        if pd.notna(full_seq) and isinstance(full_seq, str) and len(full_seq.strip()) > 0:
            seq = full_seq.strip()
            is_full_seq = True
        elif pd.notna(mature_seq) and isinstance(mature_seq, str) and len(mature_seq.strip()) > 0:
            seq = mature_seq.strip()
        
        # If a valid sequence was found from either column, proceed
        if seq is not None:
            # Extract the ID between the 1st and 2nd '|' (e.g., 'P01385')
            if pd.notna(identifier) and '|' in str(identifier):
                parts = str(identifier).split('|')
                fasta_header = parts[1] if len(parts) > 1 else str(identifier)
            else:
                # Fallback to id_new if the identifier isn't structured with pipes
                fasta_header = str(id_new) if pd.notna(id_new) else f"seq_{idx}"
            
            # Append to our combined valid list
            valid_sequences.append((fasta_header, seq))
            
            # If it was a full sequence, also append to our specific full_seq list
            if is_full_seq:
                full_seq_only.append((fasta_header, seq))

    # 3. Write all sequences to the main file
    print(f"Generating combined FASTA file: {output_fasta_all}...")
    with open(output_fasta_all, "w") as f:
        for header, seq in valid_sequences:
            f.write(f">{header}\n{seq}\n")

    # 4. Write only the full sequences to the second file
    print(f"Generating full_seq ONLY FASTA file: {output_fasta_full}...")
    with open(output_fasta_full, "w") as f:
        for header, seq in full_seq_only:
            f.write(f">{header}\n{seq}\n")

    print(f"Done! Successfully wrote {len(valid_sequences)} total sequences to {output_fasta_all}")
    print(f"Done! Successfully wrote {len(full_seq_only)} full sequences to {output_fasta_full}")

# --- Execute the script ---
if __name__ == "__main__":
    input_path = "datasets/3FTx_raw_data.xlsx"
    
    # Define output paths
    output_path_all = "datasets/3FTx.fasta"
    output_path_full_only = "datasets/3FTx_full_only.fasta"
    
    generate_fasta_from_excel(input_path, output_path_all, output_path_full_only)