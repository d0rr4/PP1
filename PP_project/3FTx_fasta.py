import pandas as pd
import os

def generate_split_fasta_from_excel(input_excel, output_fasta_1, output_fasta_2):
    # 1. Load the Excel data
    print("Loading data...")
    df = pd.read_excel(input_excel)

    valid_sequences = []

    # 2. Extract and format sequences into memory
    for idx, row in df.iterrows():
        seq = row.get('mature_seq')
        identifier = row.get('identifier')
        id_new = row.get('id_new')
        
        if pd.notna(seq) and isinstance(seq, str) and len(seq.strip()) > 0:
            seq = seq.strip()
            
            # Extract the ID between the 1st and 2nd '|' (e.g., 'P01385')
            if pd.notna(identifier) and '|' in str(identifier):
                parts = str(identifier).split('|')
                fasta_header = parts[1] if len(parts) > 1 else str(identifier)
            else:
                # Fallback to id_new if the identifier isn't structured with pipes
                fasta_header = str(id_new) if pd.notna(id_new) else f"seq_{idx}"
            
            # Append as a tuple to our valid list
            valid_sequences.append((fasta_header, seq))

    # 3. Calculate the midpoint to split the sequences
    midpoint = len(valid_sequences) // 2
    part1 = valid_sequences[:midpoint]
    part2 = valid_sequences[midpoint:]

    # 4. Write the first half to file 1
    print(f"Generating FASTA file 1: {output_fasta_1}...")
    with open(output_fasta_1, "w") as f1:
        for header, seq in part1:
            f1.write(f">{header}\n{seq}\n")

    # 5. Write the second half to file 2
    print(f"Generating FASTA file 2: {output_fasta_2}...")
    with open(output_fasta_2, "w") as f2:
        for header, seq in part2:
            f2.write(f">{header}\n{seq}\n")

    print(f"Done! Successfully wrote {len(part1)} sequences to {output_fasta_1} and {len(part2)} to {output_fasta_2}")

# --- Execute the script ---
if __name__ == "__main__":
    input_path = "datasets/3FTx_raw_data.xlsx"
    
    # Define two separate output paths
    output_path_1 = "datasets/3FTx_mature_sequences_part1.fasta"
    output_path_2 = "datasets/3FTx_mature_sequences_part2.fasta"
    
    generate_split_fasta_from_excel(input_path, output_path_1, output_path_2)