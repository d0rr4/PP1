import pandas as pd

# Define input and output filenames
input_csv = "datasets/3FTx_annotation.csv"
output_fasta = "datasets/3FTx_signal.fasta"

# Load the CSV file
df = pd.read_csv(input_csv)

# Filter for rows where 'signalp6.0' is True (case-insensitive string match for robustness)
filtered_df = df[df['signalp6.0'].astype(str).str.strip().str.lower() == 'true']

# Write the filtered rows to a FASTA file
with open(output_fasta, 'w') as fasta_file:
    for _, row in filtered_df.iterrows():
        # Uses 'ID' for the header (e.g., SP|P01385|Acanthophis_antarcticus) 
        # and 'mature_seq' for the sequence line
        fasta_file.write(f">{row['ID']}\n{row['mature_seq']}\n")

print(f"Extraction complete! Saved {len(filtered_df)} sequences to '{output_fasta}'.")