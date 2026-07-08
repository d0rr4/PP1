import pandas as pd
from Bio import SeqIO

# 1. Load your TSV
tsv_path = 'datasets/uniref50_under_2k.tsv'
df = pd.read_csv(tsv_path, sep='\t')

# 2. Parse the Swiss-Prot FASTA into a dictionary {Accession: Sequence}
fasta_path = 'datasets/uniprot_sprot.fasta'
seq_dict = {}

for record in SeqIO.parse(fasta_path, "fasta"):
    # UniProt FASTA IDs look like "sp|P12345|NAME_HUMAN". We just want the accession (P12345).
    accession = record.id.split('|')[1]
    seq_dict[accession] = str(record.seq)

# 3. Map the sequences to your dataframe using the 'From' column
df['Sequence'] = df['From'].map(seq_dict)

# 4. Check for any missing sequences
missing = df['Sequence'].isna().sum()
if missing > 0:
    print(f"Warning: {missing} sequences could not be found in the FASTA.")

# 5. Save the updated TSV
output_path = 'datasets/uniref50_under_2k_seqs.tsv'
df.to_csv(output_path, sep='\t', index=False)
print(f"Saved updated TSV to {output_path}")