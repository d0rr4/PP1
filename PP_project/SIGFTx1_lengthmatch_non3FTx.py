import os
import pandas as pd
from collections import defaultdict

# --- Configuration ---
ANNOTATION_TSV = "datasets/3FTx_annotation.tsv"
SIGNAL_FASTA = "datasets/non3FTx_signal.fasta"
REMOVED_FASTA = "datasets/non3FTx_signal_signalp6_removed.fasta"
OUTPUT_FASTA = "datasets/non3FTx_signal_signalp6_merged_lengthmatched.fasta"

BIN_SIZE = 15
MATCH_EXACT_COUNTS = True  # Set to False to filter by bin presence instead of exact counts

def read_fasta(filepath):
    """Parses a FASTA file, keeping headers completely raw and unmodified."""
    sequences = []
    current_header = None
    current_seq = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:].strip()  # Save raw header without the '>' symbol
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))
            
    return sequences

def main():
    print("1. Loading reference annotations and calculating target length bins...")
    df_ann = pd.read_csv(ANNOTATION_TSV, sep='\t')
    if 'mature_seq' not in df_ann.columns:
        raise ValueError(f"Column 'mature_seq' not found in {ANNOTATION_TSV}")
        
    mature_seqs = df_ann['mature_seq'].dropna().astype(str)
    
    target_bin_counts = defaultdict(int)
    for seq in mature_seqs:
        b_idx = len(seq) // BIN_SIZE
        target_bin_counts[b_idx] += 1

    print("2. Parsing non3FTx signal FASTA file (keeping raw headers)...")
    signal_seqs = read_fasta(SIGNAL_FASTA)
    
    signal_by_bin = defaultdict(list)
    for raw_header, seq in signal_seqs:
        b_idx = len(seq) // BIN_SIZE
        signal_by_bin[b_idx].append((raw_header, seq))

    print("3. Performing length matching selection...")
    matched_ids = set()
    matched_3ftx_records = []

    for b_idx, target_count in target_bin_counts.items():
        available_records = signal_by_bin[b_idx]
        
        if MATCH_EXACT_COUNTS:
            selected = available_records[:target_count]
        else:
            selected = available_records
            
        for raw_header, seq in selected:
            # Extract core ID dynamically for matching purposes (first word)
            seq_id = raw_header.split()[0] if raw_header else ""
            matched_ids.add(seq_id)
            matched_3ftx_records.append((raw_header, seq))

    print(f"   Selected {len(matched_ids)} unique identifiers.")

    print("4. Parsing and filtering the removed non-3FTx FASTA file...")
    removed_seqs = read_fasta(REMOVED_FASTA)
    matched_removed_records = []
    
    for raw_header, seq in removed_seqs:
        seq_id = raw_header.split()[0] if raw_header else ""
        if seq_id in matched_ids:
            matched_removed_records.append((raw_header, seq))

    print(f"   Found {len(matched_removed_records)} matching entries in the removed dataset.")

    print(f"5. Replacing spaces and slashes (keeping hyphens), then writing to {OUTPUT_FASTA}...")
    
    # Ensure parent directories exist before writing
    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)
    
    # Open the file and write out the records
    with open(OUTPUT_FASTA, 'w', encoding='utf-8') as out_f:
        # Write dataset 1
        for raw_header, seq in matched_3ftx_records:
            # ONLY spaces and forward slashes are replaced by underscores
            clean_header = raw_header.replace(" ", "_").replace("/", "_")
            out_f.write(f">{clean_header}\n{seq}\n")
            
        # Write dataset 2 (with _2 appended)
        for raw_header, seq in matched_removed_records:
            # ONLY spaces and forward slashes are replaced by underscores
            clean_header = raw_header.replace(" ", "_").replace("/", "_")
            out_f.write(f">{clean_header}_2\n{seq}\n")

    print(f"Success! Total sequences written: {len(matched_3ftx_records) + len(matched_removed_records)}")

if __name__ == "__main__":
    main()