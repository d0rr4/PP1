import os

def filter_fasta_by_signal_peptide(tsv_path, fasta_path, output_path):
    """
    Reads a UniProt TSV file to find IDs containing a 'Signal peptide' entry,
    and extracts their corresponding sequences from a FASTA file.
    """
    valid_entries = set()
    
    # Step 1: Read the TSV file and grab IDs with a signal peptide
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        
        # Locate correct column indexes dynamically
        try:
            entry_idx = header.index('Entry')
            signal_idx = header.index('Signal peptide')
        except ValueError:
            # Fallback defaults if exact header strings differ slightly
            entry_idx = 0
            signal_idx = 7
            
        for line in f:
            if not line.strip():
                continue
            cols = line.strip('\n').split('\t')
            
            # Verify row contains the target column before checking
            if len(cols) > signal_idx:
                entry_id = cols[entry_idx].strip()
                signal_val = cols[signal_idx].strip()
                
                # If the Signal peptide column is not blank, save this ID
                if signal_val: 
                    valid_entries.add(entry_id)

    # Step 2: Parse and filter the FASTA file
    count = 0
    with open(fasta_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        current_header = None
        current_seq = []
        keep_sequence = False
        
        for line in infile:
            if line.startswith('>'):
                # Save previous sequence record if it was flagged to keep
                if current_header and keep_sequence:
                    outfile.write(current_header + '\n' + '\n'.join(current_seq) + '\n')
                    count += 1
                
                # Reset for the new sequence record
                current_header = line.strip()
                current_seq = []
                
                # Parse accession ID from header (handles standard '>sp|ID|NAME' or plain '>ID')
                header_content = current_header[1:]  # strip '>'
                if '|' in header_content:
                    accession = header_content.split('|')[1].strip()
                else:
                    accession = header_content.split()[0].strip()
                    
                # Determine if this sequence matches our valid set
                keep_sequence = accession in valid_entries
            else:
                if keep_sequence:
                    current_seq.append(line.strip())
        
        # Flush the last sequence out of the loop buffer
        if current_header and keep_sequence:
            outfile.write(current_header + '\n' + '\n'.join(current_seq) + '\n')
            count += 1
            
    print(f"Processed {os.path.basename(fasta_path)}: Found {len(valid_entries)} TSV matches. Saved {count} entries to {output_path}")

# Configuration for your file pairs
files_to_process = [
    {
        "tsv": "datasets/toxins_20_2000.tsv",
        "fasta": "datasets/toxins_20_2000.fasta",
        "output": "datasets/toxins_20_2000_signal.fasta" 
    },
    {
        "tsv": "datasets/nontoxins_20_2000.tsv",
        "fasta": "datasets/nontoxins_20_2000.fasta",
        "output": "datasets/nontoxins_20_2000_signal.fasta"
    }
]

if __name__ == "__main__":
    # Ensure the datasets directory exists before running
    os.makedirs("datasets", exist_ok=True)
    
    for item in files_to_process:
        filter_fasta_by_signal_peptide(item["tsv"], item["fasta"], item["output"])