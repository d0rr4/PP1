import os

# Define your file paths
original_fasta = "datasets/metazoa_nontoxin_signalpeptide.fasta"
processed_fasta = "datasets/metazoa_nontoxin_signalpeptide_signalp6_removed.fasta"
output_fasta = "datasets/metazoa_nontoxin_signalpeptide_signalp6_merged.fasta"

def read_fasta_to_dict(file_path):
    """Reads a FASTA file into a dictionary using the core ID as the key."""
    sequences = {}
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                # Grab just the core ID (first word) to ensure a perfect match
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        # Catch the last entry
        if current_id and current_seq:
            sequences[current_id] = "".join(current_seq)
            
    return sequences

# 1. Load both sets into memory
print("📖 Reading original sequences...")
orig_seqs = read_fasta_to_dict(original_fasta)

print("📖 Reading SignalP cleaved sequences...")
proc_seqs = read_fasta_to_dict(processed_fasta)

# 2. Merge and write sequentially to a new FASTA file
pairs_written = 0
mismatch_count = 0

print("\n🤝 Merging and appending suffixes...")
with open(output_fasta, 'w') as out_file:
    for core_id, mature_seq in proc_seqs.items():
        if core_id in orig_seqs:
            # Write the original sequence (retained signal peptide)
            out_file.write(f">{core_id}\n{orig_seqs[core_id]}\n")
            
            # Write the processed sequence (removed signal peptide) with the suffix
            out_file.write(f">{core_id}_signalp6\n{mature_seq}\n")
            
            pairs_written += 1
        else:
            mismatch_count += 1

print("-" * 50)
print(f"✅ Merging complete!")
print(f"🧬 Paired entries written: {pairs_written} (Total of {pairs_written * 2} sequences in file).")
if mismatch_count > 0:
    print(f"🚨 Warning: {mismatch_count} processed sequences couldn't be mapped back to originals.")
print(f"💾 Clean FASTA file saved to: {output_fasta}")