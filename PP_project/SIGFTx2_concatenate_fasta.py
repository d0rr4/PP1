def merge_fasta_with_dedup(file1_path, file2_path, output_path):
    seen_headers = set()
    duplicate_count = 0
    total_count = 0

    with open(output_path, 'w') as outfile:
        # Process the first FASTA file
        with open(file1_path, 'r') as f1:
            for line in f1:
                if line.startswith('>'):
                    header = line.strip()
                    seen_headers.add(header)
                    total_count += 1
                outfile.write(line)

        # Process the second FASTA file
        with open(file2_path, 'r') as f2:
            for line in f2:
                if line.startswith('>'):
                    header = line.strip()
                    total_count += 1
                    # If header exists in the first file, append _2
                    if header in seen_headers:
                        header = f"{header}_2"
                        duplicate_count += 1
                    outfile.write(f"{header}\n")
                else:
                    # Write the sequence lines as they are
                    outfile.write(line)

    print("--- Merge Complete ---")
    print(f"Total sequences processed: {total_count}")
    print(f"Duplicate headers modified from second file: {duplicate_count}")
    print(f"Merged output saved to: '{output_path}'")

# Example Usage:
file1 = "datasets/3FTx_signal.fasta"
file2 = "datasets/3FTx_signal_signalp6_removed.fasta"
output = "datasets/3FTx_signal_signalp6_merged.fasta"

merge_fasta_with_dedup(file1, file2, output)