import os

# Define paths and chunk configuration
input_file = "datasets/swissprot_under_2k.tsv"
output_dir = "datasets/split_chunks"
chunk_size = 99000

# Create an output directory for the chunks if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

uniprot_ids = []

print(f"Parsing {input_file}...")

# 1. Read the master file and extract only the Entry IDs
with open(input_file, "r") as f:
    for idx, line in enumerate(f):
        # Skip the header line
        if idx == 0 and "Entry" in line:
            continue
        
        # Isolate the first column (the Entry ID)
        parts = line.strip().split("\t")
        if parts and parts[0]:
            uniprot_ids.append(parts[0])

total_ids = len(uniprot_ids)
print(f"Found {total_ids} total protein IDs.")
print(f"Splitting into text files of {chunk_size} IDs each...\n")

# 2. Slice the list into 99,000 ID blocks and save them
file_counter = 1
for i in range(0, total_ids, chunk_size):
    chunk = uniprot_ids[i:i + chunk_size]
    output_file = os.path.join(output_dir, f"uniprot_chunk_part_{file_counter}.txt")
    
    with open(output_file, "w") as out_f:
        # Write clean, newline-separated IDs
        out_f.write("\n".join(chunk) + "\n")
        
    print(f"➔ Saved {len(chunk)} IDs to: {output_file}")
    file_counter += 1

print(f"\nSuccess! All files are saved inside the '{output_dir}' folder.")