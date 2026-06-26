import os
import glob
import pandas as pd

# 1. Set your directories
input_dir = "datasets/split_chunks" 
output_file = "datasets/uniref50_under_2k.tsv"

# 2. Find all files containing '_mapped' in their filename
search_pattern = os.path.join(input_dir, "*_mapped*")
mapped_files = glob.glob(search_pattern)

# Sort them alphabetically/numerically so they merge in order
mapped_files.sort()

print(f"Found {len(mapped_files)} files containing '_mapped' in '{input_dir}':")
for f in mapped_files:
    print(f"  - {os.path.basename(f)}")

if not mapped_files:
    print("\n❌ No files found! Make sure your downloaded UniProt files have '_mapped' in the name.")
    exit()

# 3. Read and stack them up
dataframe_list = []

print("\nCombining files...")
for file_path in mapped_files:
    # UniProt web downloads use tab-separation (\t)
    df = pd.read_csv(file_path, sep="\t")
    dataframe_list.append(df)

# Concatenate everything into a single dataframe
combined_df = pd.concat(dataframe_list, ignore_index=True)
total_raw_rows = len(combined_df)

# --- NEW: Filter for lengths <= 2000 ---
if "Length" in combined_df.columns:
    print("\nFiltering sequences...")
    combined_df = combined_df[combined_df["Length"] <= 2000]
    dropped_count = total_raw_rows - len(combined_df)
    print(f"  - Removed {dropped_count:,} sequences with length > 2,000 aa.")
else:
    print("\n⚠️ Warning: 'Length' column not found in your files. Skipping length filtering.")
# ----------------------------------------

# 4. Save the clean master mapping file
combined_df.to_csv(output_file, sep="\t", index=False)

print(f"\n🎉 Success! Combined and filtered files into one master file.")
print(f"Total raw rows processed:       {total_raw_rows:,}")
print(f"Total mapped rows kept (≤2000): {len(combined_df):,}")
print(f"Saved to: {output_file}")