import os
import pandas as pd

# 1. Define file paths and target size
input_file = "datasets/uniref50_under_2k.tsv"
output_file = "datasets/uniref50_under_2k_subset.tsv"
target_size = 10000

print(f"Reading master dataset from {input_file}...")
if not os.path.exists(input_file):
    print(f"❌ Error: {input_file} not found. Ensure your concatenation/filter script ran successfully.")
    exit()

df = pd.read_csv(input_file, sep="\t")

# Safety check for column names
if "Cluster ID" not in df.columns or "From" not in df.columns:
    print("❌ Error: Missing required columns ('Cluster ID' or 'From'). Check the file headers.")
    exit()

# 2. Extract all unique clusters
unique_clusters = df["Cluster ID"].unique()
total_unique_clusters = len(unique_clusters)
print(f"Found {total_unique_clusters:,} total unique UniRef50 clusters.")

# Handle edge case where dataset has fewer than 10,000 total clusters
if total_unique_clusters < target_size:
    print(f"⚠️ Warning: Total unique clusters ({total_unique_clusters:,}) is less than your target of {target_size:,}.")
    print("Sampling one protein from every single available cluster instead.")
    selected_clusters = unique_clusters
else:
    # Step 1: Randomly sample 10,000 cluster IDs
    print(f" -> Randomly selecting {target_size:,} distinct clusters...")
    selected_clusters = pd.Series(unique_clusters).sample(n=target_size, random_state=42).values

# 3. Filter dataframe to only include rows from the selected clusters
df_subset = df[df["Cluster ID"].isin(selected_clusters)]

# 4. Step 2: Group by Cluster ID and pick exactly 1 random protein per group
print(" -> Randomly picking exactly 1 protein representative from each selected cluster...")
df_final_sample = df_subset.groupby("Cluster ID").sample(n=1, random_state=42)

# 5. Save the final clean dataset
df_final_sample.to_csv(output_file, sep="\t", index=False)

print("\n" + "=" * 50)
print("🎉 Success! Your non-redundant dataset is ready.")
print(f"Total Unique Clusters Selected: {df_final_sample['Cluster ID'].nunique():,}")
print(f"Total Protein Rows Saved:        {len(df_final_sample):,}")
print(f"Saved to:                       {output_file}")
print("=" * 50)