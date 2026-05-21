import pandas as pd

# Define your paths
csv_path = "datasets/cath_annotation.csv"
output_parquet_path = "protspace_results/cath/annotated_embeddings.parquet"

# 1. Read your CATH CSV data
df_custom = pd.read_csv(csv_path)

# 2. Rename the very first ID column to 'identifier' to match ProtSpace
first_col_name = df_custom.columns[0]
df_custom.rename(columns={first_col_name: "identifier"}, inplace=True)

# 3. Force every single column to be an 'object' (string) type
for col in df_custom.columns:
    df_custom[col] = df_custom[col].astype(str)
    # Clean up empty values so they match ProtSpace's blank text format
    df_custom[col] = df_custom[col].replace(["nan", "None", "NaN"], "")

# 4. Clean the index so it is a nameless integer range (0, 1, 2...)
df_custom.reset_index(drop=True, inplace=True)
df_custom.index.name = None

# 5. Save out to Parquet with an active index match
df_custom.to_parquet(
    output_parquet_path,
    engine="pyarrow",
    compression="snappy",
    index=True
)

print("=== DONE! CLONE FORMAT MATCHED ===")
print(f"File saved to: {output_parquet_path}")