import h5py
import psycopg2
import numpy as np

# Connect to your database
conn = psycopg2.connect("dbname=embeddings user=postgres host=localhost password=12345")
cur = conn.cursor()

# Step 1: Build an in-memory lookup dictionary (Takes ~0.5 seconds)
print("Building UniProt ID lookup map...")
map_query = """
    SELECT p.sequence_id, a.code 
    FROM protein p 
    JOIN accession a ON p.id = a.protein_id 
    WHERE a.primary = true;
"""
cur.execute(map_query)
# id_map will look like: { 1245: 'Q7Z624', 1246: 'Q7Z5W3', ... }
id_map = {row[0]: row[1] for row in cur}
print(f"Loaded {len(id_map)} ID mappings into memory.")

# Map your model database IDs to their output filenames
models_to_export = {
    1: "ESM.h5",
    2: "Prost-T5.h5",
    3: "Prot-T5.h5",
    4: "Ankh3-Large.h5"
}

# Step 2: Export embeddings using a blazing-fast single table query
for model_id, filename in models_to_export.items():
    print(f"Exporting model ID {model_id} to {filename}...")
    
    # Back to a fast single-table query, just adding the float conversion
    query = """
        SELECT sequence_id, embedding::float4[] 
        FROM sequence_embeddings 
        WHERE embedding_type_id = %s;
    """
    
    cur.execute(query, (model_id,))
    
    # Save directly to the H5 file
    with h5py.File(f"embeddings/{filename}", "w") as f:
        for row in cur:
            seq_id = row[0]
            
            # Map the integer sequence_id to its alphanumeric UniProt ID using Python RAM
            uniprot_id = id_map.get(seq_id)
            
            # Skip if this sequence doesn't have a valid primary accession mapping
            if not uniprot_id:
                continue
                
            embedding_vector = np.array(row[1], dtype=np.float32)
            f.create_dataset(uniprot_id, data=embedding_vector)

cur.close()
conn.close()
print("All embedding files fixed and ready at maximum speed!")