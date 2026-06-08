import psycopg2
import h5py
import numpy as np
import io

# --- CONFIGURATION ---
DB_PARAMS = {
    "dbname": "embeddings",
    "user": "postgres",
    "password": "12345", # <-- Update this
    "host": "localhost",
    "port": "5432"
}

# Mapping of ID from your DB to a filename suffix
MODEL_MAPPING = {
    1: "ESM",
    2: "Prost-T5",
    3: "Prot-T5",
    4: "Ankh3-Large"
}

def parse_vector(vector_data):
    """
    Converts pgvector/halfvec format to numpy array.
    If the data comes as a string '[0.1, 0.2]', we clean it.
    """
    if isinstance(vector_data, str):
        # Remove brackets and split by comma
        clean_str = vector_data.strip('[]').replace(' ', '')
        return np.array([float(x) for x in clean_str.split(',')], dtype=np.float16)
    return np.array(vector_data, dtype=np.float16)

def extract_all():
    conn = psycopg2.connect(**DB_PARAMS)
    
    for model_id, model_name in MODEL_MAPPING.items():
        filename = f"{model_name}.h5"
        print(f"--- Extracting {model_name} (ID: {model_id}) ---")
        
        # Open HDF5 file
        with h5py.File(filename, 'w') as f:
            # Use a server-side cursor to keep memory usage low
            cur = conn.cursor(name=f'cursor_{model_id}')
            cur.itersize = 1000
            
            # Fetch sequence_id and embedding
            cur.execute("SELECT sequence_id, embedding FROM sequence_embeddings WHERE embedding_type_id = %s", (model_id,))
            
            # Create dataset
            dset = f.create_dataset('embeddings', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.float16))
            
            count = 0
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                
                for seq_id, raw_emb in rows:
                    emb_array = parse_vector(raw_emb)
                    # For a real implementation, you might want to save seq_id as well
                    # For now, we are appending the raw embedding
                    # Note: H5py vlen_dtype handling
                    
                    # Logic to append to dataset
                    dset.resize((dset.shape[0] + 1,))
                    dset[-1] = emb_array
                    
                count += len(rows)
                print(f"  Processed {count} embeddings...")
            
            cur.close()
            print(f"Success! Saved to {filename}")

    conn.close()

if __name__ == "__main__":
    extract_all()