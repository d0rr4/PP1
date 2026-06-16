import numpy as np
import h5py

def clean_cath_id(cath_key):
    # 1. Strip the prefix: cath|current|12asA00 -> 12asA00
    if cath_key.startswith("cath|current|"):
        cath_key = cath_key.split("|")[-1]
        
    # 2. Strip any sub-domain formatting: 12asA00/4-330 -> 12asA00
    # THIS is the crucial line that prevents h5py from creating nested folders
    cath_key = cath_key.split("/")[0]
    
    return cath_key

def npy_to_h5_flat(npy_path, h5_path):
    print(f"Loading {npy_path}...")
    data = np.load(npy_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()

    if not isinstance(data, dict):
        raise ValueError("Expected dictionary of embeddings")

    print(f"Loaded {len(data)} embeddings. Writing flat H5...")

    with h5py.File(h5_path, "w") as dst:
        skipped = 0
        for key, value in data.items():
            clean_id = clean_cath_id(key)

            # Prevent h5py crash if multiple sub-domains resolve to the same clean_id
            if clean_id in dst:
                continue

            # value is already a numpy array — write directly
            if isinstance(value, np.ndarray):
                dst.create_dataset(clean_id, data=value)

            # value is a dict or group-like — mean pool sub-entries
            elif isinstance(value, dict):
                arrays = list(value.values())
                dst.create_dataset(clean_id, data=np.mean(arrays, axis=0))

            else:
                print(f"  Skipping {clean_id}: unexpected type {type(value)}")
                skipped += 1

        written = len(dst.keys())
        print(f"Written : {written}")
        print(f"Skipped : {skipped}")

    return written


npy_to_h5_flat(
    "embeddings/cath_s40_prott5_embeddings.npy",
    "embeddings/cath.h5",
)