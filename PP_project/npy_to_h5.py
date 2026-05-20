import numpy as np
import h5py
import argparse

def npy_to_h5(npy_path, h5_path):
    data = np.load(npy_path, allow_pickle=True)

    # if stored as object array containing dict
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()

    if not isinstance(data, dict):
        raise ValueError("Expected dictionary of embeddings")

    with h5py.File(h5_path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)

    print(f"Saved {len(data)} embeddings to {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file")
    parser.add_argument("h5_file")

    args = parser.parse_args()
    npy_to_h5(args.npy_file, args.h5_file)