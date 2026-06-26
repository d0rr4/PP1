import os
import re
import pandas as pd
import numpy as np
import h5py

# =====================================================================
# --- FILE INPUT/OUTPUT CONFIGURATION ---
# =====================================================================
TOXIN_TSV_PATH = "datasets/toxins.tsv"
TOXIN_H5_PATH = "embeddings/toxins.h5"

METAZOA_TSV_PATH = "datasets/metazoa_nontoxin_signalpeptide.tsv"
METAZOA_H5_PATH = "embeddings/metazoa_nontoxin_signalpeptide_signalp6_merged_prot_t5.h5"

OUTPUT_TOXIN_H5 = "embeddings/toxins_filtered_15_300.h5"
OUTPUT_METAZOA_H5 = "embeddings/metazoa_nontoxin_signalp6_merged_lengthmatched.h5"

# Standard delta output: full - signalp6
OUTPUT_METAZOA_DELTA_H5 = (
    "embeddings/metazoa_nontoxin_signalp6_merged_lengthmatched_"
    "delta_full_minus_signalp6.h5"
)
# =====================================================================


def load_uniprot_tsv(tsv_path):
    """Loads a standard UniProt TSV file and normalizes column names."""
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]

    if "Length" not in df.columns or "Entry" not in df.columns:
        raise KeyError(
            f"TSV file {tsv_path} must contain 'Entry' and 'Length' columns."
        )

    return df


def map_metazoa_h5_keys(h5_path):
    """
    Parses the complex or truncated H5 keys based on the middle UniProt Entry ID.

    Returns:
        {
            'Entry_ID': {
                'normal': 'full_h5_key',
                'signalp6': 'full_h5_key_signalp6'
            }
        }
    """
    mapping = {}

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

    for key in keys:
        # Split by pipe character to isolate the Entry Accession ID.
        # Example:
        # 'sp|A0A044RE18|BLI_ONCVO' -> ['sp', 'A0A044RE18', 'BLI_ONCVO']
        parts = key.split("|")

        if len(parts) >= 2:
            entry_id = parts[1]

            if entry_id not in mapping:
                mapping[entry_id] = {"normal": None, "signalp6": None}

            if key.endswith("_signalp6"):
                mapping[entry_id]["signalp6"] = key
            else:
                mapping[entry_id]["normal"] = key

    return mapping


def create_delta_dataset(
    f_in,
    f_delta_out,
    entry,
    normal_key,
    signalp6_key,
    scale_factor=1.0,
):
    """
    Creates one delta embedding dataset.

    Definition:
        delta = scale_factor * (full_sequence_embedding - signalp6_removed_embedding)

    The output H5 key is the UniProt Entry ID.
    """

    normal_embedding = np.asarray(f_in[normal_key][()], dtype=np.float32)
    signalp6_embedding = np.asarray(f_in[signalp6_key][()], dtype=np.float32)

    if normal_embedding.shape != signalp6_embedding.shape:
        raise ValueError(
            f"Embedding shape mismatch for {entry}: "
            f"{normal_key} has shape {normal_embedding.shape}, "
            f"{signalp6_key} has shape {signalp6_embedding.shape}"
        )

    delta_embedding = scale_factor * (normal_embedding - signalp6_embedding)

    dset = f_delta_out.create_dataset(
        entry,
        data=delta_embedding.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    # Store provenance directly inside the H5 file.
    dset.attrs["entry"] = entry
    dset.attrs["delta_definition"] = (
        f"{scale_factor} * (normal_embedding - signalp6_embedding)"
    )
    dset.attrs["scale_factor"] = scale_factor
    dset.attrs["normal_key"] = normal_key
    dset.attrs["signalp6_key"] = signalp6_key


def filter_and_match_distributions():
    # ---------------------------------------------------------
    # Step 1: Process and Filter Toxin Data, Length 15 to 300
    # ---------------------------------------------------------
    print("🚀 Step 1: Processing Toxin Data...")

    df_toxins = load_uniprot_tsv(TOXIN_TSV_PATH)

    df_toxins_filtered = df_toxins[
        (df_toxins["Length"] >= 15) & (df_toxins["Length"] <= 300)
    ].copy()

    with h5py.File(TOXIN_H5_PATH, "r") as f_toxin:
        toxin_h5_keys = set(f_toxin.keys())

    df_toxins_filtered = df_toxins_filtered[
        df_toxins_filtered["Entry"].isin(toxin_h5_keys)
    ].copy()

    print(
        f"   -> Retained {len(df_toxins_filtered)} toxins matching "
        "the 15-300 length criteria."
    )

    print(f"   -> Saving filtered toxin embeddings to: {OUTPUT_TOXIN_H5}")

    with h5py.File(TOXIN_H5_PATH, "r") as f_in, h5py.File(
        OUTPUT_TOXIN_H5, "w"
    ) as f_out:
        for entry in df_toxins_filtered["Entry"]:
            f_in.copy(entry, f_out)

    # ---------------------------------------------------------
    # Step 2: Filter Metazoa and Map Keys via Entry ID
    # ---------------------------------------------------------
    print("\n🚀 Step 2: Parsing and Pre-Filtering Metazoa Paired Keys...")

    df_metazoa = load_uniprot_tsv(METAZOA_TSV_PATH)

    df_metazoa = df_metazoa[
        (df_metazoa["Length"] >= 15) & (df_metazoa["Length"] <= 300)
    ].copy()

    metazoa_key_map = map_metazoa_h5_keys(METAZOA_H5_PATH)

    valid_pairs = [
        entry
        for entry, pairs in metazoa_key_map.items()
        if pairs["normal"] is not None and pairs["signalp6"] is not None
    ]

    df_metazoa_valid = df_metazoa[df_metazoa["Entry"].isin(valid_pairs)].copy()

    print(
        f"   -> Found {len(df_metazoa_valid)} fully-paired Metazoa sequences "
        "within 15-300 AA bounds."
    )

    # ---------------------------------------------------------
    # Step 3: Distribution Matching by Length Binning
    # ---------------------------------------------------------
    print("\n🚀 Step 3: Matching Metazoa Length Distribution to Toxins...")

    # Bins: 15-30, 30-45, ..., 285-300
    bin_edges = np.arange(15, 301, 15)

    toxin_counts, _ = np.histogram(df_toxins_filtered["Length"], bins=bin_edges)

    selected_metazoa_dfs = []

    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        target_count = toxin_counts[i]

        if i == len(bin_edges) - 2:
            candidates = df_metazoa_valid[
                (df_metazoa_valid["Length"] >= bin_min)
                & (df_metazoa_valid["Length"] <= bin_max)
            ]
        else:
            candidates = df_metazoa_valid[
                (df_metazoa_valid["Length"] >= bin_min)
                & (df_metazoa_valid["Length"] < bin_max)
            ]

        if len(candidates) == 0 or target_count == 0:
            continue

        if len(candidates) <= target_count:
            sampled = candidates
        else:
            sampled = candidates.sample(n=target_count, random_state=42)

        selected_metazoa_dfs.append(sampled)

    if not selected_metazoa_dfs:
        raise RuntimeError(
            "No Metazoa sequences were selected. Check length ranges, TSV entries, "
            "and H5 key matching."
        )

    df_metazoa_matched = pd.concat(selected_metazoa_dfs).copy()

    print(
        f"   -> Selected {len(df_metazoa_matched)} Metazoa sequences "
        "to match the toxin length distribution."
    )

    # ---------------------------------------------------------
    # Step 4: Write Matched Pairs and Delta
    # ---------------------------------------------------------
    print(
        f"\n🚀 Step 4: Extracting and Saving Paired Metazoa Embeddings to: "
        f"{OUTPUT_METAZOA_H5}"
    )

    print(
        f"🚀 Step 4b: Creating Delta Embeddings H5 file: "
        f"{OUTPUT_METAZOA_DELTA_H5}"
    )

    with h5py.File(METAZOA_H5_PATH, "r") as f_in, \
         h5py.File(OUTPUT_METAZOA_H5, "w") as f_pair_out, \
         h5py.File(OUTPUT_METAZOA_DELTA_H5, "w") as f_delta_out:

        for entry in df_metazoa_matched["Entry"]:
            normal_key = metazoa_key_map[entry]["normal"]
            signalp6_key = metazoa_key_map[entry]["signalp6"]

            # Copy both original paired embeddings using their exact original keys.
            f_in.copy(normal_key, f_pair_out)
            f_in.copy(signalp6_key, f_pair_out)

            # Standard delta:
            # full-sequence embedding minus signal-peptide-removed embedding.
            create_delta_dataset(
                f_in=f_in,
                f_delta_out=f_delta_out,
                entry=entry,
                normal_key=normal_key,
                signalp6_key=signalp6_key,
                scale_factor=1.0,
            )

    print("\n🎉 Pipeline Processing Complete!")
    print("-" * 60)

    # ---------------------------------------------------------
    # Step 5: Post-Write Verification Check
    # ---------------------------------------------------------
    print("🔍 Step 5: Verifying Output H5 File Entry Counts...")

    with h5py.File(OUTPUT_TOXIN_H5, "r") as f_tox_check:
        actual_toxin_count = len(f_tox_check.keys())

    with h5py.File(OUTPUT_METAZOA_H5, "r") as f_meta_check:
        actual_metazoa_count = len(f_meta_check.keys())

    with h5py.File(OUTPUT_METAZOA_DELTA_H5, "r") as f_delta_check:
        actual_delta_count = len(f_delta_check.keys())

    print(
        f"   📊 [CONFIRMED] Total unique entries in "
        f"'{OUTPUT_TOXIN_H5}': {actual_toxin_count}"
    )

    print(
        f"   📊 [CONFIRMED] Total unique entries in "
        f"'{OUTPUT_METAZOA_H5}': {actual_metazoa_count}"
    )

    print(
        f"       Note: Metazoa paired H5 contains exactly "
        f"{actual_metazoa_count // 2} matched full/signalp6 pairs."
    )

    print(
        f"   📊 [CONFIRMED] Total delta embeddings in "
        f"'{OUTPUT_METAZOA_DELTA_H5}': {actual_delta_count}"
    )

    print(
        "       Note: Delta H5 contains one dataset per matched Metazoa protein pair."
    )

    print("-" * 60)


if __name__ == "__main__":
    filter_and_match_distributions()