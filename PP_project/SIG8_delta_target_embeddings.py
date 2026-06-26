import os
import re
import numpy as np
import h5py

# =====================================================================
# --- FILE INPUT/OUTPUT CONFIGURATION ---
# =====================================================================

# Full toxin embeddings with signal peptide still present
FULL_SIGNALPEPTIDE_H5 = "embeddings/toxins_signalpeptide.h5"

# Toxin embeddings after SignalP removal
SIGNALP_REMOVED_H5 = "embeddings/toxins_signalpeptide_signalp9_removed_prot_t5.h5"

# One dataset per matched toxin:
#   ENTRY = full_signalpeptide_embedding - signalp_removed_embedding
OUTPUT_TOXIN_DELTA_H5 = (
    "embeddings/toxins_signalpeptide_signalp9_removed_"
    "delta_full_minus_removed_by_uniprot_id.h5"
)

# Symmetric RhoPCA background:
#   ENTRY_delta_pos = full_signalpeptide_embedding - signalp_removed_embedding
#   ENTRY_delta_neg = -(full_signalpeptide_embedding - signalp_removed_embedding)
OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5 = (
    "embeddings/toxins_signalpeptide_signalp9_removed_"
    "delta_symmetric_background_by_uniprot_id.h5"
)

# If duplicate UniProt accessions occur inside one H5, set this to True
# to keep the first one instead of raising an error.
ALLOW_DUPLICATES_KEEP_FIRST = False

# =====================================================================


def ensure_parent_dir(path):
    """Creates the parent directory of an output path if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def strip_known_suffixes(text):
    """
    Removes common suffixes that may have been added to H5 keys.
    This is mainly useful when keys are plain accessions like:
        A0A088MIT0_signalp9_removed
    """
    suffixes = [
        "_signalp9_removed",
        "_signalp9",
        "_signalp6_removed",
        "_signalp6",
        "_signalp_removed",
        "_removed",
        "_prot_t5",
    ]

    stripped = str(text).strip()

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)]
                changed = True

    return stripped


def extract_uniprot_accession_from_key(key):
    """
    Extracts the UniProt accession from common FASTA/H5 key formats.

    Supported examples:
        sp|A0A088MIT0|BRKP2_PHYNA Bradykinin-related peptides OS=...
            -> A0A088MIT0

        tr|A0A1234567|SOME_PROTEIN OS=...
            -> A0A1234567

        A0A088MIT0
            -> A0A088MIT0

        A0A088MIT0_signalp9_removed
            -> A0A088MIT0
    """
    key = str(key).strip()

    # Case 1: UniProt FASTA-style header/key with pipes.
    # Example: sp|A0A088MIT0|BRKP2_PHYNA ...
    if "|" in key:
        parts = key.split("|")
        if len(parts) >= 3 and parts[1].strip():
            return parts[1].strip()

    # Case 2: plain key, possibly with suffix.
    key = strip_known_suffixes(key)

    # Case 3: key may contain whitespace after accession.
    # Example: A0A088MIT0 something else
    key_first_token = key.split()[0]

    return key_first_token.strip()


def map_h5_keys_by_uniprot_id(h5_path):
    """
    Builds a map:

        UniProt accession -> actual H5 key

    Example:
        A0A088MIT0 -> sp|A0A088MIT0|BRKP2_PHYNA Bradykinin-related peptides OS=...

    This allows the full and SignalP-removed H5 files to be matched even when
    the literal H5 keys are not identical.
    """
    accession_to_key = {}
    duplicates = {}

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

    for key in keys:
        accession = extract_uniprot_accession_from_key(key)

        if accession not in accession_to_key:
            accession_to_key[accession] = key
        else:
            duplicates.setdefault(accession, []).append(key)

    if duplicates:
        msg = (
            f"Found {len(duplicates)} duplicated UniProt accessions in {h5_path}. "
            "This means multiple H5 keys map to the same accession."
        )

        if ALLOW_DUPLICATES_KEEP_FIRST:
            print(f"⚠️  Warning: {msg}")
            print("   Keeping the first key for each duplicated accession.")
        else:
            example_accession = next(iter(duplicates))
            raise ValueError(
                f"{msg}\n"
                f"Example duplicated accession: {example_accession}\n"
                f"First key kept candidate: {accession_to_key[example_accession]}\n"
                f"Other duplicate keys: {duplicates[example_accession]}\n"
                "Set ALLOW_DUPLICATES_KEEP_FIRST = True if this is expected."
            )

    return accession_to_key


def read_embedding(h5_file, key):
    """Reads one embedding from an H5 file as float32."""
    arr = np.asarray(h5_file[key][()], dtype=np.float32)

    if arr.ndim == 0:
        raise ValueError(f"Dataset at key '{key}' is scalar, not an embedding.")

    return arr


def write_delta_outputs(
    f_full,
    f_removed,
    f_delta_out,
    f_sym_out,
    accession,
    full_key,
    removed_key,
):
    """
    Computes:

        delta = full_signalpeptide_embedding - signalp_removed_embedding

    Writes:
        1. standard delta:
            accession

        2. symmetric RhoPCA background:
            accession_delta_pos
            accession_delta_neg
    """
    full_embedding = read_embedding(f_full, full_key)
    removed_embedding = read_embedding(f_removed, removed_key)

    if full_embedding.shape != removed_embedding.shape:
        raise ValueError(
            f"Embedding shape mismatch for accession {accession}:\n"
            f"   full key:    {full_key}\n"
            f"   full shape:  {full_embedding.shape}\n"
            f"   removed key: {removed_key}\n"
            f"   removed shape: {removed_embedding.shape}"
        )

    delta = full_embedding - removed_embedding

    # ---------------------------------------------------------
    # Standard one-delta-per-entry output
    # ---------------------------------------------------------
    dset_delta = f_delta_out.create_dataset(
        accession,
        data=delta.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    dset_delta.attrs["entry"] = accession
    dset_delta.attrs["delta_definition"] = (
        "full_signalpeptide_embedding - signalp_removed_embedding"
    )
    dset_delta.attrs["full_signalpeptide_key"] = full_key
    dset_delta.attrs["signalp_removed_key"] = removed_key
    dset_delta.attrs["full_signalpeptide_source_h5"] = FULL_SIGNALPEPTIDE_H5
    dset_delta.attrs["signalp_removed_source_h5"] = SIGNALP_REMOVED_H5

    # ---------------------------------------------------------
    # Symmetric RhoPCA background output
    # ---------------------------------------------------------
    pos_key = f"{accession}_delta_pos"
    neg_key = f"{accession}_delta_neg"

    dset_pos = f_sym_out.create_dataset(
        pos_key,
        data=delta.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    dset_neg = f_sym_out.create_dataset(
        neg_key,
        data=(-delta).astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    for dset, sign_label in [(dset_pos, "+delta"), (dset_neg, "-delta")]:
        dset.attrs["entry"] = accession
        dset.attrs["background_type"] = "symmetric_toxin_signalpeptide_delta"
        dset.attrs["sign"] = sign_label
        dset.attrs["delta_definition"] = (
            "full_signalpeptide_embedding - signalp_removed_embedding"
        )
        dset.attrs["full_signalpeptide_key"] = full_key
        dset.attrs["signalp_removed_key"] = removed_key
        dset.attrs["full_signalpeptide_source_h5"] = FULL_SIGNALPEPTIDE_H5
        dset.attrs["signalp_removed_source_h5"] = SIGNALP_REMOVED_H5


def main():
    print("🚀 Building toxin-specific signal-peptide delta background...")
    print("   Matching H5 entries by UniProt accession between pipe characters.")

    ensure_parent_dir(OUTPUT_TOXIN_DELTA_H5)
    ensure_parent_dir(OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5)

    # ---------------------------------------------------------
    # Step 1: Map both H5 files by UniProt accession
    # ---------------------------------------------------------
    print("\n🚀 Step 1: Mapping full-signalpeptide H5 keys...")
    full_key_map = map_h5_keys_by_uniprot_id(FULL_SIGNALPEPTIDE_H5)

    print("🚀 Step 2: Mapping SignalP-removed H5 keys...")
    removed_key_map = map_h5_keys_by_uniprot_id(SIGNALP_REMOVED_H5)

    full_accessions = set(full_key_map.keys())
    removed_accessions = set(removed_key_map.keys())

    shared_accessions = sorted(full_accessions & removed_accessions)

    only_full = sorted(full_accessions - removed_accessions)
    only_removed = sorted(removed_accessions - full_accessions)

    print(f"\n📊 Full signal-peptide H5 accessions: {len(full_accessions)}")
    print(f"📊 SignalP-removed H5 accessions: {len(removed_accessions)}")
    print(f"📊 Shared accessions: {len(shared_accessions)}")
    print(f"📊 Only in full H5: {len(only_full)}")
    print(f"📊 Only in removed H5: {len(only_removed)}")

    if len(shared_accessions) == 0:
        raise RuntimeError(
            "No shared UniProt accessions found between the two H5 files. "
            "Check the H5 key formats."
        )

    if only_full:
        print("\n⚠️  Example accessions only in full H5:")
        for accession in only_full[:10]:
            print(f"   {accession}")

    if only_removed:
        print("\n⚠️  Example accessions only in removed H5:")
        for accession in only_removed[:10]:
            print(f"   {accession}")

    # ---------------------------------------------------------
    # Step 2: Compute delta and symmetric delta background
    # ---------------------------------------------------------
    print("\n🚀 Step 3: Writing delta and symmetric RhoPCA background H5 files...")

    print(f"   -> Standard delta output:")
    print(f"      {OUTPUT_TOXIN_DELTA_H5}")

    print(f"   -> Symmetric RhoPCA background output:")
    print(f"      {OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5}")

    with h5py.File(FULL_SIGNALPEPTIDE_H5, "r") as f_full, \
         h5py.File(SIGNALP_REMOVED_H5, "r") as f_removed, \
         h5py.File(OUTPUT_TOXIN_DELTA_H5, "w") as f_delta_out, \
         h5py.File(OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5, "w") as f_sym_out:

        for accession in shared_accessions:
            full_key = full_key_map[accession]
            removed_key = removed_key_map[accession]

            write_delta_outputs(
                f_full=f_full,
                f_removed=f_removed,
                f_delta_out=f_delta_out,
                f_sym_out=f_sym_out,
                accession=accession,
                full_key=full_key,
                removed_key=removed_key,
            )

    # ---------------------------------------------------------
    # Step 3: Verify outputs
    # ---------------------------------------------------------
    print("\n🎉 Done!")
    print("-" * 60)

    with h5py.File(OUTPUT_TOXIN_DELTA_H5, "r") as f_delta_check:
        delta_count = len(f_delta_check.keys())

    with h5py.File(OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5, "r") as f_sym_check:
        symmetric_count = len(f_sym_check.keys())

    print(f"📊 Standard delta embeddings: {delta_count}")
    print(f"📊 Symmetric RhoPCA background entries: {symmetric_count}")
    print(f"📊 Expected symmetric count: {2 * delta_count}")

    if symmetric_count != 2 * delta_count:
        raise RuntimeError(
            f"Symmetric background count mismatch: expected {2 * delta_count}, "
            f"found {symmetric_count}"
        )

    print("-" * 60)
    print("Use this as the RhoPCA background:")
    print(f"   {OUTPUT_TOXIN_SYMMETRIC_DELTA_BACKGROUND_H5}")
    print("-" * 60)


if __name__ == "__main__":
    main()