"""Build toxin target and background embedding datasets.

The script expects the project data below ``PP_project/`` and writes two target
datasets plus three background datasets. Run it from any working directory:

    uv run TOX_backgrounds.py

Use ``--overwrite`` to replace outputs from an earlier run.
"""

from __future__ import annotations

import argparse
import bisect
import random
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

ROOT = Path(__file__).resolve().parent
PP_PROJECT = ROOT / "PP_project"
DATASETS = PP_PROJECT / "datasets"
EMBEDDINGS = PP_PROJECT / "embeddings"

TOXINS_TSV = DATASETS / "toxins_20_2000.tsv"
NONTOXINS_TSV = DATASETS / "nontoxins_20_2000.tsv"
TOXINS_PROCESSED_FASTA = DATASETS / "toxins_20_2000_signal_processed.fasta"
NONTOXINS_PROCESSED_FASTA = (
    DATASETS / "nontoxins_20_2000_signal_processed.fasta"
)

MASTER_H5 = EMBEDDINGS / "per-protein.h5"
TOXINS_PROCESSED_H5 = (
    EMBEDDINGS / "toxins_20_2000_signal_processed_prot_t5.h5"
)
NONTOXINS_PROCESSED_H5 = (
    EMBEDDINGS / "nontoxins_20_2000_signal_processed_prot_t5.h5"
)

TARGET_ALL_H5 = EMBEDDINGS / "toxins_20_2000.h5"
TARGET_MIXED_H5 = (
    EMBEDDINGS / "toxins_20_2000_signal_processed_puls_nonsignal.h5"
)
TARGET_MIXED_TSV = (
    DATASETS / "toxins_20_2000_signal_processed_puls_nonsignal.tsv"
)

BACKGROUND_5050_H5 = EMBEDDINGS / "nontoxins_20_2000_5050_lengthmatched.h5"
BACKGROUND_5050_TSV = DATASETS / "nontoxins_20_2000_5050_lengthmatched.tsv"

BACKGROUND_NONTOXIN_PROCESSED_H5 = (
    EMBEDDINGS / "nontoxins_20_2000_5050_processed.h5"
)
BACKGROUND_NONTOXIN_PROCESSED_TSV = (
    DATASETS / "nontoxins_20_2000_5050_processed.tsv"
)

BACKGROUND_TOXIN_PROCESSED_H5 = EMBEDDINGS / "toxins_20_2000_5050_processed.h5"
BACKGROUND_TOXIN_PROCESSED_TSV = DATASETS / "toxins_20_2000_5050_processed.tsv"

INPUTS = [
    TOXINS_TSV,
    NONTOXINS_TSV,
    TOXINS_PROCESSED_FASTA,
    NONTOXINS_PROCESSED_FASTA,
    MASTER_H5,
    TOXINS_PROCESSED_H5,
    NONTOXINS_PROCESSED_H5,
]

OUTPUTS = [
    TARGET_ALL_H5,
    TARGET_MIXED_H5,
    TARGET_MIXED_TSV,
    BACKGROUND_5050_H5,
    BACKGROUND_5050_TSV,
    BACKGROUND_NONTOXIN_PROCESSED_H5,
    BACKGROUND_NONTOXIN_PROCESSED_TSV,
    BACKGROUND_TOXIN_PROCESSED_H5,
    BACKGROUND_TOXIN_PROCESSED_TSV,
]


@dataclass(frozen=True)
class EmbeddingRecord:
    """One source HDF5 dataset copied to one output key."""

    source_path: Path
    source_key: str
    output_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic length matching (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested output files if they already exist.",
    )
    return parser.parse_args()


def accession_from_identifier(identifier: str) -> str:
    """Return a bare accession from a bare or ``sp|ACCESSION|NAME`` ID."""
    token = str(identifier).strip().split(maxsplit=1)[0]
    parts = token.split("|")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return token


def read_metadata(path: Path) -> pd.DataFrame:
    """Load and validate a UniProt TSV while preserving blank signal cells."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Entry", "Length", "Signal peptide"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    frame["Entry"] = frame["Entry"].str.strip()
    if frame["Entry"].eq("").any():
        raise ValueError(f"{path} contains blank Entry values")
    duplicates = frame.loc[frame["Entry"].duplicated(), "Entry"].unique().tolist()
    if duplicates:
        raise ValueError(f"{path} contains duplicate entries: {duplicates[:10]}")

    lengths = pd.to_numeric(frame["Length"], errors="coerce")
    invalid_lengths = frame.loc[lengths.isna() | (lengths <= 0), "Entry"].tolist()
    if invalid_lengths:
        raise ValueError(f"{path} has invalid lengths for: {invalid_lengths[:10]}")
    frame["Length"] = lengths.astype(int)
    frame["Signal peptide"] = frame["Signal peptide"].str.strip()
    return frame


def read_fasta_lengths(path: Path) -> dict[str, int]:
    """Read FASTA sequence lengths, indexed by bare accession."""
    lengths: dict[str, int] = {}
    current_accession: str | None = None
    current_length = 0

    def store_record() -> None:
        if current_accession is None:
            return
        if current_accession in lengths:
            raise ValueError(f"Duplicate FASTA accession {current_accession!r} in {path}")
        lengths[current_accession] = current_length

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                store_record()
                header_id = line[1:].split(maxsplit=1)[0]
                current_accession = accession_from_identifier(header_id)
                if not current_accession:
                    raise ValueError(f"Blank FASTA ID in {path} at line {line_number}")
                current_length = 0
            else:
                if current_accession is None:
                    raise ValueError(
                        f"Sequence before first FASTA header in {path} at line {line_number}"
                    )
                current_length += len("".join(line.split()))
    store_record()
    return lengths


def embedding_group(handle: h5py.File) -> h5py.Group | h5py.File:
    """Return the dataset-containing group for flat or ``prot_t5`` HDF5 files."""
    if "prot_t5" in handle and isinstance(handle["prot_t5"], h5py.Group):
        return handle["prot_t5"]
    return handle


def index_h5(path: Path) -> tuple[dict[str, str], int, tuple[int, ...], str]:
    """Map bare accessions to actual HDF5 keys and report the embedding signature."""
    with h5py.File(path, "r") as handle:
        group = embedding_group(handle)
        total = len(group)
        index: dict[str, str] = {}
        first_dataset: h5py.Dataset | None = None
        for key in group.keys():
            item = group[key]
            if not isinstance(item, h5py.Dataset):
                continue
            accession = accession_from_identifier(key)
            if accession in index:
                raise ValueError(
                    f"{path} has multiple datasets for accession {accession!r}"
                )
            index[accession] = key
            if first_dataset is None:
                first_dataset = item
        if first_dataset is None:
            raise ValueError(f"{path} contains no embedding datasets")
        return index, total, tuple(first_dataset.shape), str(first_dataset.dtype)


def index_master_h5(
    path: Path, requested_accessions: set[str]
) -> tuple[dict[str, str], int, tuple[int, ...], str]:
    """Index only requested bare-accession keys in the large master HDF5.

    Avoiding a full Python-level scan is important for the large, flat
    ``per-protein.h5`` file, especially when it is stored on OneDrive.
    """
    with h5py.File(path, "r") as handle:
        group = embedding_group(handle)
        index = {
            accession: accession
            for accession in requested_accessions
            if accession in group and isinstance(group[accession], h5py.Dataset)
        }
        if not index:
            raise ValueError(f"{path} contains none of the requested accessions")
        first_dataset = group[next(iter(index.values()))]
        return index, len(group), tuple(first_dataset.shape), str(first_dataset.dtype)


def require_accessions(
    accessions: Iterable[str], index: dict[str, str], context: str
) -> None:
    missing = sorted(set(accessions) - set(index))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{context}: {len(missing)} accessions are missing from the HDF5 file"
            f" (first: {preview})"
        )


def validate_inputs_and_outputs(overwrite: bool) -> None:
    missing = [path for path in INPUTS if not path.is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required input files are missing:\n{listing}")
    existing = [path for path in OUTPUTS if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist; rerun with --overwrite to replace them:\n"
            f"{listing}"
        )


def length_match_without_replacement(
    target_lengths: list[int], candidates: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, list[int]]:
    """Nearest-neighbor match candidate lengths to targets without replacement.

    Target lengths farthest from the target median are matched first. This protects
    sparse distribution tails from being consumed by central observations. Within
    each candidate length and for equal-distance ties, selection is seeded.
    """
    if len(candidates) < len(target_lengths):
        raise ValueError(
            f"Need {len(target_lengths)} candidates but only {len(candidates)} exist"
        )

    rng = random.Random(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for row_index, length in zip(
        candidates.index.tolist(), candidates["Length"].tolist(), strict=True
    ):
        buckets[int(length)].append(row_index)
    for row_indices in buckets.values():
        rng.shuffle(row_indices)

    available_lengths = sorted(buckets)
    sorted_targets = sorted(target_lengths)
    median = sorted_targets[len(sorted_targets) // 2]
    processing_order = sorted(
        enumerate(target_lengths),
        key=lambda item: (-abs(item[1] - median), item[1], item[0]),
    )

    selected_by_target: list[int | None] = [None] * len(target_lengths)
    deltas_by_target: list[int | None] = [None] * len(target_lengths)

    for target_index, target_length in processing_order:
        position = bisect.bisect_left(available_lengths, target_length)
        choices = []
        if position > 0:
            choices.append(available_lengths[position - 1])
        if position < len(available_lengths):
            choices.append(available_lengths[position])
        best_distance = min(abs(length - target_length) for length in choices)
        nearest = [
            length
            for length in choices
            if abs(length - target_length) == best_distance
        ]
        chosen_length = rng.choice(nearest)
        chosen_index = buckets[chosen_length].pop()
        selected_by_target[target_index] = chosen_index
        deltas_by_target[target_index] = abs(chosen_length - target_length)
        if not buckets[chosen_length]:
            del buckets[chosen_length]
            available_lengths.pop(bisect.bisect_left(available_lengths, chosen_length))

    selected_indices = [index for index in selected_by_target if index is not None]
    deltas = [delta for delta in deltas_by_target if delta is not None]
    if len(selected_indices) != len(target_lengths):
        raise RuntimeError("Internal length-matching error: incomplete selection")
    return candidates.loc[selected_indices].copy(), deltas


def report_match(label: str, deltas: list[int]) -> None:
    ordered = sorted(deltas)
    median = ordered[len(ordered) // 2]
    exact = sum(delta == 0 for delta in deltas)
    mean = sum(deltas) / len(deltas)
    print(
        f"  {label}: n={len(deltas):,}, exact={exact:,} "
        f"({exact / len(deltas):.1%}), mean absolute length delta={mean:.2f}, "
        f"median={median}, max={max(deltas)}"
    )


def records_for_master(
    entries: list[str], master_index: dict[str, str]
) -> list[EmbeddingRecord]:
    require_accessions(entries, master_index, "Master embedding selection")
    return [
        EmbeddingRecord(MASTER_H5, master_index[entry], entry) for entry in entries
    ]


def records_for_processed(
    entries: list[str],
    processed_path: Path,
    processed_index: dict[str, str],
    *,
    suffix: str = "",
) -> list[EmbeddingRecord]:
    require_accessions(entries, processed_index, f"Processed selection from {processed_path}")
    return [
        EmbeddingRecord(processed_path, processed_index[entry], f"{entry}{suffix}")
        for entry in entries
    ]


def write_h5(path: Path, records: list[EmbeddingRecord], overwrite: bool) -> None:
    output_keys = [record.output_key for record in records]
    if len(output_keys) != len(set(output_keys)):
        duplicates = pd.Series(output_keys)[pd.Series(output_keys).duplicated()].unique()
        raise ValueError(f"Duplicate output HDF5 keys for {path}: {duplicates[:10]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with ExitStack() as stack:
        handles = {
            source: stack.enter_context(h5py.File(source, "r"))
            for source in {record.source_path for record in records}
        }
        groups = {source: embedding_group(handle) for source, handle in handles.items()}
        output = stack.enter_context(h5py.File(path, mode))
        output.attrs["model_name"] = "prot_t5"
        for record in records:
            groups[record.source_path].copy(
                record.source_key, output, name=record.output_key
            )
    print(f"  Wrote {path.relative_to(ROOT)}: {len(records):,} embeddings")


def write_tsv(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    print(f"  Wrote {path.relative_to(ROOT)}: {len(frame):,} rows")


def validate_output_h5(path: Path, expected_entries: Iterable[str]) -> None:
    """Verify output keys, embedding dimensions, and model metadata."""
    expected = list(expected_entries)
    if len(expected) != len(set(expected)):
        raise ValueError(f"Expected output entries for {path} are not unique")
    with h5py.File(path, "r") as handle:
        actual = list(handle.keys())
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"Output verification failed for {path}: missing={missing[:10]}, "
                f"extra={extra[:10]}"
            )
        if str(handle.attrs.get("model_name", "")) != "prot_t5":
            raise ValueError(f"Output {path} is missing model_name=prot_t5")
        invalid_shapes = [key for key in actual if tuple(handle[key].shape) != (1024,)]
        if invalid_shapes:
            raise ValueError(
                f"Output {path} has non-1024 embeddings: {invalid_shapes[:10]}"
            )
    print(f"  Verified {path.relative_to(ROOT)}: {len(actual):,} unique embeddings")


def validate_output_pair(h5_path: Path, tsv_path: Path) -> None:
    """Verify that a metadata TSV has one row for every output HDF5 key."""
    metadata = pd.read_csv(tsv_path, sep="\t", dtype={"Entry": str})
    if "Entry" not in metadata or metadata["Entry"].isna().any():
        raise ValueError(f"Output metadata {tsv_path} has invalid Entry values")
    validate_output_h5(h5_path, metadata["Entry"].tolist())
    print(f"  Verified {tsv_path.relative_to(ROOT)}: {len(metadata):,} matching rows")


def paired_metadata(
    entries: list[str],
    original_lengths: dict[str, int],
    processed_lengths: dict[str, int],
) -> pd.DataFrame:
    original = pd.DataFrame(
        {"Entry": entries, "Length": [original_lengths[entry] for entry in entries]}
    )
    processed = pd.DataFrame(
        {
            "Entry": [f"{entry}_processed" for entry in entries],
            "Length": [processed_lengths[entry] for entry in entries],
        }
    )
    return pd.concat([original, processed], ignore_index=True)


def main() -> None:
    args = parse_args()
    validate_inputs_and_outputs(args.overwrite)

    print("Loading metadata and processed FASTA lengths...")
    toxins = read_metadata(TOXINS_TSV)
    nontoxins = read_metadata(NONTOXINS_TSV)
    toxin_processed_lengths = read_fasta_lengths(TOXINS_PROCESSED_FASTA)
    nontoxin_processed_lengths = read_fasta_lengths(NONTOXINS_PROCESSED_FASTA)

    toxin_signal = toxins["Signal peptide"].ne("")
    nontoxin_signal = nontoxins["Signal peptide"].ne("")
    toxins_signal = toxins.loc[toxin_signal].copy()
    toxins_nonsignal = toxins.loc[~toxin_signal].copy()
    nontoxins_signal = nontoxins.loc[nontoxin_signal].copy()
    nontoxins_nonsignal = nontoxins.loc[~nontoxin_signal].copy()

    print(
        f"  {TOXINS_TSV.relative_to(ROOT)}: {len(toxins):,} total "
        f"({len(toxins_signal):,} signal+, {len(toxins_nonsignal):,} signal-)"
    )
    print(
        f"  {NONTOXINS_TSV.relative_to(ROOT)}: {len(nontoxins):,} total "
        f"({len(nontoxins_signal):,} signal+, {len(nontoxins_nonsignal):,} signal-)"
    )
    print(
        f"  {TOXINS_PROCESSED_FASTA.relative_to(ROOT)}: "
        f"{len(toxin_processed_lengths):,} processed sequences"
    )
    print(
        f"  {NONTOXINS_PROCESSED_FASTA.relative_to(ROOT)}: "
        f"{len(nontoxin_processed_lengths):,} processed sequences"
    )

    print("Indexing embedding files...")
    requested_master_ids = set(toxins["Entry"]) | set(nontoxins["Entry"])
    master_index, master_total, master_shape, master_dtype = index_master_h5(
        MASTER_H5, requested_master_ids
    )
    toxin_processed_index, toxin_processed_total, toxin_shape, toxin_dtype = (
        index_h5(TOXINS_PROCESSED_H5)
    )
    (
        nontoxin_processed_index,
        nontoxin_processed_total,
        nontoxin_shape,
        nontoxin_dtype,
    ) = index_h5(
        NONTOXINS_PROCESSED_H5
    )
    shapes = {master_shape, toxin_shape, nontoxin_shape}
    if len(shapes) != 1:
        raise ValueError(
            "Embedding files have incompatible shapes: "
            f"master={master_shape}, toxins processed={toxin_shape}, "
            f"nontoxins processed={nontoxin_shape}"
        )
    dtypes = {master_dtype, toxin_dtype, nontoxin_dtype}
    if len(dtypes) > 1:
        print(
            "  INFO: source dtypes differ "
            f"(master={master_dtype}, toxins processed={toxin_dtype}, "
            f"nontoxins processed={nontoxin_dtype}); datasets are copied as-is "
            "and ProtSpace promotes the combined array to float32 when loading"
        )
    print(
        f"  {MASTER_H5.relative_to(ROOT)}: {master_total:,} embeddings; "
        f"{len(master_index):,}/{len(requested_master_ids):,} requested accessions "
        f"found; shape={master_shape}, dtype={master_dtype}"
    )
    print(
        f"  {TOXINS_PROCESSED_H5.relative_to(ROOT)}: "
        f"{toxin_processed_total:,} embeddings"
    )
    print(
        f"  {NONTOXINS_PROCESSED_H5.relative_to(ROOT)}: "
        f"{nontoxin_processed_total:,} embeddings"
    )

    missing_toxin_master = sorted(set(toxins["Entry"]) - set(master_index))
    missing_nontoxin_master = sorted(set(nontoxins["Entry"]) - set(master_index))
    if missing_toxin_master:
        print(
            f"  WARNING: excluding {len(missing_toxin_master):,} toxin entries missing "
            f"from the master HDF5 (first: {', '.join(missing_toxin_master[:10])})"
        )
    if missing_nontoxin_master:
        print(
            f"  WARNING: excluding {len(missing_nontoxin_master):,} nontoxin entries "
            f"missing from the master HDF5 (first: "
            f"{', '.join(missing_nontoxin_master[:10])})"
        )
    toxins_available = toxins[toxins["Entry"].isin(master_index)].copy()
    toxin_processed_difference = set(toxin_processed_index) ^ set(
        toxin_processed_lengths
    )
    if toxin_processed_difference:
        raise ValueError(
            "Toxin processed HDF5 and FASTA accessions differ: "
            f"{sorted(toxin_processed_difference)[:10]}"
        )
    nontoxin_processed_difference = set(nontoxin_processed_index) ^ set(
        nontoxin_processed_lengths
    )
    if nontoxin_processed_difference:
        raise ValueError(
            "Nontoxin processed HDF5 and FASTA accessions differ: "
            f"{sorted(nontoxin_processed_difference)[:10]}"
        )

    toxin_signal_ids = set(toxins_signal["Entry"])
    toxin_processed_ids = set(toxin_processed_index) & set(toxin_processed_lengths)
    unexpected_toxin_processed = sorted(toxin_processed_ids - toxin_signal_ids)
    if unexpected_toxin_processed:
        raise ValueError(
            "Toxin processed inputs contain accessions not marked signal-positive: "
            f"{unexpected_toxin_processed[:10]}"
        )
    toxin_pair_ids = [
        entry
        for entry in toxins_signal["Entry"]
        if entry in toxin_processed_ids and entry in master_index
    ]
    print(
        f"  Toxin processable pairs: {len(toxin_pair_ids):,}; skipped "
        f"{len(toxins_signal) - len(toxin_pair_ids):,} signal+ rows without both "
        "original and processed embeddings/FASTA records"
    )

    nontoxin_processed_ids = set(nontoxin_processed_index) & set(
        nontoxin_processed_lengths
    )
    nontoxin_signal_ids = set(nontoxins_signal["Entry"])
    unexpected_nontoxin_processed = sorted(
        nontoxin_processed_ids - nontoxin_signal_ids
    )
    if unexpected_nontoxin_processed:
        raise ValueError(
            "Nontoxin processed inputs contain accessions not marked signal-positive: "
            f"{unexpected_nontoxin_processed[:10]}"
        )

    signal_pool = nontoxins_signal[
        nontoxins_signal["Entry"].isin(
            nontoxin_processed_ids & set(master_index)
        )
    ].copy()
    nonsignal_pool = nontoxins_nonsignal[
        nontoxins_nonsignal["Entry"].isin(master_index)
    ].copy()
    print(
        f"  Eligible nontoxin pools: {len(signal_pool):,} signal+ with processed "
        f"counterparts; {len(nonsignal_pool):,} signal- with master embeddings"
    )

    print("Matching both nontoxin classes to the toxin length distribution...")
    target_lengths = toxins["Length"].astype(int).tolist()
    matched_signal, signal_deltas = length_match_without_replacement(
        target_lengths, signal_pool, args.seed
    )
    matched_nonsignal, nonsignal_deltas = length_match_without_replacement(
        target_lengths, nonsignal_pool, args.seed + 1
    )
    report_match("signal+", signal_deltas)
    report_match("signal-", nonsignal_deltas)

    matched_signal_ids = matched_signal["Entry"].tolist()
    matched_nonsignal_ids = matched_nonsignal["Entry"].tolist()
    matched_5050 = pd.concat([matched_signal, matched_nonsignal], ignore_index=True)
    matched_5050 = matched_5050.sample(frac=1, random_state=args.seed).reset_index(
        drop=True
    )

    toxin_lengths = dict(zip(toxins["Entry"], toxins["Length"], strict=True))
    nontoxin_lengths = dict(
        zip(nontoxins["Entry"], nontoxins["Length"], strict=True)
    )

    print("Writing Target 1: all toxin embeddings...")
    toxin_all_ids = toxins_available["Entry"].tolist()
    write_h5(
        TARGET_ALL_H5,
        records_for_master(toxin_all_ids, master_index),
        args.overwrite,
    )

    print("Writing Target 2: signal- toxins plus processed signal+ toxins...")
    toxins_nonsignal_available = toxins_nonsignal[
        toxins_nonsignal["Entry"].isin(master_index)
    ].copy()
    target_mixed_records = records_for_master(
        toxins_nonsignal_available["Entry"].tolist(), master_index
    ) + records_for_processed(
        sorted(toxin_processed_ids),
        TOXINS_PROCESSED_H5,
        toxin_processed_index,
    )
    target_mixed_metadata = pd.concat(
        [
            toxins_nonsignal_available[["Entry", "Length"]],
            pd.DataFrame(
                {
                    "Entry": sorted(toxin_processed_ids),
                    "Length": [
                        toxin_processed_lengths[entry]
                        for entry in sorted(toxin_processed_ids)
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    write_h5(TARGET_MIXED_H5, target_mixed_records, args.overwrite)
    write_tsv(TARGET_MIXED_TSV, target_mixed_metadata, args.overwrite)

    print("Writing Background 1: 50/50 length-matched nontoxins...")
    background_5050_ids = matched_5050["Entry"].tolist()
    write_tsv(BACKGROUND_5050_TSV, matched_5050, args.overwrite)
    write_h5(
        BACKGROUND_5050_H5,
        records_for_master(background_5050_ids, master_index),
        args.overwrite,
    )

    print("Writing Background 2: paired original/processed matched nontoxins...")
    background_nontoxin_records = records_for_master(
        matched_signal_ids, master_index
    ) + records_for_processed(
        matched_signal_ids,
        NONTOXINS_PROCESSED_H5,
        nontoxin_processed_index,
        suffix="_processed",
    )
    background_nontoxin_metadata = paired_metadata(
        matched_signal_ids, nontoxin_lengths, nontoxin_processed_lengths
    )
    write_h5(
        BACKGROUND_NONTOXIN_PROCESSED_H5,
        background_nontoxin_records,
        args.overwrite,
    )
    write_tsv(
        BACKGROUND_NONTOXIN_PROCESSED_TSV,
        background_nontoxin_metadata,
        args.overwrite,
    )

    print("Writing Background 3: paired original/processed signal+ toxins...")
    background_toxin_records = records_for_master(
        toxin_pair_ids, master_index
    ) + records_for_processed(
        toxin_pair_ids,
        TOXINS_PROCESSED_H5,
        toxin_processed_index,
        suffix="_processed",
    )
    background_toxin_metadata = paired_metadata(
        toxin_pair_ids, toxin_lengths, toxin_processed_lengths
    )
    write_h5(
        BACKGROUND_TOXIN_PROCESSED_H5,
        background_toxin_records,
        args.overwrite,
    )
    write_tsv(
        BACKGROUND_TOXIN_PROCESSED_TSV,
        background_toxin_metadata,
        args.overwrite,
    )

    print("Verifying all output files...")
    validate_output_h5(TARGET_ALL_H5, toxin_all_ids)
    validate_output_pair(TARGET_MIXED_H5, TARGET_MIXED_TSV)
    validate_output_pair(BACKGROUND_5050_H5, BACKGROUND_5050_TSV)
    validate_output_pair(
        BACKGROUND_NONTOXIN_PROCESSED_H5,
        BACKGROUND_NONTOXIN_PROCESSED_TSV,
    )
    validate_output_pair(
        BACKGROUND_TOXIN_PROCESSED_H5,
        BACKGROUND_TOXIN_PROCESSED_TSV,
    )

    print("\nDone. Output summary:")
    print(f"  Target 1: {len(toxin_all_ids):,} embeddings")
    print(f"  Target 2: {len(target_mixed_records):,} embeddings/metadata rows")
    print(
        f"  Background 1: {len(background_5050_ids):,} embeddings/metadata rows "
        f"({len(matched_signal_ids):,} signal+, {len(matched_nonsignal_ids):,} signal-)"
    )
    print(
        f"  Background 2: {len(background_nontoxin_records):,} paired "
        "embeddings/metadata rows"
    )
    print(
        f"  Background 3: {len(background_toxin_records):,} paired "
        "embeddings/metadata rows"
    )


if __name__ == "__main__":
    main()
