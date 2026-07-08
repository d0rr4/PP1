"""Build 3FTx-specific target/background embedding datasets.

This script creates the 3FTx-specific analogues of the toxin background files:

1. ``non3FTx_5050``
   A 50/50 signal+/signal- nontoxin background. Both classes are independently
   length-matched without replacement to the length distribution in
   ``datasets/3FTx_full_fallback.fasta``.

2. ``non3FTx_5050_processed``
   The matched signal+ nontoxins from ``non3FTx_5050`` paired with their
   processed signal-peptide embeddings. Processed entries get a ``_processed``
   suffix.

3. ``3FTx_5050_processed``
   3FTx entries that have a true full sequence according to
   ``datasets/3FTx_annotation.csv`` paired with their mature embeddings.
   Mature entries get a ``_processed`` suffix.

Run from any directory:

    uv run PP_project/3FTx_background.py

Use ``--overwrite`` to replace outputs from an earlier run.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import random
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py

PP_PROJECT = Path(__file__).resolve().parent
ROOT = PP_PROJECT.parent
DATASETS = PP_PROJECT / "datasets"
EMBEDDINGS = PP_PROJECT / "embeddings"

ANNOTATION_CSV = DATASETS / "3FTx_annotation.csv"
FULL_FASTA = DATASETS / "3FTx_full_fallback.fasta"
MATURE_FASTA = DATASETS / "3FTx_mature.fasta"

NONTOXINS_TSV = DATASETS / "nontoxins_20_2000.tsv"
NONTOXINS_PROCESSED_FASTA = DATASETS / "nontoxins_20_2000_signal_processed.fasta"

MASTER_H5 = EMBEDDINGS / "per-protein.h5"
FULL_H5 = EMBEDDINGS / "3FTx_full_fallback_prot_t5.h5"
MATURE_H5 = EMBEDDINGS / "3FTx_mature_prot_t5.h5"
NONTOXINS_PROCESSED_H5 = (
    EMBEDDINGS / "nontoxins_20_2000_signal_processed_prot_t5.h5"
)

NON3FTX_5050_H5 = EMBEDDINGS / "non3FTx_5050.h5"
NON3FTX_5050_TSV = DATASETS / "non3FTx_5050.tsv"

NON3FTX_5050_PROCESSED_H5 = EMBEDDINGS / "non3FTx_5050_processed.h5"
NON3FTX_5050_PROCESSED_TSV = DATASETS / "non3FTx_5050_processed.tsv"

FTX_5050_PROCESSED_H5 = EMBEDDINGS / "3FTx_5050_processed.h5"
FTX_5050_PROCESSED_TSV = DATASETS / "3FTx_5050_processed.tsv"

INPUTS = [
    ANNOTATION_CSV,
    FULL_FASTA,
    MATURE_FASTA,
    NONTOXINS_TSV,
    NONTOXINS_PROCESSED_FASTA,
    MASTER_H5,
    FULL_H5,
    MATURE_H5,
    NONTOXINS_PROCESSED_H5,
]
OUTPUTS = [
    NON3FTX_5050_H5,
    NON3FTX_5050_TSV,
    NON3FTX_5050_PROCESSED_H5,
    NON3FTX_5050_PROCESSED_TSV,
    FTX_5050_PROCESSED_H5,
    FTX_5050_PROCESSED_TSV,
]


@dataclass(frozen=True)
class EmbeddingRecord:
    """One source HDF5 dataset copied to one output key."""

    source_path: Path
    source_key: str
    output_key: str


@dataclass(frozen=True)
class MetadataRow:
    """One UniProt metadata row with the fields needed for matching."""

    entry: str
    length: int
    signal_peptide: str
    raw: dict[str, str]


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


def parse_bool(value: object) -> bool:
    """Parse the boolean spellings used in the annotation table."""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def read_full_sequence_entries(path: Path) -> list[str]:
    """Read unique annotation IDs where ``has_full`` is true."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        required = {"ID", "has_full"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

        entries: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            entry = accession_from_identifier(row["ID"])
            if not entry:
                raise ValueError(f"{path} has a blank ID at row {row_number}")
            if entry in seen:
                raise ValueError(f"{path} contains duplicate ID {entry!r}")
            seen.add(entry)
            if parse_bool(row["has_full"]):
                entries.append(entry)
    return entries


def read_nontoxin_metadata(path: Path) -> tuple[list[MetadataRow], list[str]]:
    """Load nontoxin metadata while preserving all original TSV columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        fieldnames = list(reader.fieldnames)
        required = {"Entry", "Length", "Signal peptide"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

        rows: list[MetadataRow] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            entry = accession_from_identifier(row["Entry"])
            if not entry:
                raise ValueError(f"{path} has a blank Entry at row {row_number}")
            if entry in seen:
                raise ValueError(f"{path} contains duplicate Entry {entry!r}")
            seen.add(entry)
            try:
                length = int(row["Length"])
            except ValueError as error:
                raise ValueError(
                    f"{path} has invalid Length for {entry!r} at row {row_number}"
                ) from error
            if length <= 0:
                raise ValueError(
                    f"{path} has non-positive Length for {entry!r} at row {row_number}"
                )
            normalized_row = {field: row.get(field, "") for field in fieldnames}
            normalized_row["Entry"] = entry
            normalized_row["Length"] = str(length)
            rows.append(
                MetadataRow(
                    entry=entry,
                    length=length,
                    signal_peptide=row["Signal peptide"].strip(),
                    raw=normalized_row,
                )
            )
    return rows, fieldnames


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
                        f"Sequence before first FASTA header in {path} at line "
                        f"{line_number}"
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
    """Map bare accessions to actual HDF5 keys and report embedding signature."""
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
    """Index only requested bare-accession keys in the large master HDF5."""
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


def length_match_without_replacement(
    target_lengths: Sequence[int], pool: Sequence[MetadataRow], seed: int
) -> tuple[list[MetadataRow], list[int]]:
    """Greedily choose nearest-length rows from a pool without replacement."""
    if len(pool) < len(target_lengths):
        raise ValueError(
            f"Pool has only {len(pool):,} rows but {len(target_lengths):,} are needed"
        )

    rng = random.Random(seed)
    candidates = list(pool)
    rng.shuffle(candidates)
    candidates.sort(key=lambda row: row.length)
    candidate_lengths = [row.length for row in candidates]

    shuffled_targets = list(target_lengths)
    rng.shuffle(shuffled_targets)

    selected: list[MetadataRow] = []
    deltas: list[int] = []
    for target_length in shuffled_targets:
        insert_at = bisect.bisect_left(candidate_lengths, target_length)
        options = []
        if insert_at < len(candidates):
            options.append(insert_at)
        if insert_at > 0:
            options.append(insert_at - 1)
        best_index = min(
            options,
            key=lambda index: (abs(candidate_lengths[index] - target_length), rng.random()),
        )
        selected_row = candidates.pop(best_index)
        selected_length = candidate_lengths.pop(best_index)
        selected.append(selected_row)
        deltas.append(abs(selected_length - target_length))
    return selected, deltas


def report_match(label: str, deltas: Sequence[int]) -> None:
    exact = sum(delta == 0 for delta in deltas)
    mean = sum(deltas) / len(deltas)
    sorted_deltas = sorted(deltas)
    median = sorted_deltas[len(sorted_deltas) // 2]
    print(
        f"  {label}: n={len(deltas):,}, exact={exact:,} "
        f"({exact / len(deltas):.1%}), mean_abs_delta={mean:.2f}, "
        f"median_abs_delta={median}, max_abs_delta={max(deltas)}"
    )


def records_from_index(
    entries: Sequence[str],
    source_path: Path,
    source_index: dict[str, str],
    *,
    suffix: str = "",
) -> list[EmbeddingRecord]:
    require_accessions(entries, source_index, f"Embedding selection from {source_path}")
    return [
        EmbeddingRecord(source_path, source_index[entry], f"{entry}{suffix}")
        for entry in entries
    ]


def write_h5(path: Path, records: list[EmbeddingRecord], overwrite: bool) -> None:
    output_keys = [record.output_key for record in records]
    if not output_keys:
        raise ValueError("No embedding records selected for output")
    seen: set[str] = set()
    duplicates: list[str] = []
    for key in output_keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
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


def write_metadata_tsv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path.relative_to(ROOT)}: {len(rows):,} rows")


def paired_metadata(
    entries: Sequence[str],
    original_lengths: dict[str, int],
    processed_lengths: dict[str, int],
) -> list[dict[str, object]]:
    return [
        {"Entry": entry, "Length": original_lengths[entry]} for entry in entries
    ] + [
        {"Entry": f"{entry}_processed", "Length": processed_lengths[entry]}
        for entry in entries
    ]


def validate_output_pair(h5_path: Path, tsv_path: Path) -> None:
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        expected = [row["Entry"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(expected) != len(set(expected)):
        raise ValueError(f"Expected output entries for {tsv_path} are not unique")

    with h5py.File(h5_path, "r") as handle:
        actual = list(handle.keys())
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"Output verification failed for {h5_path}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        if str(handle.attrs.get("model_name", "")) != "prot_t5":
            raise ValueError(f"Output {h5_path} is missing model_name=prot_t5")
        invalid_shapes = [key for key in actual if tuple(handle[key].shape) != (1024,)]
        if invalid_shapes:
            raise ValueError(
                f"Output {h5_path} has non-1024 embeddings: {invalid_shapes[:10]}"
            )
    print(
        f"  Verified {h5_path.relative_to(ROOT)} and {tsv_path.relative_to(ROOT)}: "
        f"{len(expected):,} matching entries"
    )


def main() -> None:
    args = parse_args()
    validate_inputs_and_outputs(args.overwrite)

    print("Loading 3FTx annotation, target lengths, and nontoxin metadata...")
    full_entries = read_full_sequence_entries(ANNOTATION_CSV)
    full_lengths = read_fasta_lengths(FULL_FASTA)
    mature_lengths = read_fasta_lengths(MATURE_FASTA)
    target_lengths = list(full_lengths.values())
    nontoxins, nontoxin_fieldnames = read_nontoxin_metadata(NONTOXINS_TSV)
    nontoxin_processed_lengths = read_fasta_lengths(NONTOXINS_PROCESSED_FASTA)

    nontoxins_signal = [row for row in nontoxins if row.signal_peptide]
    nontoxins_nonsignal = [row for row in nontoxins if not row.signal_peptide]
    print(
        f"  {FULL_FASTA.relative_to(ROOT)}: {len(full_lengths):,} sequences used as "
        "the target length distribution"
    )
    print(
        f"  {ANNOTATION_CSV.relative_to(ROOT)}: {len(full_entries):,} entries "
        "marked has_full=True for paired 3FTx full/mature output"
    )
    print(f"  {MATURE_FASTA.relative_to(ROOT)}: {len(mature_lengths):,} sequences")
    print(
        f"  {NONTOXINS_TSV.relative_to(ROOT)}: {len(nontoxins):,} total "
        f"({len(nontoxins_signal):,} signal+, {len(nontoxins_nonsignal):,} signal-)"
    )
    print(
        f"  {NONTOXINS_PROCESSED_FASTA.relative_to(ROOT)}: "
        f"{len(nontoxin_processed_lengths):,} processed signal+ sequences"
    )

    print("Indexing embedding files...")
    requested_master_ids = {row.entry for row in nontoxins}
    master_index, master_total, master_shape, master_dtype = index_master_h5(
        MASTER_H5, requested_master_ids
    )
    full_index, full_total, full_shape, full_dtype = index_h5(FULL_H5)
    mature_index, mature_total, mature_shape, mature_dtype = index_h5(MATURE_H5)
    (
        nontoxin_processed_index,
        nontoxin_processed_total,
        nontoxin_processed_shape,
        nontoxin_processed_dtype,
    ) = index_h5(NONTOXINS_PROCESSED_H5)

    shapes = {master_shape, full_shape, mature_shape, nontoxin_processed_shape}
    if len(shapes) != 1:
        raise ValueError(
            "Embedding files have incompatible shapes: "
            f"master={master_shape}, full_fallback={full_shape}, "
            f"mature={mature_shape}, nontoxin_processed={nontoxin_processed_shape}"
        )
    dtypes = {master_dtype, full_dtype, mature_dtype, nontoxin_processed_dtype}
    if len(dtypes) > 1:
        print(
            "  INFO: source dtypes differ "
            f"(master={master_dtype}, full_fallback={full_dtype}, "
            f"mature={mature_dtype}, nontoxin_processed={nontoxin_processed_dtype}); "
            "datasets are copied as-is and ProtSpace promotes the combined array "
            "to float32 when loading"
        )
    print(
        f"  {MASTER_H5.relative_to(ROOT)}: {master_total:,} embeddings; "
        f"{len(master_index):,}/{len(requested_master_ids):,} requested nontoxins "
        f"found; shape={master_shape}, dtype={master_dtype}"
    )
    print(
        f"  {FULL_H5.relative_to(ROOT)}: {full_total:,} embeddings; "
        f"shape={full_shape}, dtype={full_dtype}"
    )
    print(
        f"  {MATURE_H5.relative_to(ROOT)}: {mature_total:,} embeddings; "
        f"shape={mature_shape}, dtype={mature_dtype}"
    )
    print(
        f"  {NONTOXINS_PROCESSED_H5.relative_to(ROOT)}: "
        f"{nontoxin_processed_total:,} embeddings; "
        f"shape={nontoxin_processed_shape}, dtype={nontoxin_processed_dtype}"
    )

    nontoxin_processed_ids = set(nontoxin_processed_index) & set(
        nontoxin_processed_lengths
    )
    signal_pool = [
        row
        for row in nontoxins_signal
        if row.entry in master_index and row.entry in nontoxin_processed_ids
    ]
    nonsignal_pool = [
        row for row in nontoxins_nonsignal if row.entry in master_index
    ]
    print(
        f"  Eligible nontoxin pools: {len(signal_pool):,} signal+ with original "
        f"and processed embeddings; {len(nonsignal_pool):,} signal- with original "
        "embeddings"
    )

    print("Matching nontoxin pools to the 3FTx_full_fallback length distribution...")
    matched_signal, signal_deltas = length_match_without_replacement(
        target_lengths, signal_pool, args.seed
    )
    matched_nonsignal, nonsignal_deltas = length_match_without_replacement(
        target_lengths, nonsignal_pool, args.seed + 1
    )
    report_match("signal+", signal_deltas)
    report_match("signal-", nonsignal_deltas)

    matched_5050 = matched_signal + matched_nonsignal
    random.Random(args.seed).shuffle(matched_5050)
    matched_signal_ids = [row.entry for row in matched_signal]
    matched_5050_ids = [row.entry for row in matched_5050]
    nontoxin_lengths = {row.entry: row.length for row in nontoxins}

    available_3ftx_entries = [
        entry
        for entry in full_entries
        if entry in full_index
        and entry in mature_index
        and entry in full_lengths
        and entry in mature_lengths
    ]
    skipped_3ftx = sorted(set(full_entries) - set(available_3ftx_entries))
    if skipped_3ftx:
        print(
            f"  WARNING: excluding {len(skipped_3ftx):,} has_full=True 3FTx entries "
            "missing from at least one HDF5/FASTA input "
            f"(first: {', '.join(skipped_3ftx[:10])})"
        )

    print("Writing Background 1: non3FTx_5050 length-matched nontoxins...")
    write_metadata_tsv(
        NON3FTX_5050_TSV,
        [row.raw for row in matched_5050],
        nontoxin_fieldnames,
        args.overwrite,
    )
    write_h5(
        NON3FTX_5050_H5,
        records_from_index(matched_5050_ids, MASTER_H5, master_index),
        args.overwrite,
    )

    print("Writing Background 2: non3FTx_5050_processed paired nontoxins...")
    nontoxin_processed_records = records_from_index(
        matched_signal_ids, MASTER_H5, master_index
    ) + records_from_index(
        matched_signal_ids,
        NONTOXINS_PROCESSED_H5,
        nontoxin_processed_index,
        suffix="_processed",
    )
    write_h5(
        NON3FTX_5050_PROCESSED_H5,
        nontoxin_processed_records,
        args.overwrite,
    )
    write_metadata_tsv(
        NON3FTX_5050_PROCESSED_TSV,
        paired_metadata(
            matched_signal_ids,
            nontoxin_lengths,
            nontoxin_processed_lengths,
        ),
        ["Entry", "Length"],
        args.overwrite,
    )

    print("Writing Background 3: 3FTx_5050_processed paired full/mature 3FTx...")
    ftx_processed_records = records_from_index(
        available_3ftx_entries, FULL_H5, full_index
    ) + records_from_index(
        available_3ftx_entries,
        MATURE_H5,
        mature_index,
        suffix="_processed",
    )
    write_h5(FTX_5050_PROCESSED_H5, ftx_processed_records, args.overwrite)
    write_metadata_tsv(
        FTX_5050_PROCESSED_TSV,
        paired_metadata(available_3ftx_entries, full_lengths, mature_lengths),
        ["Entry", "Length"],
        args.overwrite,
    )

    print("Verifying output files...")
    validate_output_pair(NON3FTX_5050_H5, NON3FTX_5050_TSV)
    validate_output_pair(NON3FTX_5050_PROCESSED_H5, NON3FTX_5050_PROCESSED_TSV)
    validate_output_pair(FTX_5050_PROCESSED_H5, FTX_5050_PROCESSED_TSV)

    print("\nDone. Output summary:")
    print(
        f"  Background 1 non3FTx_5050: {len(matched_5050_ids):,} entries "
        f"({len(matched_signal_ids):,} signal+, {len(matched_nonsignal):,} signal-)"
    )
    print(
        "  Background 2 non3FTx_5050_processed: "
        f"{len(nontoxin_processed_records):,} paired entries"
    )
    print(
        "  Background 3 3FTx_5050_processed: "
        f"{len(ftx_processed_records):,} paired entries"
    )


if __name__ == "__main__":
    main()
