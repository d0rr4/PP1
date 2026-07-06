from pathlib import Path

import h5py
import pandas as pd

import TOX_backgrounds as backgrounds


def test_accession_from_identifier_handles_bare_and_uniprot_ids():
    assert backgrounds.accession_from_identifier("P12345") == "P12345"
    assert backgrounds.accession_from_identifier("sp|P12345|TEST_HUMAN") == "P12345"
    assert (
        backgrounds.accession_from_identifier("sp|P12345|TEST_HUMAN description")
        == "P12345"
    )


def test_read_fasta_lengths_uses_bare_accessions(tmp_path: Path):
    fasta = tmp_path / "processed.fasta"
    fasta.write_text(
        ">sp|P1|ONE description\nAAAA\nAA\n>P2\nGGG\n",
        encoding="utf-8",
    )

    assert backgrounds.read_fasta_lengths(fasta) == {"P1": 6, "P2": 3}


def test_length_match_is_unique_deterministic_and_nearest():
    candidates = pd.DataFrame(
        {
            "Entry": ["A", "B", "C", "D", "E", "F"],
            "Length": [9, 10, 19, 21, 30, 31],
        }
    )
    targets = [10, 20, 30]

    selected_a, deltas_a = backgrounds.length_match_without_replacement(
        targets, candidates, seed=42
    )
    selected_b, deltas_b = backgrounds.length_match_without_replacement(
        targets, candidates, seed=42
    )

    assert selected_a["Entry"].tolist() == selected_b["Entry"].tolist()
    assert deltas_a == deltas_b
    assert selected_a["Entry"].is_unique
    assert len(selected_a) == len(targets)
    assert max(deltas_a) <= 1


def test_h5_index_normalization_and_copy(tmp_path: Path):
    processed = tmp_path / "processed.h5"
    output = tmp_path / "output.h5"
    with h5py.File(processed, "w") as handle:
        dataset = handle.create_dataset("sp|P1|ONE", data=[1.0, 2.0])
        dataset.attrs["source"] = "test"

    index, total, shape, dtype = backgrounds.index_h5(processed)

    assert index == {"P1": "sp|P1|ONE"}
    assert total == 1
    assert shape == (2,)
    assert dtype.startswith("float")

    records = backgrounds.records_for_processed(
        ["P1"], processed, index, suffix="_processed"
    )
    backgrounds.write_h5(output, records, overwrite=False)

    with h5py.File(output, "r") as handle:
        assert list(handle) == ["P1_processed"]
        assert handle.attrs["model_name"] == "prot_t5"
        assert handle["P1_processed"][:].tolist() == [1.0, 2.0]
        assert handle["P1_processed"].attrs["source"] == "test"


def test_master_index_only_scans_requested_accessions(tmp_path: Path):
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset("P1", data=[1.0, 2.0])
        handle.create_dataset("P2", data=[3.0, 4.0])

    index, total, shape, dtype = backgrounds.index_master_h5(
        master, {"P2", "MISSING"}
    )

    assert index == {"P2": "P2"}
    assert total == 2
    assert shape == (2,)
    assert dtype.startswith("float")
