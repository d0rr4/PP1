"""Tests for projection evaluation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import protspace.data.evaluation.reduction as reduction_evaluation
from protspace.data.evaluation.reduction import (
    _distance_correlation,
    _parse_label_specs,
    _prepare_labels_lookup,
    run_reduction_evaluation,
)
from protspace.data.loaders import EmbeddingSet


def test_prepare_labels_lookup_strips_evidence_suffix():
    metadata = pd.DataFrame(
        {
            "identifier": ["A", "B", "C", "D"],
            "protein_families": ["Kinase|ISS", "Hydrolase|IEA", "", np.nan],
        }
    )

    labels = _prepare_labels_lookup(metadata, "protein_families")

    assert labels.to_dict() == {"A": "Kinase", "B": "Hydrolase"}


def test_parse_label_specs_supports_typed_and_legacy_labels():
    specs = _parse_label_specs(
        ["protein_families", "continuous:sequence_length"]
    )

    assert [(spec.kind, spec.column) for spec in specs] == [
        ("categorical", "protein_families"),
        ("continuous", "sequence_length"),
    ]


def test_distance_correlation_detects_deterministic_dependence():
    values = np.arange(20, dtype=float)

    score = _distance_correlation(values[:, None], values)

    assert score == pytest.approx(1.0)


def test_run_reduction_evaluation_writes_expected_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    rng = np.random.default_rng(0)
    n = 24
    dim = 16
    headers = [f"P{i}" for i in range(n)]

    emb_set = EmbeddingSet(
        name="prot_t5",
        data=rng.normal(size=(n, dim)).astype(np.float32),
        headers=headers,
    )

    metadata = pd.DataFrame(
        {
            "identifier": headers,
            "protein_families": ["ClassA|ISS"] * 12 + ["ClassB|IEA"] * 12,
        }
    )

    reductions = [
        {
            "name": "ProtT5 - PCA 2",
            "dimensions": 2,
            "info": {},
            "data": rng.normal(size=(n, 2)).astype(np.float32),
            "source_embedding": "prot_t5",
        },
        {
            "name": "ProtT5 - UMAP 2",
            "dimensions": 2,
            "info": {},
            "data": rng.normal(size=(n, 2)).astype(np.float32),
            "source_embedding": "prot_t5",
        },
    ]

    out_path = tmp_path / "output" / "data.parquetbundle"
    run_reduction_evaluation(
        embedding_sets=[emb_set],
        reductions=reductions,
        metadata=metadata,
        output_path=out_path,
        bundled=True,
        label_column="protein_families",
        min_class_size=0,
    )

    eval_dir = tmp_path / "output" / "eval" / "prot_t5"
    expected_files = [
        "summary.tsv",
        "recall.png",
        "trustworthiness.png",
        "continuity.png",
        "knn_accuracy.png",
        "silhouette.png",
    ]
    for filename in expected_files:
        assert (eval_dir / filename).exists()

    output = capsys.readouterr().out
    assert "1. Total proteins and label categories: 24 total proteins; 2 unique" in output
    assert "2. Proteins with NA label: 0 proteins" in output
    assert "3. Proteins removed by rare-class filter: 0 proteins removed" in output
    assert "4. Proteins considered for evaluation: 24 proteins remain" in output


def test_run_reduction_evaluation_prints_filter_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    rng = np.random.default_rng(1)
    headers = [f"P{i}" for i in range(12)]
    emb_set = EmbeddingSet(
        name="prot_t5",
        data=rng.normal(size=(12, 8)).astype(np.float32),
        headers=headers,
    )
    metadata = pd.DataFrame(
        {
            "identifier": headers,
            "protein_families": ["ClassA"] * 8 + ["ClassB"] * 3 + [pd.NA],
        }
    )
    reductions = [
        {
            "name": "ProtT5 - PCA 2",
            "dimensions": 2,
            "info": {},
            "data": rng.normal(size=(12, 2)).astype(np.float32),
            "source_embedding": "prot_t5",
        }
    ]

    run_reduction_evaluation(
        embedding_sets=[emb_set],
        reductions=reductions,
        metadata=metadata,
        output_path=tmp_path / "output" / "data.parquetbundle",
        bundled=True,
        label_column="protein_families",
        min_class_size=4,
    )

    output = capsys.readouterr().out
    assert "1. Total proteins and label categories: 12 total proteins; 2 unique" in output
    assert "2. Proteins with NA label: 1 proteins" in output
    assert "3. Proteins removed by rare-class filter: 3 proteins are in 1 categories" in output
    assert "4. Proteins considered for evaluation: 8 proteins remain" in output


def test_run_reduction_evaluation_raises_for_missing_label_column(tmp_path: Path):
    emb_set = EmbeddingSet(
        name="prot_t5",
        data=np.random.rand(8, 4).astype(np.float32),
        headers=[f"P{i}" for i in range(8)],
    )
    reductions = [
        {
            "name": "ProtT5 - PCA 2",
            "dimensions": 2,
            "info": {},
            "data": np.random.rand(8, 2).astype(np.float32),
            "source_embedding": "prot_t5",
        }
    ]
    metadata = pd.DataFrame({"identifier": emb_set.headers, "other": ["x"] * 8})

    with pytest.raises(ValueError, match="Label column 'protein_families'"):
        run_reduction_evaluation(
            embedding_sets=[emb_set],
            reductions=reductions,
            metadata=metadata,
            output_path=tmp_path / "data.parquetbundle",
            bundled=True,
            label_column="protein_families",
            min_class_size=0,
        )


def test_multiple_labels_use_separate_supervised_output_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(reduction_evaluation, "_MAX_DIST_SAMPLES", 12)
    rng = np.random.default_rng(7)
    n = 30
    headers = [f"P{i}" for i in range(n)]
    data = rng.normal(size=(n, 8)).astype(np.float32)
    emb_set = EmbeddingSet(name="prot_t5", data=data, headers=headers)
    metadata = pd.DataFrame(
        {
            "identifier": headers,
            "protein_families": ["ClassA"] * 15 + ["ClassB"] * 15,
            "sequence_length": data[:, 0] * 100 + 500,
        }
    )
    reductions = [
        {
            "name": "ProtT5 - PCA 2",
            "dimensions": 2,
            "info": {},
            "data": data[:, :2],
            "source_embedding": "prot_t5",
        }
    ]

    run_reduction_evaluation(
        embedding_sets=[emb_set],
        reductions=reductions,
        metadata=metadata,
        output_path=tmp_path / "output" / "data.parquetbundle",
        bundled=True,
        label_columns=[
            "categorical:protein_families",
            "continuous:sequence_length",
        ],
        min_class_size=0,
    )

    eval_dir = tmp_path / "output" / "eval" / "prot_t5"
    root_summary = pd.read_csv(eval_dir / "summary.tsv", sep="\t")
    categorical_summary = pd.read_csv(
        eval_dir / "protein_families" / "summary.tsv", sep="\t"
    )
    continuous_dir = eval_dir / "sequence_length"
    continuous_summary = pd.read_csv(continuous_dir / "summary.tsv", sep="\t")

    assert set(root_summary["type"]) == {"unsupervised"}
    assert set(categorical_summary["type"]) == {"supervised_categorical"}
    assert set(categorical_summary["n_samples"]) == {12}
    assert {"knn_acc_mean", "silhouette", "concordex"}.issubset(
        categorical_summary.columns
    )
    assert set(continuous_summary["type"]) == {"supervised_continuous"}
    assert set(continuous_summary["n_samples"]) == {12}
    assert {"linear_r2", "distance_correlation"}.issubset(
        continuous_summary.columns
    )
    assert (continuous_dir / "linear_r2.png").exists()
    assert (continuous_dir / "distance_correlation.png").exists()
    assert (continuous_dir / "heatmap_supervised.png").exists()
