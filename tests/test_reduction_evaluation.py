"""Tests for projection evaluation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protspace.data.evaluation.reduction import (
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
