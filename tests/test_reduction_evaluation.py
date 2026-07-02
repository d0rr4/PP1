"""Tests for projection evaluation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import protspace.data.evaluation.reduction as reduction_evaluation
from protspace.data.evaluation.reduction import (
    DEFAULT_K_VALUES,
    _classification_splits,
    _concordex_score,
    _continuous_space_scores,
    _distance_correlation,
    _linear_classifier_scores,
    _neighborhood_consolidation_matrix,
    _parse_label_specs,
    _prepare_labels_lookup,
    _raw_concordex_coefficient,
    _regression_splits,
    _spearman_dist_correlation,
    run_reduction_evaluation,
)
from protspace.data.loaders import EmbeddingSet


def _read_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
    )


def _read_summary_metadata(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# ") or line.startswith("# metadata_key"):
            continue
        key, label, value = line[2:].split("\t", maxsplit=2)
        rows.append({"key": key, "label": label, "value": value})
    return pd.DataFrame(rows)


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


def test_evaluation_uses_shared_k_values():
    assert DEFAULT_K_VALUES == (5, 10, 20, 30, 50)


def test_wide_summary_does_not_need_matrix_column():
    metrics = {
        "recall_summary_mean": 0.9,
        "recall_summary_std": 0.1,
        "trust_summary_mean": 0.8,
        "trust_summary_std": 0.2,
        "cont_summary_mean": 0.7,
        "cont_summary_std": 0.3,
    }
    evaluation = reduction_evaluation.EmbeddingEvaluation(
        matrix="prot_t5",
        unsupervised={"UMAP2 - Full": metrics},
        supervised={},
        k_values=[5],
        n_full=20,
        n_eval=20,
    )

    summary = reduction_evaluation._wide_summary_frame(evaluation)

    assert "matrix" not in summary
    assert summary["space"].tolist() == ["UMAP2"]


def test_linear_classifier_scores_detect_separable_labels():
    rng = np.random.default_rng(11)
    labels = np.repeat([0, 1], 30)
    features = rng.normal(scale=0.2, size=(60, 4))
    features[:, 0] += np.where(labels == 0, -2.0, 2.0)

    auc, f1 = _linear_classifier_scores(
        features, labels, _classification_splits(labels)
    )

    assert auc > 0.99
    assert f1 > 0.95


def test_distance_correlation_detects_deterministic_dependence():
    values = np.arange(20, dtype=float)

    score = _distance_correlation(values[:, None], values)

    assert score == pytest.approx(1.0)


def test_continuous_metrics_detect_predictive_and_geometric_signal():
    targets = np.linspace(0.0, 1.0, 100)
    features = targets[:, None]
    splits = _regression_splits(len(targets))

    linear_r2, knn_r2, distance_corr, spearman_corr = (
        _continuous_space_scores(
            features,
            features,
            targets,
            splits,
            knn_k=5,
        )
    )

    assert linear_r2 == pytest.approx(1.0)
    assert knn_r2 > 0.99
    assert distance_corr == pytest.approx(1.0)
    assert spearman_corr == pytest.approx(1.0)


def test_spearman_distance_correlation_is_one_for_matching_distances():
    values = np.arange(20, dtype=float)

    score = _spearman_dist_correlation(values[:, None], values)

    assert score == pytest.approx(1.0)


def test_concordex_builds_neighborhood_consolidation_matrix():
    neighbors = np.array([[1], [0], [3], [0]])
    labels = np.array([0, 0, 1, 1])

    consolidation, encoded = _neighborhood_consolidation_matrix(
        neighbors, labels, k=1
    )

    np.testing.assert_array_equal(encoded, labels)
    np.testing.assert_allclose(
        consolidation,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_raw_concordex_averages_similarity_diagonal_by_class():
    neighbors = np.array([[1], [0], [0], [0]])
    labels = np.array([0, 0, 0, 1])

    coefficient = _raw_concordex_coefficient(neighbors, labels, k=1)

    # Class 0 has diagonal similarity 1 and class 1 has 0. The paper gives
    # both classes equal weight, rather than weighting by their sample counts.
    assert coefficient == pytest.approx(0.5)


def test_corrected_concordex_uses_mean_permuted_coefficient():
    neighbors = np.array([[1], [0], [3], [2]])
    labels = np.array([0, 0, 1, 1])
    n_permutations = 20
    random_state = 7
    rng = np.random.default_rng(random_state)
    null_coefficients = []
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        class_scores = []
        for label in np.unique(permuted):
            members = np.flatnonzero(permuted == label)
            same_label = permuted[neighbors[members, 0]] == label
            class_scores.append(same_label.mean())
        null_coefficients.append(np.mean(class_scores))

    coefficient = _concordex_score(
        neighbors,
        labels,
        k=1,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    assert coefficient == pytest.approx(1.0 / np.mean(null_coefficients))


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

    eval_root = tmp_path / "output" / "eval"
    eval_dir = eval_root / "prot_t5"
    expected_files = [
        "recall.png",
        "trustworthiness.png",
        "continuity.png",
        "knn_accuracy.png",
        "silhouette.png",
    ]
    for filename in expected_files:
        assert (eval_dir / filename).exists()
    assert (eval_dir / "summary.tsv").exists()
    assert not (eval_root / "summary.tsv").exists()

    output = capsys.readouterr().out
    assert "1. Total proteins and label categories: 24 total proteins; 2 unique" in output
    assert "2. Proteins with NA label: 0 proteins" in output
    assert "3. Proteins removed by rare-class filter: 0 proteins removed" in output
    assert "4. Proteins considered for evaluation: 24 proteins remain" in output


def test_robustness_runs_are_aggregated_with_sample_standard_deviation(
    tmp_path: Path,
):
    rng = np.random.default_rng(123)
    n = 20
    headers = [f"P{i}" for i in range(n)]
    emb_set = EmbeddingSet(
        name="prot_t5",
        data=rng.normal(size=(n, 8)).astype(np.float32),
        headers=headers,
    )
    metadata = pd.DataFrame(
        {
            "identifier": headers,
            "protein_families": ["ClassA"] * 10 + ["ClassB"] * 10,
        }
    )
    group_name = "ProtT5 — UMAP 2"
    reductions = [
        {
            "name": group_name,
            "dimensions": 2,
            "info": {},
            "data": rng.normal(size=(n, 2)).astype(np.float32),
            "source_embedding": "prot_t5",
            "robustness_group": group_name,
            "robustness_run": run,
        }
        for run in range(2)
    ]

    run_reduction_evaluation(
        embedding_sets=[emb_set],
        reductions=reductions,
        metadata=metadata,
        output_path=tmp_path / "output" / "data.parquetbundle",
        bundled=True,
        label_column="protein_families",
        min_class_size=0,
    )

    summary_path = tmp_path / "output" / "eval" / "prot_t5" / "summary.tsv"
    summary = _read_summary(summary_path)
    umap_rows = summary[summary["space"].str.contains("UMAP", na=False)]

    assert len(umap_rows) == 1
    assert "type" not in summary
    assert "robustness_runs" not in summary
    assert float(umap_rows["recall_mean_std"].iloc[0]) > 0
    assert float(umap_rows["silhouette_std:protein_families"].iloc[0]) > 0
    metadata_rows = _read_summary_metadata(summary_path)
    robustness = metadata_rows[metadata_rows["key"] == "robustness"]
    assert robustness["value"].tolist() == ["2"]


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


def test_multiple_labels_share_one_wide_summary(
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

    eval_root = tmp_path / "output" / "eval"
    eval_dir = eval_root / "prot_t5"
    summary_path = eval_dir / "summary.tsv"
    summary = _read_summary(summary_path)
    continuous_dir = eval_dir / "sequence_length"
    categorical_dir = eval_dir / "protein_families"

    assert list(eval_root.rglob("summary.tsv")) == [summary_path]
    assert {"label", "robustness", "robustness_runs", "k_values", "type"}.isdisjoint(
        summary.columns
    )
    assert "matrix" not in summary
    assert not summary["space"].str.contains(" - Full", regex=False).any()
    categorical_metrics = {
        "knn_acc_mean",
        "silhouette",
        "concordex",
        "linear_auc",
        "linear_f1_macro",
    }
    assert set(reduction_evaluation._CATEGORICAL_COL_LABELS) == categorical_metrics
    assert {
        f"{metric}:protein_families" for metric in categorical_metrics
    }.issubset(summary.columns)
    continuous_metrics = {
        "linear_r2",
        "knn_r2",
        "distance_correlation",
        "spearman_dcorr",
    }
    assert set(reduction_evaluation._CONTINUOUS_COL_LABELS) == continuous_metrics
    assert {
        f"{metric}:sequence_length" for metric in continuous_metrics
    }.issubset(summary.columns)
    projection_row = summary[summary["space"] == "PCA2"].iloc[0]
    assert projection_row["recall_mean"] != "-"
    assert projection_row["knn_acc_mean:protein_families"] != "-"
    assert projection_row["linear_r2:sequence_length"] != "-"
    original_row = summary[summary["space"] == "Original (8)"].iloc[0]
    assert set(original_row[list(reduction_evaluation._UNSUPERVISED_METRICS)]) == {
        "-"
    }

    metadata_rows = _read_summary_metadata(summary_path)
    categorical_metadata = metadata_rows[
        metadata_rows["label"] == "protein_families"
    ].set_index("key")["value"]
    assert categorical_metadata["samples_after_filter"] == "30"
    assert categorical_metadata["samples_evaluated"] == "12"
    assert categorical_metadata["k_values"] == "5"
    continuous_metadata = metadata_rows[
        metadata_rows["label"] == "sequence_length"
    ].set_index("key")["value"]
    assert continuous_metadata["samples_after_filter"] == "30"
    assert continuous_metadata["samples_evaluated"] == "12"
    assert continuous_metadata["regression_folds"] == "5"
    assert continuous_metadata["knn_k"] == "5"
    assert (continuous_dir / "linear_r2.png").exists()
    assert (continuous_dir / "knn_r2.png").exists()
    assert (continuous_dir / "distance_correlation.png").exists()
    assert (continuous_dir / "spearman_distance_correlation.png").exists()
    assert (continuous_dir / "heatmap_supervised.png").exists()
    assert (categorical_dir / "linear_classifier_auc.png").exists()
    assert (categorical_dir / "linear_classifier_f1.png").exists()
