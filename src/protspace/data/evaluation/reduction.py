"""Evaluation utilities for DR projections produced by ``protspace prepare``."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, normalize

from protspace.data.loaders import EmbeddingSet

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 2, 5, 10, 15, 20, 30, 50)
DEFAULT_K_STORED = 300
DEFAULT_PCA_COMPONENTS = 50
PROJECTION_COLORS = ("#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#17BECF")
BASE_COLORS = {"Full": "#1F77B4", "PCA": "#2CA02C"}
LINESTYLES_GT = {"Full": "-", "PCA": "--"}


def run_reduction_evaluation(
    *,
    embedding_sets: list[EmbeddingSet],
    reductions: list[dict[str, Any]],
    metadata: pd.DataFrame,
    output_path: Path,
    bundled: bool,
    label_column: str,
    min_class_size: int,
) -> None:
    """Run DR evaluation and write per-embedding plots/summary files.

    Output layout:
        {output}/eval/{embedding_name}/
            summary.tsv
            recall.png
            trustworthiness.png
            continuity.png
            knn_accuracy.png
            silhouette.png
    """
    if min_class_size < 0:
        raise ValueError("--filter must be >= 0.")
    if "identifier" not in metadata.columns:
        raise ValueError("Expected metadata to contain an 'identifier' column.")
    if label_column not in metadata.columns:
        available = ", ".join(sorted(c for c in metadata.columns if c != "identifier"))
        raise ValueError(
            f"Label column '{label_column}' was not found in annotations. "
            f"Available columns: {available or '(none)'}."
        )

    eval_root = _resolve_output_dir(output_path, bundled) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    labels_lookup = _prepare_labels_lookup(metadata, label_column)
    reductions_by_embedding = _group_reductions_by_embedding(reductions)

    evaluated = 0
    for emb_set in embedding_sets:
        if emb_set.precomputed:
            logger.warning(
                "Skipping evaluation for '%s': precomputed similarity matrices are "
                "not supported.",
                emb_set.name,
            )
            continue

        emb_reductions = reductions_by_embedding.get(emb_set.name, [])
        if not emb_reductions:
            logger.warning(
                "Skipping evaluation for '%s': no projections were generated.",
                emb_set.name,
            )
            continue

        out_dir = eval_root / _safe_dir_name(emb_set.name)
        try:
            _evaluate_single_embedding(
                emb_set=emb_set,
                reductions=emb_reductions,
                labels_lookup=labels_lookup,
                label_column=label_column,
                min_class_size=min_class_size,
                out_dir=out_dir,
            )
            evaluated += 1
        except ValueError as exc:
            logger.warning("Skipping evaluation for '%s': %s", emb_set.name, exc)

    if evaluated == 0:
        logger.warning("Evaluation requested, but no embedding set could be evaluated.")
    else:
        logger.info("Evaluation outputs saved to: %s", eval_root)


def _resolve_output_dir(output_path: Path, bundled: bool) -> Path:
    if bundled:
        return output_path.parent if output_path.suffix else output_path
    return output_path


def _safe_dir_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "embedding"


def _prepare_labels_lookup(metadata: pd.DataFrame, label_column: str) -> pd.Series:
    labels = metadata[["identifier", label_column]].copy()
    labels[label_column] = (
        labels[label_column]
        .fillna("")
        .astype(str)
        .str.split("|", n=1, regex=False)
        .str[0]
        .str.strip()
    )
    labels[label_column] = labels[label_column].replace("", pd.NA)
    labels = labels.dropna(subset=[label_column]).drop_duplicates("identifier")
    return labels.set_index("identifier")[label_column].astype(str)


def _group_reductions_by_embedding(
    reductions: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for reduction in reductions:
        source = reduction.get("source_embedding")
        if not source:
            logger.warning(
                "Skipping reduction without source embedding marker: %s",
                reduction.get("name", "<unnamed>"),
            )
            continue
        grouped.setdefault(source, []).append(reduction)
    return grouped


def _evaluate_single_embedding(
    *,
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    labels_lookup: pd.Series,
    label_column: str,
    min_class_size: int,
    out_dir: Path,
) -> None:
    id_to_idx = {identifier: i for i, identifier in enumerate(emb_set.headers)}
    shared_ids = [
        identifier for identifier in emb_set.headers if identifier in labels_lookup.index
    ]

    labels = labels_lookup.loc[shared_ids]
    total_proteins = len(emb_set.headers)
    proteins_with_labels = len(labels)
    unique_categories = labels.nunique()
    proteins_with_na = total_proteins - proteins_with_labels

    class_counts = labels.value_counts()
    if min_class_size > 0:
        rare_class_counts = class_counts[class_counts < min_class_size]
        valid_classes = class_counts[class_counts >= min_class_size].index
        labels = labels[labels.isin(valid_classes)]
    else:
        rare_class_counts = class_counts.iloc[0:0]

    proteins_below_filter = int(rare_class_counts.sum())
    categories_below_filter = len(rare_class_counts)
    remaining_proteins = len(labels)

    _print_label_summary(
        embedding_name=emb_set.name,
        label_column=label_column,
        min_class_size=min_class_size,
        total_proteins=total_proteins,
        unique_categories=unique_categories,
        proteins_with_na=proteins_with_na,
        proteins_below_filter=proteins_below_filter,
        categories_below_filter=categories_below_filter,
        remaining_proteins=remaining_proteins,
    )

    if not shared_ids:
        raise ValueError("no overlap between embeddings and labeled proteins")

    active_ids = labels.index.tolist()
    if len(active_ids) < 3:
        raise ValueError(
            "not enough proteins after label filtering (need at least 3 proteins)"
        )

    logger.info(
        "Evaluation '%s': labels=%d, dropped_unlabeled=%d, dropped_by_filter=%d",
        emb_set.name,
        proteins_with_labels,
        proteins_with_na,
        proteins_below_filter,
    )

    indices = [id_to_idx[identifier] for identifier in active_ids]
    embeddings = emb_set.data[indices].astype(np.float32, copy=False)

    encoder = LabelEncoder()
    labels_int = encoder.fit_transform(labels.values)

    k_stored = min(DEFAULT_K_STORED, len(active_ids) - 1)
    if k_stored < 1:
        raise ValueError("not enough proteins to compute nearest-neighbor metrics")

    k_values = _valid_k_values(len(active_ids), k_stored)
    if not k_values:
        raise ValueError("no valid k values for trustworthiness/continuity")

    full_neighbors = _neighbors(normalize(embeddings, norm="l2"), k_stored, "cosine")
    pca_components = min(
        DEFAULT_PCA_COMPONENTS,
        embeddings.shape[1],
        embeddings.shape[0] - 1,
    )
    if pca_components < 1:
        raise ValueError("cannot compute PCA ground truth for this embedding set")

    pca = PCA(n_components=pca_components, svd_solver="auto", random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    pca_neighbors = _neighbors(normalize(emb_pca, norm="l2"), k_stored, "cosine")

    full_name = f"Full {embeddings.shape[1]}-d"
    pca_name = f"PCA-{pca_components}"

    unsupervised: dict[str, dict[str, Any]] = {}
    knn_accuracy: dict[str, list[float]] = {}
    silhouette: dict[str, float] = {}

    knn_accuracy[full_name] = [
        _knn_accuracy_at_k(full_neighbors, labels_int, k) for k in k_values
    ]
    knn_accuracy[pca_name] = [
        _knn_accuracy_at_k(pca_neighbors, labels_int, k) for k in k_values
    ]

    silhouette[full_name] = _safe_silhouette(embeddings, labels_int, metric="cosine")
    silhouette[pca_name] = _safe_silhouette(emb_pca, labels_int, metric="euclidean")

    projection_entries: list[tuple[str, dict[str, Any]]] = []
    seen_labels: set[str] = set()
    for reduction in reductions:
        label = _projection_label(reduction["name"])
        if label in seen_labels:
            suffix = 2
            while f"{label} [{suffix}]" in seen_labels:
                suffix += 1
            label = f"{label} [{suffix}]"
        seen_labels.add(label)
        projection_entries.append((label, reduction))

    projection_colors = {
        label: PROJECTION_COLORS[i % len(PROJECTION_COLORS)]
        for i, (label, _) in enumerate(projection_entries)
    }

    for proj_label, reduction in projection_entries:
        coords = reduction["data"][indices].astype(np.float32, copy=False)
        dims = 3 if reduction["dimensions"] == 3 else 2
        coords = coords[:, :dims]
        proj_neighbors = _neighbors(coords, k_stored, "euclidean")

        knn_accuracy[proj_label] = [
            _knn_accuracy_at_k(proj_neighbors, labels_int, k) for k in k_values
        ]
        silhouette[proj_label] = _safe_silhouette(
            coords,
            labels_int,
            metric="euclidean",
        )

        for gt_label, gt_neighbors in (("Full", full_neighbors), ("PCA", pca_neighbors)):
            line_label = f"{proj_label} - {gt_label}"
            recalls: list[float] = []
            trusts: list[float] = []
            conts: list[float] = []
            for k in k_values:
                recalls.append(_knn_recall_at_k(gt_neighbors, proj_neighbors, k))
                trusts.append(
                    _trustworthiness_at_k(gt_neighbors, proj_neighbors, k, len(active_ids))
                )
                conts.append(
                    _continuity_at_k(proj_neighbors, gt_neighbors, k, len(active_ids))
                )
            unsupervised[line_label] = {
                "recall": recalls,
                "trust": trusts,
                "cont": conts,
                "projection_label": proj_label,
                "gt_label": gt_label,
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(
        out_dir=out_dir,
        unsupervised=unsupervised,
        knn_accuracy=knn_accuracy,
        silhouette=silhouette,
    )
    _plot_unsupervised_lines(
        out_dir=out_dir,
        k_values=k_values,
        unsupervised=unsupervised,
        projection_colors=projection_colors,
    )
    _plot_knn_accuracy(
        out_dir=out_dir,
        k_values=k_values,
        knn_accuracy=knn_accuracy,
        projection_colors=projection_colors,
        full_name=full_name,
        pca_name=pca_name,
        label_column=label_column,
    )
    _plot_silhouette(
        out_dir=out_dir,
        silhouette=silhouette,
        projection_colors=projection_colors,
        full_name=full_name,
        pca_name=pca_name,
        label_column=label_column,
    )


def _print_label_summary(
    *,
    embedding_name: str,
    label_column: str,
    min_class_size: int,
    total_proteins: int,
    unique_categories: int,
    proteins_with_na: int,
    proteins_below_filter: int,
    categories_below_filter: int,
    remaining_proteins: int,
) -> None:
    print(
        f"\nEvaluation summary "
        f"(label: {label_column}, filter minimum: {min_class_size})"
    )
    print(
        f"Total proteins:{total_proteins}  \nTotal unique categories  in '{label_column}' label: {unique_categories}"
    )
    print(
        f"Proteins without a category in '{label_column}' label: {proteins_with_na}"
    )

    print(
        f"Proteins removed by rare-class (<{min_class_size}) filter: {proteins_below_filter} "
        f"\nRare-class categories (<{min_class_size}) removed: {categories_below_filter}"
    )

    print(

        f"Final proteins considered for evaluation: {remaining_proteins}"
    )


def _projection_label(projection_name: str) -> str:
    return projection_name.strip()


def _valid_k_values(n_samples: int, k_stored: int) -> list[int]:
    valid: list[int] = []
    for k in DEFAULT_K_VALUES:
        if k > k_stored:
            continue
        norm = k * n_samples * (2 * n_samples - 3 * k - 1) / 2
        if norm > 0:
            valid.append(k)
    return valid


def _neighbors(X: np.ndarray, k_stored: int, metric: str) -> np.ndarray:
    n_samples = X.shape[0]
    n_neighbors = min(k_stored + 1, n_samples)
    if n_neighbors <= 1:
        raise ValueError("Need at least 2 samples to compute neighbor graph.")
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices[:, 1:].astype(np.int32, copy=False)


def _knn_recall_at_k(
    neighbors_gt: np.ndarray,
    neighbors_proj: np.ndarray,
    k: int,
) -> float:
    gt_k = neighbors_gt[:, :k]
    proj_k = neighbors_proj[:, :k]
    hits = np.array(
        [len(np.intersect1d(gt_k[i], proj_k[i])) for i in range(gt_k.shape[0])]
    )
    return float(hits.mean() / k)


def _trustworthiness_at_k(
    neighbors_highd: np.ndarray,
    neighbors_proj: np.ndarray,
    k: int,
    n_samples: int,
) -> float:
    k_stored = neighbors_highd.shape[1]
    penalty = 0.0
    for i in range(n_samples):
        hd_rank = {j: r for r, j in enumerate(neighbors_highd[i])}
        for j in neighbors_proj[i, :k]:
            r = hd_rank.get(j, k_stored)
            if r >= k:
                penalty += r - k
    norm = k * n_samples * (2 * n_samples - 3 * k - 1) / 2
    return float(1 - (2 / norm) * penalty)


def _continuity_at_k(
    neighbors_proj: np.ndarray,
    neighbors_highd: np.ndarray,
    k: int,
    n_samples: int,
) -> float:
    k_stored = neighbors_proj.shape[1]
    penalty = 0.0
    for i in range(n_samples):
        proj_rank = {j: r for r, j in enumerate(neighbors_proj[i])}
        for j in neighbors_highd[i, :k]:
            r = proj_rank.get(j, k_stored)
            if r >= k:
                penalty += r - k
    norm = k * n_samples * (2 * n_samples - 3 * k - 1) / 2
    return float(1 - (2 / norm) * penalty)


def _knn_accuracy_at_k(neighbors: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_classes = int(labels.max()) + 1
    correct = 0
    for i in range(neighbors.shape[0]):
        neighbor_labels = labels[neighbors[i, :k]]
        majority = np.bincount(neighbor_labels, minlength=n_classes).argmax()
        correct += int(majority == labels[i])
    return correct / neighbors.shape[0]


def _safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric=metric))
    except ValueError:
        return float("nan")


def _write_summary(
    *,
    out_dir: Path,
    unsupervised: dict[str, dict[str, Any]],
    knn_accuracy: dict[str, list[float]],
    silhouette: dict[str, float],
) -> None:
    rows: list[dict[str, Any]] = []
    for space, metrics in unsupervised.items():
        rows.append(
            {
                "space": space,
                "type": "unsupervised",
                "recall_mean": round(float(np.mean(metrics["recall"])), 4),
                "trust_mean": round(float(np.mean(metrics["trust"])), 4),
                "cont_mean": round(float(np.mean(metrics["cont"])), 4),
                "knn_acc_mean": "",
                "silhouette": "",
            }
        )
    for space, acc_values in knn_accuracy.items():
        rows.append(
            {
                "space": space,
                "type": "supervised",
                "recall_mean": "",
                "trust_mean": "",
                "cont_mean": "",
                "knn_acc_mean": round(float(np.mean(acc_values)), 4),
                "silhouette": round(float(silhouette.get(space, float("nan"))), 4),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "summary.tsv", sep="\t", index=False)


def _plot_unsupervised_lines(
    *,
    out_dir: Path,
    k_values: list[int],
    unsupervised: dict[str, dict[str, Any]],
    projection_colors: dict[str, str],
) -> None:
    def get_line_color(projection_label: str) -> str:
        return projection_colors[projection_label]

    def get_linestyle(gt_label: str) -> str:
        return LINESTYLES_GT.get(gt_label, "-")

    def style_axis(ax: Any, title: str) -> None:
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(k_values)
        ax.legend(fontsize=9, framealpha=0.7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    def save_line_plot(metric_key: str, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for line_label, metrics in unsupervised.items():
            ax.plot(
                k_values,
                metrics[metric_key],
                color=get_line_color(metrics["projection_label"]),
                linestyle=get_linestyle(metrics["gt_label"]),
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=line_label,
            )
        style_axis(ax, title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_line_plot("recall", "kNN Recall", "recall.png")
    save_line_plot("trust", "Trustworthiness", "trustworthiness.png")
    save_line_plot("cont", "Continuity", "continuity.png")


def _plot_knn_accuracy(
    *,
    out_dir: Path,
    k_values: list[int],
    knn_accuracy: dict[str, list[float]],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    label_column: str,
) -> None:
    def get_space_color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        return projection_colors.get(space, "#333333")

    fig, ax = plt.subplots(figsize=(10, 5))
    for space, acc_values in knn_accuracy.items():
        ax.plot(
            k_values,
            acc_values,
            color=get_space_color(space),
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=space,
        )
    ax.set_title(f"kNN Accuracy ({label_column})", fontsize=13, fontweight="bold")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(k_values)
    ax.legend(fontsize=9, framealpha=0.7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "knn_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_silhouette(
    *,
    out_dir: Path,
    silhouette: dict[str, float],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    label_column: str,
) -> None:
    def get_space_color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        return projection_colors.get(space, "#333333")

    spaces = list(silhouette.keys())
    values = list(silhouette.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        spaces,
        values,
        color=[get_space_color(space) for space in spaces],
        edgecolor="white",
        width=0.5,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_title(f"Silhouette Score ({label_column})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Space", fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
