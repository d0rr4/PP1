"""Evaluation utilities for DR projections produced by ``protspace prepare``."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, normalize

from protspace.data.loaders import EmbeddingSet
from protspace.data.loaders.embedding_set import MODEL_DISPLAY_NAMES

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 2, 5, 10, 15, 20, 30, 50)
DEFAULT_K_STORED = 300
DEFAULT_PCA_COMPONENTS = 50
PROJECTION_COLORS = (
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#009988",  # teal
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
    "#AA3377",  # purple
    "#CCBB44",  # yellow
    "#228833",  # green
)
BASE_COLORS = {"Full": "#000000", "PCA": "#666666", "PCA10": "#999999"}
LINESTYLES_GT = {"Full": "-"}

# Hard cap on the number of points used for ALL metrics.  When the labelled
# set exceeds this size a single stratified random subsample is drawn once
# and every subsequent computation (neighbour graphs, silhouette, CONCORDEX,
# kNN accuracy, line plots, heatmaps) operates on that fixed subset.
_MAX_DIST_SAMPLES = 10_000


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
            concordex.png
            heatmap_unsupervised.png
            heatmap_supervised.png
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


def _stratified_subsample(
    indices: list[int],
    labels_int: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray]:
    """Return a stratified random subsample capped at *max_samples*.

    Samples are drawn proportionally per class so that the class distribution
    of the subsample mirrors the full set as closely as possible.  The
    original ordering within each class is preserved.

    Args:
        indices:     Original positional indices into the embedding matrix.
        labels_int:  Integer class labels aligned with *indices*.
        max_samples: Hard upper bound on the returned sample size.
        rng:         NumPy random generator (for reproducibility).

    Returns:
        A (sub_indices, sub_labels) tuple, both sorted by their position in
        the original *indices* list.
    """
    n = len(indices)
    if n <= max_samples:
        return indices, labels_int

    classes, counts = np.unique(labels_int, return_counts=True)
    # Allocate quota per class proportionally, ensuring at least 1 per class.
    quota = np.maximum(1, np.round(counts / n * max_samples).astype(int))
    # If rounding pushed the total over the cap, trim from the largest classes.
    while quota.sum() > max_samples:
        quota[quota.argmax()] -= 1

    chosen: list[int] = []
    for cls, q in zip(classes, quota):
        cls_positions = np.where(labels_int == cls)[0]
        take = min(q, len(cls_positions))
        chosen.extend(rng.choice(cls_positions, size=take, replace=False).tolist())

    chosen_sorted = sorted(chosen)
    sub_indices = [indices[i] for i in chosen_sorted]
    sub_labels = labels_int[chosen_sorted]
    return sub_indices, sub_labels


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

    encoder = LabelEncoder()
    all_labels_int = encoder.fit_transform(labels.values)

    # ------------------------------------------------------------------
    # Universal subsampling — enforced here, once, before any computation.
    # Every metric, neighbour graph, and plot operates on this fixed subset.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_full = len(active_ids)
    all_indices = [id_to_idx[identifier] for identifier in active_ids]

    if n_full > _MAX_DIST_SAMPLES:
        logger.info(
            "Subsampling %d → %d points for '%s' (stratified, seed=42).",
            n_full,
            _MAX_DIST_SAMPLES,
            emb_set.name,
        )
        indices, labels_int = _stratified_subsample(
            all_indices, all_labels_int, _MAX_DIST_SAMPLES, rng
        )
        n_eval = len(indices)
    else:
        indices, labels_int = all_indices, all_labels_int
        n_eval = n_full

    logger.info(
        "Evaluation '%s': labels=%d, dropped_unlabeled=%d, dropped_by_filter=%d, "
        "eval_n=%d",
        emb_set.name,
        proteins_with_labels,
        proteins_with_na,
        proteins_below_filter,
        n_eval,
    )

    embeddings = emb_set.data[indices].astype(np.float32, copy=False)

    k_stored = min(DEFAULT_K_STORED, n_eval - 1)
    if k_stored < 1:
        raise ValueError("not enough proteins to compute nearest-neighbor metrics")

    k_values = _valid_k_values(n_eval, k_stored)
    if not k_values:
        raise ValueError("no valid k values for trustworthiness/continuity")

    full_neighbors = _neighbors(normalize(embeddings, norm="l2"), k_stored, "cosine")
    pca_components = min(
        DEFAULT_PCA_COMPONENTS,
        embeddings.shape[1],
        n_eval - 1,
    )
    if pca_components < 1:
        raise ValueError("cannot compute PCA ground truth for this embedding set")

    pca = PCA(n_components=pca_components, svd_solver="auto", random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    pca_neighbors = _neighbors(normalize(emb_pca, norm="l2"), k_stored, "cosine")

    pca10_components = min(10, embeddings.shape[1], n_eval - 1)
    if pca10_components >= 1:
        pca10 = PCA(n_components=pca10_components, svd_solver="auto", random_state=42)
        emb_pca10 = pca10.fit_transform(embeddings)
        pca10_neighbors = _neighbors(normalize(emb_pca10, norm="l2"), k_stored, "cosine")

    full_name = f"Original ({embeddings.shape[1]})"
    pca_name = f"PCA{pca_components}"
    pca10_name = f"PCA{pca10_components}" if pca10_components >= 1 else ""

    # ------------------------------------------------------------------
    # Per-space scalar metrics
    # ------------------------------------------------------------------
    unsupervised: dict[str, dict[str, Any]] = {}
    knn_accuracy: dict[str, list[float]] = {}
    silhouette: dict[str, float] = {}
    concordex: dict[str, float] = {}

    knn_accuracy[full_name] = [
        _knn_accuracy_at_k(full_neighbors, labels_int, k) for k in k_values
    ]
    knn_accuracy[pca_name] = [
        _knn_accuracy_at_k(pca_neighbors, labels_int, k) for k in k_values
    ]
    if pca10_components >= 1:
        knn_accuracy[pca10_name] = [
            _knn_accuracy_at_k(pca10_neighbors, labels_int, k) for k in k_values
        ]

    silhouette[full_name] = _safe_silhouette(embeddings, labels_int, metric="cosine")
    silhouette[pca_name] = _safe_silhouette(emb_pca, labels_int, metric="euclidean")
    if pca10_components >= 1:
        silhouette[pca10_name] = _safe_silhouette(emb_pca10, labels_int, metric="euclidean")

    concordex[full_name] = _concordex_score(full_neighbors, labels_int, k=k_values[-1])
    concordex[pca_name] = _concordex_score(pca_neighbors, labels_int, k=k_values[-1])
    if pca10_components >= 1:
        concordex[pca10_name] = _concordex_score(pca10_neighbors, labels_int, k=k_values[-1])

    # ------------------------------------------------------------------
    # Build projection entries
    # ------------------------------------------------------------------
    projection_entries: list[tuple[str, dict[str, Any]]] = []
    seen_labels: set[str] = set()
    for reduction in reductions:
        label = _projection_label(reduction["name"], emb_set.name)
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

    # ------------------------------------------------------------------
    # Per-projection metrics  (indices already subsampled above)
    # ------------------------------------------------------------------
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
        concordex[proj_label] = _concordex_score(
            proj_neighbors, labels_int, k=k_values[-1]
        )

        for gt_label, gt_neighbors in (("Full", full_neighbors),):
            line_label = f"{proj_label} - {gt_label}"
            recalls: list[float] = []
            trusts: list[float] = []
            conts: list[float] = []
            for k in k_values:
                recalls.append(_knn_recall_at_k(gt_neighbors, proj_neighbors, k))
                trusts.append(
                    _trustworthiness_at_k(
                        gt_neighbors, proj_neighbors, k, n_eval
                    )
                )
                conts.append(
                    _continuity_at_k(
                        proj_neighbors, gt_neighbors, k, n_eval
                    )
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
        concordex=concordex,
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
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_silhouette(
        out_dir=out_dir,
        silhouette=silhouette,
        projection_colors=projection_colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_concordex(
        out_dir=out_dir,
        concordex=concordex,
        projection_colors=projection_colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_summary_heatmaps(
        out_dir=out_dir,
        k_values=k_values,
        label_column=label_column,
        n_full=n_full,
        n_eval=n_eval,
    )


# ---------------------------------------------------------------------------
# CONCORDEX implementation
# ---------------------------------------------------------------------------


def _concordex_score(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """CONCORDEX neighbourhood-label-consistency score.

    For each point, counts how many of its k nearest neighbours share the
    same class label, then normalises by the expected proportion under a
    random (null) assignment based on class frequencies.

    A score > 1 means labels cluster better than chance; a score of 1 means
    random; a score < 1 means anti-clustering.

    Reference: Kalinichenko et al., *Bioinformatics* (2023) — CONCORDEX.

    Args:
        neighbors: kNN index array of shape (n, k_stored); only the first
                   ``k`` columns are used.
        labels:    Integer class labels of length n.
        k:         Neighbourhood size to evaluate.

    Returns:
        CONCORDEX ratio (float), or NaN if fewer than 2 classes are present.
    """
    n_classes = int(labels.max()) + 1
    if n_classes < 2:
        return float("nan")

    k_use = min(k, neighbors.shape[1])
    neighbor_labels = labels[neighbors[:, :k_use]]  # (n, k_use)

    same_class = (neighbor_labels == labels[:, None]).sum(axis=1)
    observed = same_class.mean() / k_use

    class_freq = np.bincount(labels, minlength=n_classes) / len(labels)
    expected = float((class_freq**2).sum())

    if expected == 0:
        return float("nan")
    return float(observed / expected)


# ---------------------------------------------------------------------------
# Existing metric implementations
# ---------------------------------------------------------------------------


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


def _projection_label(projection_name: str, source_embedding: str = "") -> str:
    """Return a clean display label for a projection."""
    label = projection_name.strip()
    if source_embedding:
        sources = [source_embedding]
        display = MODEL_DISPLAY_NAMES.get(source_embedding, "")
        if display and display != source_embedding:
            sources.append(display)
        for src in sources:
            for sep in (" — ", " —", "— ", "—", " - ", " -", "- ", "-"):
                candidate = f"{src}{sep}"
                if label.startswith(candidate):
                    label = label[len(candidate):].strip()
                    break
            else:
                continue
            break
    label = label.replace(" ", "")
    return label


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


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    *,
    out_dir: Path,
    unsupervised: dict[str, dict[str, Any]],
    knn_accuracy: dict[str, list[float]],
    silhouette: dict[str, float],
    concordex: dict[str, float],
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
                "concordex": "",
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
                "concordex": round(float(concordex.get(space, float("nan"))), 4),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "summary.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Heatmap summary plots
# ---------------------------------------------------------------------------

# Human-readable column labels for the heatmap axes.
_UNSUPERVISED_COL_LABELS: dict[str, str] = {
    "recall_mean": "kNN Recall\n(mean)",
    "trust_mean": "Trustworthiness\n(mean)",
    "cont_mean": "Continuity\n(mean)",
}
_SUPERVISED_COL_LABELS: dict[str, str] = {
    "knn_acc_mean": "kNN Accuracy\n(mean)",
    "silhouette": "Silhouette\nScore",
    "concordex": "CONCORDEX",
}


def _plot_summary_heatmaps(
    *,
    out_dir: Path,
    k_values: list[int],
    label_column: str,
    n_full: int,
    n_eval: int,
) -> None:
    """Read the just-written summary TSV and produce two polished heatmaps.

    One heatmap covers unsupervised structure-preservation metrics
    (Recall, Trustworthiness, Continuity), the other covers supervised
    label-aware metrics (kNN Accuracy, Silhouette, CONCORDEX).  Both use
    column-wise min-max normalisation for the colour mapping so that the
    best value in each metric always stands out, while annotating cells
    with the original numeric values.

    The k range over which means were computed is shown in the figure
    subtitle for the three mean-based metrics.  When subsampling was applied
    (n_full > n_eval) the subtitle also states the effective sample size.
    """
    tsv_path = out_dir / "summary.tsv"
    if not tsv_path.exists():
        logger.warning("summary.tsv not found; skipping heatmap generation.")
        return

    df = pd.read_csv(tsv_path, sep="\t")

    k_range_str = f"k ∈ {{{', '.join(str(k) for k in k_values)}}}"
    sample_note = (
        f" · n = {n_eval:,} (subsampled from {n_full:,})"
        if n_eval < n_full
        else f" · n = {n_eval:,}"
    )

    # ---- Unsupervised ----
    df_uns = df[df["type"] == "unsupervised"].copy()
    df_uns["space"] = df_uns["space"].str.replace(" - Full", "", regex=False)
    uns_cols = [c for c in _UNSUPERVISED_COL_LABELS if c in df_uns.columns]
    if not df_uns.empty and uns_cols:
        df_uns = df_uns[["space"] + uns_cols].set_index("space")
        df_uns.columns = [_UNSUPERVISED_COL_LABELS[c] for c in uns_cols]
        df_uns = df_uns.apply(pd.to_numeric, errors="coerce")
        subtitle = (
            f"Means computed over {k_range_str}{sample_note} · "
            f"Colour normalised column-wise"
        )
        _render_heatmap(
            data=df_uns,
            title=f"Unsupervised DR Evaluation  ·  {label_column}",
            subtitle=subtitle,
            out_path=out_dir / "heatmap_unsupervised.png",
            cmap="YlGnBu",
        )

    # ---- Supervised ----
    df_sup = df[df["type"] == "supervised"].copy()
    sup_cols = [c for c in _SUPERVISED_COL_LABELS if c in df_sup.columns]
    if not df_sup.empty and sup_cols:
        df_sup = df_sup[["space"] + sup_cols].set_index("space")
        df_sup.columns = [_SUPERVISED_COL_LABELS[c] for c in sup_cols]
        df_sup = df_sup.apply(pd.to_numeric, errors="coerce")
        subtitle = (
            f"kNN Accuracy mean over {k_range_str}{sample_note} · "
            f"Colour normalised column-wise"
        )
        _render_heatmap(
            data=df_sup,
            title=f"Supervised DR Evaluation  ·  {label_column}",
            subtitle=subtitle,
            out_path=out_dir / "heatmap_supervised.png",
            cmap="YlOrRd",
        )


def _render_heatmap(
    *,
    data: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: Path,
    cmap: str,
) -> None:
    """Render a single publication-quality heatmap to *out_path*.

    Design choices
    --------------
    * Crisp white grid lines separate cells.
    * Annotation text is black on light cells and white on dark cells for
      maximum contrast (WCAG AA).
    * A compact horizontal colour bar sits below the axes with a label that
      clarifies the normalisation.
    * Typography uses a narrow sans-serif stack so long row/column names fit
      comfortably.
    * The figure background is white (#FFFFFF) with a subtle outer border.
    """
    n_rows, n_cols = data.shape

    # Column-wise min-max normalisation — NaN cells stay NaN (rendered grey).
    norm_data = (data - data.min()) / (data.max() - data.min())

    # ---- Figure geometry ----
    cell_w = 2.0          # inches per column
    cell_h = 0.55         # inches per row
    left_margin = 2.6     # room for row labels
    right_margin = 0.35
    top_margin = 1.05     # room for title + subtitle
    bottom_margin = 1.10  # room for column labels + colour bar

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bottom_margin
    fig_w = max(fig_w, 6.0)
    fig_h = max(fig_h, 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Axes: leave space at bottom for the colour bar.
    cbar_height_frac = 0.06
    cbar_pad_frac = 0.08
    ax_bottom = (bottom_margin) / fig_h
    ax_height = (n_rows * cell_h) / fig_h
    ax_left = left_margin / fig_w
    ax_width = (n_cols * cell_w) / fig_w

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # ---- Draw cells ----
    cm = plt.get_cmap(cmap)
    norm = Normalize(vmin=0.0, vmax=1.0)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            raw_val = data.iloc[row_idx, col_idx]
            norm_val = norm_data.iloc[row_idx, col_idx]

            if pd.isna(norm_val):
                face_color = "#D8D8D8"
                text_color = "#555555"
                cell_text = "N/A"
            else:
                face_color = cm(norm(norm_val))
                # Luminance-based contrast: use white text on dark cells.
                r, g, b, _ = face_color
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.45 else "#1a1a1a"
                cell_text = f"{raw_val:.4f}"

            rect = plt.Rectangle(
                (col_idx, n_rows - row_idx - 1),
                1, 1,
                facecolor=face_color,
                edgecolor="white",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                col_idx + 0.5,
                n_rows - row_idx - 0.5,
                cell_text,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=text_color,
                fontfamily="DejaVu Sans",
            )

    # ---- Axes cosmetics ----
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(
        data.columns,
        fontsize=9.5,
        fontfamily="DejaVu Sans",
        ha="center",
        va="top",
    )
    ax.set_yticklabels(
        data.index[::-1],
        fontsize=9.5,
        fontfamily="DejaVu Sans",
        ha="right",
        va="center",
    )
    ax.tick_params(axis="both", which="both", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Titles ----
    title_y = ax_bottom + ax_height + 0.20 / fig_h
    fig.text(
        ax_left + ax_width / 2,
        title_y + 0.25 / fig_h,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="normal",
        fontfamily="DejaVu Sans",
        color="#111111",
    )
    fig.text(
        ax_left + ax_width / 2,
        title_y,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=9,
        fontstyle="italic",
        fontfamily="DejaVu Sans",
        color="#555555",
    )

    # ---- Colour bar ----
    cbar_ax = fig.add_axes(
        [
            ax_left,
            ax_bottom - cbar_pad_frac - cbar_height_frac,
            ax_width,
            cbar_height_frac,
        ]
    )
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Column-normalised score  (0 = worst, 1 = best)",
        fontsize=9,
        fontfamily="DejaVu Sans",
        color="#444444",
        labelpad=4,
    )
    cbar.ax.tick_params(labelsize=9, colors="#444444", length=2)
    cbar.outline.set_visible(False)

    # ---- Save ----
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Heatmap saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


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
        ax.legend(fontsize=9, framealpha=0.7, bbox_to_anchor=(1.02, 1), loc="upper left")
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
    pca10_name: str,
    label_column: str,
) -> None:
    def get_space_color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        if pca10_name and space == pca10_name:
            return BASE_COLORS["PCA10"]
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
    ax.legend(fontsize=9, framealpha=0.7, bbox_to_anchor=(1.02, 1), loc="upper left")
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
    pca10_name: str,
    label_column: str,
) -> None:
    def get_space_color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        if pca10_name and space == pca10_name:
            return BASE_COLORS["PCA10"]
        return projection_colors.get(space, "#333333")

    spaces = list(silhouette.keys())
    values = list(silhouette.values())

    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_xticks(range(len(spaces)))
    ax.set_xticklabels(spaces, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_concordex(
    *,
    out_dir: Path,
    concordex: dict[str, float],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    pca10_name: str,
    label_column: str,
) -> None:
    """Bar chart of CONCORDEX scores, one bar per space.

    A dashed horizontal line at y=1 marks the random-assignment baseline.
    Scores above 1 indicate better-than-chance neighbourhood purity.
    """

    def get_space_color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        if pca10_name and space == pca10_name:
            return BASE_COLORS["PCA10"]
        return projection_colors.get(space, "#333333")

    spaces = list(concordex.keys())
    values = list(concordex.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        spaces,
        values,
        color=[get_space_color(s) for s in spaces],
        edgecolor="white",
        width=0.5,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="Random baseline (1.0)")
    ax.set_title(
        f"CONCORDEX Score ({label_column})", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Score (ratio to random)", fontsize=11)
    ax.set_xlabel("Space", fontsize=11)
    ax.set_xticks(range(len(spaces)))
    ax.set_xticklabels(spaces, rotation=15, ha="right")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "concordex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)