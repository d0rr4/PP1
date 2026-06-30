"""Evaluation utilities for DR projections produced by ``protspace prepare``."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

from protspace.data.loaders import EmbeddingSet
from protspace.data.loaders.embedding_set import MODEL_DISPLAY_NAMES

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (5, 10, 20, 30, 50)
DEFAULT_K_STORED = 300
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_CONCORDEX_PERMUTATIONS = 100
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

# Hard cap shared by all metrics. Distance correlation has O(n^2) cost, so a
# single cap keeps every reported sample size comparable and memory bounded.
_MAX_DIST_SAMPLES = 10_000
_KNN_REGRESSION_K = 5


@dataclass(frozen=True)
class EvaluationLabel:
    """A metadata column and the supervised metric family applied to it."""

    kind: str
    column: str


def _parse_label_specs(raw_labels: Iterable[str]) -> list[EvaluationLabel]:
    """Parse ``[categorical|continuous]:column`` label specifications."""
    specs: list[EvaluationLabel] = []
    seen: set[str] = set()
    for raw in raw_labels:
        value = raw.strip()
        if not value:
            raise ValueError("--label must not be empty.")
        prefix, separator, column = value.partition(":")
        if separator and prefix.lower() in {"categorical", "continuous"}:
            kind = prefix.lower()
            column = column.strip()
        else:
            kind = "categorical"
            column = value
        if not column:
            raise ValueError(f"Invalid --label specification '{raw}': missing column.")
        if column in seen:
            raise ValueError(f"Label column '{column}' was specified more than once.")
        seen.add(column)
        specs.append(EvaluationLabel(kind=kind, column=column))
    if not specs:
        specs.append(EvaluationLabel("categorical", "protein_families"))
    return specs


def run_reduction_evaluation(
    *,
    embedding_sets: list[EmbeddingSet],
    reductions: list[dict[str, Any]],
    metadata: pd.DataFrame,
    output_path: Path,
    bundled: bool,
    min_class_size: int,
    label_columns: list[str] | None = None,
    label_column: str | None = None,
) -> None:
    """Run label-independent and per-label DR evaluation."""
    if min_class_size < 0:
        raise ValueError("--filter must be >= 0.")
    if "identifier" not in metadata.columns:
        raise ValueError("Expected metadata to contain an 'identifier' column.")
    raw_labels = label_columns
    if raw_labels is None:
        raw_labels = [label_column or "protein_families"]
    label_specs = _parse_label_specs(raw_labels)
    missing = [spec.column for spec in label_specs if spec.column not in metadata.columns]
    if missing:
        available = ", ".join(sorted(c for c in metadata.columns if c != "identifier"))
        if len(missing) == 1:
            raise ValueError(
                f"Label column '{missing[0]}' was not found in annotations. "
                "Available columns: {}.".format(available or "(none)")
            )
        raise ValueError(
            f"Label column(s) {', '.join(repr(column) for column in missing)} "
            "were not found in annotations. "
            f"Available columns: {available or '(none)'}."
        )

    eval_root = _resolve_output_dir(output_path, bundled) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

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

        embedding_dir = eval_root / _safe_dir_name(emb_set.name)
        try:
            unsupervised, k_values, n_full, n_eval = _evaluate_unsupervised(
                emb_set=emb_set, reductions=emb_reductions, out_dir=embedding_dir
            )
            single_label_rows: list[dict[str, Any]] | None = None
            for spec in label_specs:
                label_dir = (
                    embedding_dir
                    if len(label_specs) == 1
                    else embedding_dir / _safe_dir_name(spec.column)
                )
                try:
                    rows = _evaluate_supervised_label(
                        emb_set=emb_set,
                        reductions=emb_reductions,
                        metadata=metadata,
                        spec=spec,
                        min_class_size=min_class_size,
                        out_dir=label_dir,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Skipping label '%s' for '%s': %s",
                        spec.column,
                        emb_set.name,
                        exc,
                    )
                    continue
                if len(label_specs) == 1:
                    single_label_rows = rows
                else:
                    _write_summary(out_dir=label_dir, supervised=rows)
                    _plot_summary_heatmaps(
                        out_dir=label_dir,
                        k_values=[],
                        label_column=spec.column,
                        supervised_kind=spec.kind,
                        n_full=0,
                        n_eval=0,
                    )

            _write_summary(
                out_dir=embedding_dir,
                unsupervised=unsupervised,
                supervised=single_label_rows,
            )
            _plot_summary_heatmaps(
                out_dir=embedding_dir,
                k_values=k_values,
                label_column=label_specs[0].column if len(label_specs) == 1 else None,
                supervised_kind=label_specs[0].kind if len(label_specs) == 1 else None,
                n_full=n_full,
                n_eval=n_eval,
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


def _prepare_continuous_lookup(metadata: pd.DataFrame, column: str) -> pd.Series:
    values = metadata[["identifier", column]].copy()
    values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=[column]).drop_duplicates("identifier")
    return values.set_index("identifier")[column].astype(float)


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
    for cls, q in zip(classes, quota, strict=True):
        cls_positions = np.where(labels_int == cls)[0]
        take = min(q, len(cls_positions))
        chosen.extend(rng.choice(cls_positions, size=take, replace=False).tolist())

    chosen_sorted = sorted(chosen)
    sub_indices = [indices[i] for i in chosen_sorted]
    sub_labels = labels_int[chosen_sorted]
    return sub_indices, sub_labels


def _random_subsample_indices(n_samples: int, max_samples: int) -> list[int]:
    if n_samples <= max_samples:
        return list(range(n_samples))
    rng = np.random.default_rng(42)
    return sorted(rng.choice(n_samples, size=max_samples, replace=False).tolist())


def _build_projection_entries(
    reductions: list[dict[str, Any]], embedding_name: str
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, str]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for reduction in reductions:
        robustness_group = reduction.get("robustness_group")
        label = _projection_label(
            str(robustness_group or reduction["name"]), embedding_name
        )
        if not robustness_group and label in seen:
            suffix = 2
            while f"{label} [{suffix}]" in seen:
                suffix += 1
            label = f"{label} [{suffix}]"
        seen.add(label)
        entries.append((label, reduction))
    unique_labels = list(dict.fromkeys(label for label, _ in entries))
    colors = {
        label: PROJECTION_COLORS[i % len(PROJECTION_COLORS)]
        for i, label in enumerate(unique_labels)
    }
    return entries, colors


def _mean_and_sample_std(values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-aware means and sample SDs along the run axis."""
    array = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(array), axis=0)
    mean = np.full(np.shape(counts), np.nan, dtype=float)
    np.divide(
        np.nansum(array, axis=0),
        counts,
        out=mean,
        where=counts > 0,
    )
    squared = np.nansum((array - mean) ** 2, axis=0)
    std = np.full(np.shape(mean), np.nan, dtype=float)
    np.divide(
        squared,
        counts - 1,
        out=std,
        where=counts > 1,
    )
    return np.asarray(mean), np.sqrt(std)


def _aggregate_scalar_runs(
    values: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for space, runs in values.items():
        mean, std = _mean_and_sample_std(runs)
        means[space] = float(mean)
        stds[space] = float(std)
    return means, stds


def _aggregate_vector_runs(
    values: dict[str, list[list[float]]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    for space, runs in values.items():
        mean, std = _mean_and_sample_std(runs)
        means[space] = mean.tolist()
        stds[space] = std.tolist()
    return means, stds


def _evaluate_unsupervised(
    *,
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[int], int, int]:
    """Evaluate structure preservation once, independently of labels."""
    n_full = len(emb_set.headers)
    if n_full < 3:
        raise ValueError("not enough proteins for evaluation (need at least 3)")
    indices = _random_subsample_indices(n_full, _MAX_DIST_SAMPLES)
    n_eval = len(indices)
    embeddings = emb_set.data[indices].astype(np.float32, copy=False)
    k_stored = min(DEFAULT_K_STORED, n_eval - 1)
    k_values = _valid_k_values(n_eval, k_stored)
    if not k_values:
        raise ValueError("no valid k values for trustworthiness/continuity")

    full_neighbors = _neighbors(normalize(embeddings, norm="l2"), k_stored, "cosine")
    projection_entries, projection_colors = _build_projection_entries(
        reductions, emb_set.name
    )
    run_metrics: dict[str, list[dict[str, Any]]] = {}
    for proj_label, reduction in projection_entries:
        dims = 3 if reduction["dimensions"] == 3 else 2
        coords = reduction["data"][indices, :dims].astype(np.float32, copy=False)
        proj_neighbors = _neighbors(coords, k_stored, "euclidean")
        recalls: list[float] = []
        trusts: list[float] = []
        continuities: list[float] = []
        for k in k_values:
            recalls.append(_knn_recall_at_k(full_neighbors, proj_neighbors, k))
            trusts.append(
                _trustworthiness_at_k(full_neighbors, proj_neighbors, k, n_eval)
            )
            continuities.append(
                _continuity_at_k(proj_neighbors, full_neighbors, k, n_eval)
            )
        run_metrics.setdefault(f"{proj_label} - Full", []).append({
            "recall": recalls,
            "trust": trusts,
            "cont": continuities,
            "projection_label": proj_label,
            "gt_label": "Full",
        })

    unsupervised: dict[str, dict[str, Any]] = {}
    for line_label, runs in run_metrics.items():
        aggregated: dict[str, Any] = {
            "projection_label": runs[0]["projection_label"],
            "gt_label": runs[0]["gt_label"],
            "robustness_runs": len(runs),
        }
        for metric in ("recall", "trust", "cont"):
            mean, std = _mean_and_sample_std([run[metric] for run in runs])
            aggregated[metric] = mean.tolist()
            aggregated[f"{metric}_std"] = std.tolist()
            run_means = [float(np.mean(run[metric])) for run in runs]
            summary_mean, summary_std = _mean_and_sample_std(run_means)
            aggregated[f"{metric}_summary_mean"] = float(summary_mean)
            aggregated[f"{metric}_summary_std"] = float(summary_std)
        unsupervised[line_label] = aggregated

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_unsupervised_lines(
        out_dir=out_dir,
        k_values=k_values,
        unsupervised=unsupervised,
        projection_colors=projection_colors,
    )
    return unsupervised, k_values, n_full, n_eval


def _supervised_spaces(
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    indices: list[int],
) -> tuple[list[tuple[str, np.ndarray]], dict[str, str], str, str, str]:
    embeddings = emb_set.data[indices].astype(np.float32, copy=False)
    n_eval = len(indices)
    pca_components = min(DEFAULT_PCA_COMPONENTS, embeddings.shape[1], n_eval - 1)
    if pca_components < 1:
        raise ValueError("cannot compute PCA ground truth for this embedding set")
    emb_pca = PCA(
        n_components=pca_components, svd_solver="auto", random_state=42
    ).fit_transform(embeddings)
    pca10_components = min(10, embeddings.shape[1], n_eval - 1)
    emb_pca10 = PCA(
        n_components=pca10_components, svd_solver="auto", random_state=42
    ).fit_transform(embeddings)

    full_name = f"Original ({embeddings.shape[1]})"
    pca_name = f"PCA{pca_components}"
    pca10_name = f"PCA{pca10_components}"
    spaces = [
        (full_name, embeddings),
        (pca_name, emb_pca),
        (pca10_name, emb_pca10),
    ]
    entries, colors = _build_projection_entries(reductions, emb_set.name)
    for label, reduction in entries:
        dims = 3 if reduction["dimensions"] == 3 else 2
        spaces.append(
            (
                label,
                reduction["data"][indices, :dims].astype(np.float32, copy=False),
            )
        )
    return spaces, colors, full_name, pca_name, pca10_name


def _evaluate_supervised_label(
    *,
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    metadata: pd.DataFrame,
    spec: EvaluationLabel,
    min_class_size: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    if spec.kind == "continuous":
        return _evaluate_continuous_label(
            emb_set=emb_set,
            reductions=reductions,
            values_lookup=_prepare_continuous_lookup(metadata, spec.column),
            label_column=spec.column,
            out_dir=out_dir,
        )
    return _evaluate_categorical_label(
        emb_set=emb_set,
        reductions=reductions,
        labels_lookup=_prepare_labels_lookup(metadata, spec.column),
        label_column=spec.column,
        min_class_size=min_class_size,
        out_dir=out_dir,
    )


def _evaluate_categorical_label(
    *,
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    labels_lookup: pd.Series,
    label_column: str,
    min_class_size: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    id_to_idx = {identifier: i for i, identifier in enumerate(emb_set.headers)}
    shared_ids = [
        identifier for identifier in emb_set.headers if identifier in labels_lookup.index
    ]
    labels = labels_lookup.loc[shared_ids]
    class_counts = labels.value_counts()
    rare = (
        class_counts[class_counts < min_class_size]
        if min_class_size
        else class_counts.iloc[0:0]
    )
    if min_class_size:
        valid = class_counts[class_counts >= min_class_size].index
        labels = labels[labels.isin(valid)]
    _print_label_summary(
        embedding_name=emb_set.name,
        label_column=label_column,
        min_class_size=min_class_size,
        total_proteins=len(emb_set.headers),
        unique_categories=class_counts.size,
        proteins_with_na=len(emb_set.headers) - len(shared_ids),
        proteins_below_filter=int(rare.sum()),
        categories_below_filter=len(rare),
        remaining_proteins=len(labels),
    )
    if len(labels) < 3:
        raise ValueError(f"not enough proteins for label '{label_column}'")
    labels_int_all = LabelEncoder().fit_transform(labels.to_numpy())
    all_indices = [id_to_idx[identifier] for identifier in labels.index]
    indices, labels_int = _stratified_subsample(
        all_indices,
        labels_int_all,
        _MAX_DIST_SAMPLES,
        np.random.default_rng(42),
    )
    n_eval = len(indices)
    k_stored = min(DEFAULT_K_STORED, n_eval - 1)
    k_values = _valid_k_values(n_eval, k_stored)
    if not k_values:
        raise ValueError(
            "not enough proteins for the minimum evaluation neighborhood (k=5)"
        )
    spaces, colors, full_name, pca_name, pca10_name = _supervised_spaces(
        emb_set, reductions, indices
    )

    accuracy_runs: dict[str, list[list[float]]] = {}
    silhouette_runs: dict[str, list[float]] = {}
    concordex_runs: dict[str, list[list[float]]] = {}
    linear_auc_runs: dict[str, list[float]] = {}
    linear_f1_runs: dict[str, list[float]] = {}
    classifier_splits = _classification_splits(labels_int)
    classifier_folds = len(classifier_splits) if classifier_splits else 0
    for space, values in spaces:
        metric = "cosine" if space == full_name else "euclidean"
        neighbor_values = normalize(values, norm="l2") if metric == "cosine" else values
        neighbors = _neighbors(neighbor_values, k_stored, metric)
        accuracy_runs.setdefault(space, []).append(
            [_knn_accuracy_at_k(neighbors, labels_int, k) for k in k_values]
        )
        silhouette_runs.setdefault(space, []).append(
            _safe_silhouette(values, labels_int, metric=metric)
        )
        concordex_runs.setdefault(space, []).append(
            _concordex_scores(neighbors, labels_int, k_values)
        )
        auc, f1 = _linear_classifier_scores(
            values, labels_int, classifier_splits
        )
        linear_auc_runs.setdefault(space, []).append(auc)
        linear_f1_runs.setdefault(space, []).append(f1)

    accuracies, accuracy_stds = _aggregate_vector_runs(accuracy_runs)
    silhouettes, silhouette_stds = _aggregate_scalar_runs(silhouette_runs)
    concordex, concordex_stds = _aggregate_vector_runs(concordex_runs)
    linear_auc, linear_auc_stds = _aggregate_scalar_runs(linear_auc_runs)
    linear_f1, linear_f1_stds = _aggregate_scalar_runs(linear_f1_runs)

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_knn_accuracy(
        out_dir=out_dir,
        k_values=k_values,
        knn_accuracy=accuracies,
        knn_accuracy_std=accuracy_stds,
        projection_colors=colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_silhouette(
        out_dir=out_dir,
        silhouette=silhouettes,
        silhouette_std=silhouette_stds,
        projection_colors=colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_concordex(
        out_dir=out_dir,
        k_values=k_values,
        concordex=concordex,
        concordex_std=concordex_stds,
        projection_colors=colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    _plot_categorical_classifier_metrics(
        out_dir=out_dir,
        auc_scores=linear_auc,
        f1_scores=linear_f1,
        auc_stds=linear_auc_stds,
        f1_stds=linear_f1_stds,
        projection_colors=colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    return [
        {
            "space": space,
            "type": "supervised_categorical",
            "label": label_column,
            "n_samples": n_eval,
            "k_values": ",".join(str(k) for k in k_values),
            "classifier_folds": classifier_folds,
            "robustness_runs": len(accuracy_runs[space]),
            "knn_acc_mean": round(float(np.mean(accuracies[space])), 4),
            "knn_acc_mean_std": round(
                float(
                    _mean_and_sample_std(
                        [np.mean(run) for run in accuracy_runs[space]]
                    )[1]
                ),
                4,
            ),
            "silhouette": round(float(silhouettes[space]), 4),
            "silhouette_std": round(float(silhouette_stds[space]), 4),
            "concordex": round(float(np.mean(concordex[space])), 4),
            "concordex_std": round(
                float(
                    _mean_and_sample_std(
                        [np.mean(run) for run in concordex_runs[space]]
                    )[1]
                ),
                4,
            ),
            "linear_auc": round(float(linear_auc[space]), 4),
            "linear_auc_std": round(float(linear_auc_stds[space]), 4),
            "linear_f1_macro": round(float(linear_f1[space]), 4),
            "linear_f1_macro_std": round(float(linear_f1_stds[space]), 4),
        }
        for space in accuracy_runs
    ]


def _classification_splits(
    labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Create deterministic shared folds, adapting to the smallest class."""
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        logger.warning(
            "Linear classifier metrics require at least two classes with "
            "two samples each; reporting NaN."
        )
        return None
    splitter = StratifiedKFold(
        n_splits=min(5, int(counts.min())),
        shuffle=True,
        random_state=42,
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def _linear_classifier_scores(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[float, float]:
    """Return leakage-safe cross-validated macro ROC-AUC and macro F1."""
    if splits is None:
        return float("nan"), float("nan")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=splits,
        method="predict_proba",
    )
    predictions = probabilities.argmax(axis=1)
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            labels,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return float(auc), float(f1)


def _regression_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic shared folds for continuous predictive metrics."""
    splitter = KFold(
        n_splits=min(5, n_samples),
        shuffle=True,
        random_state=42,
    )
    return list(splitter.split(np.zeros(n_samples)))


def _continuous_space_scores(
    features: np.ndarray,
    geometry_features: np.ndarray,
    targets: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    knn_k: int,
) -> tuple[float, float, float, float]:
    """Compute the four continuous-label metrics for one embedding space."""
    linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_predictions = cross_val_predict(
        linear_model,
        features,
        targets,
        cv=splits,
    )
    linear_r2 = r2_score(targets, linear_predictions)

    knn_model = KNeighborsRegressor(
        n_neighbors=knn_k,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    )
    knn_predictions = cross_val_predict(
        knn_model,
        geometry_features,
        targets,
        cv=splits,
    )
    knn_r2 = r2_score(targets, knn_predictions)
    distance_correlation = _distance_correlation(geometry_features, targets)
    spearman_distance = _spearman_dist_correlation(geometry_features, targets)
    return (
        float(linear_r2),
        float(knn_r2),
        distance_correlation,
        spearman_distance,
    )


def _evaluate_continuous_label(
    *,
    emb_set: EmbeddingSet,
    reductions: list[dict[str, Any]],
    values_lookup: pd.Series,
    label_column: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    id_to_idx = {identifier: i for i, identifier in enumerate(emb_set.headers)}
    active_ids = [
        identifier for identifier in emb_set.headers if identifier in values_lookup.index
    ]
    if len(active_ids) < 3:
        raise ValueError(f"not enough numeric values for label '{label_column}'")
    all_indices = [id_to_idx[identifier] for identifier in active_ids]
    chosen = _random_subsample_indices(len(all_indices), _MAX_DIST_SAMPLES)
    indices = [all_indices[i] for i in chosen]
    targets = values_lookup.loc[active_ids].to_numpy(dtype=float)[chosen]
    if np.unique(targets).size < 2:
        raise ValueError(f"continuous label '{label_column}' is constant")
    spaces, colors, full_name, pca_name, pca10_name = _supervised_spaces(
        emb_set, reductions, indices
    )
    folds = _regression_splits(len(targets))
    knn_k = min(_KNN_REGRESSION_K, min(len(train) for train, _ in folds))
    r2_runs: dict[str, list[float]] = {}
    knn_r2_runs: dict[str, list[float]] = {}
    distance_correlation_runs: dict[str, list[float]] = {}
    spearman_dcorr_runs: dict[str, list[float]] = {}
    for space, values in spaces:
        # Normalize the full embedding so euclidean distance ≡ cosine, matching
        # the metric used by the unsupervised and categorical evaluations.
        norm_values = normalize(values, norm="l2") if space == full_name else values

        linear_r2, knn_r2, distance_correlation, spearman_distance = (
            _continuous_space_scores(
            values,
            norm_values,
            targets,
            folds,
            knn_k,
            )
        )
        r2_runs.setdefault(space, []).append(linear_r2)
        knn_r2_runs.setdefault(space, []).append(knn_r2)
        distance_correlation_runs.setdefault(space, []).append(distance_correlation)
        spearman_dcorr_runs.setdefault(space, []).append(spearman_distance)

    r2_scores, r2_stds = _aggregate_scalar_runs(r2_runs)
    knn_r2_scores, knn_r2_stds = _aggregate_scalar_runs(knn_r2_runs)
    distance_correlations, distance_correlation_stds = _aggregate_scalar_runs(
        distance_correlation_runs
    )
    spearman_dcorr, spearman_dcorr_stds = _aggregate_scalar_runs(
        spearman_dcorr_runs
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_continuous_metrics(
        out_dir=out_dir,
        r2_scores=r2_scores,
        r2_stds=r2_stds,
        knn_r2_scores=knn_r2_scores,
        knn_r2_stds=knn_r2_stds,
        distance_correlations=distance_correlations,
        distance_correlation_stds=distance_correlation_stds,
        spearman_dcorr=spearman_dcorr,
        spearman_dcorr_stds=spearman_dcorr_stds,
        projection_colors=colors,
        full_name=full_name,
        pca_name=pca_name,
        pca10_name=pca10_name,
        label_column=label_column,
    )
    logger.info(
        "Continuous evaluation '%s' / '%s': labels=%d, dropped=%d, eval_n=%d",
        emb_set.name,
        label_column,
        len(active_ids),
        len(emb_set.headers) - len(active_ids),
        len(indices),
    )
    return [
        {
            "space": space,
            "type": "supervised_continuous",
            "label": label_column,
            "n_samples": len(indices),
            "regression_folds": len(folds),
            "knn_k": knn_k,
            "robustness_runs": len(r2_runs[space]),
            "linear_r2": round(r2_scores[space], 4),
            "linear_r2_std": round(r2_stds[space], 4),
            "knn_r2": round(knn_r2_scores[space], 4),
            "knn_r2_std": round(knn_r2_stds[space], 4),
            "distance_correlation": round(distance_correlations[space], 4),
            "distance_correlation_std": round(
                distance_correlation_stds[space], 4
            ),
            "spearman_dcorr": round(spearman_dcorr[space], 4),
            "spearman_dcorr_std": round(spearman_dcorr_stds[space], 4),
        }
        for space in r2_runs
    ]


def _distance_correlation(features: np.ndarray, target: np.ndarray) -> float:
    """Return the biased sample distance correlation in [0, 1]."""
    distances_x = squareform(pdist(features, metric="euclidean"))
    distances_y = np.abs(target[:, None] - target[None, :])
    centered_x = (
        distances_x
        - distances_x.mean(axis=0)[None, :]
        - distances_x.mean(axis=1)[:, None]
        + distances_x.mean()
    )
    centered_y = (
        distances_y
        - distances_y.mean(axis=0)[None, :]
        - distances_y.mean(axis=1)[:, None]
        + distances_y.mean()
    )
    covariance_sq = max(float(np.mean(centered_x * centered_y)), 0.0)
    variance_x_sq = max(float(np.mean(centered_x * centered_x)), 0.0)
    variance_y_sq = max(float(np.mean(centered_y * centered_y)), 0.0)
    denominator = np.sqrt(variance_x_sq * variance_y_sq)
    if denominator == 0:
        return float("nan")
    return float(np.clip(np.sqrt(covariance_sq / denominator), 0.0, 1.0))


def _spearman_dist_correlation(features: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation between pairwise feature distances and absolute target differences.

    A model-free alternative to distance correlation: tests whether points that
    are close in feature space also tend to have similar target values, using
    rank correlation so monotone non-linearities don't inflate the score.
    """
    feat_dists = pdist(features, metric="euclidean")
    target_diffs = pdist(target[:, None], metric="cityblock")
    rho, _ = spearmanr(feat_dists, target_diffs)
    return float(np.clip(rho, -1.0, 1.0))


# ---------------------------------------------------------------------------
# CONCORDEX implementation
# ---------------------------------------------------------------------------


def _concordex_score(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    n_permutations: int = DEFAULT_CONCORDEX_PERMUTATIONS,
    random_state: int = 42,
) -> float:
    """Return the permutation-corrected CONCORDEX coefficient.

    The implementation follows Jackson et al. (2023): construct the
    neighborhood consolidation matrix, average it by source label to obtain
    the label-similarity matrix, and use the mean of that matrix's diagonal as
    the raw coefficient. The corrected coefficient divides the raw value by
    the mean raw coefficient from permuted label assignments.

    Args:
        neighbors: kNN index array of shape (n, k_stored); only the first
                   ``k`` columns are used.
        labels:    Integer class labels of length n.
        k: Number of nearest neighbors to evaluate.
        n_permutations: Number of label permutations used for correction.
        random_state: Seed used to make the null distribution reproducible.

    Returns:
        Corrected CONCORDEX coefficient, or NaN when it is undefined.
    """
    return _concordex_scores(
        neighbors,
        labels,
        [k],
        n_permutations=n_permutations,
        random_state=random_state,
    )[0]


def _concordex_scores(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k_values: list[int],
    *,
    n_permutations: int = DEFAULT_CONCORDEX_PERMUTATIONS,
    random_state: int = 42,
) -> list[float]:
    """Compute corrected CONCORDEX for several k values using one null draw."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != neighbors.shape[0]:
        raise ValueError("labels must be one-dimensional and aligned with neighbors")
    if np.unique(labels).size < 2:
        return [float("nan")] * len(k_values)

    raw = _raw_concordex_coefficients(neighbors, labels, k_values)
    rng = np.random.default_rng(random_state)
    null_coefficients = np.empty((n_permutations, len(k_values)), dtype=float)
    for iteration in range(n_permutations):
        permuted = rng.permutation(labels)
        null_coefficients[iteration] = _raw_concordex_coefficients(
            neighbors, permuted, k_values
        )
    mean_null = null_coefficients.mean(axis=0)
    return [
        float(value / expected) if expected != 0 else float("nan")
        for value, expected in zip(raw, mean_null, strict=True)
    ]


def _raw_concordex_coefficients(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k_values: list[int],
) -> np.ndarray:
    """Compute class-balanced diagonal means for multiple neighborhood sizes."""
    if not k_values or min(k_values) < 1 or max(k_values) > neighbors.shape[1]:
        raise ValueError("k values must be within the stored neighbor range")
    _, encoded = np.unique(labels, return_inverse=True)
    n_classes = int(encoded.max()) + 1
    class_sizes = np.bincount(encoded, minlength=n_classes)
    neighbor_labels = encoded[neighbors[:, : max(k_values)]]
    cumulative_matches = np.cumsum(
        neighbor_labels == encoded[:, None], axis=1, dtype=np.int32
    )

    coefficients = np.empty(len(k_values), dtype=float)
    for index, k in enumerate(k_values):
        point_fractions = cumulative_matches[:, k - 1] / k
        class_fractions = (
            np.bincount(encoded, weights=point_fractions, minlength=n_classes)
            / class_sizes
        )
        coefficients[index] = class_fractions.mean()
    return coefficients


def _neighborhood_consolidation_matrix(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the paper's n-by-m neighborhood consolidation matrix K."""
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    k_use = min(k, neighbors.shape[1])
    if k_use < 1:
        raise ValueError("k must be at least 1 and neighbors must not be empty")

    neighbor_labels = encoded[neighbors[:, :k_use]]
    consolidation = np.zeros(
        (neighbors.shape[0], unique_labels.size), dtype=np.float64
    )
    rows = np.repeat(np.arange(neighbors.shape[0]), k_use)
    np.add.at(consolidation, (rows, neighbor_labels.ravel()), 1.0)
    consolidation /= k_use
    return consolidation, encoded


def _raw_concordex_coefficient(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Compute the uncorrected, class-balanced CONCORDEX coefficient."""
    consolidation, encoded = _neighborhood_consolidation_matrix(
        neighbors, labels, k
    )
    n_classes = consolidation.shape[1]
    similarity = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(similarity, encoded, consolidation)
    class_sizes = np.bincount(encoded, minlength=n_classes)
    similarity /= class_sizes[:, None]
    return float(np.trace(similarity) / n_classes)


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
        f"\nEvaluation summary for {embedding_name} "
        f"(label: {label_column}, filter: {min_class_size})"
    )
    print(
        "1. Total proteins and label categories: "
        f"{total_proteins} total proteins; {unique_categories} unique categories"
    )
    print(f"2. Proteins with NA label: {proteins_with_na} proteins")
    if proteins_below_filter:
        filter_summary = (
            f"{proteins_below_filter} proteins are in "
            f"{categories_below_filter} categories"
        )
    else:
        filter_summary = "0 proteins removed"
    print(f"3. Proteins removed by rare-class filter: {filter_summary}")
    print(
        "4. Proteins considered for evaluation: "
        f"{remaining_proteins} proteins remain"
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
    unsupervised: dict[str, dict[str, Any]] | None = None,
    supervised: list[dict[str, Any]] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for space, metrics in (unsupervised or {}).items():
        rows.append(
            {
                "space": space,
                "type": "unsupervised",
                "robustness_runs": metrics.get("robustness_runs", 1),
                "recall_mean": round(metrics["recall_summary_mean"], 4),
                "recall_mean_std": round(metrics["recall_summary_std"], 4),
                "trust_mean": round(metrics["trust_summary_mean"], 4),
                "trust_mean_std": round(metrics["trust_summary_std"], 4),
                "cont_mean": round(metrics["cont_summary_mean"], 4),
                "cont_mean_std": round(metrics["cont_summary_std"], 4),
            }
        )
    rows.extend(supervised or [])

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
_CATEGORICAL_COL_LABELS: dict[str, str] = {
    "knn_acc_mean": "kNN Accuracy\n(mean)",
    "silhouette": "Silhouette\nScore",
    "concordex": "CONCORDEX\n(mean)",
    "linear_auc": "Linear Classifier\nROC-AUC",
    "linear_f1_macro": "Linear Classifier\nMacro-F1",
}
_CONTINUOUS_COL_LABELS: dict[str, str] = {
    "linear_r2": "Linear\nR²",
    "knn_r2": f"kNN\nR² (k={_KNN_REGRESSION_K})",
    "distance_correlation": "Distance\nCorrelation",
    "spearman_dcorr": "Spearman\nDist. Corr.",
}


def _plot_summary_heatmaps(
    *,
    out_dir: Path,
    k_values: list[int] | None = None,
    label_column: str | None = None,
    supervised_kind: str | None = None,
    n_full: int | None = None,
    n_eval: int | None = None,
) -> None:
    """Read the summary TSV and produce available metric heatmaps.

    Categorical and continuous labels use their respective supervised metric
    columns. Both heatmaps use column-wise min-max normalisation for colour
    mapping while annotating cells with the original numeric values.

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
        uns_indexed = df_uns.set_index("space")
        std_uns = pd.DataFrame(
            {
                _UNSUPERVISED_COL_LABELS[column]: pd.to_numeric(
                    uns_indexed.get(f"{column}_std"), errors="coerce"
                )
                for column in uns_cols
            },
            index=uns_indexed.index,
        )
        df_uns = uns_indexed[uns_cols]
        df_uns.columns = [_UNSUPERVISED_COL_LABELS[c] for c in uns_cols]
        df_uns = df_uns.apply(pd.to_numeric, errors="coerce")
        subtitle = f"Means computed over {k_range_str}{sample_note}"
        if int(pd.to_numeric(uns_indexed["robustness_runs"]).max()) > 1:
            subtitle += " · mean ± sample SD across runs"
        _render_heatmap(
            data=df_uns,
            std_data=std_uns,
            title="Unsupervised DR Evaluation",
            subtitle=subtitle,
            out_path=out_dir / "heatmap_unsupervised.png",
            cmap="YlGnBu",
        )

    # ---- Supervised ----
    df_sup = df[df["type"].astype(str).str.startswith("supervised_")].copy()
    column_labels = (
        _CONTINUOUS_COL_LABELS
        if supervised_kind == "continuous"
        else _CATEGORICAL_COL_LABELS
    )
    sup_cols = [c for c in column_labels if c in df_sup.columns]
    if not df_sup.empty and sup_cols:
        stored_k = (
            df_sup["k_values"].dropna()
            if "k_values" in df_sup.columns
            else pd.Series(dtype=str)
        )
        stored_folds = (
            df_sup["classifier_folds"].dropna()
            if "classifier_folds" in df_sup.columns
            else pd.Series(dtype=int)
        )
        stored_regression_folds = (
            df_sup["regression_folds"].dropna()
            if "regression_folds" in df_sup.columns
            else pd.Series(dtype=int)
        )
        stored_knn_k = (
            df_sup["knn_k"].dropna()
            if "knn_k" in df_sup.columns
            else pd.Series(dtype=int)
        )
        sup_indexed = df_sup.set_index("space")
        std_sup = pd.DataFrame(
            {
                column_labels[column]: pd.to_numeric(
                    sup_indexed.get(f"{column}_std"), errors="coerce"
                )
                for column in sup_cols
            },
            index=sup_indexed.index,
        )
        df_sup = sup_indexed[sup_cols]
        df_sup.columns = [column_labels[c] for c in sup_cols]
        df_sup = df_sup.apply(pd.to_numeric, errors="coerce")
        if supervised_kind == "continuous":
            folds = (
                int(stored_regression_folds.iloc[0])
                if not stored_regression_folds.empty
                else 0
            )
            knn_k = int(stored_knn_k.iloc[0]) if not stored_knn_k.empty else 0
            subtitle = (
                f"{folds}-fold CV for linear/kNN R2 (kNN k={knn_k}); "
                "dCor and Spearman correlation of pairwise distances"
            )
        else:
            used_k = str(stored_k.iloc[0]).split(",") if not stored_k.empty else []
            categorical_k = f"k in {{{', '.join(used_k)}}}"
            folds = int(stored_folds.iloc[0]) if not stored_folds.empty else 0
            subtitle = (
                f"kNN Accuracy and CONCORDEX means over {categorical_k}; "
                f"linear classifier uses {folds}-fold CV"
            )
        if int(pd.to_numeric(sup_indexed["robustness_runs"]).max()) > 1:
            subtitle += " · mean ± sample SD across runs"
        _render_heatmap(
            data=df_sup,
            std_data=std_sup,
            title=f"Supervised DR Evaluation  ·  {label_column}",
            subtitle=subtitle,
            out_path=out_dir / "heatmap_supervised.png",
            cmap="YlOrRd",
            lower_is_better=False,
        )


def _render_heatmap(
    *,
    data: pd.DataFrame,
    std_data: pd.DataFrame | None = None,
    title: str,
    subtitle: str,
    out_path: Path,
    cmap: str,
    lower_is_better: bool = False,
) -> None:
    """Render a single publication-quality heatmap to *out_path*.

    Design choices
    --------------
    * Crisp white grid lines separate cells.
    * Annotation text is black on light cells and white on dark cells for
      maximum contrast (WCAG AA).
    * A compact vertical colour bar sits to the right of the axes with a label
      that clarifies the normalisation.
    * Typography uses a narrow sans-serif stack so long row/column names fit
      comfortably.
    * The figure background is white (#FFFFFF) with a subtle outer border.
    """
    n_rows, n_cols = data.shape

    # Column-wise min-max normalisation — NaN cells stay NaN (rendered grey).
    column_ranges = data.max() - data.min()
    norm_data = (data - data.min()) / column_ranges.replace(0.0, np.nan)
    for column in data.columns[column_ranges == 0.0]:
        norm_data.loc[data[column].notna(), column] = 0.5
    if lower_is_better:
        norm_data = 1.0 - norm_data

    # ---- Figure geometry ----
    cell_w = 2.0          # inches per column
    cell_h = 0.55         # inches per row
    left_margin = 2.6     # room for row labels
    right_margin = 1.05   # room for vertical colour bar + its labels
    top_margin = 1.05     # room for title + subtitle
    bottom_margin = 0.90  # room for column labels

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bottom_margin
    fig_w = max(fig_w, 6.0)
    fig_h = max(fig_h, 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

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
                cell_text = f"{raw_val:.2f}"
                if std_data is not None:
                    std_val = std_data.iloc[row_idx, col_idx]
                    if pd.notna(std_val):
                        cell_text += f"\n± {std_val:.2f}"

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
                fontsize=13.5,
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
        fontsize=13.5,
        fontfamily="DejaVu Sans",
        ha="center",
        va="top",
    )
    ax.set_yticklabels(
        data.index[::-1],
        fontsize=13.5,
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
        fontsize=18,
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
        fontsize=13,
        fontstyle="italic",
        fontfamily="DejaVu Sans",
        color="#555555",
    )

    # ---- Colour bar (vertical, right side) ----
    cbar_gap_in = 0.18     # gap between ax right edge and colourbar left, in inches
    cbar_w_in   = 0.20     # colourbar width (thin), in inches
    cbar_left   = (left_margin + n_cols * cell_w + cbar_gap_in) / fig_w
    cbar_ax = fig.add_axes(
        [cbar_left, ax_bottom, cbar_w_in / fig_w, ax_height]
    )
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(
        "Column-normalised desirability\n(0 = worst, 1 = best)",
        fontsize=15,
    )
    cbar.ax.tick_params(labelsize=15, colors="#444444", length=2)
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
            std_values = np.asarray(metrics[f"{metric_key}_std"], dtype=float)
            yerr = np.nan_to_num(std_values, nan=0.0)
            ax.errorbar(
                k_values,
                metrics[metric_key],
                yerr=yerr,
                color=get_line_color(metrics["projection_label"]),
                linestyle=get_linestyle(metrics["gt_label"]),
                marker="o",
                markersize=4,
                linewidth=1.8,
                capsize=3,
                label=line_label,
            )
        style_axis(ax, title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_line_plot("recall", "kNN Recall", "recall.png")
    save_line_plot("trust", "Trustworthiness", "trustworthiness.png")
    save_line_plot("cont", "Continuity", "continuity.png")


def _plot_continuous_metrics(
    *,
    out_dir: Path,
    r2_scores: dict[str, float],
    r2_stds: dict[str, float],
    knn_r2_scores: dict[str, float],
    knn_r2_stds: dict[str, float],
    distance_correlations: dict[str, float],
    distance_correlation_stds: dict[str, float],
    spearman_dcorr: dict[str, float],
    spearman_dcorr_stds: dict[str, float],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    pca10_name: str,
    label_column: str,
) -> None:
    """Plot linear/kNN predictive power and pairwise distance correlations by space."""

    def color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        if space == pca10_name:
            return BASE_COLORS["PCA10"]
        return projection_colors.get(space, "#333333")

    def plot_metric(
        values: dict[str, float],
        stds: dict[str, float],
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        spaces = list(values)
        errors = [0.0 if np.isnan(stds[space]) else stds[space] for space in spaces]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            spaces,
            list(values.values()),
            yerr=errors,
            capsize=4,
            color=[color(space) for space in spaces],
            edgecolor="white",
            width=0.5,
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_title(f"{title} ({label_column})", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Space", fontsize=11)
        ax.set_xticks(range(len(spaces)))
        ax.set_xticklabels(spaces, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_metric(
        r2_scores,
        r2_stds,
        "Linear Regression Predictive Power",
        "Cross-validated R²",
        "linear_r2.png",
    )
    plot_metric(
        knn_r2_scores,
        knn_r2_stds,
        f"kNN Regression Predictive Power (k={_KNN_REGRESSION_K})",
        "Cross-validated R²",
        "knn_r2.png",
    )
    plot_metric(
        distance_correlations,
        distance_correlation_stds,
        "Distance Correlation",
        "Distance correlation",
        "distance_correlation.png",
    )
    plot_metric(
        spearman_dcorr,
        spearman_dcorr_stds,
        "Spearman Distance Correlation",
        "Spearman ρ",
        "spearman_distance_correlation.png",
    )


def _plot_categorical_classifier_metrics(
    *,
    out_dir: Path,
    auc_scores: dict[str, float],
    f1_scores: dict[str, float],
    auc_stds: dict[str, float],
    f1_stds: dict[str, float],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    pca10_name: str,
    label_column: str,
) -> None:
    """Plot cross-validated linear-classifier predictive power by space."""

    def color(space: str) -> str:
        if space == full_name:
            return BASE_COLORS["Full"]
        if space == pca_name:
            return BASE_COLORS["PCA"]
        if space == pca10_name:
            return BASE_COLORS["PCA10"]
        return projection_colors.get(space, "#333333")

    def plot_metric(
        values: dict[str, float],
        stds: dict[str, float],
        title: str,
        ylabel: str,
        filename: str,
        baseline: float | None = None,
    ) -> None:
        spaces = list(values)
        errors = [0.0 if np.isnan(stds[space]) else stds[space] for space in spaces]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            spaces,
            list(values.values()),
            yerr=errors,
            capsize=4,
            color=[color(space) for space in spaces],
            edgecolor="white",
            width=0.5,
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        if baseline is not None:
            ax.axhline(
                baseline,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label=f"Random baseline ({baseline:.1f})",
            )
            ax.legend(fontsize=9, framealpha=0.7)
        ax.set_title(f"{title} ({label_column})", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Space", fontsize=11)
        ax.set_xticks(range(len(spaces)))
        ax.set_xticklabels(spaces, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_metric(
        auc_scores,
        auc_stds,
        "Linear Classifier ROC-AUC",
        "Cross-validated ROC-AUC",
        "linear_classifier_auc.png",
        baseline=0.5,
    )
    plot_metric(
        f1_scores,
        f1_stds,
        "Linear Classifier Macro-F1",
        "Cross-validated Macro-F1",
        "linear_classifier_f1.png",
    )


def _plot_knn_accuracy(
    *,
    out_dir: Path,
    k_values: list[int],
    knn_accuracy: dict[str, list[float]],
    knn_accuracy_std: dict[str, list[float]],
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
        ax.errorbar(
            k_values,
            acc_values,
            yerr=np.nan_to_num(knn_accuracy_std[space], nan=0.0),
            color=get_space_color(space),
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
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
    silhouette_std: dict[str, float],
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
    errors = [
        0.0 if np.isnan(silhouette_std[space]) else silhouette_std[space]
        for space in spaces
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        spaces,
        values,
        yerr=errors,
        capsize=4,
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
    k_values: list[int],
    concordex: dict[str, list[float]],
    concordex_std: dict[str, list[float]],
    projection_colors: dict[str, str],
    full_name: str,
    pca_name: str,
    pca10_name: str,
    label_column: str,
) -> None:
    """Plot permutation-corrected CONCORDEX across neighborhood sizes.

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

    fig, ax = plt.subplots(figsize=(10, 5))
    for space, scores in concordex.items():
        ax.errorbar(
            k_values,
            scores,
            yerr=np.nan_to_num(concordex_std[space], nan=0.0),
            color=get_space_color(space),
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            label=space,
        )
    ax.axhline(
        1.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="Random baseline (1.0)",
    )
    ax.set_title(
        f"CONCORDEX Score ({label_column})", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Score (ratio to random)", fontsize=11)
    ax.set_xlabel("k", fontsize=11)
    ax.set_xticks(k_values)
    ax.legend(fontsize=9, framealpha=0.7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "concordex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
