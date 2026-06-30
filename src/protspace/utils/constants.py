"""Lightweight constants and config — no heavy dependencies (sklearn, umap, pacmap).

Import this module freely without triggering numba/pynndescent compilation.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_args

# Method name constants
PCA_NAME = "pca"
TSNE_NAME = "tsne"
UMAP_NAME = "umap"
PACMAP_NAME = "pacmap"
MDS_NAME = "mds"
LOCALMAP_NAME = "localmap"
RHOPCA_NAME = "rhopca"
DENSEMAP_NAME = "densmap"
TRIMAP_NAME = "trimap"
PHATE_NAME = "phate"
CPCA_NAME = "cpca"

REDUCER_METHODS = [
    PCA_NAME,
    TSNE_NAME,
    UMAP_NAME,
    PACMAP_NAME,
    MDS_NAME,
    LOCALMAP_NAME,
    RHOPCA_NAME,
    DENSEMAP_NAME,
    TRIMAP_NAME,
    PHATE_NAME,
    CPCA_NAME,
]

# Metric types
METRIC_TYPES = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class DimensionReductionConfig:
    """Configuration for dimension reduction methods.

    Parameters:
        n_components: Number of dimensions in reduced space (2 or 3)
        n_neighbors: Number of neighbors for manifold learning (>0)
        metric: Distance metric to use
        precomputed: Whether distances are precomputed
        min_dist: Minimum distance for UMAP (0-1)
        perplexity: Perplexity for t-SNE (5-50)
        learning_rate: Learning rate for t-SNE optimization (>0)
        mn_ratio: Ratio for PaCMAP (0-1)
        fp_ratio: Ratio for PaCMAP (>0)
        n_init: Number of initializations for MDS (>0)
        max_iter: Maximum iterations (>0)
        eps: Convergence tolerance (>0)
        random_state: Random seed for reproducibility (>= 0)
    """

    # ------------------------------------------------------------------
    # Shared / universal parameters
    # ------------------------------------------------------------------
    # change needed to enable rhoPCA50+umap2
    #n_components: int = field(default=2, metadata={"allowed": [2, 3]})
    n_components: int = field(default=2, metadata={"gt": 0})
    # ^ PCA, t-SNE, UMAP, PaCMAP, MDS, LocalMAP, rhoPCA

    random_state: int = field(default=42, metadata={"gte": 0})
    # ^ t-SNE, UMAP, PaCMAP, LocalMAP, MDS

    # ------------------------------------------------------------------
    # PCA-family parameters
    # ------------------------------------------------------------------
    component_start: int = field(default=1, metadata={"gte": 1})
    # ^ PCA, rhoPCA — 1-indexed starting component

    # ------------------------------------------------------------------
    # Neighbor-graph / manifold parameters
    # ------------------------------------------------------------------
    n_neighbors: int = field(default=15, metadata={"gt": 0})
    # ^ UMAP, PaCMAP, LocalMAP

    metric: METRIC_TYPES = field(
        default="euclidean", metadata={"allowed": list(get_args(METRIC_TYPES))}
    )
    # ^ t-SNE, UMAP

    min_dist: float = field(default=0.1, metadata={"gte": 0, "lte": 1})
    # ^ UMAP — minimum distance between points in low-dimensional space

    densmap: bool = field(default=False)
    # ^ densMAP — enable density-preserving mode (umap-learn >= 0.5)

    # ------------------------------------------------------------------
    # TriMAP parameters
    # ------------------------------------------------------------------
    trimap_n_inliers: int = field(default=12, metadata={"gt": 0})
    # ^ TriMAP — number of nearest neighbors for local structure preservation

    trimap_n_outliers: int = field(default=4, metadata={"gt": 0})
    # ^ TriMAP — number of outlier pairs for global structure

    trimap_lr: float = field(default=0.1, metadata={"gt": 0})
    # ^ TriMAP — learning rate (note: TriMAP default 0.1, unlike t-SNE's 200)

    trimap_n_iters: int = field(default=400, metadata={"gt": 0})
    # ^ TriMAP — number of optimization iterations

    trimap_apply_pca: bool = field(default=True)
    # ^ TriMAP — whether to apply PCA pre-processing before embedding

    # ------------------------------------------------------------------
    # PHATE parameters
    # ------------------------------------------------------------------
    phate_knn: int = field(default=5, metadata={"gt": 0})
    # ^ PHATE — number of nearest neighbors (note: PHATE default 5, unlike UMAP's 15)

    phate_decay: int = field(default=40, metadata={"gt": 0})
    # ^ PHATE — decay rate for the diffusion kernel

    phate_n_landmark: int = field(default=None)
    # ^ PHATE — landmark subsampling (None = no subsampling; set to e.g. 2000 for large datasets)

    phate_t: str = field(default="auto")
    # ^ PHATE — diffusion time scale: "auto" or an integer number of steps

    phate_gamma: float = field(default=1.0, metadata={"gt": 0})
    # ^ PHATE — information distance gamma parameter

    phate_n_pca: int = field(default=100, metadata={"gt": 0})
    # ^ PHATE — number of PCA components for pre-processing

    # ------------------------------------------------------------------
    # t-SNE parameters
    # ------------------------------------------------------------------
    perplexity: int = field(default=30, metadata={"gte": 5, "lte": 50})
    # ^ t-SNE

    learning_rate: int = field(default=200, metadata={"gt": 0})
    # ^ t-SNE

    # ------------------------------------------------------------------
    # PaCMAP / LocalMAP parameters
    # ------------------------------------------------------------------
    mn_ratio: float = field(default=0.5, metadata={"gte": 0, "lte": 1})
    # ^ PaCMAP, LocalMAP — mid-near pair ratio

    fp_ratio: float = field(default=2.0, metadata={"gt": 0})
    # ^ PaCMAP, LocalMAP — further pair ratio

    # ------------------------------------------------------------------
    # MDS parameters
    # ------------------------------------------------------------------
    precomputed: bool = field(default=False)
    # ^ MDS — whether input is a precomputed distance matrix

    n_init: int = field(default=4, metadata={"gt": 0})
    # ^ MDS — number of initializations

    max_iter: int = field(default=300, metadata={"gt": 0})
    # ^ MDS — maximum iterations

    eps: float = field(default=1e-3, metadata={"gt": 0})
    # ^ MDS — convergence tolerance

    # ------------------------------------------------------------------
    # rhoPCA / cPCA (contrastive) parameters
    # ------------------------------------------------------------------
    scale_variance: bool = field(default=True)
    # ^ rhoPCA, cPCA — whether to scale variance

    cpca_alpha: float = field(default=None)
    # ^ cPCA — contrast strength α (None = auto-select via spectral gap)

    background: str = field(default=None)
    # ^ rhoPCA, cPCA — path to background HDF5 file (set by pipeline)

    background_matrix: Any = field(default=None)
    # ^ rhoPCA, cPCA — loaded background data matrix (set by pipeline)

    def __post_init__(self):
        """Validate configuration parameters."""
        for data_field in fields(self):
            value = getattr(self, data_field.name)
            if value is None:
                continue
            metadata = data_field.metadata

            if "allowed" in metadata:
                if value not in metadata["allowed"]:
                    raise ValueError(
                        f"{data_field.name} must be one of {metadata['allowed']}"
                    )

            if "gt" in metadata:
                if value <= metadata["gt"]:
                    raise ValueError(
                        f"{data_field.name} must be greater than {metadata['gt']}"
                    )

            if "lt" in metadata:
                if value >= metadata["lt"]:
                    raise ValueError(
                        f"{data_field.name} must be less than {metadata['lt']}"
                    )

            if "gte" in metadata:
                if value < metadata["gte"]:
                    raise ValueError(
                        f"{data_field.name} must be greater than or equal to {metadata['gte']}"
                    )

            if "lte" in metadata:
                if value > metadata["lte"]:
                    raise ValueError(
                        f"{data_field.name} must be less than or equal to {metadata['lte']}"
                    )
