import inspect
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from typing import Any, get_type_hints

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from rhopca.methods import rhoPCA
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Re-export constants and config from lightweight module
from protspace.utils.constants import (  # noqa: F401
    CPCA_NAME,
    DENSEMAP_NAME,
    LOCALMAP_NAME,
    MDS_NAME,
    METRIC_TYPES,
    PACMAP_NAME,
    PCA_NAME,
    PHATE_NAME,
    REDUCER_METHODS,
    RHOPCA_NAME,
    TRIMAP_NAME,
    TSNE_NAME,
    UMAP_NAME,
    DimensionReductionConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# annoy compatibility shim — annoy can segfault or return empty results on
# certain platforms (notably macOS ARM64).  We detect that at first use and
# transparently swap in an sklearn-based replacement so PaCMAP / LocalMAP
# keep working everywhere.
# ---------------------------------------------------------------------------
_annoy_checked: bool = False


def _ensure_annoy_or_fallback() -> None:
    """Patch pacmap to use sklearn if annoy is broken. Only runs once."""
    global _annoy_checked
    if _annoy_checked:
        return
    _annoy_checked = True

    import subprocess
    import sys

    # Run the check in a subprocess so a segfault doesn't kill the main process
    code = (
        "from annoy import AnnoyIndex; import random; random.seed(0); "
        "d=10; t=AnnoyIndex(d,'euclidean'); "
        "[t.add_item(i,[random.gauss(0,1) for _ in range(d)]) for i in range(50)]; "
        "t.build(5); "
        "exit(0 if len(t.get_nns_by_item(0,10))>=10 else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], timeout=10, capture_output=True
        )
        if result.returncode == 0:
            return
    except Exception:
        pass

    # annoy is broken — swap in sklearn fallback
    import pacmap.pacmap as _pm
    from sklearn.neighbors import NearestNeighbors

    class _SklearnAnnoyIndex:
        """Drop-in AnnoyIndex replacement backed by sklearn NearestNeighbors."""

        def __init__(self, dim: int, metric: str = "euclidean"):
            self.dim = dim
            self.metric = metric
            self._items: list = []

        def add_item(self, i: int, vec) -> None:
            while len(self._items) <= i:
                self._items.append(None)
            self._items[i] = vec

        def set_seed(self, seed: int) -> None:
            pass  # determinism handled by sklearn

        def build(self, n_trees: int) -> None:
            self._data = np.array(self._items, dtype=np.float64)
            self._nn = NearestNeighbors(metric=self.metric, algorithm="auto")
            self._nn.fit(self._data)

        def get_nns_by_item(self, i: int, n: int) -> list[int]:
            k = min(n, len(self._data))
            _, idx = self._nn.kneighbors(self._data[i : i + 1], n_neighbors=k)
            return idx[0].tolist()

        def get_distance(self, i: int, j: int) -> float:
            return float(np.linalg.norm(self._data[i] - self._data[j]))

    _pm.AnnoyIndex = _SklearnAnnoyIndex  # type: ignore[attr-defined]
    logger.warning(
        "annoy is non-functional on this platform; "
        "using sklearn NearestNeighbors fallback for PaCMAP/LocalMAP"
    )

    # Constants and DimensionReductionConfig are imported from constants.py above

    def parameters_by_method(self, method: str) -> list[dict[str, Any]]:
        from pacmap import LocalMAP, PaCMAP
        from rhopca.methods import rhoPCA
        from umap import UMAP

        method_map = {
            TSNE_NAME: TSNE,
            PCA_NAME: PCA,
            UMAP_NAME: UMAP,
            PACMAP_NAME: PaCMAP,
            MDS_NAME: MDS,
            LOCALMAP_NAME: LocalMAP,
            RHOPCA_NAME: rhoPCA
        }

        if method not in method_map:
            return []

        def _get_parameter_desc_from_docstring(parameter: str, docstring: str) -> str:
            large_splits = []
            possible_split_variants = [
                f"{parameter} : ",
                f"{parameter}: ",
                f"{parameter}:",
                parameter,
            ]
            for split_variant in possible_split_variants:
                if split_variant in docstring:
                    large_splits = docstring.split(split_variant)
                    break
            if len(large_splits) == 0:
                return ""
            large_split = large_splits[0] if len(large_splits) == 1 else large_splits[1]
            param_split = (
                large_split.split("\n\n")[0]
                if "\n" in large_split
                else large_split.split("\n")[0]
            )
            param_split_cleaned = (
                param_split.replace("\n\n", "")
                .replace("\t", "")
                .replace("  ", " ")
                .replace("   ", " ")
                .strip()
            )
            return param_split_cleaned

        type_hints = get_type_hints(self.__class__)

        try:
            method_function = method_map[method]
            method_signature = inspect.signature(method_function)
            docstring = inspect.getdoc(method_function)
            method_parameters = list(method_signature.parameters.keys())
            # Create a dictionary of lowercase attribute names to their original names
            lowercase_fields = {
                data_field.name.lower(): data_field for data_field in fields(self)
            }
            result = []
            for param in method_parameters:
                # Exclude parameters not relevant for certain methods
                if method == MDS_NAME and param == "metric":
                    continue
                if method == UMAP_NAME and param == "learning_rate":
                    continue
                if method == LOCALMAP_NAME and param == "metric":
                    continue

                if param.lower() in lowercase_fields:
                    data_field = lowercase_fields[param.lower()]
                    field_type_hint = type_hints.get(data_field.name, Any)
                    field_type_name = getattr(
                        field_type_hint, "__name__", str(field_type_hint)
                    )
                    if hasattr(
                        field_type_hint, "__args__"
                    ):  # Handle Literal, Union etc.
                        field_type_name = str(field_type_hint).replace("typing.", "")

                    doc_desc = _get_parameter_desc_from_docstring(
                        parameter=param, docstring=docstring
                    )
                    description = (
                        doc_desc
                        if doc_desc
                        else f"{data_field.name}: Config parameter. Default: {data_field.default}"
                    )

                    result.append(
                        {
                            "name": param.lower(),
                            "default": data_field.default,
                            "description": description,
                            "constraints": {
                                "type": field_type_name,
                                **data_field.metadata,
                            },
                        }
                    )
            return result
        except Exception as e:
            logger.error("Failed to extract parameter info: %s", e)
            return []


class DimensionReducer(ABC):
    """Abstract base class for dimension reduction methods."""

    def __init__(self, config: DimensionReductionConfig):
        self.config = config

    @abstractmethod
    def fit_transform(self, data: np.ndarray, background_data: np.ndarray=None) -> np.ndarray:
        """Transform data to lower dimensions."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the reduction."""
        pass


class PCAReducer(DimensionReducer):
    """Principal Component Analysis reduction, preferring ARPACK solver."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        solver = "arpack"
        n_samples, n_annotations = data.shape
        n_out = self.config.n_components
        cs = getattr(self.config, "component_start", 1)
        # Total PCs to compute = output dims + (start offset - 1)
        k = n_out + cs - 1

        # ARPACK requires n_components < min(shape), fallback to full SVD otherwise
        if k >= min(n_samples, n_annotations):
            logger.warning(
                f"PCA: n_components ({k}) >= min(shape) ({min(n_samples, n_annotations)}). "
                f"'arpack' solver unavailable, falling back to 'full' solver."
            )
            solver = "full"

        pca = PCA(n_components=k, svd_solver=solver)
        try:
            result = pca.fit_transform(data)
            self.explained_variance = pca.explained_variance_ratio_.tolist()
            self.used_solver = solver  # Store the solver that was actually used
            # Slice to the requested component range (1-indexed)
            return result[:, cs - 1: cs - 1 + n_out]
        except Exception as e:
            logger.error(f"PCA failed using '{solver}' solver: {e}")
            raise

    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the reduction."""
        params = {
            "n_components": self.config.n_components,
            "component_start": self.config.component_start,
            # Report the solver used, default to 'arpack' if not set yet
            "svd_solver": getattr(self, "used_solver", "arpack"),
        }
        if hasattr(self, "explained_variance"):
            params["explained_variance_ratio"] = self.explained_variance
        return params


class rhoPCAReducer(DimensionReducer):
    """rhoPCA - contrastive dimensionality reduction method.

    This reducer handles parsing background data explicitly from the CLI,
    formats matrices into AnnData structures, runs the contrastive PCA model,
    and extracts background-aware projections directly from the fitted model attributes.
    """

    def __init__(self, config: DimensionReductionConfig):
        super().__init__(config)

    def fit_transform(self, data: np.ndarray, background_data: np.ndarray = None) -> np.ndarray:
        # Use background_data if already passed by the pipeline (avoids double-load).
        # Only fall back to CLI arg parsing when the reducer is called standalone.
        if background_data is None:
            bg_path_str = None
            if "--background" in sys.argv:
                try:
                    idx = sys.argv.index("--background")
                    bg_path_str = sys.argv[idx + 1]
                except IndexError:
                    pass

            if bg_path_str:
                bg_path = Path(bg_path_str)
                if not bg_path.exists():
                    raise FileNotFoundError(
                        f"The background file specified in the command line does not exist: {bg_path.resolve()}"
                    )

                logger.info("rhoPCA: reading background matrix from CLI: %s", bg_path)
                with h5py.File(bg_path, "r") as f:
                    first_key = list(f.keys())[0]
                    if f[first_key].ndim == 1:
                        background_data = np.vstack([f[key][:] for key in f.keys()])
                    else:
                        background_data = np.array(f[first_key])

        if background_data is None:
            raise ValueError(
                "rhoPCA requires background data. The pipeline failed to parse "
                "the '--background' flag path directly from your terminal input."
            )
        target_data = data
        X = np.vstack([target_data, background_data])

        labels = (
            ["target"] * len(target_data)
            + ["background"] * len(background_data)
        )

        adata = ad.AnnData(X)
        adata.obs["group"] = pd.Categorical(labels)

        scale_var = getattr(self.config, "scale_variance", True)
        dims = getattr(self.config, "n_components", 2)
        cs = getattr(self.config, "component_start", 1)
        total_needed = dims + cs - 1

        # testing
        #print(f"Doing rhoPCA with dims: {dims}")

        model = rhoPCA(
            adata,
            contrast_column="group",
            target="target",
            background="background",
            scale_variance=scale_var
        )
        model.fit()

        full_embeddings = None

        if hasattr(model, 'target_proj') and model.target_proj is not None:
            full_embeddings = np.array(model.target_proj)
            full_embeddings = full_embeddings[:, :total_needed]
        if full_embeddings is None and hasattr(model, 'loadings'):
            loadings = model.loadings
            if loadings is not None:
                v_slice = loadings[:, :total_needed]
                full_embeddings = np.dot(X, v_slice)[:len(target_data)]

        if full_embeddings is None:
            raise KeyError(
                f"Could not extract target projections or loadings from rhoPCA.\n"
                f"Available attributes: {[a for a in dir(model) if not a.startswith('__')]}"
            )

        # Slice to the requested component range (1-indexed)
        return full_embeddings[:, cs - 1: cs - 1 + dims]

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": getattr(self.config, "n_components", 2),
            "scale_variance": getattr(self.config, "scale_variance", True),
            "component_start": getattr(self.config, "component_start", 1),
        }


class CPCAReducer(DimensionReducer):
    """cPCA — contrastive PCA via the `contrastive` package.

    C = cov(foreground) − α · cov(background).  α controls the contrast
    strength: positive suppresses background variance, negative boosts it,
    zero recovers standard PCA.

    Default α = None → CPCA auto-selects α via spectral gap heuristics.
    """

    def fit_transform(
        self, data: np.ndarray, background_data: np.ndarray = None
    ) -> np.ndarray:
        from contrastive import CPCA

        # --- Load background ---
        # Use background_data if already passed by the pipeline (avoids double-load).
        # Only fall back to CLI arg parsing when the reducer is called standalone.
        if background_data is None:
            bg_path_str = None
            if "--background" in sys.argv:
                try:
                    idx = sys.argv.index("--background")
                    bg_path_str = sys.argv[idx + 1]
                except IndexError:
                    pass

            if bg_path_str:
                bg_path = Path(bg_path_str)
                if not bg_path.exists():
                    raise FileNotFoundError(
                        f"The background file specified in the command line "
                        f"does not exist: {bg_path.resolve()}"
                    )
                logger.info(
                    "cPCA: reading background matrix from CLI: %s", bg_path
                )
                with h5py.File(bg_path, "r") as f:
                    first_key = list(f.keys())[0]
                    if f[first_key].ndim == 1:
                        background_data = np.vstack(
                            [f[key][:] for key in f.keys()]
                        )
                    else:
                        background_data = np.array(f[first_key])

        if background_data is None:
            raise ValueError(
                "cPCA requires background data. Provide it via the "
                "--background CLI flag."
            )

        target_data = data
        dims = self.config.n_components
        cs = getattr(self.config, "component_start", 1)
        total_needed = dims + cs - 1
        standardize = getattr(self.config, "scale_variance", True)
        alpha = getattr(self.config, "cpca_alpha", None)

        # --- Fit + transform ---
        # CPCA requires d < n_bg for its internal eigendecomposition.
        # When d > n_bg (common for protein embeddings: 1024 dims vs ~1000
        # background samples), we PCA-preprocess first to avoid a shape
        # mismatch in the contrastive package.
        n_bg = background_data.shape[0]
        if n_bg <= background_data.shape[1]:
            pca_dim = min(n_bg - 1, 100)
            logger.info(
                "cPCA: n_background (%d) <= d (%d) — applying PCA to %d dims "
                "before contrastive analysis.",
                n_bg, background_data.shape[1], pca_dim,
            )
            pca_pre = PCA(n_components=pca_dim, random_state=42)
            target_data = pca_pre.fit_transform(target_data)
            background_data = pca_pre.transform(background_data)

        model = CPCA(
            n_components=total_needed,
            standardize=standardize,
            verbose=False,
        )
        scores = model.fit_transform(
            target_data,
            background_data,
            alpha_selection="manual" if alpha is not None else "auto",
            alpha_value=alpha,
        )

        # When auto-selecting, CPCA returns a list of arrays (one per α).
        # Take the first (best) one.
        if isinstance(scores, list):
            scores = scores[0]
        else:
            scores = np.array(scores)

        # Slice to the requested component range (1-indexed)
        return scores[:, cs - 1: cs - 1 + dims]

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": getattr(self.config, "n_components", 2),
            "standardize": getattr(self.config, "scale_variance", True),
            "component_start": getattr(self.config, "component_start", 1),
            "alpha": getattr(self.config, "cpca_alpha", None),
        }


class TSNEReducer(DimensionReducer):
    """t-SNE (t-Distributed Stochastic Neighbor Embedding) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return TSNE(
            n_components=self.config.n_components,
            perplexity=self.config.perplexity,
            learning_rate=self.config.learning_rate,
            metric=self.config.metric,
            random_state=self.config.random_state,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "perplexity": self.config.perplexity,
            "learning_rate": self.config.learning_rate,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
        }


class UMAPReducer(DimensionReducer):
    """UMAP (Uniform Manifold Approximation and Projection) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from umap import UMAP
        # testing
        #print(f"Doing UMAP.")
        return UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
        ).fit_transform(data)



    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "min_dist": self.config.min_dist,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
        }


class DensMAPReducer(DimensionReducer):
    """densMAP — density-preserving UMAP reduction.

    Built into umap-learn >= 0.5.0.  Uses the same parameters as UMAP
    but sets densmap=True to preserve local density information.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from umap import UMAP

        return UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
            densmap=True,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "min_dist": self.config.min_dist,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
            "densmap": True,
        }


class TrimapReducer(DimensionReducer):
    """TriMAP — triplet-based manifold reduction.

    Uses TriMAP's own defaults for all behavioural parameters (n_inliers=12,
    n_outliers=4, lr=0.1, n_iters=400).  Only n_components and metric are
    shared with the general config — everything else respects TriMAP defaults
    unless explicitly overridden via the CLI.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        import trimap

        np.random.seed(self.config.random_state)
        return trimap.TRIMAP(
            n_dims=self.config.n_components,
            n_inliers=self.config.trimap_n_inliers,
            n_outliers=self.config.trimap_n_outliers,
            distance=self.config.metric,
            lr=self.config.trimap_lr,
            n_iters=self.config.trimap_n_iters,
            apply_pca=self.config.trimap_apply_pca,
            n_random=3,
            weight_temp=0.5,
            opt_method="dbd",
            verbose=False,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_inliers": self.config.trimap_n_inliers,
            "n_outliers": self.config.trimap_n_outliers,
            "metric": self.config.metric,
            "lr": self.config.trimap_lr,
            "n_iters": self.config.trimap_n_iters,
            "apply_pca": self.config.trimap_apply_pca,
            "random_state": self.config.random_state,
        }


class PhateReducer(DimensionReducer):
    """PHATE — Potential of Heat-diffusion for Affinity-based Transition Embedding.

    Uses PHATE's own defaults for all behavioural parameters (knn=5,
    decay=40, t="auto", gamma=1.0).  Only n_components and metric are
    shared with the general config.

    Note: PHATE subsamples to n_landmark points by default (2000).
    For datasets larger than this, increase phate_n_landmark or the
    output will represent only a subset of your proteins.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        import phate

        # Resolve t: "auto" stays as "auto", numeric strings become int
        t_val = self.config.phate_t
        if t_val != "auto":
            try:
                t_val = int(t_val)
            except ValueError:
                pass  # keep as string (e.g. malformed, let PHATE error)

        return phate.PHATE(
            n_components=self.config.n_components,
            knn=self.config.phate_knn,
            decay=self.config.phate_decay,
            n_landmark=self.config.phate_n_landmark,
            t=t_val,
            gamma=self.config.phate_gamma,
            n_pca=self.config.phate_n_pca,
            knn_dist=self.config.metric,
            n_jobs=1,
            random_state=self.config.random_state,
            verbose=False,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "knn": self.config.phate_knn,
            "decay": self.config.phate_decay,
            "n_landmark": self.config.phate_n_landmark,
            "t": self.config.phate_t,
            "gamma": self.config.phate_gamma,
            "n_pca": self.config.phate_n_pca,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
        }


class PaCMAPReducer(DimensionReducer):
    """PaCMAP (Pairwise Controlled Manifold Approximation) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from pacmap import PaCMAP

        _ensure_annoy_or_fallback()
        return PaCMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            MN_ratio=self.config.mn_ratio,
            FP_ratio=self.config.fp_ratio,
            random_state=self.config.random_state,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "MN_ratio": self.config.mn_ratio,
            "FP_ratio": self.config.fp_ratio,
            "random_state": self.config.random_state,
        }


class LocalMAPReducer(DimensionReducer):
    """LocalMAP (Local Manifold Approximation) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from pacmap import LocalMAP

        _ensure_annoy_or_fallback()
        return LocalMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            MN_ratio=self.config.mn_ratio,
            FP_ratio=self.config.fp_ratio,
            random_state=self.config.random_state,
        ).fit_transform(data, init="pca")

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "MN_ratio": self.config.mn_ratio,
            "FP_ratio": self.config.fp_ratio,
            "random_state": self.config.random_state,
        }


class MDSReducer(DimensionReducer):
    """Multidimensional Scaling reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return MDS(
            n_components=self.config.n_components,
            metric=True,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            eps=self.config.eps,
            random_state=self.config.random_state,
            dissimilarity=("precomputed" if self.config.precomputed else "euclidean"),
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_init": self.config.n_init,
            "max_iter": self.config.max_iter,
            "eps": self.config.eps,
            "random_state": self.config.random_state,
        }
