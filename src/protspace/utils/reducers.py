import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, get_type_hints

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

import anndata as ad
import pandas as pd
import h5py

import sys
from pathlib import Path
from rhopca.methods import rhoPCA

# Re-export constants and config from lightweight module
from protspace.utils.constants import (  # noqa: F401
    LOCALMAP_NAME,
    MDS_NAME,
    METRIC_TYPES,
    PACMAP_NAME,
    PCA_NAME,
    REDUCER_METHODS,
    TSNE_NAME,
    UMAP_NAME,
    RHOPCA_NAME,
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
        from umap import UMAP
        from rhopca.methods import rhoPCA

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
        k = self.config.n_components

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
            return result
        except Exception as e:
            logger.error(f"PCA failed using '{solver}' solver: {e}")
            raise

    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the reduction."""
        params = {
            "n_components": self.config.n_components,
            # Report the solver used, default to 'arpack' if not set yet
            "svd_solver": getattr(self, "used_solver", "arpack"),
        }
        if hasattr(self, "explained_variance"):
            params["explained_variance_ratio"] = self.explained_variance
        return params

#class rhoPCAReducer(DimensionReducer):
#    """rhoPCA - contrastive dimentionality reduction method."""
#    
#    def fit_transform(self, target_data: np.ndarray, background_data: np.ndarray) -> np.ndarray:
#        # combine both datasets
#        X = np.vstack([target_data, background_data])
#        # labels for contrast groups
#        labels = (
#            ["target"] * len(target_data)
#            + ["background"] * len(background_data)
#        )
#        # create AnnData
#        adata = ad.AnnData(X)
#        adata.obs["group"] = pd.Categorical(labels)
#        # instantiate model
#        model = rhoPCA(
#            adata,
#            contrast_column="group",
#            target="target",
#            background="background",
#            scale_variance=True
#        )
#        model.fit()
#        # transform all data
#        embedding = model.transform()
#        return embedding
#    
#    def get_params(self) -> dict[str, Any]:
#        return {
#            "n_components": self.config.n_components,
#            "scale_variance": self.config.scale_variance
#        }

class rhoPCAReducer(DimensionReducer):
    """rhoPCA - contrastive dimensionality reduction method.
    
    This reducer handles parsing background data explicitly from the CLI,
    formats matrices into AnnData structures, runs the contrastive PCA model,
    and extracts background-aware projections directly from the fitted model attributes.
    """

    def __init__(self, config: DimensionReductionConfig):
        super().__init__(config)

    def fit_transform(self, data: np.ndarray, background_data: np.ndarray = None) -> np.ndarray:
        # --- 1. Parse CLI Arguments for Background Matrix ---
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
                
                print(f"--> rhoPCA explicitly reading background matrix from CLI: {bg_path}")
                with h5py.File(bg_path, "r") as f:
                    first_key = list(f.keys())[0]
                    if f[first_key].ndim == 1:
                        background_data = np.vstack([f[key][:] for key in f.keys()])
                    else:
                        background_data = np.array(f[first_key])

        # Final validation check
        if background_data is None:
            raise ValueError(
                "rhoPCA requires background data. The pipeline failed to parse "
                "the '--background' flag path directly from your terminal input."
            )

        # --- 2. Data Assembly & Labeling ---
        target_data = data
        X = np.vstack([target_data, background_data])
        
        labels = (
            ["target"] * len(target_data)
            + ["background"] * len(background_data)
        )
        
        # Instantiate AnnData structure required by the rhoPCA package
        adata = ad.AnnData(X)
        adata.obs["group"] = pd.Categorical(labels)
        
        # Safely extract configuration variables
        scale_var = getattr(self.config, "scale_variance", True)
        dims = getattr(self.config, "n_components", 2)
        
        # --- 3. Instantiate and Fit Model ---
        model = rhoPCA(
            adata,
            contrast_column="group",
            target="target",
            background="background",
            scale_variance=scale_var
        )
        model.fit()
        
        # --- 4. Coordinate Retrieval & Eigenvector Projection ---
        full_embeddings = None
        
        # Strategy A: Check pre-computed target projection matrix first
        if hasattr(model, 'target_proj') and model.target_proj is not None:
            # target_proj contains only the target slice coordinates, which is perfect
            full_embeddings = np.array(model.target_proj)
            # Restrict columns to requested dimensions if it computed more than needed
            full_embeddings = full_embeddings[:, :dims]
            
        # Strategy B: Fallback to manual projection using model's loadings matrix
        if full_embeddings is None and hasattr(model, 'loadings'):
            loadings = model.loadings
            if loadings is not None:
                v_slice = loadings[:, :dims]
                # Matrix multiply the combined matrix X, then slice target lines
                full_embeddings = np.dot(X, v_slice)[:len(target_data)]
                
        # Strategy C: Fail safely if everything is missing
        if full_embeddings is None:
            raise KeyError(
                f"Could not extract target projections or loadings from rhoPCA.\n"
                f"Available attributes: {[a for a in dir(model) if not a.startswith('__')]}"
            )
            
        # --- 5. Return Array ---
        # Because Strategy A already targets the exact subset, full_embeddings is 
        # already sized to match target_data length perfectly.
        return full_embeddings
    
    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": getattr(self.config, "n_components", 2),
            "scale_variance": getattr(self.config, "scale_variance", True)
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
