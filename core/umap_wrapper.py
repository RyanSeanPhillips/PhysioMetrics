"""
UMAP wrapper with graceful fallback for PhysioMetrics.

Provides UMAP dimensionality reduction when umap-learn is installed,
with automatic fallback to PCA if not available.
"""

from typing import Tuple, Optional
import numpy as np

# Track availability
UMAP_AVAILABLE = False
_UMAP_CLASS = None

try:
    from umap import UMAP as _UMAPClass
    UMAP_AVAILABLE = True
    _UMAP_CLASS = _UMAPClass
except ImportError:
    pass


def get_umap_availability() -> Tuple[bool, str]:
    """
    Check if UMAP is available.

    Returns:
        Tuple of (is_available: bool, message: str)
    """
    if UMAP_AVAILABLE:
        return True, "UMAP available (umap-learn installed)"
    else:
        return False, "umap-learn not installed. Using PCA fallback. Install with: pip install umap-learn"


def compute_embedding(
    feature_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[np.ndarray, str]:
    """
    Compute dimensionality reduction embedding.

    Uses UMAP if available, otherwise falls back to PCA.

    Args:
        feature_matrix: Scaled feature matrix, shape (N, F)
        n_components: Output dimensions (2 or 3)
        n_neighbors: UMAP n_neighbors parameter (ignored for PCA)
        min_dist: UMAP min_dist parameter (ignored for PCA)
        metric: Distance metric for UMAP (ignored for PCA)
        random_state: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Tuple of:
        - embedding: np.ndarray of shape (N, n_components)
        - method: 'umap' or 'pca'

    Raises:
        ValueError: If feature_matrix has invalid shape or contains NaN/Inf
    """
    # Validate input
    if feature_matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {feature_matrix.shape}")

    n_samples, n_features = feature_matrix.shape

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for embedding, got {n_samples}")

    if not np.all(np.isfinite(feature_matrix)):
        # Count and report NaN/Inf
        n_nan = np.sum(np.isnan(feature_matrix))
        n_inf = np.sum(np.isinf(feature_matrix))
        raise ValueError(f"feature_matrix contains {n_nan} NaN and {n_inf} Inf values")

    if UMAP_AVAILABLE:
        # Adjust n_neighbors if we have fewer samples
        effective_neighbors = min(n_neighbors, n_samples - 1)
        if effective_neighbors < 2:
            effective_neighbors = 2

        if verbose:
            print(f"[UMAP] Computing embedding: {n_samples} samples, {n_features} features")
            print(f"[UMAP] Parameters: n_components={n_components}, n_neighbors={effective_neighbors}, min_dist={min_dist}")

        reducer = _UMAP_CLASS(
            n_components=n_components,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=verbose
        )
        embedding = reducer.fit_transform(feature_matrix)
        return embedding, 'umap'
    else:
        # Fallback to PCA
        from sklearn.decomposition import PCA

        # PCA can't have more components than min(n_samples, n_features)
        max_components = min(n_samples, n_features)
        effective_components = min(n_components, max_components)

        if verbose:
            print(f"[PCA] Computing embedding: {n_samples} samples, {n_features} features")
            print(f"[PCA] n_components={effective_components}")

        pca = PCA(n_components=effective_components, random_state=random_state)
        embedding = pca.fit_transform(feature_matrix)

        # If we got fewer components than requested, pad with zeros
        if effective_components < n_components:
            padding = np.zeros((n_samples, n_components - effective_components))
            embedding = np.hstack([embedding, padding])

        return embedding, 'pca'


def compute_embedding_async(
    feature_matrix: np.ndarray,
    callback: callable,
    error_callback: callable = None,
    **kwargs
) -> None:
    """
    Compute embedding asynchronously using QThread.

    Args:
        feature_matrix: Scaled feature matrix
        callback: Called with (embedding, method) when complete
        error_callback: Called with exception if error occurs
        **kwargs: Passed to compute_embedding()

    Note:
        This function returns immediately. Results are delivered via callbacks.
        Must be called from a Qt application context.
    """
    from PyQt6.QtCore import QThread, pyqtSignal, QObject

    class EmbeddingWorker(QObject):
        finished = pyqtSignal(object, str)
        error = pyqtSignal(Exception)

        def __init__(self, matrix, params):
            super().__init__()
            self.matrix = matrix
            self.params = params

        def run(self):
            try:
                embedding, method = compute_embedding(self.matrix, **self.params)
                self.finished.emit(embedding, method)
            except Exception as e:
                self.error.emit(e)

    # Create worker and thread
    thread = QThread()
    worker = EmbeddingWorker(feature_matrix, kwargs)
    worker.moveToThread(thread)

    # Connect signals
    thread.started.connect(worker.run)
    worker.finished.connect(lambda emb, meth: callback(emb, meth))
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    if error_callback:
        worker.error.connect(error_callback)
    worker.error.connect(thread.quit)
    worker.error.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    # Start
    thread.start()

    # Return thread reference so caller can track it
    return thread
