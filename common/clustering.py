"""
Unsupervised learning: dimensionality reduction + clustering.

Pipeline:
  1. Identify extinct configurations as a separate phase (label = -2).
  2. Standardise features (z-score) on the alive subset.
  3. UMAP to 2-D (or n-D) embedding.
  4. HDBSCAN to find clusters in the embedding.
  5. Optionally compare with k-means + silhouette analysis.
  6. Merge labels back: extinct = special label, alive = cluster labels.
"""

import hdbscan
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common.config import ClusteringConfig

# Label used for configurations that went fully extinct.
EXTINCT_LABEL = -2


def build_pipeline(
    feature_matrix: np.ndarray,
    cfg: ClusteringConfig,
    feature_names: list = None,
    exclude_features: list = None,
):
    """
    Run the full unsupervised-learning pipeline.

    Extinct grids (rho_final_mean == 0) are separated out first and
    assigned ``EXTINCT_LABEL``.  UMAP + clustering operates only on the
    alive samples so that trivial settling-time variation among dead grids
    can no longer dominate the clustering.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_samples, n_features)
    cfg : ClusteringConfig
    feature_names : list of str, optional
    exclude_features : list of str, optional

    Returns
    -------
    dict with keys:
        "scaler"        : fitted StandardScaler (on alive subset)
        "scaled"        : standardised feature matrix (alive only)
        "umap_model"    : fitted UMAP reducer
        "embedding"     : np.ndarray, shape (n_samples, 2) — extinct samples at origin
        "hdbscan_model" : fitted HDBSCAN clusterer
        "labels_hdbscan": np.ndarray of cluster labels (EXTINCT_LABEL for dead)
        "kmeans_results": dict
        "used_columns"  : np.ndarray
        "alive_mask"    : np.ndarray[bool]
        "n_extinct"     : int
    """
    # ── 0. Separate extinct from alive ────────────────────────────────
    if feature_names and "rho_final_mean" in feature_names:
        rho_idx = feature_names.index("rho_final_mean")
        alive_mask = feature_matrix[:, rho_idx] > 0
    else:
        alive_mask = np.ones(feature_matrix.shape[0], dtype=bool)

    n_extinct = int((~alive_mask).sum())
    alive_features = feature_matrix[alive_mask]

    # Optionally exclude control-parameter columns
    if exclude_features and feature_names:
        keep = [i for i, n in enumerate(feature_names) if n not in exclude_features]
    else:
        keep = list(range(feature_matrix.shape[1]))
    X_alive = alive_features[:, keep]

    # ── 1. Standardise (alive only) ──────────────────────────────────
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_alive)

    # ── 2. UMAP ──────────────────────────────────────────────────────
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=min(cfg.umap_n_neighbors, max(2, X_alive.shape[0] - 1)),
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=42,
    )
    embedding_alive = reducer.fit_transform(scaled)

    # ── 3. HDBSCAN on alive embedding ────────────────────────────────
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        cluster_selection_method=cfg.hdbscan_selection_method,
    )
    labels_alive = clusterer.fit_predict(embedding_alive)

    # ── 4. k-means sweep on alive embedding ──────────────────────────
    kmeans_results_alive = _kmeans_sweep(embedding_alive, cfg.kmeans_k_range)

    # ── 5. Build full-sample arrays ──────────────────────────────────
    n_total = feature_matrix.shape[0]

    # Full embedding: extinct samples placed at the embedding origin
    embedding_full = np.zeros((n_total, cfg.umap_n_components))
    embedding_full[alive_mask] = embedding_alive
    # Shift extinct points to a clearly separated region
    if n_extinct > 0 and embedding_alive.shape[0] > 0:
        x_min = embedding_alive[:, 0].min()
        embedding_full[~alive_mask, 0] = x_min - 3.0
        embedding_full[~alive_mask, 1] = embedding_alive[:, 1].mean()

    # Full labels
    labels_full = np.full(n_total, EXTINCT_LABEL, dtype=int)
    labels_full[alive_mask] = labels_alive

    # Full k-means labels
    km_labels_full = np.full(n_total, EXTINCT_LABEL, dtype=int)
    km_labels_full[alive_mask] = kmeans_results_alive["best_labels"]
    kmeans_results_full = dict(kmeans_results_alive)
    kmeans_results_full["best_labels"] = km_labels_full

    return {
        "scaler": scaler,
        "scaled": scaled,
        "umap_model": reducer,
        "embedding": embedding_full,
        "hdbscan_model": clusterer,
        "labels_hdbscan": labels_full,
        "kmeans_results": kmeans_results_full,
        "used_columns": np.array(keep),
        "alive_mask": alive_mask,
        "n_extinct": n_extinct,
    }


def _kmeans_sweep(embedding: np.ndarray, k_range: tuple) -> dict:
    """
    Run k-means for k in k_range and compute silhouette scores.

    Returns dict with:
        "k_values"          : list of int
        "silhouette_scores" : list of float
        "best_k"            : int
        "best_labels"       : np.ndarray
    """
    k_lo, k_hi = k_range
    k_values = list(range(k_lo, k_hi + 1))
    silhouette_scores = []
    all_labels = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(embedding)
        if len(set(lbl)) > 1:
            score = silhouette_score(embedding, lbl)
        else:
            score = -1.0
        silhouette_scores.append(score)
        all_labels.append(lbl)

    best_idx = int(np.argmax(silhouette_scores))
    return {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "best_k": k_values[best_idx],
        "best_labels": all_labels[best_idx],
    }
