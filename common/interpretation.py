"""
Cluster interpretation module.

Given clustering results and the feature matrix, this module provides
tools to understand *what* the discovered clusters correspond to
physically:

  * Per-cluster feature statistics (mean ± std of every feature).
  * Feature importance via inter-cluster variance / total variance
    (a simple η² measure, analogous to ANOVA).
  * Representative sample selection (closest to each cluster centroid
    in UMAP space).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def cluster_feature_stats(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-cluster mean and std for every feature.

    Returns a DataFrame with MultiIndex columns: (cluster_id, 'mean'/'std').
    Rows are feature names.
    """
    unique = sorted(set(labels))
    records = {}

    for lab in unique:
        mask = labels == lab
        name = f"cluster_{lab}" if lab != -1 else "noise"
        subset = feature_matrix[mask]
        records[(name, "mean")] = subset.mean(axis=0)
        records[(name, "std")] = subset.std(axis=0)
        records[(name, "count")] = np.full(len(feature_names), mask.sum())

    df = pd.DataFrame(records, index=feature_names)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def feature_importance(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Rank features by how much they vary *between* clusters relative to
    total variance (η² = SS_between / SS_total).

    Returns a DataFrame sorted by η² descending.
    """
    unique = sorted([l for l in set(labels) if l != -1])
    grand_mean = feature_matrix.mean(axis=0)
    ss_total = ((feature_matrix - grand_mean) ** 2).sum(axis=0)

    ss_between = np.zeros(feature_matrix.shape[1])
    for lab in unique:
        mask = labels == lab
        n_k = mask.sum()
        cluster_mean = feature_matrix[mask].mean(axis=0)
        ss_between += n_k * (cluster_mean - grand_mean) ** 2

    eta_sq = np.where(ss_total > 1e-10, ss_between / ss_total, 0.0)
    eta_sq = np.clip(eta_sq, 0.0, 1.0)  # guard against floating-point artifacts

    df = pd.DataFrame({"feature": feature_names, "eta_squared": eta_sq}).sort_values(
        "eta_squared", ascending=False
    )
    df = df.reset_index(drop=True)
    return df


def representative_samples(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int = 3,
) -> Dict[int, np.ndarray]:
    """
    For each cluster, find the `n_per_cluster` samples closest to the
    cluster centroid in the UMAP embedding.

    Returns dict: cluster_label -> array of sample indices.
    """
    reps = {}
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        mask = labels == lab
        indices = np.where(mask)[0]
        centroid = embedding[mask].mean(axis=0)
        dists = np.linalg.norm(embedding[indices] - centroid, axis=1)
        order = np.argsort(dists)[:n_per_cluster]
        reps[lab] = indices[order]
    return reps


def cluster_transition_matrix(
    labels: np.ndarray,
    density_init: np.ndarray,
) -> pd.DataFrame:
    """
    Cross-tabulation of cluster label vs initial density.
    Shows how cluster membership shifts as ρ₀ changes — the "phase diagram."
    """
    df = pd.DataFrame({"density_init": density_init, "cluster": labels})
    ct = pd.crosstab(df["density_init"], df["cluster"], normalize="index")
    return ct


def print_interpretation_summary(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    density_init: np.ndarray,
    method_name: str = "HDBSCAN",
):
    """Print a human-readable interpretation of the clusters."""
    unique = sorted([l for l in set(labels) if l != -1])
    n_clusters = len(unique)

    print(f"\n{'─'*60}")
    print(f"  Cluster Interpretation ({method_name}, {n_clusters} clusters)")
    print(f"{'─'*60}")

    # Feature importance
    imp = feature_importance(feature_matrix, feature_names, labels)
    print(f"\n  Top-5 discriminating features (η²):")
    for _, row in imp.head(5).iterrows():
        print(f"    {row['feature']:25s}  η² = {row['eta_squared']:.3f}")

    # Per-cluster summary
    key_features = [
        "density_init",
        "rho_final_mean",
        "n_clusters",
        "settled",
        "settling_time",
        "fourier_k_peak",
    ]
    key_features = [f for f in key_features if f in feature_names]
    key_idx = [feature_names.index(f) for f in key_features]

    for lab in unique:
        mask = labels == lab
        n = mask.sum()
        rho0_range = (density_init[mask].min(), density_init[mask].max())
        print(
            f"\n  Cluster {lab}  (n={n}, ρ₀ ∈ [{rho0_range[0]:.2f}, {rho0_range[1]:.2f}]):"
        )
        for fi, fname in zip(key_idx, key_features):
            vals = feature_matrix[mask, fi]
            print(f"    {fname:25s}  {vals.mean():.4f} ± {vals.std():.4f}")

    print(f"\n{'─'*60}\n")
