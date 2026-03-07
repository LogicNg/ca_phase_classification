"""
Plotting utilities for the CA Phase Classification project.

All figures are saved to the configured figure_dir. Each function returns
the matplotlib Figure so callers can customise further if desired.
"""

import os
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ── 1. Density time-series ────────────────────────────────────────────────


def plot_density_traces(
    results: List[Dict],
    densities_to_show: Optional[List[float]] = None,
    max_traces: int = 10,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Plot population-density vs time for a few representative initial densities.
    """
    ensure_dir(fig_dir)
    if densities_to_show is None:
        all_d = sorted({r["density_init"] for r in results})
        densities_to_show = all_d[:: max(1, len(all_d) // 5)]  # ~5 panels

    fig, axes = plt.subplots(
        1,
        len(densities_to_show),
        figsize=(4 * len(densities_to_show), 3.5),
        sharey=True,
    )
    if len(densities_to_show) == 1:
        axes = [axes]

    for ax, d in zip(axes, densities_to_show):
        subset = [r for r in results if r["density_init"] == d][:max_traces]
        for r in subset:
            ax.plot(r["sim"]["density"], alpha=0.5, linewidth=0.7)
        ax.set_title(f"$\\rho_0 = {d:.2f}$")
        ax.set_xlabel("time step")
    axes[0].set_ylabel("density $\\rho(t)$")
    fig.suptitle("Density time-series", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "density_traces.png"), dpi=150, bbox_inches="tight"
    )
    return fig


# ── 2. Steady-state density vs initial density ───────────────────────────


def plot_rho_vs_rho0(
    feature_matrix: np.ndarray, feature_names: List[str], fig_dir: str = "figures"
) -> plt.Figure:
    """Scatter plot: final steady-state density vs initial density."""
    ensure_dir(fig_dir)
    i_rho0 = feature_names.index("density_init")
    i_rho = feature_names.index("rho_final_mean")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(feature_matrix[:, i_rho0], feature_matrix[:, i_rho], s=6, alpha=0.4)
    ax.set_xlabel("initial density $\\rho_0$")
    ax.set_ylabel("steady-state density $\\langle\\rho\\rangle$")
    ax.set_title("Phase diagram: $\\rho_{\\mathrm{ss}}$ vs $\\rho_0$")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rho_vs_rho0.png"), dpi=150, bbox_inches="tight")
    return fig


# ── 3. UMAP embedding ────────────────────────────────────────────────────


def plot_umap_embedding(
    embedding: np.ndarray,
    color_by: np.ndarray,
    color_label: str = "initial density",
    labels: Optional[np.ndarray] = None,
    fig_dir: str = "figures",
    filename: str = "umap_embedding.png",
) -> plt.Figure:
    """
    2-D UMAP scatter coloured by a continuous variable or cluster label.
    """
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=color_by, cmap="viridis", s=8, alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label=color_label)

    if labels is not None:
        unique = sorted(set(labels))
        for lab in unique:
            mask = labels == lab
            cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
            ax.annotate(
                str(lab),
                (cx, cy),
                fontsize=12,
                fontweight="bold",
                ha="center",
                color="red",
            )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP embedding")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), dpi=150, bbox_inches="tight")
    return fig


# ── 4. Cluster-coloured UMAP ─────────────────────────────────────────────


def plot_umap_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    method_name: str = "HDBSCAN",
    fig_dir: str = "figures",
) -> plt.Figure:
    """UMAP scatter coloured by cluster assignment."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        colour = "lightgrey" if lab == -1 else cmap(i)
        name = "noise" if lab == -1 else f"cluster {lab}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colour],
            s=8,
            alpha=0.6,
            label=name,
        )

    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Clusters ({method_name})")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, f"umap_clusters_{method_name.lower()}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    return fig


# ── 5. Silhouette score vs k ─────────────────────────────────────────────


def plot_silhouette(kmeans_results: dict, fig_dir: str = "figures") -> plt.Figure:
    """Plot silhouette score vs number of clusters k."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(
        kmeans_results["k_values"],
        kmeans_results["silhouette_scores"],
        "o-",
        markersize=5,
    )
    best = kmeans_results["best_k"]
    ax.axvline(best, color="red", linestyle="--", label=f"best k = {best}")
    ax.set_xlabel("number of clusters $k$")
    ax.set_ylabel("silhouette score")
    ax.set_title("k-means: optimal cluster count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "silhouette.png"), dpi=150, bbox_inches="tight")
    return fig


# ── 6. Example grid snapshots ────────────────────────────────────────────


def plot_grid_snapshots(
    results: List[Dict],
    densities: Optional[List[float]] = None,
    fig_dir: str = "figures",
) -> plt.Figure:
    """Show final grid states for a few initial densities."""
    ensure_dir(fig_dir)
    if densities is None:
        all_d = sorted({r["density_init"] for r in results})
        densities = all_d[:: max(1, len(all_d) // 6)]

    fig, axes = plt.subplots(1, len(densities), figsize=(3 * len(densities), 3))
    if len(densities) == 1:
        axes = [axes]

    for ax, d in zip(axes, densities):
        sample = next((r for r in results if r["density_init"] == d), None)
        if sample is None:
            continue
        ax.imshow(sample["sim"]["final"], cmap="binary", interpolation="nearest")
        ax.set_title(f"$\\rho_0 = {d:.2f}$", fontsize=10)
        ax.axis("off")

    fig.suptitle("Final configurations", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "grid_snapshots.png"), dpi=150, bbox_inches="tight"
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Cluster interpretation plots
# ══════════════════════════════════════════════════════════════════════════


def plot_feature_importance(
    importance_df,
    top_n: int = 10,
    fig_dir: str = "figures",
) -> plt.Figure:
    """Horizontal bar chart of top-N features by η²."""
    ensure_dir(fig_dir)
    df = importance_df.head(top_n).iloc[::-1]  # reverse for bottom-to-top

    fig, ax = plt.subplots(figsize=(6, 0.4 * top_n + 1))
    ax.barh(df["feature"], df["eta_squared"], color="steelblue")
    ax.set_xlabel("$\\eta^2$ (inter-cluster variance fraction)")
    ax.set_title("Feature importance for cluster separation")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    return fig


def plot_cluster_feature_distributions(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    features_to_show: Optional[List[str]] = None,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Violin / box plots showing how key features distribute across clusters.
    """
    ensure_dir(fig_dir)
    if features_to_show is None:
        features_to_show = [
            "rho_final_mean",
            "n_clusters",
            "settling_time",
            "fourier_k_peak",
            "acf_r1",
        ]
    features_to_show = [f for f in features_to_show if f in feature_names]
    n_feats = len(features_to_show)

    unique_labels = sorted([l for l in set(labels) if l != -1])
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(1, n_feats, figsize=(3.5 * n_feats, 4))
    if n_feats == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab10", n_clusters)

    for ax, fname in zip(axes, features_to_show):
        fi = feature_names.index(fname)
        data = []
        positions = []
        colors = []
        for j, lab in enumerate(unique_labels):
            mask = labels == lab
            data.append(feature_matrix[mask, fi])
            positions.append(j)
            colors.append(cmap(j))

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"C{l}" for l in unique_labels])
        ax.set_title(fname, fontsize=9)
        ax.set_xlabel("cluster")

    fig.suptitle("Feature distributions by cluster", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "cluster_feature_distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    return fig


def plot_representative_grids(
    results: List[Dict],
    embedding: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict],
    n_per_cluster: int = 3,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Show representative final grids (closest to centroid) for each cluster.
    """
    ensure_dir(fig_dir)
    from common.interpretation import representative_samples

    reps = representative_samples(embedding, labels, n_per_cluster)
    n_clusters = len(reps)

    fig, axes = plt.subplots(
        n_clusters,
        n_per_cluster,
        figsize=(3 * n_per_cluster, 3 * n_clusters),
    )
    if n_clusters == 1:
        axes = [axes]

    for row, (lab, indices) in enumerate(sorted(reps.items())):
        for col in range(n_per_cluster):
            ax = axes[row][col] if n_clusters > 1 else axes[col]
            if col < len(indices):
                idx = indices[col]
                grid = results[idx]["sim"]["final"]
                rho0 = metadata[idx]["density_init"]
                ax.imshow(grid, cmap="binary", interpolation="nearest")
                ax.set_title(f"C{lab}, $\\rho_0$={rho0:.2f}", fontsize=8)
            ax.axis("off")

    fig.suptitle("Representative configurations per cluster", y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "representative_grids.png"),
        dpi=150,
        bbox_inches="tight",
    )
    return fig


def plot_phase_diagram_heatmap(
    transition_matrix,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Heatmap showing cluster membership fraction vs initial density.
    This is the 'phase diagram'.
    """
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    data = transition_matrix.values
    densities = transition_matrix.index.values
    clusters = transition_matrix.columns.values

    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap="YlOrRd",
        extent=[densities[0], densities[-1], -0.5, len(clusters) - 0.5],
    )
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"cluster {c}" for c in clusters])
    ax.set_xlabel("initial density $\\rho_0$")
    ax.set_title("Phase diagram: cluster fraction vs $\\rho_0$")
    plt.colorbar(im, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "phase_diagram_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Perturbation response plots
# ══════════════════════════════════════════════════════════════════════════


def plot_hamming_traces(
    pert_results: List[Dict],
    densities_to_show: Optional[List[float]] = None,
    max_traces: int = 15,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Plot Hamming distance d(t) for perturbation responses, grouped by ρ₀.
    """
    ensure_dir(fig_dir)
    all_d = sorted({r["density_init"] for r in pert_results})
    if densities_to_show is None:
        densities_to_show = all_d[:: max(1, len(all_d) // 5)]

    fig, axes = plt.subplots(
        1,
        len(densities_to_show),
        figsize=(4 * len(densities_to_show), 3.5),
        sharey=True,
    )
    if len(densities_to_show) == 1:
        axes = [axes]

    for ax, d in zip(axes, densities_to_show):
        subset = [r for r in pert_results if r["density_init"] == d][:max_traces]
        for r in subset:
            ax.plot(r["response"]["hamming"], alpha=0.4, linewidth=0.7)
        ax.set_title(f"$\\rho_0 = {d:.2f}$")
        ax.set_xlabel("time after perturbation")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)

    axes[0].set_ylabel("Hamming distance $d(t)$")
    fig.suptitle("Perturbation response: damage spreading", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "hamming_traces.png"), dpi=150, bbox_inches="tight"
    )
    return fig


def plot_susceptibility(
    pert_feature_names: List[str],
    pert_feature_matrix: np.ndarray,
    pert_densities: np.ndarray,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Plot perturbation 'susceptibility' measures vs initial density:
    peak Hamming, final Hamming, damage radius.
    """
    ensure_dir(fig_dir)
    measures = ["hamming_max", "hamming_final", "damage_radius_max"]
    measures = [m for m in measures if m in pert_feature_names]

    fig, axes = plt.subplots(1, len(measures), figsize=(4.5 * len(measures), 4))
    if len(measures) == 1:
        axes = [axes]

    for ax, mname in zip(axes, measures):
        idx = pert_feature_names.index(mname)
        # Compute mean ± std per density
        unique_d = sorted(set(pert_densities))
        means, stds = [], []
        for d in unique_d:
            mask = pert_densities == d
            vals = pert_feature_matrix[mask, idx]
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(
            unique_d, means, yerr=stds, fmt="o-", markersize=4, capsize=3, alpha=0.8
        )
        ax.set_xlabel("initial density $\\rho_0$")
        ax.set_ylabel(mname.replace("_", " "))
        ax.set_title(mname.replace("_", " ").title())

    fig.suptitle("Perturbation susceptibility vs $\\rho_0$", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "susceptibility.png"), dpi=150, bbox_inches="tight"
    )
    return fig


def plot_response_type_diagram(
    pert_feature_names: List[str],
    pert_feature_matrix: np.ndarray,
    pert_densities: np.ndarray,
    fig_dir: str = "figures",
) -> plt.Figure:
    """
    Stacked bar: fraction of responses that are healed / partially recovered /
    sustained, as a function of initial density.
    """
    ensure_dir(fig_dir)
    rt_idx = pert_feature_names.index("response_type")
    unique_d = sorted(set(pert_densities))

    fractions = {0: [], 1: [], 2: []}
    for d in unique_d:
        mask = pert_densities == d
        types = pert_feature_matrix[mask, rt_idx]
        n = len(types)
        for t in [0, 1, 2]:
            fractions[t].append((types == t).sum() / n if n > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(unique_d))
    labels_map = {0: "healed", 1: "partial recovery", 2: "sustained damage"}
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    for t in [0, 1, 2]:
        vals = np.array(fractions[t])
        ax.bar(
            unique_d,
            vals,
            bottom=bottom,
            width=0.04,
            label=labels_map[t],
            color=colors[t],
            alpha=0.8,
        )
        bottom += vals

    ax.set_xlabel("initial density $\\rho_0$")
    ax.set_ylabel("fraction")
    ax.set_title("Response type classification vs $\\rho_0$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "response_type_diagram.png"),
        dpi=150,
        bbox_inches="tight",
    )
    return fig


def plot_damage_maps(
    pert_results: List[Dict],
    densities: Optional[List[float]] = None,
    fig_dir: str = "figures",
) -> plt.Figure:
    """Show cumulative damage maps for representative densities."""
    ensure_dir(fig_dir)
    all_d = sorted({r["density_init"] for r in pert_results})
    if densities is None:
        densities = all_d[:: max(1, len(all_d) // 6)]

    fig, axes = plt.subplots(1, len(densities), figsize=(3 * len(densities), 3))
    if len(densities) == 1:
        axes = [axes]

    for ax, d in zip(axes, densities):
        sample = next((r for r in pert_results if r["density_init"] == d), None)
        if sample is None:
            continue
        dmap = sample["response"]["damage_map"]
        ax.imshow(dmap, cmap="hot", interpolation="nearest")
        ax.set_title(f"$\\rho_0 = {d:.2f}$", fontsize=10)
        ax.axis("off")

    fig.suptitle("Cumulative damage maps", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "damage_maps.png"), dpi=150, bbox_inches="tight")
    return fig
