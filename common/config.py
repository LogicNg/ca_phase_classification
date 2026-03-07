"""
Configuration for the Cellular Automata Phase Classification project.

This file centralises all tuneable parameters so experiments are reproducible
and easy to sweep over.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimulationConfig:
    """Parameters controlling the Game-of-Life simulation."""

    grid_size: int = 100  # L x L grid
    n_steps: int = 500  # number of time-steps per run
    boundary: str = "wrap"  # "wrap" (periodic) or "fill" (dead boundary)

    # Initial-condition sweep: densities to sample
    initial_densities: List[float] = field(
        default_factory=lambda: [round(0.05 * i, 2) for i in range(1, 20)]
    )
    n_samples_per_density: int = 50  # independent random ICs per density

    # Rules  (birth / survival neighbour counts)
    # Standard Conway: birth=[3], survive=[2,3]
    birth: List[int] = field(default_factory=lambda: [3])
    survive: List[int] = field(default_factory=lambda: [2, 3])

    seed: int = 42  # master RNG seed for reproducibility


@dataclass
class FeatureConfig:
    """Parameters for feature extraction from simulation trajectories."""

    # How many final time-steps to average over for "steady-state" features
    steady_state_window: int = 50

    # Spatial autocorrelation: max lag (in cells)
    max_correlation_lag: int = 25

    # Fourier spectrum: whether to include radial power spectrum
    include_fourier: bool = True


@dataclass
class ClusteringConfig:
    """Parameters for unsupervised learning."""

    # Dimensionality reduction
    umap_n_components: int = 2
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # HDBSCAN clustering on the UMAP embedding
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    hdbscan_selection_method: str = "eom"  # "eom" or "leaf"

    # Alternatively, k-means (if you want to compare)
    kmeans_k_range: Tuple[int, int] = (2, 10)


@dataclass
class ProjectConfig:
    """Top-level config aggregating all sub-configs."""

    sim: SimulationConfig = field(default_factory=SimulationConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    clust: ClusteringConfig = field(default_factory=ClusteringConfig)

    output_dir: str = "results"
    figure_dir: str = "figures"
