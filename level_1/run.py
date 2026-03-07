#!/usr/bin/env python
"""
level_1/run.py  –  Phase Classification driver
===============================================

Level 1: Simulate Conway's Game of Life across initial densities, extract
physics-motivated features, and apply unsupervised clustering (UMAP +
HDBSCAN / k-means) to discover dynamical phases.

Usage (from project root)
-------------------------
    python level_1/run.py                      # defaults (100*100, 500 steps)
    python level_1/run.py --quick              # fast test (50*50, 200 steps)
    python level_1/run.py --grid_size 50 --n_steps 300 --n_samples 20
"""

import argparse
import os
import sys
import time

# Allow running from the project root or from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.batch import sweep_densities
from common.clustering import build_pipeline
from common.config import ProjectConfig
from common.features import extract_all
from common.interpretation import (
    cluster_transition_matrix,
    feature_importance,
    print_interpretation_summary,
)
from common.plotting import (
    plot_cluster_feature_distributions,
    plot_density_traces,
    plot_feature_importance,
    plot_grid_snapshots,
    plot_phase_diagram_heatmap,
    plot_representative_grids,
    plot_rho_vs_rho0,
    plot_silhouette,
    plot_umap_clusters,
    plot_umap_embedding,
)

# ── Output paths ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")


def parse_args():
    p = argparse.ArgumentParser(description="CA Phase Classification – Level 1")
    p.add_argument("--grid_size", type=int, default=None, help="Grid side length L")
    p.add_argument("--n_steps", type=int, default=None, help="Simulation steps")
    p.add_argument("--n_samples", type=int, default=None, help="Samples per density")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: 42)")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 50×50 grid, 200 steps, 10 samples/density",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ProjectConfig()

    # Override from CLI
    if args.quick:
        cfg.sim.grid_size = 50
        cfg.sim.n_steps = 200
        cfg.sim.n_samples_per_density = 10
    if args.grid_size is not None:
        cfg.sim.grid_size = args.grid_size
    if args.n_steps is not None:
        cfg.sim.n_steps = args.n_steps
    if args.n_samples is not None:
        cfg.sim.n_samples_per_density = args.n_samples
    if args.seed is not None:
        cfg.sim.seed = args.seed

    cfg.output_dir = DEFAULT_OUTPUT_DIR
    cfg.figure_dir = DEFAULT_FIGURE_DIR
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.figure_dir, exist_ok=True)

    n_total = len(cfg.sim.initial_densities) * cfg.sim.n_samples_per_density
    print(f"\n{'='*60}")
    print(f" Level 1: CA Phase Classification")
    print(f"{'='*60}")
    print(f" Grid:        {cfg.sim.grid_size} × {cfg.sim.grid_size}")
    print(f" Steps:       {cfg.sim.n_steps}")
    print(
        f" Densities:   {len(cfg.sim.initial_densities)} values in "
        f"[{cfg.sim.initial_densities[0]:.2f}, {cfg.sim.initial_densities[-1]:.2f}]"
    )
    print(f" Samples/ρ₀:  {cfg.sim.n_samples_per_density}")
    print(f" Total runs:  {n_total}")
    print(
        f" Rules:       B{''.join(map(str, cfg.sim.birth))}"
        f"/S{''.join(map(str, cfg.sim.survive))}"
    )
    print(f"{'='*60}\n")

    # ── Step 1: Simulate ──────────────────────────────────────────────
    t0 = time.time()
    results = sweep_densities(cfg.sim, save_trajectory=False, verbose=True)
    t_sim = time.time() - t0
    print(f"\n  Simulation complete in {t_sim:.1f}s")

    # ── Step 2: Feature extraction ────────────────────────────────────
    print("\n  Extracting features ...")
    t0 = time.time()
    feature_names, feature_matrix, metadata = extract_all(
        results, cfg.feat, cfg.sim.n_steps
    )
    t_feat = time.time() - t0
    print(
        f"  {len(feature_names)} features × {feature_matrix.shape[0]} samples"
        f"  ({t_feat:.1f}s)"
    )

    np.savez(
        os.path.join(cfg.output_dir, "features.npz"),
        names=feature_names,
        matrix=feature_matrix,
        density_init=[m["density_init"] for m in metadata],
    )
    print(f"  Saved → {cfg.output_dir}/features.npz")

    # ── Step 3: Unsupervised learning ─────────────────────────────────
    print("\n  Running UMAP + clustering ...")
    t0 = time.time()
    cl = build_pipeline(
        feature_matrix,
        cfg.clust,
        feature_names=feature_names,
        exclude_features=["density_init"],
    )
    t_cl = time.time() - t0

    n_hdb_clusters = (
        len(set(cl["labels_hdbscan"]))
        - (1 if -1 in cl["labels_hdbscan"] else 0)
        - (1 if -2 in cl["labels_hdbscan"] else 0)
    )
    n_noise = (cl["labels_hdbscan"] == -1).sum()
    n_extinct = cl.get("n_extinct", 0)
    print(f"  Extinct configs separated: {n_extinct}")
    print(f"  HDBSCAN: {n_hdb_clusters} clusters, {n_noise} noise points")
    print(
        f"  k-means best k = {cl['kmeans_results']['best_k']}  "
        f"(silhouette = {max(cl['kmeans_results']['silhouette_scores']):.3f})"
    )
    print(f"  Clustering complete in {t_cl:.1f}s\n")

    # ── Step 4: Cluster interpretation ───────────────────────────────
    print("  Interpreting clusters ...")
    density_init_arr = np.array([m["density_init"] for m in metadata])

    imp_df = feature_importance(feature_matrix, feature_names, cl["labels_hdbscan"])
    trans = cluster_transition_matrix(cl["labels_hdbscan"], density_init_arr)

    print_interpretation_summary(
        feature_matrix,
        feature_names,
        cl["labels_hdbscan"],
        density_init_arr,
        method_name="HDBSCAN",
    )

    # ── Step 5: Figures ───────────────────────────────────────────────
    print("  Generating figures ...")

    plot_density_traces(results, fig_dir=cfg.figure_dir)
    plot_rho_vs_rho0(feature_matrix, feature_names, fig_dir=cfg.figure_dir)
    plot_grid_snapshots(results, fig_dir=cfg.figure_dir)
    plot_umap_embedding(
        cl["embedding"],
        color_by=density_init_arr,
        color_label="initial density $\\rho_0$",
        fig_dir=cfg.figure_dir,
    )
    plot_umap_clusters(
        cl["embedding"],
        cl["labels_hdbscan"],
        method_name="HDBSCAN",
        fig_dir=cfg.figure_dir,
    )
    plot_umap_clusters(
        cl["embedding"],
        cl["kmeans_results"]["best_labels"],
        method_name="KMeans",
        fig_dir=cfg.figure_dir,
    )
    plot_silhouette(cl["kmeans_results"], fig_dir=cfg.figure_dir)
    plot_feature_importance(imp_df, fig_dir=cfg.figure_dir)
    plot_cluster_feature_distributions(
        feature_matrix,
        feature_names,
        cl["labels_hdbscan"],
        fig_dir=cfg.figure_dir,
    )
    plot_representative_grids(
        results,
        cl["embedding"],
        cl["labels_hdbscan"],
        metadata,
        fig_dir=cfg.figure_dir,
    )
    plot_phase_diagram_heatmap(trans, fig_dir=cfg.figure_dir)

    print(f"  Figures saved → {cfg.figure_dir}/\n")

    print(f"{'='*60}")
    print(f" Done!  Level 1 results:")
    print(f"   • Feature matrix:   {cfg.output_dir}/features.npz")
    print(f"   • HDBSCAN clusters: {n_hdb_clusters}")
    print(f"   • k-means best k:   {cl['kmeans_results']['best_k']}")
    print(f"   • Figures:          {cfg.figure_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
