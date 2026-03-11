#!/usr/bin/env python
"""
level_2/run.py  –  Perturbation Response Analysis driver
=========================================================

Level 2: Apply block-flip perturbations to steady-state configurations
from a fresh simulation sweep and measure damage spreading (Hamming
distance) over 200 steps.

Usage (from project root)
-------------------------
    python level_2/run.py                  # defaults (100×100, 500 steps)
    python level_2/run.py --quick          # fast test (50×50, 200 steps)
    python level_2/run.py --perturbation_type single
    python level_2/run.py --n_perturbations 10
"""

import argparse
import os
import sys
import time
from collections import Counter

# Allow running from the project root or from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.batch import sweep_densities
from common.config import ProjectConfig
from common.plotting import (
    plot_damage_maps,
    plot_hamming_traces,
    plot_response_type_diagram,
    plot_susceptibility,
)
from level_2.perturbation import aggregate_perturbation_features, perturbation_sweep

# ── Output paths ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")


def parse_args():
    p = argparse.ArgumentParser(description="CA Phase Classification – Level 2")
    p.add_argument("--grid_size", type=int, default=None, help="Grid side length L")
    p.add_argument("--n_steps", type=int, default=None, help="Simulation steps")
    p.add_argument("--n_samples", type=int, default=None, help="Samples per density")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: 42)")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 50×50 grid, 200 steps, 10 samples/density",
    )
    p.add_argument(
        "--n_perturbations",
        type=int,
        default=5,
        help="Perturbation replicates per sample (default: 5)",
    )
    p.add_argument(
        "--perturbation_type",
        type=str,
        default="block",
        choices=["single", "block", "noise_patch"],
        help="Type of perturbation to apply (default: block)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ProjectConfig()

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
    print(f" Level 2: Perturbation Response Analysis")
    print(f"{'='*60}")
    print(f" Grid:          {cfg.sim.grid_size} × {cfg.sim.grid_size}")
    print(f" Steps:         {cfg.sim.n_steps}")
    print(f" Total sims:    {n_total}")
    print(f" Pert. type:    {args.perturbation_type}")
    print(f" Pert./sample:  {args.n_perturbations}")
    print(f" Response steps: 200")
    print(f"{'='*60}\n")

    # ── Step 1: Re-run simulation sweep ───────────────────────────────
    print("  Running simulation sweep (needed for perturbation analysis) ...")
    t0 = time.time()
    results = sweep_densities(cfg.sim, save_trajectory=False, verbose=True)
    t_sim = time.time() - t0
    print(f"\n  Simulation complete in {t_sim:.1f}s")

    # ── Step 2: Subsample alive configurations ────────────────────────
    pert_sample = []
    density_counts = Counter()
    max_per_density = 5
    for r in results:
        d = r["density_init"]
        if r["sim"]["final"].sum() == 0:
            continue
        if density_counts[d] < max_per_density:
            density_counts[d] += 1
            pert_sample.append(r)

    n_alive = len(pert_sample)
    n_dead = sum(1 for r in results if r["sim"]["final"].sum() == 0)
    print(f"\n  Subsampled {n_alive} alive configs (skipped {n_dead} extinct)\n")

    # ── Step 3: Perturbation sweep ────────────────────────────────────
    t0 = time.time()
    pert_results = perturbation_sweep(
        pert_sample,
        birth=cfg.sim.birth,
        survive=cfg.sim.survive,
        n_perturbations=args.n_perturbations,
        n_response_steps=200,
        perturbation_type=args.perturbation_type,
        block_size=5,
        boundary=cfg.sim.boundary,
        verbose=True,
    )
    t_pert = time.time() - t0
    print(f"\n  Perturbation sweep complete in {t_pert:.1f}s")

    # ── Step 4: Aggregate features and save ───────────────────────────
    pf_names, pf_matrix, pf_densities = aggregate_perturbation_features(pert_results)

    np.savez(
        os.path.join(cfg.output_dir, "perturbation_features.npz"),
        names=pf_names,
        matrix=pf_matrix,
        density_init=pf_densities,
    )
    print(f"  Saved → {cfg.output_dir}/perturbation_features.npz")

    # ── Step 5: Response type summary ────────────────────────────────
    rt_idx = pf_names.index("response_type")
    n_healed = (pf_matrix[:, rt_idx] == 0).sum()
    n_partial = (pf_matrix[:, rt_idx] == 1).sum()
    n_sustained = (pf_matrix[:, rt_idx] == 2).sum()
    print(
        f"\n  Response types: healed={n_healed}, "
        f"partial={n_partial}, sustained={n_sustained}"
    )

    # ── Step 6: Figures ───────────────────────────────────────────────
    print("\n  Generating figures ...")
    plot_hamming_traces(pert_results, fig_dir=cfg.figure_dir)
    plot_susceptibility(pf_names, pf_matrix, pf_densities, fig_dir=cfg.figure_dir)
    plot_response_type_diagram(
        pf_names, pf_matrix, pf_densities, fig_dir=cfg.figure_dir
    )
    plot_damage_maps(pert_results, fig_dir=cfg.figure_dir)

    print(f"  Figures saved → {cfg.figure_dir}/\n")

    print(f"{'='*60}")
    print(f" Done!  Level 2 results:")
    print(f"   • Perturbation features: {cfg.output_dir}/perturbation_features.npz")
    print(
        f"   • Response types: healed={n_healed}, partial={n_partial}, sustained={n_sustained}"
    )
    print(f"   • Figures: {cfg.figure_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
