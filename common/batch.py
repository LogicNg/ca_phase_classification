"""
Batch simulation engine.

Sweeps over initial densities, runs many independent realisations for each,
and collects the raw results for downstream feature extraction.
"""

from typing import Dict, List

import numpy as np
from tqdm import tqdm

from common.config import SimulationConfig
from common.simulator import random_initial, run


def sweep_densities(
    cfg: SimulationConfig, save_trajectory: bool = False, verbose: bool = True
) -> List[Dict]:
    """
    Run simulations across a grid of initial densities.

    Returns a list of dicts, one per (density, sample) pair, each containing:
        "density_init" : float   – initial alive-cell fraction
        "sample_idx"   : int     – replicate index
        "sim"          : dict    – output from simulator.run(...)
    """
    rng = np.random.default_rng(cfg.seed)
    results = []

    combos = [
        (d, s) for d in cfg.initial_densities for s in range(cfg.n_samples_per_density)
    ]

    iterator = tqdm(combos, desc="Simulating", disable=not verbose)

    for density, sample_idx in iterator:
        iterator.set_postfix(density=f"{density:.2f}", sample=sample_idx)

        ic = random_initial(cfg.grid_size, density, rng)
        sim = run(
            initial=ic,
            birth=cfg.birth,
            survive=cfg.survive,
            n_steps=cfg.n_steps,
            boundary=cfg.boundary,
            save_trajectory=save_trajectory,
        )

        results.append(
            {
                "density_init": density,
                "sample_idx": sample_idx,
                "sim": sim,
            }
        )

    return results
