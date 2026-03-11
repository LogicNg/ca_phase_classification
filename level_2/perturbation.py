"""
Level 2: Perturbation response analysis.

Given a steady-state configuration, we perturb it and measure how the
system responds.  This gives us analogues of physical susceptibility
and correlation lengths, and lets us classify the *stability* of
different phases.

Perturbation types:
  * Single-cell flip (minimal perturbation)
  * Block insertion (localised seed)
  * Random noise patch (mesoscale perturbation)

Response measures:
  * Hamming distance  d(t) = fraction of cells that differ from the
    unperturbed trajectory at each time step.
  * Damage spreading exponent: how d(t) grows at early times.
  * Spatial extent of the damage (bounding box / radius).
  * Recovery: does d(t) → 0, or does it saturate?
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from common.simulator import run, step

# ── Perturbation generators ──────────────────────────────────────────────


def flip_single_cell(
    grid: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Flip a single random cell."""
    perturbed = grid.copy()
    y, x = rng.integers(0, grid.shape[0]), rng.integers(0, grid.shape[1])
    perturbed[y, x] = 1 - perturbed[y, x]
    return perturbed, (y, x)


def flip_block(
    grid: np.ndarray, block_size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Flip all cells in a block_size × block_size region."""
    perturbed = grid.copy()
    L = grid.shape[0]
    y0 = rng.integers(0, L - block_size + 1)
    x0 = rng.integers(0, L - block_size + 1)
    perturbed[y0 : y0 + block_size, x0 : x0 + block_size] = (
        1 - perturbed[y0 : y0 + block_size, x0 : x0 + block_size]
    )
    return perturbed, (y0, x0)


def add_noise_patch(
    grid: np.ndarray, patch_size: int, noise_density: float, rng: np.random.Generator
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Randomise cells in a patch_size × patch_size region."""
    perturbed = grid.copy()
    L = grid.shape[0]
    y0 = rng.integers(0, L - patch_size + 1)
    x0 = rng.integers(0, L - patch_size + 1)
    patch = (rng.random((patch_size, patch_size)) < noise_density).astype(np.int8)
    perturbed[y0 : y0 + patch_size, x0 : x0 + patch_size] = patch
    return perturbed, (y0, x0)


# ── Response measurement ─────────────────────────────────────────────────


def measure_perturbation_response(
    grid_unperturbed: np.ndarray,
    grid_perturbed: np.ndarray,
    birth: List[int],
    survive: List[int],
    n_response_steps: int = 200,
    boundary: str = "wrap",
) -> Dict[str, np.ndarray]:
    """
    Evolve both the unperturbed and perturbed grids in parallel and
    track how the damage (difference) evolves.

    Returns
    -------
    dict with keys:
        "hamming"     : np.ndarray, shape (n_response_steps+1,)
        "damage_map"  : np.ndarray, shape (L, L)
        "damage_radius": np.ndarray, shape (n_response_steps+1,)
        "recovered"   : bool
        "recovery_time": int or None
    """
    L = grid_unperturbed.shape[0]
    N = L * L

    g_ref = grid_unperturbed.astype(np.int8).copy()
    g_pert = grid_perturbed.astype(np.int8).copy()

    hamming = np.zeros(n_response_steps + 1)
    damage_radius = np.zeros(n_response_steps + 1)
    damage_map = np.zeros((L, L), dtype=np.int32)

    diff = np.abs(g_ref - g_pert)
    hamming[0] = diff.sum() / N
    damage_map += diff
    damage_radius[0] = _damage_effective_radius(diff)

    recovered = False
    recovery_time = None

    for t in range(1, n_response_steps + 1):
        g_ref = step(g_ref, birth, survive, boundary)
        g_pert = step(g_pert, birth, survive, boundary)

        diff = np.abs(g_ref - g_pert)
        hamming[t] = diff.sum() / N
        damage_map += diff
        damage_radius[t] = _damage_effective_radius(diff)

        if not recovered and hamming[t] == 0:
            recovered = True
            recovery_time = t

    return {
        "hamming": hamming,
        "damage_map": damage_map,
        "damage_radius": damage_radius,
        "recovered": recovered,
        "recovery_time": recovery_time,
    }


def _damage_effective_radius(diff: np.ndarray) -> float:
    """
    Effective radius of the damaged region: sqrt(second moment of the
    damage distribution about its centroid).
    """
    if diff.sum() == 0:
        return 0.0
    L = diff.shape[0]
    ys, xs = np.where(diff > 0)
    cy, cx = ys.mean(), xs.mean()
    r2 = ((ys - cy) ** 2 + (xs - cx) ** 2).mean()
    return float(np.sqrt(r2))


# ── Perturbation feature extraction ──────────────────────────────────────


def extract_perturbation_features(response: dict) -> Dict[str, float]:
    """
    Summarise a perturbation response into a feature vector.
    """
    h = response["hamming"]
    dr = response["damage_radius"]

    h_max = float(h.max())
    h_max_t = int(np.argmax(h))
    h_final = float(h[-1])

    early_mask = h[1:] > 0
    if early_mask.sum() > 2:
        first_nonzero = np.where(early_mask)[0]
        if len(first_nonzero) >= 3:
            end = min(len(first_nonzero), 20)
            ts = first_nonzero[:end] + 1
            ds = h[ts]
            valid = ds > 0
            if valid.sum() >= 2:
                growth_exp = float(
                    np.polyfit(np.log(ts[valid]), np.log(ds[valid]), 1)[0]
                )
            else:
                growth_exp = 0.0
        else:
            growth_exp = 0.0
    else:
        growth_exp = 0.0

    r_max = float(dr.max())
    r_final = float(dr[-1])

    if response["recovered"]:
        response_type = 0
    elif h_final < 0.001:
        response_type = 0
    elif h_final < h_max * 0.1:
        response_type = 1
    else:
        response_type = 2

    return {
        "hamming_max": h_max,
        "hamming_max_t": float(h_max_t),
        "hamming_final": h_final,
        "growth_exponent": growth_exp,
        "damage_radius_max": r_max,
        "damage_radius_final": r_final,
        "recovered": float(response["recovered"]),
        "recovery_time": float(
            response["recovery_time"]
            if response["recovery_time"] is not None
            else len(h)
        ),
        "response_type": float(response_type),
    }


# ── Batch perturbation sweep ─────────────────────────────────────────────


def perturbation_sweep(
    results: list,
    birth: List[int],
    survive: List[int],
    n_perturbations: int = 5,
    n_response_steps: int = 200,
    perturbation_type: str = "single",
    block_size: int = 3,
    boundary: str = "wrap",
    seed: int = 123,
    verbose: bool = True,
) -> List[Dict]:
    """
    For each simulation result, apply perturbations to the final grid
    and measure the response.
    """
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    all_results = []

    combos = [(r, p_idx) for r in results for p_idx in range(n_perturbations)]
    iterator = tqdm(combos, desc="Perturbation sweep", disable=not verbose)

    for r, p_idx in iterator:
        final_grid = r["sim"]["final"]

        if perturbation_type == "single":
            perturbed, _ = flip_single_cell(final_grid, rng)
        elif perturbation_type == "block":
            perturbed, _ = flip_block(final_grid, block_size, rng)
        elif perturbation_type == "noise_patch":
            perturbed, _ = add_noise_patch(final_grid, block_size, 0.5, rng)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        response = measure_perturbation_response(
            final_grid, perturbed, birth, survive, n_response_steps, boundary
        )
        features = extract_perturbation_features(response)

        all_results.append(
            {
                "density_init": r["density_init"],
                "sample_idx": r["sample_idx"],
                "perturbation_idx": p_idx,
                "response": response,
                "features": features,
            }
        )

    return all_results


def aggregate_perturbation_features(pert_results: List[Dict]):
    """
    Build a feature matrix from perturbation results, averaging over
    perturbation replicates for each (density, sample) pair.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in pert_results:
        key = (r["density_init"], r["sample_idx"])
        grouped[key].append(r["features"])

    feature_names = list(pert_results[0]["features"].keys())
    rows = []
    densities = []

    for (d, s), feat_list in sorted(grouped.items()):
        avg = {}
        for fname in feature_names:
            avg[fname] = np.mean([f[fname] for f in feat_list])
        rows.append([avg[k] for k in feature_names])
        densities.append(d)

    return feature_names, np.array(rows), np.array(densities)
