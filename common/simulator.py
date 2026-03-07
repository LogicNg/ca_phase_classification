"""
Fast 2-D cellular automaton simulator.

Uses scipy.signal.convolve2d for neighbour counting, which is much faster
than naive Python loops.  Supports arbitrary birth/survival rule sets and
both periodic and fixed-boundary conditions.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import convolve2d

# 3x3 kernel that counts the 8 neighbours (Moore neighbourhood)
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)


def step(
    grid: np.ndarray, birth: List[int], survive: List[int], boundary: str = "wrap"
) -> np.ndarray:
    """
    Advance the grid by one time-step.

    Parameters
    ----------
    grid : np.ndarray, shape (L, L), dtype bool or int
        Current state (1 = alive, 0 = dead).
    birth : list of int
        Neighbour counts that cause a dead cell to become alive.
    survive : list of int
        Neighbour counts that let a living cell stay alive.
    boundary : str
        "wrap" for periodic (toroidal) boundaries,
        "fill" for fixed dead boundaries.

    Returns
    -------
    np.ndarray  – the next-generation grid (same shape, dtype np.int8).
    """
    neighbour_count = convolve2d(
        grid, _NEIGHBOUR_KERNEL, mode="same", boundary=boundary, fillvalue=0
    )
    new_grid = np.zeros_like(grid, dtype=np.int8)

    # Birth rule: dead cell with correct neighbour count comes alive
    birth_mask = grid == 0
    for b in birth:
        new_grid[birth_mask & (neighbour_count == b)] = 1

    # Survival rule: live cell with correct neighbour count stays alive
    alive_mask = grid == 1
    for s in survive:
        new_grid[alive_mask & (neighbour_count == s)] = 1

    return new_grid


def run(
    initial: np.ndarray,
    birth: List[int],
    survive: List[int],
    n_steps: int,
    boundary: str = "wrap",
    save_every: int = 1,
    save_trajectory: bool = False,
) -> dict:
    """
    Run a full simulation from an initial configuration.

    Parameters
    ----------
    initial : np.ndarray, shape (L, L)
        Starting grid.
    birth, survive : lists of int
        Rule definition.
    n_steps : int
        Total number of time-steps.
    boundary : str
        Boundary condition ("wrap" or "fill").
    save_every : int
        Store a snapshot every *save_every* steps (to control memory).
    save_trajectory : bool
        If True, store all saved snapshots (can be memory-heavy for large grids).

    Returns
    -------
    dict with keys:
        "final"       : np.ndarray – final grid state
        "density"     : np.ndarray, shape (n_steps+1,) – population density at each step
        "trajectory"  : list of np.ndarray (only if save_trajectory=True)
        "settled_step": int or None – step at which the pattern became periodic / static
    """
    L = initial.shape[0]
    N = L * L
    grid = initial.astype(np.int8).copy()

    densities = np.empty(n_steps + 1, dtype=np.float32)
    densities[0] = grid.sum() / N

    trajectory = [grid.copy()] if save_trajectory else []
    hash_history = {_grid_hash(grid): 0}  # hash → first step seen
    settled_step = None
    period = None

    prev_grid = grid.copy()
    for t in range(1, n_steps + 1):
        prev_grid = grid.copy()
        grid = step(grid, birth, survive, boundary)
        densities[t] = grid.sum() / N

        if save_trajectory and (t % save_every == 0):
            trajectory.append(grid.copy())

        # Periodicity detection — arbitrary period via hash dictionary
        h = _grid_hash(grid)
        if settled_step is None:
            if h in hash_history:
                settled_step = t
                period = t - hash_history[h]
            else:
                hash_history[h] = t

    result = {
        "final": grid,
        "density": densities,
        "settled_step": settled_step,
        "period": period if period is not None else 0,
        "penultimate": prev_grid,
    }
    if save_trajectory:
        result["trajectory"] = trajectory
    return result


def random_initial(L: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random L x L binary grid with given alive-cell density."""
    return (rng.random((L, L)) < density).astype(np.int8)


def _grid_hash(grid: np.ndarray) -> int:
    """Fast hash of a grid state for periodicity detection."""
    return hash(grid.tobytes())
