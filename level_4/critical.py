"""
Level 4 — Critical phenomena and finite-size scaling.

Provides tools for studying whether the ML-discovered phases in cellular
automata correspond to genuine thermodynamic phase transitions:

  1. Running CA simulations with detailed observable collection
  2. Computing order parameters, susceptibilities, Binder cumulants
  3. Spatial correlation functions and correlation lengths
  4. Finite-size scaling analysis and critical exponent estimation
  5. Comparison with directed percolation universality class
"""

from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from common.simulator import random_initial, step

# ── 2D Directed Percolation critical exponents ────────────────────────────

DP_EXPONENTS_2D = {
    "beta": 0.584,
    "nu_perp": 0.734,
    "gamma_over_nu": 0.901,
    "z": 1.764,
}

MEAN_FIELD_EXPONENTS = {
    "beta": 1.0,
    "nu_perp": 0.5,
    "gamma_over_nu": 2.0,
}


# ── Detailed simulation with observables ──────────────────────────────────


def run_with_observables(
    ic: np.ndarray,
    birth: List[int],
    survive: List[int],
    n_steps: int,
    boundary: str = "wrap",
) -> Dict:
    """
    Run a CA simulation collecting time series of density and activity
    (fraction of cells changing per step), plus the final grid.
    """
    L = ic.shape[0]
    N = L * L
    grid = ic.copy()

    density = np.empty(n_steps + 1)
    activity = np.empty(n_steps)
    density[0] = grid.sum() / N

    for t in range(n_steps):
        new_grid = step(grid, birth, survive, boundary)
        density[t + 1] = new_grid.sum() / N
        activity[t] = np.sum(grid != new_grid) / N
        grid = new_grid

    return {
        "density": density,
        "activity": activity,
        "final": grid,
        "rho_final": float(density[-1]),
        "survived": bool(density[-1] > 0),
    }


# ── Spatial correlation function ──────────────────────────────────────────


def spatial_correlation_fft(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radially-averaged connected two-point correlation function C(r)
    computed via FFT (Wiener-Khinchin theorem) on a periodic grid.

    Returns (r_values, C_values) arrays.
    """
    L = grid.shape[0]
    s = grid.astype(np.float64)
    mean_s = s.mean()

    if mean_s < 1e-10 or mean_s > 1 - 1e-10:
        max_r = L // 2
        return np.arange(max_r + 1, dtype=float), np.zeros(max_r + 1)

    delta = s - mean_s
    fft_delta = np.fft.fft2(delta)
    power = np.abs(fft_delta) ** 2
    acf_2d = np.real(np.fft.ifft2(power)) / (L * L)

    max_r = L // 2
    yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    dx = np.minimum(xx, L - xx)
    dy = np.minimum(yy, L - yy)
    r_grid = np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2)

    r_values = []
    C_values = []
    for r_int in range(max_r + 1):
        mask = (r_grid >= r_int - 0.5) & (r_grid < r_int + 0.5)
        if mask.any():
            r_values.append(float(r_int))
            C_values.append(float(acf_2d[mask].mean()))

    return np.array(r_values), np.array(C_values)


def correlation_length(r: np.ndarray, Cr: np.ndarray) -> float:
    """
    Second-moment correlation length:
      xi^2 = sum(r^2 * C(r)) / sum(C(r))   for r > 0, C(r) > 0.
    """
    mask = (r > 0) & (Cr > 0)
    if mask.sum() < 3:
        return 0.0
    rp, Cp = r[mask], Cr[mask]
    denom = np.sum(Cp)
    if denom < 1e-15:
        return 0.0
    return float(np.sqrt(max(np.sum(rp**2 * Cp) / denom, 0)))


# ── Statistical mechanics observables ─────────────────────────────────────


def binder_cumulant(values: np.ndarray) -> float:
    """
    Fourth-order Binder cumulant: U_4 = 1 - <m^4> / (3 <m^2>^2).
    L-independent at a continuous critical point (curves cross).
    Shows a minimum at a first-order transition.
    """
    if len(values) < 2:
        return 0.0
    m2 = np.mean(values**2)
    m4 = np.mean(values**4)
    if m2 < 1e-15:
        return 0.0
    return 1.0 - m4 / (3.0 * m2**2)


def susceptibility(values: np.ndarray, L: int) -> float:
    """
    Susceptibility chi = L^d * Var(order_parameter).
    At a continuous transition: chi_max ~ L^{gamma/nu}.
    At a first-order transition: chi_max ~ L^d.
    """
    return float(L**2 * np.var(values))


# ── Finite-size scaling sweep ─────────────────────────────────────────────


def finite_size_sweep(
    grid_sizes: List[int],
    densities: np.ndarray,
    n_samples: int,
    birth: List[int],
    survive: List[int],
    n_steps: int,
    boundary: str = "wrap",
    seed: int = 42,
    steady_window: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Run simulations across multiple grid sizes and initial densities,
    collecting all observables needed for finite-size scaling analysis.

    Returns results[L][rho_0] = dict of ensemble-averaged observables.
    """
    rng = np.random.default_rng(seed)
    results = {}
    total = len(grid_sizes) * len(densities) * n_samples
    pbar = tqdm(total=total, desc="FSS sweep", disable=not verbose)

    for L in grid_sizes:
        results[L] = {}
        for rho0 in densities:
            rho0 = float(round(rho0, 6))
            rho_finals = []
            activities_ss = []
            survival = []
            corr_lengths = []
            corr_accum = None
            corr_count = 0

            ss_start = max(0, n_steps - steady_window)

            for _ in range(n_samples):
                ic = random_initial(L, rho0, rng)
                obs = run_with_observables(ic, birth, survive, n_steps, boundary)

                rho_finals.append(obs["rho_final"])
                survival.append(float(obs["survived"]))

                ss_act = obs["activity"][ss_start:]
                activities_ss.append(float(np.mean(ss_act)))

                if obs["survived"]:
                    r_arr, C_arr = spatial_correlation_fft(obs["final"])
                    corr_lengths.append(correlation_length(r_arr, C_arr))
                    if corr_accum is None:
                        corr_accum = np.zeros(len(C_arr))
                    n = min(len(C_arr), len(corr_accum))
                    corr_accum[:n] += C_arr[:n]
                    corr_count += 1

                pbar.update(1)

            rho_arr = np.array(rho_finals)
            act_arr = np.array(activities_ss)

            if corr_accum is not None and corr_count > 0:
                avg_C = corr_accum / corr_count
                avg_r = np.arange(len(avg_C), dtype=float)
            else:
                avg_r = np.array([0.0])
                avg_C = np.array([0.0])

            results[L][rho0] = {
                "rho_finals": rho_arr,
                "rho_mean": float(np.mean(rho_arr)),
                "rho_std": float(np.std(rho_arr)),
                "activities": act_arr,
                "activity_mean": float(np.mean(act_arr)),
                "activity_std": float(np.std(act_arr)),
                "survival_frac": float(np.mean(survival)),
                "binder_rho": binder_cumulant(rho_arr),
                "binder_act": binder_cumulant(act_arr),
                "chi_rho": susceptibility(rho_arr, L),
                "chi_act": susceptibility(act_arr, L),
                "xi_mean": float(np.mean(corr_lengths)) if corr_lengths else 0.0,
                "xi_std": float(np.std(corr_lengths)) if corr_lengths else 0.0,
                "corr_r": avg_r,
                "corr_C": avg_C,
            }

    pbar.close()
    return results


# ── Critical point estimation ─────────────────────────────────────────────


def find_binder_crossings(
    results: Dict,
    grid_sizes: List[int],
    densities: np.ndarray,
    binder_key: str = "binder_rho",
) -> Tuple[float, List[float]]:
    """
    Estimate rho_c from pairwise Binder cumulant crossings between
    consecutive system sizes. Returns (rho_c, list_of_crossings).
    """
    crossings = []
    for i in range(len(grid_sizes) - 1):
        L1, L2 = grid_sizes[i], grid_sizes[i + 1]
        b1 = np.array([results[L1][float(round(d, 6))][binder_key] for d in densities])
        b2 = np.array([results[L2][float(round(d, 6))][binder_key] for d in densities])
        diff = b1 - b2
        for j in range(len(diff) - 1):
            if diff[j] * diff[j + 1] < 0:
                frac = abs(diff[j]) / (abs(diff[j]) + abs(diff[j + 1]))
                rho_cross = densities[j] + frac * (densities[j + 1] - densities[j])
                crossings.append(float(rho_cross))

    rho_c = float(np.mean(crossings)) if crossings else float(np.nan)
    return rho_c, crossings


def chi_peak_scaling(
    results: Dict,
    grid_sizes: List[int],
    densities: np.ndarray,
    chi_key: str = "chi_rho",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract the susceptibility peak height at each L and fit
    log(chi_max) vs log(L) to get gamma/nu.

    Returns (L_array, chi_max_array, gamma_over_nu_estimate).
    """
    L_arr = np.array(grid_sizes, dtype=float)
    chi_max = np.array(
        [
            max(results[L][float(round(d, 6))][chi_key] for d in densities)
            for L in grid_sizes
        ]
    )

    mask = (L_arr > 0) & (chi_max > 0)
    if mask.sum() >= 2:
        slope, _ = np.polyfit(np.log(L_arr[mask]), np.log(chi_max[mask]), 1)
    else:
        slope = float(np.nan)

    return L_arr, chi_max, float(slope)


def order_param_at_critical(
    results: Dict,
    grid_sizes: List[int],
    densities: np.ndarray,
    rho_c: float,
    obs_key: str = "rho_mean",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract the order parameter at the density nearest to rho_c for
    each L. Fit log(m) vs log(L) to get -beta/nu.

    Returns (L_array, m_array, beta_over_nu_estimate).
    """
    idx_c = int(np.argmin(np.abs(densities - rho_c)))
    rho0_c = float(round(densities[idx_c], 6))

    L_arr = np.array(grid_sizes, dtype=float)
    m_arr = np.array([results[L][rho0_c][obs_key] for L in grid_sizes])

    mask = (L_arr > 0) & (m_arr > 1e-10)
    if mask.sum() >= 2:
        slope, _ = np.polyfit(np.log(L_arr[mask]), np.log(m_arr[mask]), 1)
        beta_over_nu = -slope
    else:
        beta_over_nu = float(np.nan)

    return L_arr, m_arr, float(beta_over_nu)
