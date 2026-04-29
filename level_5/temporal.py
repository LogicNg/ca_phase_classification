"""
Level 5 — Temporal Scaling & Dynamic Critical Phenomena.

Provides tools for:
  1. Temporal autocorrelation function C_t(tau) in steady state
  2. Temporal correlation time tau_c from exponential decay fit
  3. Augmented FSS sweep that collects spatial + temporal observables jointly
  4. xi/L convergence table (tests SOC vs finite intrinsic length scale)
  5. Dynamic exponent z: tau_c ~ L^z at the transition density
  6. Data collapse: xi(rho_0, L)/L vs (rho_0 - rho_c)*L^(1/nu)

Physical context
----------------
Level 4 established that the GoL extinction transition is first-order and
that the correlation length xi scales approximately as xi ~ L in the active
phase (possible SOC). Level 5 settles two open questions:

  (a) Does xi/L converge to a constant as L -> infinity (true SOC), or does
      it slowly vanish (finite intrinsic xi that is merely large)?

  (b) What is the dynamic exponent z connecting spatial and temporal
      correlations via tau_c ~ xi^z ~ L^z at the transition?

These results, combined with Level 4's spatial FSS, complete the finite-size
scaling picture needed for a physics journal submission.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from common.simulator import random_initial, step
from level_4.critical import (
    binder_cumulant,
    correlation_length,
    spatial_correlation_fft,
    susceptibility,
)

# ── 2D Directed Percolation dynamic exponent ──────────────────────────────
DP_DYNAMIC_EXPONENT_2D = 1.764  # z = nu_parallel / nu_perp


# ── Temporal autocorrelation ──────────────────────────────────────────────


def temporal_autocorrelation(ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalised temporal autocorrelation function of a 1-D time series.

    C_t(tau) = <delta_a(t) * delta_a(t+tau)> / Var(a)
             = (auto-covariance at lag tau) / (auto-covariance at lag 0)

    Computed via FFT for efficiency (Wiener-Khinchin theorem). Only lags
    0 … len(ts)//2 are returned (the reliable half of the circular ACF).

    Parameters
    ----------
    ts : np.ndarray, shape (T,)
        Scalar time series (e.g. activity or density) in steady state.

    Returns
    -------
    tau_arr : np.ndarray, shape (T//2 + 1,)
        Lag values 0, 1, 2, …, T//2.
    Ct_arr  : np.ndarray, shape (T//2 + 1,)
        Normalised autocorrelation, Ct[0] = 1 by construction.
    """
    T = len(ts)
    if T < 4:
        return np.array([0.0]), np.array([1.0])

    delta = ts - ts.mean()
    var = np.var(ts)
    if var < 1e-15:
        tau_arr = np.arange(T // 2 + 1, dtype=float)
        return tau_arr, np.zeros(len(tau_arr))

    # Zero-pad to next power-of-2 for FFT speed and to avoid circular wrap
    n_fft = 1
    while n_fft < 2 * T:
        n_fft <<= 1

    F = np.fft.rfft(delta, n=n_fft)
    acf_full = np.fft.irfft(np.abs(F) ** 2)[:T] / (T * var)

    max_lag = T // 2
    tau_arr = np.arange(max_lag + 1, dtype=float)
    return tau_arr, acf_full[: max_lag + 1]


def temporal_correlation_time(
    tau_arr: np.ndarray,
    Ct_arr: np.ndarray,
    min_lag: int = 1,
    min_points: int = 4,
) -> float:
    """
    Estimate the temporal correlation time tau_c by fitting
    C_t(tau) ~ exp(-tau / tau_c) on the positive-correlation tail.

    A log-linear (least-squares) fit is applied to lags where C_t > 0.
    Returns 0.0 if the fit cannot be performed reliably.

    Parameters
    ----------
    tau_arr   : lag values (output of temporal_autocorrelation).
    Ct_arr    : autocorrelation values (output of temporal_autocorrelation).
    min_lag   : start fitting from this lag (skip tau=0 which is always 1).
    min_points: minimum number of usable points for a fit attempt.

    Returns
    -------
    tau_c : float — exponential decay time (steps). 0.0 on failure.
    """
    mask = (tau_arr >= min_lag) & (Ct_arr > 0)
    if mask.sum() < min_points:
        return 0.0

    log_Ct = np.log(Ct_arr[mask])
    taus = tau_arr[mask]
    # Fit: log(C_t) = -tau / tau_c + const  =>  slope = -1/tau_c
    slope, _ = np.polyfit(taus, log_Ct, 1)
    if slope >= 0:
        return 0.0
    return float(-1.0 / slope)


# ── Augmented FSS sweep ───────────────────────────────────────────────────


def temporal_fss_sweep(
    grid_sizes: List[int],
    densities: np.ndarray,
    n_samples: int,
    birth: List[int],
    survive: List[int],
    n_steps: int,
    steady_start: int,
    boundary: str = "wrap",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Augmented finite-size scaling sweep that collects spatial *and* temporal
    observables for every (L, rho_0) combination.

    Extends `level_4.critical.finite_size_sweep` by keeping the full steady-
    state activity time series and computing:
      - Ensemble-averaged temporal autocorrelation C_t(tau)
      - Temporal correlation time tau_c

    All Level 4 observables are also computed so results are self-contained.

    Parameters
    ----------
    grid_sizes   : list of system side lengths L.
    densities    : 1-D array of initial densities rho_0.
    n_samples    : number of independent replicates per (L, rho_0).
    birth        : birth counts for the CA rule.
    survive      : survival counts for the CA rule.
    n_steps      : total simulation steps.
    steady_start : step index from which steady-state sampling begins.
    boundary     : "wrap" (periodic) or "fill" (fixed dead).
    seed         : RNG seed for reproducibility.
    verbose      : show tqdm progress bar.

    Returns
    -------
    results : dict
        results[L][rho_0] contains:
          rho_finals, rho_mean, rho_std,
          activities, activity_mean, activity_std,
          survival_frac,
          binder_rho, binder_act,
          chi_rho, chi_act,
          xi_mean, xi_std,
          corr_r, corr_C,          (spatial, same as Level 4)
          Ct_mean, tau_arr,         (temporal autocorrelation)
          tau_c_mean, tau_c_std,    (temporal correlation time)
          xi_over_L                 (xi_mean / L, key SOC diagnostic)
    """
    rng = np.random.default_rng(seed)
    results: Dict = {}
    total = len(grid_sizes) * len(densities) * n_samples
    pbar = tqdm(total=total, desc="Temporal FSS sweep", disable=not verbose)

    for L in grid_sizes:
        N = L * L
        results[L] = {}

        for rho0 in densities:
            rho0 = float(round(rho0, 6))

            # Accumulators
            rho_finals: List[float] = []
            activities_ss: List[float] = []
            survival: List[float] = []
            corr_lengths: List[float] = []
            corr_accum: Optional[np.ndarray] = None
            corr_count: int = 0

            ts_accum: Optional[np.ndarray] = None  # sum of per-run C_t
            ts_count: int = 0
            tau_c_vals: List[float] = []

            for _ in range(n_samples):
                grid = random_initial(L, rho0, rng)
                density_ts = np.empty(n_steps + 1)
                activity_ts = np.empty(n_steps)
                density_ts[0] = grid.sum() / N

                for t in range(n_steps):
                    new_grid = step(grid, birth, survive, boundary)
                    activity_ts[t] = float(np.sum(grid != new_grid)) / N
                    density_ts[t + 1] = float(new_grid.sum()) / N
                    grid = new_grid

                rho_final = float(density_ts[-1])
                survived = rho_final > 0

                rho_finals.append(rho_final)
                survival.append(float(survived))

                ss_act = activity_ts[steady_start:]
                activities_ss.append(float(ss_act.mean()))

                # Spatial correlation (only for surviving runs)
                if survived:
                    r_arr, C_arr = spatial_correlation_fft(grid)
                    xi = correlation_length(r_arr, C_arr)
                    corr_lengths.append(xi)
                    if corr_accum is None:
                        corr_accum = np.zeros(len(C_arr))
                    n = min(len(C_arr), len(corr_accum))
                    corr_accum[:n] += C_arr[:n]
                    corr_count += 1

                # Temporal autocorrelation from steady-state activity
                if len(ss_act) >= 8:
                    tau_arr, Ct = temporal_autocorrelation(ss_act)
                    tau_c = temporal_correlation_time(tau_arr, Ct)
                    tau_c_vals.append(tau_c)
                    if ts_accum is None:
                        ts_accum = np.zeros(len(Ct))
                    n = min(len(Ct), len(ts_accum))
                    ts_accum[:n] += Ct[:n]
                    ts_count += 1

                pbar.update(1)

            rho_arr = np.array(rho_finals)
            act_arr = np.array(activities_ss)

            # Spatial correlation averages
            if corr_accum is not None and corr_count > 0:
                avg_C = corr_accum / corr_count
                avg_r = np.arange(len(avg_C), dtype=float)
            else:
                avg_r = np.array([0.0])
                avg_C = np.array([0.0])

            xi_mean = float(np.mean(corr_lengths)) if corr_lengths else 0.0
            xi_std = float(np.std(corr_lengths)) if corr_lengths else 0.0

            # Temporal autocorrelation averages
            if ts_accum is not None and ts_count > 0:
                Ct_mean = ts_accum / ts_count
                tau_arr_out = np.arange(len(Ct_mean), dtype=float)
            else:
                Ct_mean = np.array([1.0])
                tau_arr_out = np.array([0.0])

            tau_c_mean = float(np.mean(tau_c_vals)) if tau_c_vals else 0.0
            tau_c_std = float(np.std(tau_c_vals)) if tau_c_vals else 0.0

            results[L][rho0] = {
                # ── Density / activity (Level 4-compatible) ──
                "rho_finals": rho_arr,
                "rho_mean": float(np.mean(rho_arr)),
                "rho_std": float(np.std(rho_arr)),
                "activities": act_arr,
                "activity_mean": float(np.mean(act_arr)),
                "activity_std": float(np.std(act_arr)),
                "survival_frac": float(np.mean(survival)),
                # ── Statistical mechanics observables ──
                "binder_rho": binder_cumulant(rho_arr),
                "binder_act": binder_cumulant(act_arr),
                "chi_rho": susceptibility(rho_arr, L),
                "chi_act": susceptibility(act_arr, L),
                # ── Spatial correlations ──
                "xi_mean": xi_mean,
                "xi_std": xi_std,
                "corr_r": avg_r,
                "corr_C": avg_C,
                "xi_over_L": xi_mean / L if L > 0 else 0.0,
                # ── Temporal correlations ──
                "Ct_mean": Ct_mean,
                "tau_arr": tau_arr_out,
                "tau_c_mean": tau_c_mean,
                "tau_c_std": tau_c_std,
            }

    pbar.close()
    return results


# ── xi/L convergence analysis ─────────────────────────────────────────────


def xi_over_L_table(
    results: Dict,
    grid_sizes: List[int],
    densities: np.ndarray,
    active_rho_max: float = 0.80,
) -> Dict:
    """
    Build a table of xi/L values for active-phase densities.

    If the active phase is truly scale-free (SOC), xi/L should converge
    to a constant as L increases. If xi/L -> 0, there is a finite intrinsic
    correlation length that only appears scale-free on small grids.

    Parameters
    ----------
    results      : output of temporal_fss_sweep.
    grid_sizes   : list of L values.
    densities    : density array used in the sweep.
    active_rho_max : upper rho_0 cutoff for the active phase (default 0.80).

    Returns
    -------
    table : dict
        table[L] = {"rhos": array, "xi_over_L": array}
        for rho_0 <= active_rho_max.
    """
    table: Dict = {}
    active_mask = densities <= active_rho_max
    active_rhos = np.array(sorted([float(round(d, 6)) for d in densities[active_mask]]))

    for L in grid_sizes:
        xi_L = np.array([
            results[L][float(round(d, 6))]["xi_over_L"]
            for d in active_rhos
            if float(round(d, 6)) in results[L]
        ])
        table[L] = {"rhos": active_rhos[:len(xi_L)], "xi_over_L": xi_L}

    return table


# ── Dynamic exponent ──────────────────────────────────────────────────────


def dynamic_exponent_fit(
    L_arr: np.ndarray,
    tau_vals: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate the dynamic exponent z from tau_c ~ L^z.

    Performs a log-log least-squares fit on the provided (L, tau_c) data.
    Only uses points where both L > 0 and tau_c > 0.

    Parameters
    ----------
    L_arr    : system sizes.
    tau_vals : temporal correlation times at those system sizes (at rho_c).

    Returns
    -------
    (z, log_prefactor) : float, float
        z is the dynamic exponent. log_prefactor is the intercept of the
        log-log fit (log of the amplitude A in tau_c = A * L^z).
        Returns (nan, nan) if fewer than 2 valid points.
    """
    mask = (L_arr > 0) & (tau_vals > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")

    log_L = np.log(L_arr[mask])
    log_tau = np.log(tau_vals[mask])
    z, intercept = np.polyfit(log_L, log_tau, 1)
    return float(z), float(intercept)


# ── Data collapse ─────────────────────────────────────────────────────────


def data_collapse_xy(
    results: Dict,
    grid_sizes: List[int],
    densities: np.ndarray,
    rho_c: float,
    nu: float,
    obs_key: str = "xi_over_L",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute finite-size scaling data collapse variables.

    Scaled x-variable: x = (rho_0 - rho_c) * L^(1/nu)
    Scaled y-variable: y = results[L][rho_0][obs_key]  (default: xi/L)

    For a continuous transition with exponent nu, curves for different L
    should collapse onto a single universal function y = f(x).
    Failure to collapse is itself evidence of a first-order or non-standard
    transition.

    Parameters
    ----------
    results    : output of temporal_fss_sweep.
    grid_sizes : list of L values.
    densities  : density array used in the sweep.
    rho_c      : estimated critical density.
    nu         : correlation length exponent (try 0.5 for first-order,
                 0.734 for 2D DP, or scan for best collapse).
    obs_key    : which observable to use as y (default "xi_over_L").

    Returns
    -------
    x_all  : np.ndarray — scaled density deviations (all L concatenated).
    y_all  : np.ndarray — observable values.
    L_labels: np.ndarray — L value for each point (for colouring).
    """
    x_list, y_list, L_list = [], [], []

    for L in grid_sizes:
        for rho0 in sorted(results[L].keys()):
            val = results[L][rho0].get(obs_key)
            if val is None or not np.isfinite(val):
                continue
            x = (rho0 - rho_c) * (L ** (1.0 / nu))
            x_list.append(x)
            y_list.append(float(val))
            L_list.append(float(L))

    return np.array(x_list), np.array(y_list), np.array(L_list)
