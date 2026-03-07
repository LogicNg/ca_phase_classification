"""
Feature extraction from simulation results.

Converts raw simulation output into a fixed-length feature vector suitable
for unsupervised learning.  Features are chosen to capture the physics:

  * Density statistics  – mean, variance, and derivative of the population
    density time-series in the steady-state window.
  * Spatial features    – connected-component statistics, spatial auto-
    correlation function, and radially averaged Fourier power spectrum of
    the final grid.
  * Temporal features   – whether the system reached a fixed point or a
    short-period cycle, and the settling time.
"""

from typing import Dict, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.signal import convolve2d

from common.config import FeatureConfig

# ── density-based features ───────────────────────────────────────────────


def _density_features(density_ts: np.ndarray, window: int) -> Dict[str, float]:
    """Extract features from the tail of the density time-series."""
    tail = density_ts[-window:]
    full = density_ts

    return {
        "rho_final_mean": float(tail.mean()),
        "rho_final_std": float(tail.std()),
        "rho_final_min": float(tail.min()),
        "rho_final_max": float(tail.max()),
        "rho_range": float(tail.max() - tail.min()),
        # overall density drop from initial to final
        "rho_drop": float(full[0] - tail.mean()),
        # derivative in steady-state (slope of linear fit)
        "rho_slope": float(np.polyfit(np.arange(len(tail)), tail, 1)[0]),
    }


# ── spatial features ─────────────────────────────────────────────────────


def _connected_component_features(grid: np.ndarray) -> Dict[str, float]:
    """Statistics of connected components (clusters) on the final grid."""
    labelled, n_clusters = ndimage.label(grid)
    if n_clusters == 0:
        return {
            "n_clusters": 0,
            "mean_cluster_size": 0.0,
            "max_cluster_size": 0.0,
            "std_cluster_size": 0.0,
        }
    sizes = ndimage.sum(grid, labelled, range(1, n_clusters + 1))
    return {
        "n_clusters": float(n_clusters),
        "mean_cluster_size": float(np.mean(sizes)),
        "max_cluster_size": float(np.max(sizes)),
        "std_cluster_size": float(np.std(sizes)),
    }


def _spatial_autocorrelation(grid: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Radially averaged spatial autocorrelation C(r) of the binary grid.
    Uses FFT-based method for speed.
    """
    g = grid.astype(np.float64) - grid.mean()
    # 2D autocorrelation via FFT
    ft = np.fft.fft2(g)
    power = np.abs(ft) ** 2
    acf_2d = np.fft.ifft2(power).real / g.size

    # Normalize
    var = acf_2d[0, 0]
    if var > 0:
        acf_2d /= var

    # Radial average
    L = grid.shape[0]
    cy, cx = L // 2, L // 2
    acf_2d_shifted = np.fft.fftshift(acf_2d)

    # Clamp max_lag so we don't go out of bounds on small grids
    max_lag = min(max_lag, L // 2 - 1)
    if max_lag < 1:
        return np.zeros(1, dtype=np.float64)

    radii = np.zeros(max_lag, dtype=np.float64)
    counts = np.zeros(max_lag, dtype=np.int64)

    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            r = int(np.sqrt(dy**2 + dx**2))
            if 0 < r < max_lag:
                radii[r] += acf_2d_shifted[cy + dy, cx + dx]
                counts[r] += 1

    mask = counts > 0
    radii[mask] /= counts[mask]
    return radii  # C(r) for r = 0 .. max_lag-1


def _fourier_features(grid: np.ndarray) -> Dict[str, float]:
    """Radially averaged power spectrum and derived features."""
    g = grid.astype(np.float64) - grid.mean()
    ft = np.fft.fft2(g)
    power = np.abs(np.fft.fftshift(ft)) ** 2

    L = grid.shape[0]
    cy, cx = L // 2, L // 2
    max_k = L // 2

    radial_power = np.zeros(max_k)
    counts = np.zeros(max_k, dtype=np.int64)

    for y in range(L):
        for x in range(L):
            k = int(np.sqrt((y - cy) ** 2 + (x - cx) ** 2))
            if k < max_k:
                radial_power[k] += power[y, x]
                counts[k] += 1

    mask = counts > 0
    radial_power[mask] /= counts[mask]

    # Summary statistics of the power spectrum
    total = radial_power[1:].sum()  # exclude k=0
    if total > 0:
        k_vals = np.arange(1, max_k)
        rp = radial_power[1:]
        k_mean = np.sum(k_vals * rp) / total
        k_var = np.sum((k_vals - k_mean) ** 2 * rp) / total
        k_peak = k_vals[np.argmax(rp)]
    else:
        k_mean = k_var = k_peak = 0.0

    return {
        "fourier_k_mean": float(k_mean),
        "fourier_k_var": float(k_var),
        "fourier_k_peak": float(k_peak),
        "fourier_total_power": float(total),
    }


# ── temporal / dynamical features ────────────────────────────────────────


def _temporal_features(sim: dict, n_steps: int) -> Dict[str, float]:
    """Features related to settling dynamics."""
    settled = sim.get("settled_step")
    return {
        "settled": float(settled is not None),
        "settling_time": float(settled if settled is not None else n_steps),
        "settling_fraction": float((settled / n_steps) if settled is not None else 1.0),
    }


def _activity_features(sim: dict) -> Dict[str, float]:
    """
    Activity: fraction of cells that changed between the penultimate and
    final step.  This distinguishes static (still-life) from oscillating
    (period ≥ 2) patterns even when they have similar density.
    """
    final = sim["final"]
    penultimate = sim.get("penultimate")
    if penultimate is None:
        return {"activity": 0.0}
    diff = np.abs(final.astype(np.int16) - penultimate.astype(np.int16))
    N = final.shape[0] * final.shape[1]
    return {
        "activity": float(diff.sum() / N),
    }


# ── entropy features ─────────────────────────────────────────────────────


def _spatial_entropy(grid: np.ndarray, block_size: int = 2) -> float:
    """
    Spatial block entropy: Shannon entropy over the distribution of
    overlapping block_size × block_size binary patterns.

    High for chaotic / disordered grids, low for regular / empty grids.
    Normalised to [0, 1] by dividing by max entropy (block_size²).
    """
    L = grid.shape[0]
    if L < block_size or grid.sum() == 0:
        return 0.0

    windows = sliding_window_view(grid, (block_size, block_size))
    flat = windows.reshape(-1, block_size * block_size)
    powers = 2 ** np.arange(block_size * block_size)
    codes = flat @ powers  # integer code per block

    unique, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))

    max_entropy = block_size * block_size  # log2(2^(bs^2))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _temporal_entropy(density_ts: np.ndarray, window: int) -> float:
    """
    Shannon entropy of the binned density time-series in the steady-state
    window.  High for chaotic dynamics, zero for static / periodic-1.
    Normalised to [0, 1].
    """
    tail = density_ts[-window:]
    n_bins = min(30, max(2, len(tail) // 3))

    if tail.max() - tail.min() < 1e-10:
        return 0.0  # constant density → zero entropy

    hist, _ = np.histogram(tail, bins=n_bins)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(n_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _period_features(sim: dict, n_steps: int) -> Dict[str, float]:
    """
    Period of the detected cycle (0 = never settled / aperiodic in the
    simulation window) and boolean flags for convenience.
    """
    period = sim.get("period", 0)
    return {
        "period": float(period),
        "is_static": float(period == 1),
        "is_periodic": float(1 < period <= 50),
        "is_chaotic": float(period == 0 and sim.get("settled_step") is None),
    }


# ── public API ───────────────────────────────────────────────────────────


def extract_features(
    sim: dict, density_init: float, cfg: FeatureConfig, n_steps: int
) -> Dict[str, float]:
    """
    Build a full feature dict from one simulation result.

    Parameters
    ----------
    sim : dict
        Output of simulator.run().
    density_init : float
        Initial density that was used.
    cfg : FeatureConfig
    n_steps : int

    Returns
    -------
    dict mapping feature-name → float value.
    """
    feats: Dict[str, float] = {"density_init": density_init}

    # Density time-series features
    feats.update(_density_features(sim["density"], cfg.steady_state_window))

    # Spatial features on the final grid
    final = sim["final"]
    feats.update(_connected_component_features(final))

    # Spatial autocorrelation — store first few lags as features
    acf = _spatial_autocorrelation(final, cfg.max_correlation_lag)
    for lag in [1, 2, 5, 10]:
        if lag < len(acf):
            feats[f"acf_r{lag}"] = float(acf[lag])

    # Fourier
    if cfg.include_fourier:
        feats.update(_fourier_features(final))

    # Temporal / dynamical
    feats.update(_temporal_features(sim, n_steps))

    # Activity (static vs oscillating)
    feats.update(_activity_features(sim))

    # Entropy features
    feats["spatial_entropy"] = _spatial_entropy(final)
    feats["temporal_entropy"] = _temporal_entropy(
        sim["density"], cfg.steady_state_window
    )

    # Period / cycle features
    feats.update(_period_features(sim, n_steps))

    return feats


def extract_all(results: list, cfg: FeatureConfig, n_steps: int):
    """
    Extract features for all simulation results.

    Returns
    -------
    feature_names : list of str
    feature_matrix : np.ndarray, shape (n_samples, n_features)
    metadata : list of dict  (density_init, sample_idx per row)
    """
    all_feats = []
    metadata = []

    for r in results:
        f = extract_features(r["sim"], r["density_init"], cfg, n_steps)
        all_feats.append(f)
        metadata.append(
            {
                "density_init": r["density_init"],
                "sample_idx": r["sample_idx"],
            }
        )

    feature_names = list(all_feats[0].keys())
    matrix = np.array(
        [[f[k] for k in feature_names] for f in all_feats], dtype=np.float64
    )
    return feature_names, matrix, metadata
