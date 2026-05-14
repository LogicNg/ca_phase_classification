"""
Utilities for Level 7: run and summarise FSS for multiple Life-like rules.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import ensure_dir
from level_3.rule_space import rule_to_string
from level_4.critical import chi_peak_scaling, find_binder_crossings, finite_size_sweep
from level_5.temporal import dynamic_exponent_fit, temporal_fss_sweep


def run_rule_fss(
    birth: List[int],
    survive: List[int],
    rule_name: str,
    grid_sizes: List[int],
    densities: np.ndarray,
    n_samples: int,
    n_steps: int,
    seed: int = 42,
    boundary: str = "wrap",
    verbose: bool = True,
) -> Dict:
    """
    Thin wrapper combining Level 4 + Level 5 sweeps for one rule.
    """
    steady_start = max(0, n_steps - min(200, n_steps // 3))
    spatial = finite_size_sweep(
        grid_sizes=grid_sizes,
        densities=densities,
        n_samples=n_samples,
        birth=birth,
        survive=survive,
        n_steps=n_steps,
        boundary=boundary,
        seed=seed,
        steady_window=min(100, max(20, n_steps // 4)),
        verbose=verbose,
    )
    temporal = temporal_fss_sweep(
        grid_sizes=grid_sizes,
        densities=densities,
        n_samples=n_samples,
        birth=birth,
        survive=survive,
        n_steps=n_steps,
        steady_start=steady_start,
        boundary=boundary,
        seed=seed + 1,
        verbose=verbose,
    )
    return {
        "rule_name": rule_name,
        "rule_str": rule_to_string(birth, survive),
        "birth": list(birth),
        "survive": list(survive),
        "grid_sizes": list(grid_sizes),
        "densities": np.array(densities, dtype=float),
        "n_samples": int(n_samples),
        "n_steps": int(n_steps),
        "steady_start": int(steady_start),
        "spatial": spatial,
        "temporal": temporal,
    }


def _sorted_rhos(results: Dict, L: int) -> np.ndarray:
    return np.array(sorted(results[L].keys()))


def summarize_rule_result(rule_result: Dict) -> Dict[str, float]:
    """
    Extract key scalings for one rule:
      rho_c, gamma/nu, alpha (xi scaling), z, transition type
    """
    temporal = rule_result["temporal"]
    grid_sizes = sorted(rule_result["grid_sizes"])
    densities = np.array(rule_result["densities"], dtype=float)

    densities_high = densities[densities > 0.60]
    rho_c, crossings = find_binder_crossings(
        temporal, grid_sizes, densities_high, binder_key="binder_rho"
    )
    if np.isnan(rho_c):
        rho_c = float(densities[np.argmin(np.abs(densities - 0.85))])

    _, _, gamma_over_nu = chi_peak_scaling(temporal, grid_sizes, densities, "chi_rho")

    L_sorted = np.array(grid_sizes, dtype=float)
    xi_means = np.array(
        [
            np.mean([temporal[L][r]["xi_mean"] for r in _sorted_rhos(temporal, L)])
            for L in grid_sizes
        ],
        dtype=float,
    )
    mask_xi = (L_sorted > 0) & (xi_means > 0)
    alpha = float("nan")
    if mask_xi.sum() >= 3:
        alpha, _ = np.polyfit(np.log(L_sorted[mask_xi]), np.log(xi_means[mask_xi]), 1)

    ref_rhos = _sorted_rhos(temporal, grid_sizes[0])
    rho_key = float(ref_rhos[int(np.argmin(np.abs(ref_rhos - rho_c)))])
    tau_at_rhoc = np.array([temporal[L][rho_key]["tau_c_mean"] for L in grid_sizes], dtype=float)
    z, _ = dynamic_exponent_fit(L_sorted, tau_at_rhoc)

    transition_type = "first-order-like" if crossings else "inconclusive/no-crossing"

    return {
        "rho_c": float(rho_c),
        "gamma_over_nu": float(gamma_over_nu),
        "alpha_xi": float(alpha),
        "z": float(z),
        "transition_type": transition_type,
        "rho_c_key": float(rho_key),
    }


def plot_rule_panels(rule_result: Dict, summary: Dict[str, float], out_path: str) -> None:
    """
    4-panel figure per rule:
      1) P_s vs rho_0
      2) Binder U4 vs rho_0
      3) xi/L vs rho_0
      4) xi vs L (log-log) with alpha fit
    """
    temporal = rule_result["temporal"]
    grid_sizes = sorted(rule_result["grid_sizes"])
    densities = np.array(rule_result["densities"], dtype=float)
    rule_name = rule_result["rule_name"]
    rule_str = rule_result["rule_str"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    ax_ps, ax_b, ax_xiL, ax_xi = axes.ravel()

    for L in grid_sizes:
        rhos = _sorted_rhos(temporal, L)
        ax_ps.plot(rhos, [temporal[L][r]["survival_frac"] for r in rhos], "-o", ms=3, lw=1.1, label=f"L={L}")
        ax_b.plot(rhos, [temporal[L][r]["binder_rho"] for r in rhos], "-o", ms=3, lw=1.1, label=f"L={L}")
        ax_xiL.plot(rhos, [temporal[L][r]["xi_over_L"] for r in rhos], "-o", ms=3, lw=1.1, label=f"L={L}")

    ax_ps.set_title("Survival probability")
    ax_ps.set_xlabel(r"$\rho_0$")
    ax_ps.set_ylabel(r"$P_s$")
    ax_ps.set_ylim(-0.02, 1.02)
    ax_ps.legend(fontsize=8)

    ax_b.set_title("Binder cumulant")
    ax_b.set_xlabel(r"$\rho_0$")
    ax_b.set_ylabel(r"$U_4$")
    ax_b.axhline(0.0, color="k", lw=0.5, alpha=0.4)

    ax_xiL.set_title(r"$\xi/L$ diagnostic")
    ax_xiL.set_xlabel(r"$\rho_0$")
    ax_xiL.set_ylabel(r"$\xi/L$")
    ax_xiL.axhline(0.0, color="k", lw=0.5, alpha=0.4)

    xi_points = []
    for L in grid_sizes:
        vals = [temporal[L][float(round(d, 6))]["xi_mean"] for d in densities if float(round(d, 6)) in temporal[L]]
        xi_points.append(float(np.mean([x for x in vals if x > 0])) if vals else 0.0)
    Lf = np.array(grid_sizes, dtype=float)
    xif = np.array(xi_points, dtype=float)
    m = (Lf > 0) & (xif > 0)
    ax_xi.loglog(Lf[m], xif[m], "o-", lw=1.5, ms=5, label="mean xi")
    if m.sum() >= 3 and np.isfinite(summary["alpha_xi"]):
        alpha = summary["alpha_xi"]
        c0 = np.exp(np.mean(np.log(xif[m]) - alpha * np.log(Lf[m])))
        xfit = np.linspace(Lf[m].min(), Lf[m].max(), 100)
        ax_xi.loglog(xfit, c0 * xfit**alpha, "--", lw=1.2, label=rf"fit: $\alpha={alpha:.2f}$")
    ax_xi.set_title(r"$\xi$ vs $L$")
    ax_xi.set_xlabel(r"$L$")
    ax_xi.set_ylabel(r"$\xi$")
    ax_xi.legend(fontsize=8)

    fig.suptitle(
        f"{rule_name} ({rule_str}) | rho_c={summary['rho_c']:.3f}, "
        f"gamma/nu={summary['gamma_over_nu']:.2f}, z={summary['z']:.2f}, {summary['transition_type']}",
        fontsize=10,
    )
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
