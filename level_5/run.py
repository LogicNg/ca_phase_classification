#!/usr/bin/env python
"""
level_5/run.py  –  Temporal Scaling & Dynamic Critical Phenomena driver
=======================================================================

Extends Level 4 by measuring *temporal* correlations alongside the spatial
FSS already established, and by extending to L=512 to settle the ξ/L
convergence question (SOC vs finite intrinsic length scale).

Key outputs
-----------
  1. ξ/L vs ρ₀ for each L  — SOC convergence test
  2. Temporal autocorrelation C_t(τ) in steady state
  3. Temporal correlation time τ_c(ρ₀, L)
  4. Dynamic exponent z from τ_c ~ L^z at the transition
  5. Data collapse: ξ/L vs (ρ₀ − ρ_c)·L^(1/ν)
  6. log-log ξ vs L in the active phase

Usage (from project root)
-------------------------
    python level_5/run.py                          # default: L=32,64,128,256,512
    python level_5/run.py --quick                  # fast test: L=32,64,128
    python level_5/run.py --full                   # L=32,64,128,256,512,1024
    python level_5/run.py --grid_sizes 64 128 256 512
    python level_5/run.py --rho_c 0.87 --nu 0.5   # override FSS parameters
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import ensure_dir
from level_4.critical import (
    DP_EXPONENTS_2D,
    chi_peak_scaling,
    find_binder_crossings,
    order_param_at_critical,
)
from level_5.temporal import (
    DP_DYNAMIC_EXPONENT_2D,
    data_collapse_xy,
    dynamic_exponent_fit,
    temporal_fss_sweep,
    xi_over_L_table,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")

L_COLORS = {
    32: "#1f77b4",
    64: "#ff7f0e",
    128: "#2ca02c",
    256: "#d62728",
    512: "#9467bd",
    1024: "#8c564b",
}
L_MARKERS = {32: "o", 64: "s", 128: "^", 256: "D", 512: "v", 1024: "P"}


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Level 5: Temporal Scaling & Dynamic Critical Phenomena"
    )
    p.add_argument(
        "--grid_sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512],
        help="System side lengths to sweep (default: 32 64 128 256 512)",
    )
    p.add_argument("--n_samples", type=int, default=30,
                   help="Replicates per (L, rho_0) point (default: 30)")
    p.add_argument("--n_steps", type=int, default=500,
                   help="Simulation steps per run (default: 500)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--rho_c", type=float, default=0.85,
                   help="Estimated critical density from Level 4 (default: 0.85)")
    p.add_argument("--nu", type=float, default=0.5,
                   help="Correlation length exponent for data collapse (default: 0.5)")
    p.add_argument("--quick", action="store_true",
                   help="Quick test: L=32,64,128, 15 samples, 300 steps")
    p.add_argument("--full", action="store_true",
                   help="Full run: adds L=1024, 50 samples")
    return p.parse_args()


# ── Density sweep grid ────────────────────────────────────────────────────


def build_density_sweep() -> np.ndarray:
    """Fine resolution near the extinction transition, coarser elsewhere."""
    return np.unique(np.round(np.concatenate([
        np.arange(0.02, 0.12, 0.02),
        np.arange(0.15, 0.45, 0.10),
        np.arange(0.50, 0.80, 0.10),
        np.arange(0.78, 0.96, 0.02),
    ]), 4))


def build_density_sweep_quick() -> np.ndarray:
    return np.unique(np.round(np.concatenate([
        np.array([0.05, 0.15, 0.35, 0.55]),
        np.arange(0.75, 0.96, 0.05),
    ]), 4))


# ── Plotting helpers ──────────────────────────────────────────────────────


def _sorted_rhos(results: dict, L: int) -> np.ndarray:
    return np.array(sorted(results[L].keys()))


def plot_xi_over_L(results, grid_sizes, fig_dir):
    """ξ/L vs ρ₀ for each L — the SOC convergence diagnostic."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        xi_L = [results[L][d]["xi_over_L"] for d in rhos]
        ax.plot(rhos, xi_L,
                f"-{L_MARKERS.get(L, 'o')}",
                ms=5, lw=1.4,
                color=L_COLORS.get(L, "grey"),
                label=f"L={L}", alpha=0.85)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"$\xi / L$")
    ax.set_title(r"Scale-Free Diagnostic: $\xi/L$ vs $\rho_0$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "xi_over_L.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_autocorr(results, grid_sizes, densities, fig_dir):
    """C_t(τ) for selected ρ₀ values at the largest system size."""
    ensure_dir(fig_dir)
    L_max = max(grid_sizes)
    rhos_all = _sorted_rhos(results, L_max)

    targets = [0.06, 0.20, 0.40, 0.60, 0.75, 0.82]
    selected = []
    seen = set()
    for t in targets:
        idx = int(np.argmin(np.abs(rhos_all - t)))
        r = rhos_all[idx]
        if r not in seen:
            seen.add(r)
            selected.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps.get_cmap("plasma").resampled(len(selected))
    for i, rho0 in enumerate(selected):
        data = results[L_max][rho0]
        tau = data["tau_arr"]
        Ct = data["Ct_mean"]
        mask = (tau > 0) & (Ct > 0)
        if mask.sum() > 2:
            ax.semilogy(tau[mask], Ct[mask], "-",
                        lw=1.5, color=cmap(i),
                        label=rf"$\rho_0={rho0:.2f}$", alpha=0.85)

    ax.set_xlabel(r"Lag $\tau$ (steps)")
    ax.set_ylabel(r"$C_t(\tau)$")
    ax.set_title(f"Temporal Autocorrelation in Steady State (L={L_max})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "temporal_autocorr.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_correlation_time(results, grid_sizes, fig_dir):
    """τ_c vs ρ₀ for each L with error bars."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        tau_c = [results[L][d]["tau_c_mean"] for d in rhos]
        tau_err = [results[L][d]["tau_c_std"] for d in rhos]
        ax.errorbar(rhos, tau_c, yerr=tau_err,
                    fmt=f"-{L_MARKERS.get(L, 'o')}",
                    ms=4, lw=1.2, capsize=2,
                    color=L_COLORS.get(L, "grey"),
                    label=f"L={L}", alpha=0.85)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"Temporal correlation time $\tau_c$ (steps)")
    ax.set_title("Temporal Correlation Time vs Density")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "correlation_time.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_dynamic_exponent(L_arr, tau_at_rhoc, z, intercept, rho_c, fig_dir):
    """log-log τ_c(ρ_c) vs L with z fit and DP reference line."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    mask = (L_arr > 0) & (tau_at_rhoc > 0)
    if mask.sum() >= 2:
        ax.loglog(L_arr[mask], tau_at_rhoc[mask], "ko", ms=8, zorder=3,
                  label=r"$\tau_c$ at $\rho_c$")
        L_fit = np.linspace(L_arr[mask].min() * 0.8, L_arr[mask].max() * 1.2, 100)
        A = np.exp(intercept)
        ax.loglog(L_fit, A * L_fit**z, "r--", lw=1.5,
                  label=rf"Fit: $\tau_c \sim L^{{{z:.2f}}}$")
        ax.loglog(L_fit, A * L_fit**DP_DYNAMIC_EXPONENT_2D, "b:", lw=1.5, alpha=0.7,
                  label=rf"DP: $z = {DP_DYNAMIC_EXPONENT_2D:.2f}$")
    else:
        ax.text(0.5, 0.5, "Insufficient data for fit",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\tau_c$ at $\rho_c$ (steps)")
    ax.set_title(rf"Dynamic Exponent $z$ ($\tau_c \sim L^z$, $\rho_c \approx {rho_c:.3f}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "dynamic_exponent.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_data_collapse(x_all, y_all, L_labels, grid_sizes, rho_c, nu, fig_dir):
    """ξ/L vs (ρ₀ − ρ_c)·L^(1/ν), all L overlaid — tests scaling hypothesis."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in sorted(grid_sizes):
        mask = L_labels == float(L)
        if mask.sum() == 0:
            continue
        ax.scatter(x_all[mask], y_all[mask], s=20, alpha=0.7,
                   color=L_COLORS.get(L, "grey"), label=f"L={L}")
    ax.set_xlabel(
        rf"$(\rho_0 - \rho_c)\cdot L^{{1/\nu}}$  "
        rf"($\rho_c={rho_c:.3f},\;\nu={nu:.2f}$)"
    )
    ax.set_ylabel(r"$\xi / L$")
    ax.set_title("Finite-Size Scaling Data Collapse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "data_collapse.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_xi_loglog(results, grid_sizes, densities, active_rho_max, fig_dir):
    """
    log-log ξ vs L at selected active-phase densities.
    Slope = 1 on this plot would confirm ξ ∝ L (SOC / scale-free).
    """
    ensure_dir(fig_dir)
    targets = [0.10, 0.30, 0.50, 0.70]
    rhos_ref = _sorted_rhos(results, max(grid_sizes))

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.colormaps.get_cmap("viridis").resampled(len(targets))

    for i, t in enumerate(targets):
        idx = int(np.argmin(np.abs(rhos_ref - t)))
        rho0 = rhos_ref[idx]
        if rho0 > active_rho_max:
            continue
        L_vals, xi_vals = [], []
        for L in sorted(grid_sizes):
            rho0_key = float(round(rho0, 6))
            if rho0_key in results[L]:
                xi = results[L][rho0_key]["xi_mean"]
                if xi > 0:
                    L_vals.append(L)
                    xi_vals.append(xi)
        if len(L_vals) >= 2:
            L_arr = np.array(L_vals, dtype=float)
            xi_arr = np.array(xi_vals)
            ax.loglog(L_arr, xi_arr, f"-{L_MARKERS.get(L_vals[0], 'o')}",
                      ms=6, lw=1.4, color=cmap(i),
                      label=rf"$\rho_0={rho0:.2f}$", alpha=0.85)

    # Reference slope = 1 line
    L_range = np.array([min(grid_sizes), max(grid_sizes)], dtype=float)
    ax.loglog(L_range, 0.08 * L_range, "k--", lw=1, alpha=0.5, label=r"Slope = 1 ($\xi\propto L$)")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"Correlation length $\xi$")
    ax.set_title(r"$\xi$ vs $L$ in Active Phase (log-log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "xi_L_ratio_loglog.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_summary_panel(
    results, grid_sizes, densities, rho_c, z,
    L_chi, chi_max, gamma_over_nu,
    fig_dir,
):
    """Publication-quality 2×3 summary panel."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (a) ξ/L vs ρ₀
    ax = axes[0, 0]
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        xi_L = [results[L][d]["xi_over_L"] for d in rhos]
        ax.plot(rhos, xi_L, f"-{L_MARKERS.get(L,'o')}",
                ms=3, lw=1.2, color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$\xi/L$")
    ax.set_title(r"(a) Scale-free diagnostic $\xi/L$")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)

    # (b) Temporal correlation time τ_c vs ρ₀
    ax = axes[0, 1]
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        tau_c = [results[L][d]["tau_c_mean"] for d in rhos]
        ax.plot(rhos, tau_c, f"-{L_MARKERS.get(L,'o')}",
                ms=3, lw=1.2, color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$\tau_c$ (steps)")
    ax.set_title(r"(b) Temporal correlation time $\tau_c$")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)

    # (c) Survival probability
    ax = axes[0, 2]
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        ps = [results[L][d]["survival_frac"] for d in rhos]
        ax.plot(rhos, ps, f"-{L_MARKERS.get(L,'o')}",
                ms=3, lw=1.2, color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$P_s$")
    ax.set_title("(c) Survival probability")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)

    # (d) Binder cumulant
    ax = axes[1, 0]
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        b = [results[L][d]["binder_rho"] for d in rhos]
        ax.plot(rhos, b, f"-{L_MARKERS.get(L,'o')}",
                ms=3, lw=1.2, color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    if not np.isnan(rho_c):
        ax.axvline(rho_c, ls="--", color="grey", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$U_4$")
    ax.set_title(rf"(d) Binder cumulant ($\rho_c\approx{rho_c:.3f}$)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)

    # (e) Dynamic exponent: τ_c at ρ_c vs L
    ax = axes[1, 1]
    L_arr = np.array(sorted(grid_sizes), dtype=float)
    rhos_ref = _sorted_rhos(results, int(L_arr.min()))
    idx_c = int(np.argmin(np.abs(rhos_ref - rho_c)))
    rho_c_key = float(round(rhos_ref[idx_c], 6))
    tau_at_rhoc = np.array([
        results[L].get(rho_c_key, {}).get("tau_c_mean", 0.0)
        for L in sorted(grid_sizes)
    ])
    mask = (L_arr > 0) & (tau_at_rhoc > 0)
    if mask.sum() >= 2:
        ax.loglog(L_arr[mask], tau_at_rhoc[mask], "ko", ms=7, zorder=3)
        L_fit = np.linspace(L_arr[mask].min() * 0.8, L_arr[mask].max() * 1.2, 80)
        if not np.isnan(z):
            A_fit = np.exp(
                np.mean(np.log(tau_at_rhoc[mask]) - z * np.log(L_arr[mask]))
            )
            ax.loglog(L_fit, A_fit * L_fit**z, "r--", lw=1.5,
                      label=rf"$z={z:.2f}$ (fit)")
        ax.loglog(L_fit,
                  np.exp(np.mean(np.log(tau_at_rhoc[mask])
                                 - DP_DYNAMIC_EXPONENT_2D * np.log(L_arr[mask])))
                  * L_fit**DP_DYNAMIC_EXPONENT_2D,
                  "b:", lw=1.5, alpha=0.6,
                  label=rf"DP: $z={DP_DYNAMIC_EXPONENT_2D:.2f}$")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\tau_c$ at $\rho_c$")
    ax.set_title(rf"(e) Dynamic exponent $z$")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)

    # (f) χ peak scaling
    ax = axes[1, 2]
    mask2 = (L_chi > 0) & (chi_max > 0)
    if mask2.sum() >= 2:
        ax.loglog(L_chi[mask2], chi_max[mask2], "ko", ms=7, zorder=3)
        L_fit2 = np.linspace(L_chi[mask2].min() * 0.8, L_chi[mask2].max() * 1.2, 80)
        log_ic = np.mean(np.log(chi_max[mask2]) - gamma_over_nu * np.log(L_chi[mask2]))
        ax.loglog(L_fit2, np.exp(log_ic) * L_fit2**gamma_over_nu, "r--", lw=1.5,
                  label=rf"Fit: $\gamma/\nu={gamma_over_nu:.2f}$")
        dp_gn = DP_EXPONENTS_2D["gamma_over_nu"]
        ax.loglog(L_fit2, np.exp(log_ic) * L_fit2**dp_gn, "b:", lw=1.5, alpha=0.6,
                  label=rf"DP: $\gamma/\nu={dp_gn:.2f}$")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\chi_{\max}$")
    ax.set_title(rf"(f) $\chi$ peak scaling")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Level 5 — Temporal Scaling & Dynamic Critical Phenomena",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "summary_panel.png"), dpi=200, bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(fig_dir, "summary_panel.pdf"), bbox_inches="tight"
    )
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if args.quick:
        args.grid_sizes = [32, 64, 128]
        args.n_samples = 15
        args.n_steps = 300
    elif args.full:
        args.grid_sizes = [32, 64, 128, 256, 512, 1024]
        args.n_samples = 50
        args.n_steps = 500

    grid_sizes = sorted(args.grid_sizes)
    densities = build_density_sweep_quick() if args.quick else build_density_sweep()
    steady_start = max(0, args.n_steps - min(200, args.n_steps // 3))

    fig_dir = DEFAULT_FIGURE_DIR
    out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    birth, survive = [3], [2, 3]
    total_sims = len(grid_sizes) * len(densities) * args.n_samples

    print(f"\n{'='*70}")
    print(f" Level 5: Temporal Scaling & Dynamic Critical Phenomena")
    print(f"{'='*70}")
    print(f" Grid sizes:   {grid_sizes}")
    print(f" Densities:    {len(densities)} values in [{densities[0]:.2f}, {densities[-1]:.2f}]")
    print(f" Samples/pt:   {args.n_samples}")
    print(f" Steps:        {args.n_steps}  (steady state from step {steady_start})")
    print(f" Total sims:   {total_sims}")
    print(f" Rule:         B3/S23 (Conway's Game of Life)")
    print(f" rho_c:        {args.rho_c:.3f}  (from Level 4; override with --rho_c)")
    print(f" nu:           {args.nu:.2f}   (collapse exponent; override with --nu)")
    print(f"{'='*70}\n")

    # ── 1. Temporal FSS sweep ─────────────────────────────────────────
    t0 = time.time()
    results = temporal_fss_sweep(
        grid_sizes=grid_sizes,
        densities=densities,
        n_samples=args.n_samples,
        birth=birth,
        survive=survive,
        n_steps=args.n_steps,
        steady_start=steady_start,
        boundary="wrap",
        seed=args.seed,
        verbose=True,
    )
    t_sweep = time.time() - t0
    print(f"\n  Sweep complete in {t_sweep:.1f}s\n")

    # ── 2. Critical point from Binder cumulant crossings ─────────────
    print("  Estimating rho_c from Binder cumulant crossings ...")
    rho_c_est, crossings = find_binder_crossings(
        results, grid_sizes, densities, "binder_rho"
    )
    if crossings:
        print(f"    Crossings: {[f'{x:.4f}' for x in crossings]}")
        print(f"    Binder-estimated rho_c = {rho_c_est:.4f}")
        rho_c = rho_c_est
    else:
        rho_c = args.rho_c
        print(f"    No crossings found; using --rho_c = {rho_c:.4f}")

    # ── 3. ξ/L convergence table ──────────────────────────────────────
    print("\n  xi/L convergence table (active phase, rho_0 <= 0.80):")
    xi_table = xi_over_L_table(results, grid_sizes, densities, active_rho_max=0.80)
    for L in grid_sizes:
        xi_L_vals = xi_table[L]["xi_over_L"]
        xi_L_mean = float(np.mean(xi_L_vals[xi_L_vals > 0])) if xi_L_vals.any() else 0.0
        print(f"    L={L:5d}:  mean(xi/L) = {xi_L_mean:.4f}")
    print("    [If xi/L converges -> SOC; if xi/L -> 0 -> finite intrinsic xi]")

    # ── 4. Dynamic exponent z ─────────────────────────────────────────
    print(f"\n  Estimating dynamic exponent z (tau_c ~ L^z at rho_0 ~ {rho_c:.3f}) ...")
    L_arr_full = np.array(sorted(grid_sizes), dtype=float)

    # Find the rho_0 key closest to rho_c in the results
    ref_rhos = _sorted_rhos(results, grid_sizes[0])
    idx_c = int(np.argmin(np.abs(ref_rhos - rho_c)))
    rho_c_key = float(round(ref_rhos[idx_c], 6))

    tau_at_rhoc = np.array([
        results[L].get(rho_c_key, {}).get("tau_c_mean", 0.0)
        for L in sorted(grid_sizes)
    ])
    z, z_intercept = dynamic_exponent_fit(L_arr_full, tau_at_rhoc)
    if not np.isnan(z):
        print(f"    z = {z:.3f}  (DP prediction: {DP_DYNAMIC_EXPONENT_2D:.3f})")
    else:
        print(f"    Could not fit z (insufficient non-zero tau_c values at rho_c)")
        z, z_intercept = float("nan"), float("nan")

    # ── 5. Susceptibility peak scaling ───────────────────────────────
    print("\n  Chi peak scaling (gamma/nu) ...")
    L_chi, chi_max_arr, gamma_over_nu = chi_peak_scaling(
        results, grid_sizes, densities, "chi_rho"
    )
    print(f"    gamma/nu = {gamma_over_nu:.3f}  "
          f"(DP: {DP_EXPONENTS_2D['gamma_over_nu']:.3f}, 1st-order: 2.0)")

    # ── 6. Order parameter at rho_c ───────────────────────────────────
    L_m, m_arr, beta_over_nu = order_param_at_critical(
        results, grid_sizes, densities,
        rho_c if not np.isnan(rho_c) else 0.85,
    )
    dp_bn = DP_EXPONENTS_2D["beta"] / DP_EXPONENTS_2D["nu_perp"]
    print(f"    beta/nu  = {beta_over_nu:.3f}  (DP: {dp_bn:.3f})")

    # ── 7. Data collapse ──────────────────────────────────────────────
    print(f"\n  Data collapse (xi/L vs scaled density, nu={args.nu:.2f}) ...")
    x_all, y_all, L_labels = data_collapse_xy(
        results, grid_sizes, densities, rho_c, args.nu, obs_key="xi_over_L"
    )

    # ── 8. Transition summary ─────────────────────────────────────────
    print(f"\n  Transition summary per system size:")
    for L in grid_sizes:
        rhos = _sorted_rhos(results, L)
        survs = [results[L][d]["survival_frac"] for d in rhos]
        chis = [results[L][d]["chi_rho"] for d in rhos]
        idx_peak = int(np.argmax(chis))
        rho_peak = rhos[idx_peak]

        rho_half = float("nan")
        for j in range(len(survs) - 1):
            if survs[j] > 0.5 >= survs[j + 1]:
                frac = (survs[j] - 0.5) / max(survs[j] - survs[j + 1], 1e-10)
                rho_half = rhos[j] + frac * (rhos[j + 1] - rhos[j])
                break

        xi_L_active = xi_table[L]["xi_over_L"]
        xi_L_mean = (
            float(np.mean(xi_L_active[xi_L_active > 0]))
            if xi_L_active.any() else 0.0
        )
        tau_c_rhoc = results[L].get(rho_c_key, {}).get("tau_c_mean", 0.0)
        print(f"    L={L:5d}:  chi_peak@rho={rho_peak:.2f}  "
              f"Ps=0.5@rho~{rho_half:.3f}  "
              f"mean(xi/L)={xi_L_mean:.3f}  "
              f"tau_c(rho_c)={tau_c_rhoc:.2f}")

    # ── 9. Save results ───────────────────────────────────────────────
    save_data: dict = {
        "grid_sizes": np.array(grid_sizes),
        "densities": densities,
        "rho_c": np.array([rho_c]),
        "z": np.array([z]),
        "gamma_over_nu": np.array([gamma_over_nu]),
        "beta_over_nu": np.array([beta_over_nu]),
        "L_chi": L_chi,
        "chi_max": chi_max_arr,
        "L_m": L_m,
        "m_at_rhoc": m_arr,
        "tau_at_rhoc": tau_at_rhoc,
        "collapse_x": x_all,
        "collapse_y": y_all,
        "collapse_L": L_labels,
    }
    for L in grid_sizes:
        for key in [
            "rho_mean", "rho_std",
            "activity_mean", "activity_std",
            "survival_frac",
            "binder_rho", "binder_act",
            "chi_rho", "chi_act",
            "xi_mean", "xi_std", "xi_over_L",
            "tau_c_mean", "tau_c_std",
        ]:
            rhos = _sorted_rhos(results, L)
            save_data[f"L{L}_{key}"] = np.array([results[L][d][key] for d in rhos])
        save_data[f"L{L}_rhos"] = _sorted_rhos(results, L)

    np.savez(os.path.join(out_dir, "temporal_fss.npz"), **save_data)
    print(f"\n  Saved → {out_dir}/temporal_fss.npz")

    # ── 10. Figures ───────────────────────────────────────────────────
    print(f"\n  Generating figures ...")

    plot_xi_over_L(results, grid_sizes, fig_dir)
    plot_temporal_autocorr(results, grid_sizes, densities, fig_dir)
    plot_correlation_time(results, grid_sizes, fig_dir)
    plot_dynamic_exponent(L_arr_full, tau_at_rhoc, z, z_intercept, rho_c, fig_dir)
    plot_data_collapse(x_all, y_all, L_labels, grid_sizes, rho_c, args.nu, fig_dir)
    plot_xi_loglog(results, grid_sizes, densities, active_rho_max=0.80, fig_dir=fig_dir)
    plot_summary_panel(
        results, grid_sizes, densities, rho_c, z,
        L_chi, chi_max_arr, gamma_over_nu, fig_dir,
    )

    print(f"  Figures saved → {fig_dir}/\n")

    # ── 11. Concluding summary ────────────────────────────────────────
    xi_L_trend = "converging" if (
        len(grid_sizes) >= 3 and
        xi_table[grid_sizes[-1]]["xi_over_L"].mean()
        > 0.8 * xi_table[grid_sizes[-2]]["xi_over_L"].mean()
    ) else "decreasing"
    soc_verdict = (
        "consistent with SOC (scale-free active phase)"
        if xi_L_trend == "converging"
        else "ξ/L still decreasing — finite intrinsic length scale or SOC onset not yet reached"
    )

    print(f"{'='*70}")
    print(f" Done!  Level 5 results:")
    print(f"   System sizes:   {grid_sizes}")
    print(f"   rho_c:          {rho_c:.4f}")
    print(f"   Dynamic exp z:  {z:.3f}  (DP: {DP_DYNAMIC_EXPONENT_2D:.3f})"
          if not np.isnan(z) else f"   Dynamic exp z:  not determined")
    print(f"   gamma/nu:       {gamma_over_nu:.3f}  (DP: {DP_EXPONENTS_2D['gamma_over_nu']:.3f})")
    print(f"   xi/L trend:     {xi_L_trend}  -> {soc_verdict}")
    print(f"   Data:           {out_dir}/temporal_fss.npz")
    print(f"   Figures:        {fig_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
