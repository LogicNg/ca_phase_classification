#!/usr/bin/env python
"""
level_4/run.py  –  Phase Transitions & Critical Phenomena driver
================================================================

Tests whether the ML-discovered phases from Level 1 correspond to genuine
thermodynamic phase transitions by performing finite-size scaling analysis.

Computes order parameters (density, activity), susceptibilities, Binder
cumulants, and spatial correlation functions across multiple system sizes,
then extracts critical exponents and compares them with directed percolation.

Usage (from project root)
-------------------------
    python level_4/run.py                # default: L=32,64,128,256
    python level_4/run.py --quick        # fast test: L=32,64,128
    python level_4/run.py --full         # high-res: L=32,64,128,256,512
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
    MEAN_FIELD_EXPONENTS,
    chi_peak_scaling,
    find_binder_crossings,
    finite_size_sweep,
    order_param_at_critical,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")

L_COLORS = {32: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c", 256: "#d62728", 512: "#9467bd"}
L_MARKERS = {32: "o", 64: "s", 128: "^", 128: "^", 256: "D", 512: "v"}


def parse_args():
    p = argparse.ArgumentParser(description="Level 4: Phase Transitions & Critical Phenomena")
    p.add_argument("--grid_sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Quick test: L=32,64,128, 15 samples")
    p.add_argument("--full", action="store_true", help="Full run: adds L=512, 50 samples")
    return p.parse_args()


def build_density_sweep():
    """Fine resolution near the extinction transition (~0.85) with sparser coverage elsewhere."""
    return np.unique(np.round(np.concatenate([
        np.arange(0.02, 0.12, 0.02),
        np.arange(0.15, 0.45, 0.10),
        np.arange(0.50, 0.80, 0.10),
        np.arange(0.78, 0.96, 0.02),
    ]), 4))


def build_density_sweep_quick():
    return np.unique(np.round(np.concatenate([
        np.array([0.05, 0.15, 0.35, 0.55]),
        np.arange(0.75, 0.96, 0.05),
    ]), 4))


# ── Plotting helpers ──────────────────────────────────────────────────────


def _get_sorted_densities(results, L):
    return np.array(sorted(results[L].keys()))


def plot_density_fss(results, grid_sizes, fig_dir):
    """rho_final vs rho_0 for each L, showing the transition sharpening."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        means = [results[L][d]["rho_mean"] for d in rhos]
        stds = [results[L][d]["rho_std"] for d in rhos]
        ax.errorbar(rhos, means, yerr=stds, fmt="-o", ms=4, lw=1.2, capsize=2,
                     color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"Final density $\langle\rho_\mathrm{final}\rangle$")
    ax.set_title("Finite-Size Scaling: Steady-State Density")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "density_fss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_activity_fss(results, grid_sizes, fig_dir):
    """Activity (order parameter) vs rho_0 for each L."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        means = [results[L][d]["activity_mean"] for d in rhos]
        stds = [results[L][d]["activity_std"] for d in rhos]
        ax.errorbar(rhos, means, yerr=stds, fmt="-o", ms=4, lw=1.2, capsize=2,
                     color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"Activity density $\langle a \rangle$")
    ax.set_title("Finite-Size Scaling: Activity (Order Parameter)")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "activity_fss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survival_probability(results, grid_sizes, fig_dir):
    """Survival probability P_s vs rho_0 for each L."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        ps = [results[L][d]["survival_frac"] for d in rhos]
        ax.plot(rhos, ps, "-o", ms=5, lw=1.5,
                color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.85)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"Survival probability $P_s$")
    ax.set_title("Finite-Size Scaling: Survival Probability")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "survival_probability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_susceptibility(results, grid_sizes, fig_dir):
    """Susceptibility chi vs rho_0 for each L."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "chi_rho", r"$\chi_\rho = L^2 \,\mathrm{Var}(\rho)$"),
        (axes[1], "chi_act", r"$\chi_a = L^2 \,\mathrm{Var}(a)$"),
    ]:
        for L in grid_sizes:
            rhos = _get_sorted_densities(results, L)
            chi = [results[L][d][key] for d in rhos]
            ax.plot(rhos, chi, "-o", ms=4, lw=1.2,
                    color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.85)
        ax.set_xlabel(r"$\rho_0$")
        ax.set_ylabel(title)
        ax.legend()
        ax.set_xlim(0, 1)

    fig.suptitle("Susceptibility (Fluctuations of Order Parameters)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "susceptibility.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binder_cumulant(results, grid_sizes, densities, rho_c, fig_dir):
    """Binder cumulant U_4 vs rho_0 for each L, with vertical line at rho_c."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in [
        (axes[0], "binder_rho", r"$U_4(\rho)$"),
        (axes[1], "binder_act", r"$U_4(a)$"),
    ]:
        for L in grid_sizes:
            rhos = _get_sorted_densities(results, L)
            b = [results[L][d][key] for d in rhos]
            ax.plot(rhos, b, "-o", ms=4, lw=1.2,
                    color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.85)
        if not np.isnan(rho_c):
            ax.axvline(rho_c, ls="--", color="grey", lw=1, alpha=0.7,
                       label=rf"$\rho_c \approx {rho_c:.3f}$")
        ax.set_xlabel(r"$\rho_0$")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Binder Cumulant — Crossing Identifies Critical Point", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "binder_cumulant.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_functions(results, grid_sizes, densities, fig_dir):
    """C(r) at selected densities for the largest system size."""
    ensure_dir(fig_dir)
    L_max = max(grid_sizes)
    rhos = _get_sorted_densities(results, L_max)

    targets = [0.06, 0.35, 0.70, 0.82, 0.86, 0.90]
    selected = []
    for t in targets:
        idx = int(np.argmin(np.abs(rhos - t)))
        if rhos[idx] not in [s for s, _ in selected]:
            selected.append((rhos[idx], results[L_max][rhos[idx]]))

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps.get_cmap("viridis").resampled(len(selected))
    for i, (rho0, data) in enumerate(selected):
        r, C = data["corr_r"], data["corr_C"]
        if len(r) > 1 and np.any(C > 0):
            mask = (r > 0) & (C > 0)
            if mask.sum() > 2:
                ax.semilogy(r[mask], C[mask], "-", lw=1.5, color=cmap(i),
                            label=rf"$\rho_0 = {rho0:.2f}$", alpha=0.85)

    ax.set_xlabel(r"Distance $r$")
    ax.set_ylabel(r"$C(r)$")
    ax.set_title(f"Spatial Correlation Function (L = {L_max})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_functions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_length(results, grid_sizes, fig_dir):
    """Correlation length xi vs rho_0 for each L."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        xi = [results[L][d]["xi_mean"] for d in rhos]
        xi_err = [results[L][d]["xi_std"] for d in rhos]
        ax.errorbar(rhos, xi, yerr=xi_err, fmt="-o", ms=4, lw=1.2, capsize=2,
                     color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    ax.set_xlabel(r"Initial density $\rho_0$")
    ax.set_ylabel(r"Correlation length $\xi$")
    ax.set_title("Finite-Size Scaling: Correlation Length")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_length.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chi_peak_scaling(L_arr, chi_max, gamma_over_nu, fig_dir):
    """log-log plot of chi_max vs L with power-law fit."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    mask = (L_arr > 0) & (chi_max > 0)
    if mask.sum() >= 2:
        ax.loglog(L_arr[mask], chi_max[mask], "ko", ms=8, zorder=3)
        L_fit = np.linspace(L_arr[mask].min() * 0.8, L_arr[mask].max() * 1.2, 100)
        log_intercept = np.mean(np.log(chi_max[mask]) - gamma_over_nu * np.log(L_arr[mask]))
        ax.loglog(L_fit, np.exp(log_intercept) * L_fit**gamma_over_nu,
                  "r--", lw=1.5, label=rf"$\chi_\max \sim L^{{{gamma_over_nu:.2f}}}$")

        dp_val = DP_EXPONENTS_2D["gamma_over_nu"]
        ax.loglog(L_fit, np.exp(log_intercept) * L_fit**dp_val,
                  "b:", lw=1.5, alpha=0.6, label=rf"DP: $\gamma/\nu = {dp_val:.2f}$")
        ax.loglog(L_fit, np.exp(log_intercept) * L_fit**2.0,
                  "g:", lw=1.5, alpha=0.6, label=r"1st order: $\gamma/\nu = d = 2$")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\chi_\max$")
    ax.set_title("Susceptibility Peak Scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "chi_peak_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_order_param_scaling(L_arr, m_arr, beta_over_nu, rho_c, fig_dir):
    """log-log plot of m(rho_c) vs L with power-law fit."""
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    mask = (L_arr > 0) & (m_arr > 1e-10)
    if mask.sum() >= 2:
        ax.loglog(L_arr[mask], m_arr[mask], "ko", ms=8, zorder=3)
        L_fit = np.linspace(L_arr[mask].min() * 0.8, L_arr[mask].max() * 1.2, 100)
        log_intercept = np.mean(np.log(m_arr[mask]) + beta_over_nu * np.log(L_arr[mask]))
        ax.loglog(L_fit, np.exp(log_intercept) * L_fit**(-beta_over_nu),
                  "r--", lw=1.5, label=rf"$m \sim L^{{-{beta_over_nu:.2f}}}$ (fit)")

        dp_bn = DP_EXPONENTS_2D["beta"] / DP_EXPONENTS_2D["nu_perp"]
        ax.loglog(L_fit, np.exp(log_intercept) * L_fit**(-dp_bn),
                  "b:", lw=1.5, alpha=0.6,
                  label=rf"DP: $\beta/\nu = {dp_bn:.2f}$")

    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(rf"$\langle\rho\rangle$ at $\rho_0 \approx {rho_c:.2f}$")
    ax.set_title("Order Parameter at Critical Point vs System Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "order_param_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_panel(results, grid_sizes, densities, rho_c, gamma_over_nu,
                       beta_over_nu, L_chi, chi_max, fig_dir):
    """Publication-quality 2x3 summary panel."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (a) Density FSS
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        means = [results[L][d]["rho_mean"] for d in rhos]
        axes[0, 0].plot(rhos, means, "-o", ms=3, lw=1.2,
                        color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    axes[0, 0].set_xlabel(r"$\rho_0$")
    axes[0, 0].set_ylabel(r"$\langle\rho_\mathrm{final}\rangle$")
    axes[0, 0].set_title("(a) Density vs initial density")
    axes[0, 0].legend(fontsize=7)

    # (b) Survival probability
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        ps = [results[L][d]["survival_frac"] for d in rhos]
        axes[0, 1].plot(rhos, ps, "-o", ms=3, lw=1.2,
                        color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    axes[0, 1].set_xlabel(r"$\rho_0$")
    axes[0, 1].set_ylabel(r"$P_s$")
    axes[0, 1].set_title("(b) Survival probability")
    axes[0, 1].legend(fontsize=7)

    # (c) Binder cumulant
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        b = [results[L][d]["binder_rho"] for d in rhos]
        axes[0, 2].plot(rhos, b, "-o", ms=3, lw=1.2,
                        color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    if not np.isnan(rho_c):
        axes[0, 2].axvline(rho_c, ls="--", color="grey", lw=0.8, alpha=0.7)
    axes[0, 2].set_xlabel(r"$\rho_0$")
    axes[0, 2].set_ylabel(r"$U_4$")
    axes[0, 2].set_title(rf"(c) Binder cumulant ($\rho_c \approx {rho_c:.3f}$)")
    axes[0, 2].legend(fontsize=7)

    # (d) Susceptibility
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        chi = [results[L][d]["chi_rho"] for d in rhos]
        axes[1, 0].plot(rhos, chi, "-o", ms=3, lw=1.2,
                        color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    axes[1, 0].set_xlabel(r"$\rho_0$")
    axes[1, 0].set_ylabel(r"$\chi_\rho$")
    axes[1, 0].set_title("(d) Susceptibility")
    axes[1, 0].legend(fontsize=7)

    # (e) Correlation length
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        xi = [results[L][d]["xi_mean"] for d in rhos]
        axes[1, 1].plot(rhos, xi, "-o", ms=3, lw=1.2,
                        color=L_COLORS.get(L, "grey"), label=f"L={L}", alpha=0.8)
    axes[1, 1].set_xlabel(r"$\rho_0$")
    axes[1, 1].set_ylabel(r"$\xi$")
    axes[1, 1].set_title("(e) Correlation length")
    axes[1, 1].legend(fontsize=7)

    # (f) Chi peak scaling
    mask = (L_chi > 0) & (chi_max > 0)
    if mask.sum() >= 2:
        axes[1, 2].loglog(L_chi[mask], chi_max[mask], "ko", ms=7, zorder=3)
        L_fit = np.linspace(L_chi[mask].min() * 0.8, L_chi[mask].max() * 1.2, 50)
        log_ic = np.mean(np.log(chi_max[mask]) - gamma_over_nu * np.log(L_chi[mask]))
        axes[1, 2].loglog(L_fit, np.exp(log_ic) * L_fit**gamma_over_nu,
                          "r--", lw=1.5, label=rf"fit: $\gamma/\nu={gamma_over_nu:.2f}$")
        dp_gn = DP_EXPONENTS_2D["gamma_over_nu"]
        axes[1, 2].loglog(L_fit, np.exp(log_ic) * L_fit**dp_gn,
                          "b:", lw=1.5, alpha=0.6, label=rf"DP: $\gamma/\nu={dp_gn:.2f}$")
    axes[1, 2].set_xlabel(r"$L$")
    axes[1, 2].set_ylabel(r"$\chi_\max$")
    axes[1, 2].set_title(f"(f) Peak scaling")
    axes[1, 2].legend(fontsize=7)

    fig.suptitle("Phase Transitions & Critical Phenomena — Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "summary_panel.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "summary_panel.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if args.quick:
        args.grid_sizes = [32, 64, 128]
        args.n_samples = 15
        args.n_steps = 300
    elif args.full:
        args.grid_sizes = [32, 64, 128, 256, 512]
        args.n_samples = 50
        args.n_steps = 500

    densities = build_density_sweep_quick() if args.quick else build_density_sweep()
    grid_sizes = sorted(args.grid_sizes)

    fig_dir = DEFAULT_FIGURE_DIR
    out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    birth, survive = [3], [2, 3]
    total_sims = len(grid_sizes) * len(densities) * args.n_samples

    print(f"\n{'='*65}")
    print(f" Level 4: Phase Transitions & Critical Phenomena")
    print(f"{'='*65}")
    print(f" Grid sizes:   {grid_sizes}")
    print(f" Densities:    {len(densities)} values in [{densities[0]:.2f}, {densities[-1]:.2f}]")
    print(f" Samples/pt:   {args.n_samples}")
    print(f" Steps:        {args.n_steps}")
    print(f" Total sims:   {total_sims}")
    print(f" Rule:         B3/S23 (Conway's Game of Life)")
    print(f"{'='*65}\n")

    # ── 1. Finite-size scaling sweep ──────────────────────────────────
    t0 = time.time()
    results = finite_size_sweep(
        grid_sizes=grid_sizes,
        densities=densities,
        n_samples=args.n_samples,
        birth=birth,
        survive=survive,
        n_steps=args.n_steps,
        boundary="wrap",
        seed=args.seed,
        steady_window=min(100, args.n_steps // 3),
        verbose=True,
    )
    t_sweep = time.time() - t0
    print(f"\n  Sweep complete in {t_sweep:.1f}s\n")

    # ── 2. Critical point estimation ──────────────────────────────────
    print("  Estimating critical point from Binder cumulant crossings ...")
    rho_c, crossings = find_binder_crossings(results, grid_sizes, densities, "binder_rho")
    print(f"    Binder crossings found: {len(crossings)}")
    if crossings:
        print(f"    Crossing locations: {[f'{x:.4f}' for x in crossings]}")
    print(f"    Estimated rho_c = {rho_c:.4f}")

    # Also try activity-based Binder
    rho_c_act, crossings_act = find_binder_crossings(results, grid_sizes, densities, "binder_act")
    if crossings_act:
        print(f"    (Activity-based: rho_c = {rho_c_act:.4f}, {len(crossings_act)} crossings)")

    # Use density-based if available, else activity-based
    if np.isnan(rho_c) and not np.isnan(rho_c_act):
        rho_c = rho_c_act
        print(f"    Using activity-based estimate: rho_c = {rho_c:.4f}")

    # ── 3. Critical exponent estimation ───────────────────────────────
    print("\n  Extracting critical exponents ...")

    L_chi, chi_max_arr, gamma_over_nu = chi_peak_scaling(
        results, grid_sizes, densities, "chi_rho")
    print(f"    chi_max vs L:  gamma/nu = {gamma_over_nu:.3f}")
    print(f"      (DP prediction:         gamma/nu = {DP_EXPONENTS_2D['gamma_over_nu']:.3f})")
    print(f"      (1st-order prediction:  gamma/nu = d = 2.000)")

    L_m, m_arr, beta_over_nu = order_param_at_critical(
        results, grid_sizes, densities, rho_c if not np.isnan(rho_c) else 0.85)
    print(f"    m(rho_c) vs L: beta/nu  = {beta_over_nu:.3f}")
    dp_bn = DP_EXPONENTS_2D["beta"] / DP_EXPONENTS_2D["nu_perp"]
    print(f"      (DP prediction:         beta/nu  = {dp_bn:.3f})")

    # ── 4. Summary of transition near rho_0 ~ 0.85 ───────────────────
    print(f"\n  Transition summary per system size:")
    for L in grid_sizes:
        rhos = _get_sorted_densities(results, L)
        survs = [results[L][d]["survival_frac"] for d in rhos]
        chis = [results[L][d]["chi_rho"] for d in rhos]
        idx_peak = int(np.argmax(chis))
        rho_peak = rhos[idx_peak]

        rho_half = np.nan
        for j in range(len(survs) - 1):
            if survs[j] > 0.5 and survs[j + 1] <= 0.5:
                frac = (survs[j] - 0.5) / (survs[j] - survs[j + 1])
                rho_half = rhos[j] + frac * (rhos[j + 1] - rhos[j])
                break

        print(f"    L={L:4d}:  chi_peak at rho_0={rho_peak:.2f}  "
              f"(chi_max={chis[idx_peak]:.3f}),  "
              f"P_s = 0.5 at rho_0 ~ {rho_half:.3f}")

    # ── 5. Save results ───────────────────────────────────────────────
    save_data = {
        "grid_sizes": np.array(grid_sizes),
        "densities": densities,
        "rho_c": rho_c,
        "gamma_over_nu": gamma_over_nu,
        "beta_over_nu": beta_over_nu,
        "L_chi": L_chi,
        "chi_max": chi_max_arr,
        "L_m": L_m,
        "m_at_rhoc": m_arr,
    }

    for L in grid_sizes:
        for key in ["rho_mean", "rho_std", "activity_mean", "activity_std",
                     "survival_frac", "binder_rho", "binder_act",
                     "chi_rho", "chi_act", "xi_mean"]:
            rhos = _get_sorted_densities(results, L)
            save_data[f"L{L}_{key}"] = np.array([results[L][d][key] for d in rhos])

    np.savez(os.path.join(out_dir, "fss_results.npz"), **save_data)
    print(f"\n  Saved → {out_dir}/fss_results.npz")

    # ── 6. Figures ────────────────────────────────────────────────────
    print(f"\n  Generating figures ...")

    plot_density_fss(results, grid_sizes, fig_dir)
    plot_activity_fss(results, grid_sizes, fig_dir)
    plot_survival_probability(results, grid_sizes, fig_dir)
    plot_susceptibility(results, grid_sizes, fig_dir)
    plot_binder_cumulant(results, grid_sizes, densities, rho_c, fig_dir)
    plot_correlation_functions(results, grid_sizes, densities, fig_dir)
    plot_correlation_length(results, grid_sizes, fig_dir)
    plot_chi_peak_scaling(L_chi, chi_max_arr, gamma_over_nu, fig_dir)
    plot_order_param_scaling(L_m, m_arr, beta_over_nu,
                             rho_c if not np.isnan(rho_c) else 0.85, fig_dir)
    plot_summary_panel(results, grid_sizes, densities, rho_c, gamma_over_nu,
                       beta_over_nu, L_chi, chi_max_arr, fig_dir)

    print(f"  Figures saved → {fig_dir}/\n")

    # ── 7. Print concluding summary ───────────────────────────────────
    transition_type = "first-order" if gamma_over_nu > 1.5 else "continuous (DP-like)"
    if np.isnan(gamma_over_nu):
        transition_type = "undetermined"

    print(f"{'='*65}")
    print(f" Done!  Level 4 results:")
    print(f"   - System sizes: {grid_sizes}")
    print(f"   - Critical density: rho_c ~ {rho_c:.4f}")
    print(f"   - gamma/nu = {gamma_over_nu:.3f}  (DP: {DP_EXPONENTS_2D['gamma_over_nu']:.3f}, 1st-order: 2.0)")
    print(f"   - beta/nu  = {beta_over_nu:.3f}  (DP: {dp_bn:.3f})")
    print(f"   - Transition type: {transition_type}")
    print(f"   - Data: {out_dir}/fss_results.npz")
    print(f"   - Figures: {fig_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
