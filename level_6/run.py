#!/usr/bin/env python
"""
level_6/run.py — Mean-field rate equations for Life-like CAs (Moore, n=8).

No simulations: polynomial f(ρ) and its integral Φ(ρ) explain bistability
(first-order transition in the mean-field limit).

Usage (from project root)
-------------------------
    python level_6/run.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import ensure_dir
from level_3.rule_space import langton_lambda, rule_to_string
from level_6.mean_field import (
    classify_transition_mf,
    critical_density_mf,
    find_fixed_points,
    integrate_mean_field,
    landau_free_energy,
    parse_rule_string,
    rate_equation_array,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")
LEVEL3_RESULTS = os.path.join(
    os.path.dirname(_HERE), "level_3", "results", "rule_features.npz"
)


def parse_args():
    p = argparse.ArgumentParser(description="Level 6: Mean-field theory")
    p.add_argument(
        "--fig_dir",
        type=str,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for PNG outputs",
    )
    return p.parse_args()


def _annotate_fixed_points(ax, birth, survive, rho_fine):
    fps = find_fixed_points(birth, survive)
    f_fine = rate_equation_array(rho_fine, birth, survive)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    for r, stab in fps:
        color = "#2ca02c" if stab == "stable" else "#d62728" if stab == "unstable" else "#7f7f7f"
        ax.axvline(r, color=color, ls=":", lw=1.0, alpha=0.85)
        y0 = float(np.interp(r, rho_fine, f_fine))
        ax.scatter([r], [y0], c=[color], s=40, zorder=5, edgecolors="k", linewidths=0.4)
    lines = [f"{r:.4f} ({s})" for r, s in fps]
    ax.text(
        0.98,
        0.97,
        "Fixed points:\n" + "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    return fps


def plot_rate_equation_gol(fig_dir: str):
    birth, survive = [3], [2, 3]
    rho = np.linspace(0.0, 1.0, 2000)
    f = rate_equation_array(rho, birth, survive)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(rho, f, color="#1f77b4", lw=1.8)
    _annotate_fixed_points(ax, birth, survive, rho)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$f(\rho)$")
    ax.set_title("Mean-field rate equation — Conway (B3/S23)")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    path = os.path.join(fig_dir, "rate_equation.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_free_energy_gol(fig_dir: str):
    birth, survive = [3], [2, 3]
    rho = np.linspace(0.0, 1.0, 4000)
    phi = landau_free_energy(rho, birth, survive)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(rho, phi, color="#9467bd", lw=1.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\Phi(\rho)$")
    ax.set_title(r"Landau free energy $\Phi(\rho) = -\int_0^\rho f(\rho')\,d\rho'$ — B3/S23")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.35)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    path = os.path.join(fig_dir, "free_energy.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_basin_trajectories(fig_dir: str):
    birth, survive = [3], [2, 3]
    rho0_list = [0.001, 0.05, 0.15, 0.28, 0.35, 0.5, 0.75, 0.95]
    spin = critical_density_mf(birth, survive)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = plt.cm.coolwarm
    for j, r0 in enumerate(rho0_list):
        t, r = integrate_mean_field(r0, birth, survive, t_max=250.0, dt=0.02)
        end = float(r[-1])
        basin = "extinct" if end < (spin if not np.isnan(spin) else 0.2) else "active"
        color = cmap(j / max(1, len(rho0_list) - 1))
        ax.plot(t, r, color=color, lw=1.2, label=rf"$\rho_0$={r0:g} → {basin}")

    if not np.isnan(spin):
        ax.axhline(spin, color="k", ls="--", lw=0.9, alpha=0.6, label=rf"spinodal $\approx${spin:.3f}")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\rho(t)$")
    ax.set_title("Mean-field flow: B3/S23 (Euler, clipped to [0,1])")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    path = os.path.join(fig_dir, "basin_trajectories.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rule_comparison(fig_dir: str):
    """
    2×3 panel: f(ρ) and Φ(ρ) for GoL, a Class III rule, a Class IV rule
    (examples aligned with Level 3 regime names; not re-clustered here).
    """
    rules = [
        ("GoL B3/S23", [3], [2, 3]),
        ("Class III — B36/S23", [3, 6], [2, 3]),
        ("Class IV — B4/S12345", [4], [1, 2, 3, 4, 5]),
    ]
    rho = np.linspace(0.0, 1.0, 2500)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True)
    for col, (name, b, s) in enumerate(rules):
        f = rate_equation_array(rho, b, s)
        phi = landau_free_energy(rho, b, s)
        axes[0, col].plot(rho, f, color="#1f77b4", lw=1.4)
        axes[0, col].axhline(0.0, color="k", lw=0.45, alpha=0.35)
        axes[0, col].set_title(name, fontsize=9)
        axes[0, col].set_ylabel(r"$f(\rho)$")
        for r, st in find_fixed_points(b, s):
            c = "#2ca02c" if st == "stable" else "#d62728"
            if st != "neutral":
                axes[0, col].axvline(r, color=c, ls=":", lw=0.85, alpha=0.75)

        axes[1, col].plot(rho, phi, color="#9467bd", lw=1.4)
        axes[1, col].axhline(0.0, color="k", lw=0.45, alpha=0.35)
        axes[1, col].set_xlabel(r"$\rho$")
        axes[1, col].set_ylabel(r"$\Phi(\rho)$")

    fig.suptitle("Mean-field comparison: rate law and Landau potential", fontsize=11)
    fig.tight_layout()
    path = os.path.join(fig_dir, "rule_comparison.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def maybe_plot_spinodal_vs_lambda(fig_dir: str) -> str | None:
    if not os.path.isfile(LEVEL3_RESULTS):
        return None
    data = np.load(LEVEL3_RESULTS, allow_pickle=True)
    rules = [str(x) for x in data["rule_strings"]]
    lam = np.array(data["lambdas"], dtype=float)
    spin = []
    valid_lam = []
    for rstr, lam_i in zip(rules, lam):
        try:
            b, s = parse_rule_string(rstr)
        except ValueError:
            continue
        sp = critical_density_mf(b, s)
        if not np.isnan(sp):
            spin.append(sp)
            valid_lam.append(lam_i)
    if len(spin) < 5:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(valid_lam, spin, alpha=0.45, s=18, c="#1f77b4", edgecolors="none")
    ax.set_xlabel(r"Langton $\lambda$")
    ax.set_ylabel(r"Mean-field spinodal $\rho_{\mathrm{sp}}$")
    ax.set_title("Spinodal density vs λ (Level 3 rule sample)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "spinodal_vs_lambda.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary_table(rows: list[tuple[str, list[int], list[int]]]):
    print("\n" + "=" * 100)
    print("Mean-field summary (Moore n=8, uncorrelated sites)")
    print("=" * 100)
    hdr = f"{'Rule':<24} {'lam':>6} {'Spinodal':>10} {'MF transition':<28} {'Fixed points'}"
    print(hdr)
    print("-" * 100)
    for name, b, s in rows:
        lam = langton_lambda(list(b), list(s))
        sp = critical_density_mf(b, s)
        sp_s = f"{sp:.4f}" if not np.isnan(sp) else "-"
        fps = find_fixed_points(b, s)
        fp_str = "; ".join(f"{r:.3f} ({st})" for r, st in fps[:6])
        if len(fps) > 6:
            fp_str += " …"
        cls = classify_transition_mf(b, s)
        print(f"{name:<24} {lam:6.3f} {sp_s:>10} {cls:<28} {fp_str}")
    print("=" * 100 + "\n")


def main():
    args = parse_args()
    fig_dir = args.fig_dir
    ensure_dir(fig_dir)

    rows = [
        ("GoL B3/S23", [3], [2, 3]),
        ("B36/S23 (III-like)", [3, 6], [2, 3]),
        ("B4/S12345 (IV-like)", [4], [1, 2, 3, 4, 5]),
        ("B047/S013", [0, 4, 7], [0, 1, 3]),
        ("B014/S12347", [0, 1, 4], [1, 2, 3, 4, 7]),
    ]

    print("Level 6: mean-field rate equations")
    print(f"  Figure directory: {fig_dir}/")

    p1 = plot_rate_equation_gol(fig_dir)
    print(f"  Wrote {p1}")
    p2 = plot_free_energy_gol(fig_dir)
    print(f"  Wrote {p2}")
    p3 = plot_basin_trajectories(fig_dir)
    print(f"  Wrote {p3}")
    p4 = plot_rule_comparison(fig_dir)
    print(f"  Wrote {p4}")

    p5 = maybe_plot_spinodal_vs_lambda(fig_dir)
    if p5:
        print(f"  Wrote {p5} (from {LEVEL3_RESULTS})")
    else:
        print(f"  Skipped spinodal_vs_lambda (need {LEVEL3_RESULTS} with ≥5 parseable rules)")

    print_summary_table(rows)

    # Sanity: rule_to_string round-trip
    for _, b, s in rows[:2]:
        assert parse_rule_string(rule_to_string(b, s)) == (b, s)


if __name__ == "__main__":
    main()
