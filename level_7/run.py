#!/usr/bin/env python
"""
level_7/run.py — Rule-space FSS comparison (Level 4 + Level 5 pipeline).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.plotting import ensure_dir
from level_7.fss_comparison import plot_rule_panels, run_rule_fss, summarize_rule_result

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")


def parse_args():
    p = argparse.ArgumentParser(description="Level 7: Rule-space FSS comparison")
    p.add_argument("--grid_sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Quick test for pipeline sanity")
    p.add_argument("--full", action="store_true", help="Heavier run: more L and samples")
    return p.parse_args()


def build_density_sweep() -> np.ndarray:
    return np.unique(
        np.round(
            np.concatenate(
                [
                    np.arange(0.02, 0.12, 0.02),
                    np.arange(0.15, 0.45, 0.10),
                    np.arange(0.50, 0.80, 0.10),
                    np.arange(0.78, 0.96, 0.02),
                ]
            ),
            4,
        )
    )


def build_density_sweep_quick() -> np.ndarray:
    return np.unique(np.round(np.concatenate([np.array([0.05, 0.15, 0.35, 0.55]), np.arange(0.75, 0.96, 0.05)]), 4))


def default_rules():
    """
    Temporary defaults (to be refined after Task 2 rerun class distribution).
    """
    return [
        {"name": "GoL reference", "birth": [3], "survive": [2, 3]},
        {"name": "III candidate A", "birth": [3, 6], "survive": [2, 3]},
        {"name": "III candidate B", "birth": [1, 3, 5, 7], "survive": [1, 3, 5, 7]},
        {"name": "IV candidate A", "birth": [0, 4, 7], "survive": [0, 1, 3]},
        {"name": "IV candidate B", "birth": [4], "survive": [1, 2, 3, 4, 5]},
        {"name": "II control", "birth": [2], "survive": [2, 3, 4]},
    ]


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _heartbeat_worker(
    stop_event: threading.Event,
    rule_name: str,
    rule_index: int,
    n_rules: int,
    started_at: float,
    pulse_seconds: float = 20.0,
) -> None:
    while not stop_event.wait(pulse_seconds):
        elapsed = time.time() - started_at
        print(
            f"   ... [{rule_index}/{n_rules}] still running {rule_name} "
            f"(elapsed {_format_duration(elapsed)})",
            flush=True,
        )


def main():
    args = parse_args()
    if args.quick:
        args.grid_sizes = [32, 64]
        args.n_samples = 8
        args.n_steps = 250
    elif args.full:
        args.grid_sizes = [32, 64, 128, 256, 512]
        args.n_samples = 40
        args.n_steps = 500

    grid_sizes = sorted(args.grid_sizes)
    densities = build_density_sweep_quick() if args.quick else build_density_sweep()
    rules = default_rules()

    ensure_dir(DEFAULT_FIGURE_DIR)
    ensure_dir(DEFAULT_OUTPUT_DIR)

    total = len(rules) * len(grid_sizes) * len(densities) * args.n_samples
    print(f"\n{'='*76}")
    print(" Level 7: Rule-space FSS comparison")
    print(f"{'='*76}")
    print(f" Rules:       {len(rules)}")
    print(f" Grid sizes:  {grid_sizes}")
    print(f" Densities:   {len(densities)} in [{densities[0]:.2f}, {densities[-1]:.2f}]")
    print(f" Samples/pt:  {args.n_samples}")
    print(f" Steps:       {args.n_steps}")
    print(f" Total sims:  ~{total} per sweep (spatial + temporal wrappers)")
    print(f"{'='*76}\n")

    summary_rows = []
    for i, rule in enumerate(rules, start=1):
        print(f"[{i}/{len(rules)}] Running {rule['name']} ...", flush=True)
        t0 = time.time()
        stop_event = threading.Event()
        hb_thread = threading.Thread(
            target=_heartbeat_worker,
            args=(stop_event, rule["name"], i, len(rules), t0),
            daemon=True,
        )
        hb_thread.start()
        try:
            res = run_rule_fss(
                birth=rule["birth"],
                survive=rule["survive"],
                rule_name=rule["name"],
                grid_sizes=grid_sizes,
                densities=densities,
                n_samples=args.n_samples,
                n_steps=args.n_steps,
                seed=args.seed + 100 * i,
                verbose=True,
            )
        finally:
            stop_event.set()
            hb_thread.join(timeout=0.2)
        sm = summarize_rule_result(res)
        elapsed = time.time() - t0

        out_npz = os.path.join(DEFAULT_OUTPUT_DIR, f"rule_{i:02d}_fss.npz")
        np.savez(out_npz, payload=np.array([res], dtype=object))

        fig_path = os.path.join(DEFAULT_FIGURE_DIR, f"rule_{i:02d}_panel.png")
        plot_rule_panels(res, sm, fig_path)

        summary_rows.append(
            {
                "name": rule["name"],
                "rule_str": res["rule_str"],
                "rho_c": sm["rho_c"],
                "gamma_over_nu": sm["gamma_over_nu"],
                "alpha_xi": sm["alpha_xi"],
                "z": sm["z"],
                "transition_type": sm["transition_type"],
                "seconds": elapsed,
                "npz": out_npz,
                "figure": fig_path,
            }
        )
        print(
            f"  -> rho_c={sm['rho_c']:.3f}, gamma/nu={sm['gamma_over_nu']:.2f}, "
            f"alpha={sm['alpha_xi']:.2f}, z={sm['z']:.2f}, {sm['transition_type']} "
            f"(rule elapsed {_format_duration(elapsed)})",
            flush=True,
        )

    csv_path = os.path.join(DEFAULT_OUTPUT_DIR, "rule_fss_summary.csv")
    lines = ["name,rule_str,rho_c,gamma_over_nu,alpha_xi,z,transition_type,seconds,figure,npz"]
    for r in summary_rows:
        lines.append(
            f"{r['name']},{r['rule_str']},{r['rho_c']:.6f},{r['gamma_over_nu']:.6f},"
            f"{r['alpha_xi']:.6f},{r['z']:.6f},{r['transition_type']},{r['seconds']:.2f},"
            f"{r['figure']},{r['npz']}"
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n{'='*76}")
    print(" Done! Level 7 outputs:")
    print(f"   • Summary table: {csv_path}")
    print(f"   • Per-rule npz:  {DEFAULT_OUTPUT_DIR}/rule_XX_fss.npz")
    print(f"   • Per-rule figs: {DEFAULT_FIGURE_DIR}/rule_XX_panel.png")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
