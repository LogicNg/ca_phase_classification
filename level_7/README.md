# Level 7 — Rule-Space FSS Comparison

## Overview

Level 7 runs the **same finite-size and temporal pipeline as Levels 4–5**, but **across several Life-like rules** chosen from the Level 3 narrative (GoL reference, class III/IV candidates, and a class II control). For each rule it:

1. **Spatial FSS sweep** (`level_4.critical.finite_size_sweep`): ensemble means of density, activity, survival, Binder cumulants, susceptibility χ, and spatial correlation length ξ.
2. **Temporal FSS sweep** (`level_5.temporal.temporal_fss_sweep`): steady-state activity, temporal autocorrelation, τ_c, ξ/L, etc.

Summaries are written to `results/rule_fss_summary.csv`; full per-rule payloads live in `results/rule_XX_fss.npz`; each rule gets a four-panel diagnostic figure in `figures/rule_XX_panel.png`.

**Status:** Implemented via `python level_7/run.py`. A **full** run is heavy (many hours); use `--quick` for a sanity check.

## Method

For each rule, `level_7/fss_comparison.summarize_rule_result` extracts:

| Quantity | Meaning |
| -------- | ------- |
| **ρ_c** | From Binder cumulant **crossings** between consecutive L in ρ₀ > 0.60; if none, falls back to the density closest to **0.85** (see `find_binder_crossings` + `summarize_rule_result`). |
| **γ/ν** | Slope of log χ_max vs log L over the density sweep (`chi_peak_scaling` on `chi_rho`). Can be **negative or NaN** if peaks do not grow with L in a log-linear way. |
| **α_ξ** | Log–log slope of mean ξ vs L (averaging ξ over ρ for each L). |
| **z** | From τ_c(L) at the density key closest to ρ_c (`dynamic_exponent_fit`). **NaN** if the fit cannot be formed (e.g. too many zero or invalid τ_c). |
| **transition_type** | `first-order-like` if at least one Binder crossing is found in the high-density window; otherwise `inconclusive/no-crossing`. |

### CLI presets

| Mode | L | Samples / (L, ρ₀) | Steps |
| ---- | - | ----------------- | ----- |
| Default | 32, 64, 128, 256 | 30 | 500 |
| `--quick` | 32, 64 | 8 | 250 |
| `--full` | 32, 64, 128, 256, 512 | 40 | 500 |

Density grid (default / `--full`): **21** values from about **0.02** to **0.94** (`build_density_sweep` in `run.py`). Per rule, the driver runs **two** full sweeps (spatial + temporal), so cost scales as **2 × |L| × 21 × n_samples** CA trajectories per rule.

## Code layout

| File | Role |
| ---- | ---- |
| `fss_comparison.py` | `run_rule_fss`, `summarize_rule_result`, `plot_rule_panels` |
| `run.py` | CLI, rule list, directories, CSV summary, heartbeat progress on long runs |

## Usage

From the repository root:

```bash
python level_7/run.py
python level_7/run.py --quick
python level_7/run.py --full
```

Optional overrides: `--grid_sizes`, `--n_samples`, `--n_steps`, `--seed`.

Requires: `numpy`, `matplotlib`, `tqdm`, and project modules `common`, `level_3`, `level_4`, `level_5` on `PYTHONPATH` (same as other levels: run from repo root).

## Figures

Each `figures/rule_XX_panel.png` is a **2×2** panel: P_s vs ρ₀, Binder U₄ vs ρ₀, ξ/L vs ρ₀, and log–log ξ vs L with the fitted α when available.

| File | Rule (label in run) |
| ---- | ------------------- |
| `rule_01_panel.png` | GoL reference (B3/S23) |
| `rule_02_panel.png` | III candidate A (B36/S23) |
| `rule_03_panel.png` | III candidate B (B1357/S1357) |
| `rule_04_panel.png` | IV candidate A (B047/S013) |
| `rule_05_panel.png` | IV candidate B (B4/S12345) |
| `rule_06_panel.png` | II control (B2/S234) |

## Results (representative `--full` run)

The table below is copied from the numeric columns of `results/rule_fss_summary.csv` after a full run (wall times are per rule on one machine; total wall time ≈ **34 hours** for all six rules). Paths in the CSV are absolute on the machine that ran the job; in git use **`figures/`** and **`results/`** as above.

| Rule | rule_str | ρ_c | γ/ν | α_ξ | z | transition_type | Wall (s) |
| ---- | -------- | ----- | ----- | ----- | --- | ----------------- | --------: |
| GoL reference | B3/S23 | 0.870 | 0.097 | 0.941 | — | first-order-like | ~11 200 |
| III candidate A | B36/S23 | 0.846 | −0.085 | 0.906 | 1.06 | first-order-like | ~11 800 |
| III candidate B | B1357/S1357 | 0.840 | — | — | — | inconclusive/no-crossing | ~49 500 |
| IV candidate A | B047/S013 | 0.840 | −0.022 | 0.754 | 1.15 | inconclusive/no-crossing | ~12 400 |
| IV candidate B | B4/S12345 | 0.840 | −0.083 | 0.796 | 1.18 | inconclusive/no-crossing | ~25 800 |
| II control | B2/S234 | 0.840 | −2.51 | 0.714 | 0.66 | inconclusive/no-crossing | ~12 200 |

**Reading this:**

- **GoL and B36/S23** show **Binder crossings** in the high-ρ window, so the summary flags **`first-order-like`**, consistent with the single-rule story in Levels 4–5 (discontinuous, nucleation-dominated physics; not DP-like continuous FSS).
- **ρ_c = 0.840** for several rows is the **fallback** when **no** Binder crossing is detected: the code substitutes the density in the sweep nearest **0.85**, not a fitted critical point. Treat those ρ_c entries as **placeholders**, not measured critical densities.
- **γ/ν and α_ξ as NaN** (III candidate B) mean the automated scaling fits failed or were ill-conditioned for that rule—often because χ or ξ behave erratically across L and ρ for chaotic or non–phase-transition dynamics.
- **z blank** for GoL means `dynamic_exponent_fit` returned **NaN** for τ_c(L) at the chosen ρ key (e.g. insufficient non-zero τ_c); the per-rule panel and `rule_01_fss.npz` still contain the full temporal sweep for manual checks.
- **B1357/S1357** took much longer wall time: the same step budget is far more expensive when typical activity or correlation updates are larger or noisier—compare elapsed seconds in the CSV when reproducing.

## Relation to other levels

- **Level 4–5:** Provide the **engines** (`finite_size_sweep`, `temporal_fss_sweep`) and observables; Level 7 **reuses** them without duplicating physics.
- **Level 3:** Rule strings and classes motivate **which** rules appear in `default_rules()` in `run.py`; the set can be refined after wider rule-space classification.
- **Level 6:** Mean-field bistability explains why a **continuous** second-order transition is unexpected for GoL-like rules; Level 7 checks whether **simulation-level** Binder and scaling metrics line up across rules.

## Outputs checklist

| Path | Content |
| ---- | ------- |
| `results/rule_fss_summary.csv` | One row per rule: scalars + paths to figure and npz |
| `results/rule_XX_fss.npz` | Pickled dict payload: `spatial`, `temporal`, grids, densities, metadata |
| `figures/rule_XX_panel.png` | Four-panel diagnostic for that rule |

To reload a run in Python:

```python
import numpy as np
blob = np.load("level_7/results/rule_01_fss.npz", allow_pickle=True)
res = blob["payload"].item()  # dict with keys spatial, temporal, rule_name, ...
```
