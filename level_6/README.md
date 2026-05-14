# Level 6 — Mean-Field Theory

## Overview

Level 6 answers a question reviewers will raise after Levels 4–5: **why should the extinction transition be first-order?** Here the answer is analytical, not simulational. Under a **mean-field** approximation (Moore neighbourhood, $n = 8$ neighbours, sites treated as independent Bernoulli random variables with density $\rho$), the average density obeys a closed **rate equation** $d\rho/dt = f(\rho)$ — a degree-9 polynomial for any Life-like rule.

For Conway’s Game of Life (B3/S23), $f(\rho)$ has the classic **bistable** shape: a stable fixed point at $\rho = 0$ (extinct), an **unstable** fixed point (the **spinodal**), and a stable fixed point at finite $\rho$ (active mean-field state). Bistability implies a **discontinuous** jump between basins, so the mean-field transition cannot be continuous (second-order). This is the theoretical complement to the simulation-based first-order evidence in Level 4.

**Status:** Implemented and reproducible via `python level_6/run.py`. No CA grid updates are performed in this level.

## Background

### Rate equation

Let $B$ and $S$ be the birth and survival neighbour-count sets. With binomial neighbour counts,

$$
\text{birth rate} = (1-\rho)\sum_{k \in B} \binom{8}{k}\,\rho^k (1-\rho)^{8-k},
$$

$$
\text{death rate} = \rho \sum_{k \notin S} \binom{8}{k}\,\rho^k (1-\rho)^{8-k}.
$$

The net rate is $f(\rho) = \text{birth rate} - \text{death rate}$. Fixed points satisfy $f(\rho^\ast) = 0$. Linear stability uses $f'(\rho^\ast)$: **stable** if $f' < 0$, **unstable** if $f' > 0$.

### Landau free energy

Define

$$
\Phi(\rho) = -\int_0^\rho f(\rho')\,d\rho'
$$

(up to an arbitrary constant). Local **minima** of $\Phi$ correspond to stable fixed points; a **maximum** corresponds to the spinodal between two attraction basins.

### Caveat

Real GoL is strongly **spatially correlated**; mean-field densities and spinodal positions are **not** meant to match simulation $\rho_c$ or steady-state densities quantitatively. The deliverable is **qualitative**: the mean-field rate law already forbids a continuous transition because $f$ admits **two** stable separated by an unstable root.

## Code layout

| File            | Role                                                                                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mean_field.py` | `birth_rate`, `death_rate`, `rate_equation` / `rate_equation_array`, `find_fixed_points`, `landau_free_energy`, `integrate_mean_field`, `critical_density_mf`, `classify_transition_mf`, `parse_rule_string` |
| `run.py`        | CLI, figures, summary table; optional spinodal–$\lambda$ plot from Level 3 `results/rule_features.npz`                                                                                                       |

## Usage

From the repository root:

```bash
python level_6/run.py
```

Optional output directory:

```bash
python level_6/run.py --fig_dir path/to/figures
```

Requires: `numpy`, `matplotlib`, and the existing `level_3` / `common` packages on `PYTHONPATH` (same pattern as other levels: run from root or with `sys.path` as in `run.py`).

## Figures

| Figure                   | Description                                                                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rate_equation.png`      | $f(\rho)$ for B3/S23 with fixed points marked and listed                                                                                                       |
| `free_energy.png`        | $\Phi(\rho)$ for B3/S23 (double-well when bistable)                                                                                                            |
| `basin_trajectories.png` | Euler flows $\rho(t)$ from several $\rho_0$; spinodal shown when defined                                                                                       |
| `rule_comparison.png`    | $2 \times 3$ panel: $f(\rho)$ and $\Phi(\rho)$ for GoL, B36/S23, B4/S12345                                                                                     |
| `spinodal_vs_lambda.png` | Scatter of mean-field spinodal vs Langton $\lambda$ for rules in `level_3/results/rule_features.npz` (written only if that file exists and enough rules parse) |

## Typical mean-field output (B3/S23)

Illustrative numbers from `find_fixed_points` (see console table when you run):

- $\rho^\ast \approx 0$ (stable, extinct)
- $\rho^\ast \approx 0.19$ (unstable, spinodal)
- $\rho^\ast \approx 0.37$ (stable, active mean-field state)

Exact values depend slightly on grid resolution and bisection tolerance inside `find_fixed_points`.

## Relation to other levels

- **Level 4–5:** Simulation FSS and temporal scaling show **first-order** behaviour and intrinsic scales; Level 6 gives a **mean-field mechanism** (bistability of $f$).
- **Level 3:** Optional `spinodal_vs_lambda.png` ties each sampled rule string to $(\lambda, \rho_{\mathrm{sp}}^{\mathrm{MF}})$; a future rerun with more rules improves that scatter statistically.
