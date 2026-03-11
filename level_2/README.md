# Level 2 — Perturbation Response Analysis

## Overview

Level 2 quantifies how each dynamical phase discovered in Level 1 responds to **localised perturbations**. By applying a 5×5 block flip to alive configurations from Conway's Game of Life and tracking the Hamming distance between perturbed and unperturbed trajectories, the analysis reveals that **all alive configurations exhibit sustained damage spreading** — a hallmark of chaotic dynamics. No perturbation ever heals, confirming that the active phase has a positive effective Lyapunov exponent.

## Method

### 1. Simulate

A fresh density sweep is run identically to Level 1 (same parameters, same seed), producing an ensemble of final-state grids across the full range of initial densities $\rho_0 \in [0.05, 0.95]$.

### 2. Select & Perturb

1. **Subsample** alive configurations: at most 5 per density value, skipping extinct grids ($\rho_\text{final} = 0$).
2. **Apply perturbation** to each steady-state grid. Three perturbation types are available:

| Type     | Description                                        |
| -------- | -------------------------------------------------- |
| `block`  | Flip all cells in a 5×5 contiguous patch (default) |
| `single` | Flip 1 random cell                                 |
| `noise`  | Randomise a patch with specified noise density     |

### 3. Track Response

Both the original and perturbed grids are re-evolved for 200 steps under identical GoL rules. At each step, the **Hamming distance** is computed:

$$d_H(t) = \frac{1}{N} \sum_{i,j} \left| g_{ij}^\text{ref}(t) - g_{ij}^\text{pert}(t) \right|$$

An effective **damage radius** is also tracked — the square root of the second spatial moment of the damage distribution, measuring how far the perturbation has spread.

### 4. Classify Response

Each perturbation trial is classified as:

| Response Type | Criterion                                             |
| ------------- | ----------------------------------------------------- |
| **Healed**    | $d_H \to 0$ — perturbation fully absorbed             |
| **Partial**   | Damage grows then decays but never reaches zero       |
| **Sustained** | Damage persists at finite amplitude for all 200 steps |

### 5. Extract Features

Nine perturbation features are extracted per trial:

| Feature               | Description                                     |
| --------------------- | ----------------------------------------------- |
| `hamming_max`         | Peak Hamming distance                           |
| `hamming_max_t`       | Time step at peak damage                        |
| `hamming_final`       | Final Hamming distance at $t = 200$             |
| `growth_exponent`     | Fitted early-time damage growth rate ($\gamma$) |
| `damage_radius_max`   | Maximum effective radius of damage region       |
| `damage_radius_final` | Final effective damage radius                   |
| `recovered`           | Binary: did $d_H$ ever reach 0?                 |
| `recovery_time`       | Time of recovery (200 if none)                  |
| `response_type`       | 0 = healed, 1 = partial, 2 = sustained          |

Multiple perturbation replicates per configuration are averaged to produce one feature vector per alive sample.

## Results (Full Mode — 100 × 100, Block Perturbation)

**Configuration:** 85 alive configurations subsampled from the Level 1 sweep (123 extinct skipped), 5 perturbation replicates each.

### Response Type Summary

| Response Type | Count | Fraction |
| ------------- | ----- | -------- |
| Healed        | 0     | 0.0%     |
| Partial       | 0     | 0.0%     |
| Sustained     | 78    | 100%     |

**Every tested alive configuration shows sustained damage.** The 5×5 block perturbation never heals — not even once across 78 configurations on a 100×100 grid.

### Physical Interpretation

- The positive growth exponent indicates exponential-like initial growth of the damage front, analogous to a positive Lyapunov exponent in continuous dynamical systems.
- Even "sparse static" configurations near the extinction boundary are embedded in a GoL universe that relays and amplifies perturbation damage through interspersed active regions.
- On the larger 100×100 grid, damage has more room to spread before hitting boundary effects, reinforcing the universality of the chaotic response.
- This result is consistent with the chaotic dynamics expected in the active phase: small perturbations grow exponentially and persist indefinitely.

## Figures

### Hamming Distance Trajectories

![Hamming distance d(t) trajectories for perturbation trials](figures/hamming_traces.png)

### Susceptibility

![Peak Hamming, final Hamming, and damage radius vs initial density](figures/susceptibility.png)

### Response Type Diagram

![Stacked bar chart of response types by density](figures/response_type_diagram.png)

### Damage Maps

![Accumulated spatial damage maps showing perturbation spread](figures/damage_maps.png)
