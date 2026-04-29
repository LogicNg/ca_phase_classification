# Level 5 — Temporal Scaling & Dynamic Critical Phenomena

## Overview

Level 5 closes the two most important open questions left by Level 4:

1. **Does ξ/L converge as L → ∞?**
   Level 4 measured the spatial correlation length ξ at L = 32, 64, 128, 256 and
   found ξ ∝ L approximately — suggestive of self-organised criticality (SOC) in
   the active phase. But the ratio ξ/L was *still decreasing* at L = 256 (from
   0.09 to 0.062), leaving open whether ξ/L converges to a constant (true SOC)
   or slowly vanishes (finite intrinsic length scale). Level 5 extends to L = 512
   (and optionally L = 1024) to settle this.

2. **What is the dynamic exponent z?**
   Level 4 measured only spatial correlations. The full finite-size scaling (FSS)
   picture requires the dynamic exponent z, which governs how *temporal*
   correlations grow with system size: τ_c ~ L^z at the transition density. This
   connects to the kinetics of the transition and distinguishes GoL from known
   universality classes (DP has z ≈ 1.76 in 2D).

## Physical Context

In equilibrium statistical mechanics, the dynamic exponent z links spatial and
temporal divergences at a critical point:

$$\xi_t \sim \xi^z \implies \tau_c \sim \xi^z \sim L^z \quad \text{at } \rho_0 = \rho_c$$

For a continuous (second-order) transition in the directed percolation (DP)
universality class: z ≈ 1.76 (2D). For a first-order transition, no universal z
exists in the same sense — τ_c behaviour reflects the nucleation timescale rather
than critical slowing-down.

Level 4 found that the GoL extinction transition is first-order (Binder cumulant
shows deep minima, not crossings; γ/ν ≈ 0.28 ≪ DP prediction of 0.90). Level 5
tests whether z is also inconsistent with DP, which would complete the evidence
that GoL does not belong to the DP universality class.

## Method

### 1. Extended Finite-Size Sweep

Simulations are run across **5 system sizes** (L = 32, 64, 128, 256, 512) with the
same density grid used in Level 4 (fine resolution near ρ_c ≈ 0.85):

| Parameter             | Default | Quick mode | Full mode |
| --------------------- | ------- | ---------- | --------- |
| Grid sizes L          | 32–512  | 32–128     | 32–1024   |
| Initial densities     | 21 pts  | 8 pts      | 21 pts    |
| Samples per (L, ρ₀)   | 30      | 15         | 50        |
| Steps                 | 500     | 300        | 500       |
| Total simulations     | 3,150   | 360        | 6,300     |

The sweep collects **all Level 4 observables** (ρ_final, activity, survival
probability, Binder cumulant, susceptibility, spatial correlation length ξ) plus
**temporal observables**:

### 2. Temporal Autocorrelation

For each simulation, the steady-state activity time series a(t) is saved. The
**temporal autocorrelation function** is computed via FFT (Wiener-Khinchin theorem):

$$C_t(\tau) = \frac{\langle \delta a(t)\,\delta a(t+\tau)\rangle}{\mathrm{Var}(a)}, \quad \delta a = a - \langle a\rangle$$

Only lags τ = 0, …, T/2 are used (the reliable half of the circular ACF).

### 3. Temporal Correlation Time τ_c

The correlation time is extracted by fitting an exponential decay to the
positive-correlation tail of C_t(τ):

$$C_t(\tau) \approx \exp(-\tau / \tau_c)$$

A log-linear least-squares fit is used. Runs where C_t falls below zero before a
reliable fit can be made return τ_c = 0.

### 4. Dynamic Exponent z

At the transition density ρ_c, the ensemble-averaged τ_c(L) is fit in log-log
space to extract z:

$$\tau_c(\rho_c, L) \sim L^z$$

This is compared to:
- **2D Directed Percolation**: z = 1.76
- **First-order** (nucleation-controlled): z expected to be anomalous or undefined

### 5. ξ/L Convergence

For active-phase densities (ρ₀ ≤ 0.80), the ratio ξ/L is tabulated for each L.
Convergence to a constant → SOC. Decay toward zero → finite intrinsic ξ.

### 6. Data Collapse

The scaling collapse tests whether ξ/L is a universal function of the scaled
variable (ρ₀ − ρ_c)·L^(1/ν):

$$\frac{\xi}{L} = \mathcal{F}\!\left[(ρ_0 - \rho_c) \cdot L^{1/\nu}\right]$$

A clean collapse would indicate the existence of a universal scaling function near
ρ_c. For a first-order transition, the collapse is expected to be imperfect.

## Figures

| Figure                  | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| `xi_over_L.png`         | ξ/L vs ρ₀ for each L — SOC convergence test (key result)              |
| `temporal_autocorr.png` | C_t(τ) at several ρ₀ values for the largest L, on semi-log scale      |
| `correlation_time.png`  | τ_c vs ρ₀ for each L with error bars                                  |
| `dynamic_exponent.png`  | log-log τ_c(ρ_c) vs L with z fit and DP reference (z = 1.76)         |
| `data_collapse.png`     | ξ/L vs (ρ₀ − ρ_c)·L^(1/ν), all L overlaid — tests scaling hypothesis |
| `xi_L_ratio_loglog.png` | log-log ξ vs L at active-phase densities; slope = 1 would confirm SOC |
| `summary_panel.png/pdf` | Publication-quality 2×3 panel                                          |

## Usage

```bash
# Default (L = 32, 64, 128, 256, 512; 30 samples; 500 steps)
python level_5/run.py

# Quick test (L = 32, 64, 128; 15 samples; 300 steps — ~5 min)
python level_5/run.py --quick

# Full run (adds L = 1024; 50 samples — compute-intensive)
python level_5/run.py --full

# Override critical parameters from Level 4
python level_5/run.py --rho_c 0.87 --nu 0.5

# Specify grid sizes manually
python level_5/run.py --grid_sizes 64 128 256 512 1024
```

## Expected Results

### ξ/L Convergence

| Outcome | Interpretation |
| ------- | -------------- |
| ξ/L → constant (L: 32→512) | True SOC: active phase is intrinsically scale-free |
| ξ/L → 0 slowly | Finite intrinsic ξ; the "SOC" appearance is a finite-size artefact |
| ξ/L non-monotone | Complex behaviour; ξ may depend on whether ρ₀ is above/below some crossover |

### Dynamic Exponent z

| Outcome | Interpretation |
| ------- | -------------- |
| z ≈ 1.76 | Consistent with DP universality (would conflict with first-order finding) |
| z ≫ 1.76 or z ≈ 0 | Non-standard dynamics; nucleation-controlled, not critical slowing-down |
| z not determined | τ_c too small or constant relative to L — transition is abrupt, not critical |

### Data Collapse Quality

| Outcome | Interpretation |
| ------- | -------------- |
| Clean collapse | Consistent with a continuous transition with exponent ν |
| Poor collapse | First-order transition (expected, consistent with Level 4) |

## Key Physics Summary

Level 5, combined with Level 4, completes the finite-size scaling picture:

- **Spatial FSS** (Level 4): first-order Binder cumulant, ξ ∝ L in active phase, γ/ν ≈ 0.28
- **Temporal FSS** (Level 5): dynamic exponent z, temporal correlation time τ_c(L), ξ/L convergence

Together these establish whether the GoL active phase is:

1. A **SOC state** (ξ/L converges, z well-defined but ≠ DP)
2. A **large-but-finite correlated state** (ξ/L → 0, z undefined or 0)
3. Or something more exotic

This distinction is the final empirical ingredient needed before the theoretical
framework of Level 6 can be written.

## Connection to the Project Arc

```
Level 1  Phase discovery (UMAP + HDBSCAN)
Level 2  Chaos confirmation (positive Lyapunov exponent)
Level 3  Rule-space generalisation (Wolfram classes)
Level 4  Spatial FSS — first-order extinction, xi ~ L
Level 5  Temporal FSS — z exponent, xi/L convergence      ← YOU ARE HERE
Level 6  Rule-space FSS comparison (other CAs)
Level 7  Mean-field theoretical framework
```
