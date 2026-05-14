"""
Mean-field rate equations for Life-like rules on a Moore neighbourhood (n=8).

Cells are treated as independent Bernoulli(ρ); neighbour counts are Binomial(8, ρ).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_COMB8 = np.array([math.comb(8, k) for k in range(9)], dtype=float)
_K = np.arange(9, dtype=float)

# (ρ, stability) where stability is "stable" / "unstable" / "neutral"
FixedPoint = Tuple[float, str]


def _binom_pmf(k: int, n: int, rho: float) -> float:
    if rho <= 0.0:
        return 1.0 if k == 0 else 0.0
    if rho >= 1.0:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * (rho**k) * ((1.0 - rho) ** (n - k))


def _binom_pmf_all(rho: np.ndarray) -> np.ndarray:
    """P(k | ρ) for k = 0..8; shape (..., 9)."""
    r = np.asarray(rho, dtype=float)
    r = np.clip(r, 0.0, 1.0)
    r_exp = r[..., None]
    return _COMB8 * (r_exp**_K) * ((1.0 - r_exp) ** (8.0 - _K))


def birth_rate(rho: float, birth: Sequence[int]) -> float:
    """(1−ρ) × Σ_{k ∈ B} C(8,k) ρ^k (1−ρ)^(8−k)."""
    rho = float(np.clip(rho, 0.0, 1.0))
    bset = set(int(x) for x in birth)
    return (1.0 - rho) * sum(_binom_pmf(k, 8, rho) for k in bset)


def death_rate(rho: float, survive: Sequence[int]) -> float:
    """ρ × Σ_{k ∉ S} C(8,k) ρ^k (1−ρ)^(8−k)."""
    rho = float(np.clip(rho, 0.0, 1.0))
    sset = set(int(x) for x in survive)
    return rho * sum(_binom_pmf(k, 8, rho) for k in range(9) if k not in sset)


def rate_equation_array(
    rho_arr: np.ndarray, birth: Sequence[int], survive: Sequence[int]
) -> np.ndarray:
    """f(ρ) on a grid (vectorized)."""
    rho = np.asarray(rho_arr, dtype=float)
    rho_flat = rho.ravel()
    pmf = _binom_pmf_all(rho_flat)
    bset = sorted(set(int(x) for x in birth))
    sset = set(int(x) for x in survive)
    idx_d = [k for k in range(9) if k not in sset]
    birth_sum = pmf[:, bset].sum(axis=1) if bset else np.zeros_like(rho_flat)
    death_sum = pmf[:, idx_d].sum(axis=1)
    f = (1.0 - rho_flat) * birth_sum - rho_flat * death_sum
    return f.reshape(rho.shape)


def rate_equation(rho: float, birth: Sequence[int], survive: Sequence[int]) -> float:
    """f(ρ) = dρ/dt in the mean-field ODE."""
    return float(rate_equation_array(np.array([rho], dtype=float), birth, survive)[0])


def _derivative_f(
    rho: float, birth: Sequence[int], survive: Sequence[int], eps: float = 1e-7
) -> float:
    r = float(np.clip(rho, eps, 1.0 - eps))
    return (
        rate_equation(r + eps, birth, survive) - rate_equation(r - eps, birth, survive)
    ) / (2.0 * eps)


def _bisect_root(
    birth: Sequence[int],
    survive: Sequence[int],
    a: float,
    b: float,
    fa: float,
    fb: float,
    tol: float = 1e-14,
    max_iter: int = 80,
) -> float:
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return 0.5 * (a + b)
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = rate_equation(mid, birth, survive)
        if abs(hi - lo) < tol or abs(fm) < tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def find_fixed_points(
    birth: Sequence[int],
    survive: Sequence[int],
    n_pts: int = 10000,
    f_tol: float = 1e-9,
) -> List[FixedPoint]:
    """
    Find ρ* with f(ρ*) ≈ 0 on [0, 1], classify stability via f'(ρ*).

    Stable: f' < 0; unstable: f' > 0; neutral: |f'| very small.
    """
    rho_grid = np.linspace(0.0, 1.0, n_pts)
    f_grid = rate_equation_array(rho_grid, birth, survive)

    roots: List[float] = []
    if abs(float(f_grid[0])) < f_tol:
        roots.append(0.0)
    if abs(float(f_grid[-1])) < f_tol:
        roots.append(1.0)

    for i in range(n_pts - 1):
        r0, r1 = float(rho_grid[i]), float(rho_grid[i + 1])
        f0, f1 = float(f_grid[i]), float(f_grid[i + 1])
        if f0 * f1 < 0.0:
            roots.append(_bisect_root(birth, survive, r0, r1, f0, f1))

    roots_sorted = sorted(roots)
    merged: List[float] = []
    for r in roots_sorted:
        if not merged or abs(r - merged[-1]) > 1e-7:
            merged.append(r)

    out: List[FixedPoint] = []
    for r in merged:
        fp = _derivative_f(r, birth, survive)
        if fp < -1e-8:
            stab = "stable"
        elif fp > 1e-8:
            stab = "unstable"
        else:
            stab = "neutral"
        out.append((r, stab))
    return out


def landau_free_energy(
    rho_arr: np.ndarray, birth: Sequence[int], survive: Sequence[int]
) -> np.ndarray:
    """
    Φ(ρ) = −∫₀^ρ f(ρ') dρ' (trapezoid cumulative integral on rho_arr grid).
    """
    rho_arr = np.asarray(rho_arr, dtype=float)
    if rho_arr.size < 2:
        return np.zeros_like(rho_arr)
    f_arr = rate_equation_array(rho_arr, birth, survive)
    d_rho = rho_arr[1] - rho_arr[0]
    mid = 0.5 * (f_arr[:-1] + f_arr[1:])
    cum = np.concatenate([[0.0], np.cumsum(mid * d_rho)])
    return -cum


def integrate_mean_field(
    rho0: float,
    birth: Sequence[int],
    survive: Sequence[int],
    t_max: float = 200.0,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Euler integration of dρ/dt = f(ρ); returns (t, rho(t))."""
    n_steps = int(math.ceil(t_max / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    rho = np.empty(n_steps, dtype=float)
    rho[0] = float(np.clip(rho0, 0.0, 1.0))
    for i in range(1, n_steps):
        r = rho[i - 1]
        rho[i] = r + dt * rate_equation(r, birth, survive)
        rho[i] = float(np.clip(rho[i], 0.0, 1.0))
    return t, rho


def critical_density_mf(birth: Sequence[int], survive: Sequence[int]) -> float:
    """
    Spinodal density: smallest unstable fixed point in (0, 1).

    Returns NaN if there is no such point.
    """
    fps = find_fixed_points(birth, survive)
    unst = [r for r, s in fps if s == "unstable" and 0.0 < r < 1.0]
    if not unst:
        return float("nan")
    return min(unst)


def classify_transition_mf(birth: Sequence[int], survive: Sequence[int]) -> str:
    """Coarse label from fixed-point structure."""
    fps = find_fixed_points(birth, survive)
    st = [r for r, s in fps if s == "stable"]
    unst = [r for r, s in fps if s == "unstable"]
    if unst and len(st) >= 2:
        return "first_order (MF bistability)"
    if len(fps) <= 1:
        return "trivial / monostable"
    return "other / marginal"


def parse_rule_string(rule_str: str) -> Tuple[List[int], List[int]]:
    """Parse 'B368/S245678' into birth and survive neighbour-count lists."""
    s = rule_str.strip().upper()
    if "/" not in s:
        raise ValueError(f"expected B.../S..., got {rule_str!r}")
    left, right = s.split("/", 1)
    if not left.startswith("B") or not right.startswith("S"):
        raise ValueError(f"expected B.../S..., got {rule_str!r}")
    b_digits, s_digits = left[1:], right[1:]
    birth = [int(ch) for ch in b_digits] if b_digits else []
    survive = [int(ch) for ch in s_digits] if s_digits else []
    return birth, survive
