"""
Level 3 — Rule-space exploration.

For a 2-state, Moore-neighbourhood cellular automaton there are 9 possible
neighbour counts (0–8).  An "outer totalistic" rule is defined by specifying
which counts cause birth (B) and which allow survival (S).  This gives a
discrete rule space of 2^9 × 2^9 = 262,144 rules.

This module:
  1. Samples rules from this space.
  2. For each rule, runs a small ensemble of simulations at a few initial
     densities, extracts summary features, and averages them.
  3. Computes Langton's λ parameter for each rule.
  4. Builds a feature matrix over rules for downstream UMAP + clustering.
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from common.config import FeatureConfig
from common.features import extract_features
from common.simulator import random_initial, run, step

# ── Langton's λ parameter ─────────────────────────────────────────────────


def langton_lambda(birth: List[int], survive: List[int]) -> float:
    """
    Langton's λ parameter: fraction of (state, neighbourhood) entries
    that map to the alive state.
    """
    return (len(birth) + len(survive)) / 18.0


# ── Rule encoding ─────────────────────────────────────────────────────────


def rule_to_string(birth: List[int], survive: List[int]) -> str:
    """Human-readable rule string, e.g. 'B3/S23'."""
    return f"B{''.join(map(str, sorted(birth)))}/S{''.join(map(str, sorted(survive)))}"


def rule_to_id(birth: List[int], survive: List[int]) -> int:
    """Encode a rule as a single integer (0–262143)."""
    b_bits = sum(1 << n for n in birth)
    s_bits = sum(1 << n for n in survive)
    return (b_bits << 9) | s_bits


def id_to_rule(rule_id: int) -> Tuple[List[int], List[int]]:
    """Decode a rule integer back to (birth, survive) lists."""
    s_bits = rule_id & 0x1FF
    b_bits = (rule_id >> 9) & 0x1FF
    birth = [n for n in range(9) if b_bits & (1 << n)]
    survive = [n for n in range(9) if s_bits & (1 << n)]
    return birth, survive


# ── Rule sampling strategies ──────────────────────────────────────────────


def sample_rules_random(
    n_rules: int, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Sample n_rules random (birth, survive) pairs uniformly."""
    rng = np.random.default_rng(seed)
    rules = []
    seen = set()
    while len(rules) < n_rules:
        b_bits = int(rng.integers(0, 512))
        s_bits = int(rng.integers(0, 512))
        key = (b_bits, s_bits)
        if key not in seen:
            seen.add(key)
            birth = [n for n in range(9) if b_bits & (1 << n)]
            survive = [n for n in range(9) if s_bits & (1 << n)]
            rules.append((birth, survive))
    return rules


def sample_rules_lambda_stratified(
    n_rules: int, n_lambda_bins: int = 10, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Sample rules stratified by λ so we get good coverage across the
    full range of rule complexity.
    """
    rng = np.random.default_rng(seed)
    bins = np.linspace(0, 1, n_lambda_bins + 1)
    per_bin = max(1, n_rules // n_lambda_bins)

    rules = []
    seen = set()

    for lo, hi in zip(bins[:-1], bins[1:]):
        count = 0
        attempts = 0
        while count < per_bin and attempts < 10000:
            b_bits = int(rng.integers(0, 512))
            s_bits = int(rng.integers(0, 512))
            birth = [n for n in range(9) if b_bits & (1 << n)]
            survive = [n for n in range(9) if s_bits & (1 << n)]
            lam = langton_lambda(birth, survive)
            key = (b_bits, s_bits)
            if lo <= lam < hi and key not in seen:
                seen.add(key)
                rules.append((birth, survive))
                count += 1
            attempts += 1

    return rules[:n_rules]


def sample_rules_critical(
    n_rules: int, n_lambda_bins: int = 20, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Sample rules with extra density near the critical region λ ∈ [0.3, 0.7]
    where the order-chaos phase transition is expected.
    """
    rng = np.random.default_rng(seed)

    n_critical = int(0.6 * n_rules)
    n_flanks = n_rules - n_critical

    rules = []
    seen = set()

    def _fill_range(lo, hi, target):
        n_b = max(3, n_lambda_bins // 2)
        edges = np.linspace(lo, hi, n_b + 1)
        per_bin = max(1, target // n_b)
        count_total = 0
        for a, b in zip(edges[:-1], edges[1:]):
            count = 0
            attempts = 0
            while count < per_bin and attempts < 10000 and count_total < target:
                b_bits = int(rng.integers(0, 512))
                s_bits = int(rng.integers(0, 512))
                birth = [n for n in range(9) if b_bits & (1 << n)]
                survive = [n for n in range(9) if s_bits & (1 << n)]
                lam = langton_lambda(birth, survive)
                key = (b_bits, s_bits)
                if a <= lam < b and key not in seen:
                    seen.add(key)
                    rules.append((birth, survive))
                    count += 1
                    count_total += 1
                attempts += 1

    _fill_range(0.3, 0.7, n_critical)
    _fill_range(0.0, 0.3, n_flanks // 2)
    _fill_range(0.7, 1.0, n_flanks - n_flanks // 2)

    return rules[:n_rules]


# ── Damage spreading (Lyapunov-like metric) ──────────────────────────────


def _damage_spreading(
    birth: List[int],
    survive: List[int],
    grid_size: int,
    n_steps: int,
    boundary: str,
    rng: np.random.Generator,
    n_trials: int = 3,
) -> Dict[str, float]:
    """
    Measure damage spreading by running pairs of simulations that differ
    by a single cell flip and tracking Hamming-distance divergence.
    """
    exponents = []
    saturations = []

    for _ in range(n_trials):
        ic = random_initial(grid_size, 0.5, rng)
        ic_pert = ic.copy()
        r, c = int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size))
        ic_pert[r, c] = 1 - ic_pert[r, c]

        g1, g2 = ic.copy(), ic_pert.copy()
        N = grid_size * grid_size
        hamming = np.empty(n_steps)

        for t in range(n_steps):
            g1 = step(g1, birth, survive, boundary)
            g2 = step(g2, birth, survive, boundary)
            hamming[t] = np.sum(g1 != g2) / N

        saturations.append(float(hamming[-1]))

        growth_end = (
            int(np.argmax(hamming > 0.2)) if np.any(hamming > 0.2) else len(hamming)
        )
        growth_end = max(growth_end, 5)
        h_seg = hamming[:growth_end]

        if h_seg.max() > 1e-8:
            h_safe = np.maximum(h_seg, 1e-10)
            try:
                slope, _ = np.polyfit(np.arange(len(h_safe)), np.log(h_safe), 1)
                exponents.append(float(slope))
            except (np.linalg.LinAlgError, ValueError):
                exponents.append(0.0)
        else:
            exponents.append(0.0)

    return {
        "damage_spreading_rate": float(np.mean(exponents)),
        "damage_saturation": float(np.mean(saturations)),
    }


# ── Per-rule feature extraction ───────────────────────────────────────────


def characterise_rule(
    birth: List[int],
    survive: List[int],
    grid_size: int = 50,
    n_steps: int = 300,
    densities: List[float] = None,
    n_samples: int = 5,
    boundary: str = "wrap",
    feat_cfg: FeatureConfig = None,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Run a small ensemble of simulations for one rule and return an
    averaged feature vector that characterises the rule's dynamics.
    """
    if densities is None:
        densities = [0.2, 0.4, 0.6]
    if feat_cfg is None:
        feat_cfg = FeatureConfig(steady_state_window=30)
    if rng is None:
        rng = np.random.default_rng(0)

    all_feats = []
    for d in densities:
        for _ in range(n_samples):
            ic = random_initial(grid_size, d, rng)
            sim = run(ic, birth, survive, n_steps, boundary)
            f = extract_features(sim, d, feat_cfg, n_steps)
            all_feats.append(f)

    keys = list(all_feats[0].keys())
    avg = {}
    for k in keys:
        vals = [f[k] for f in all_feats]
        avg[k] = float(np.mean(vals))
        avg[f"{k}_std"] = float(np.std(vals))

    dmg = _damage_spreading(birth, survive, grid_size, n_steps, boundary, rng)
    avg.update(dmg)

    avg["langton_lambda"] = langton_lambda(birth, survive)
    avg["n_birth"] = float(len(birth))
    avg["n_survive"] = float(len(survive))
    avg["rule_id"] = float(rule_to_id(birth, survive))

    return avg


# ── Full rule sweep ───────────────────────────────────────────────────────


def rule_sweep(
    rules: List[Tuple[List[int], List[int]]],
    grid_size: int = 50,
    n_steps: int = 300,
    densities: List[float] = None,
    n_samples: int = 5,
    boundary: str = "wrap",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[str], np.ndarray, List[Dict]]:
    """
    Characterise all rules and build a feature matrix.
    """
    rng = np.random.default_rng(seed)
    feat_cfg = FeatureConfig(steady_state_window=min(30, n_steps // 5))

    all_feats = []
    rule_info = []

    iterator = tqdm(rules, desc="Rule sweep", disable=not verbose)
    for birth, survive in iterator:
        rstr = rule_to_string(birth, survive)
        lam = langton_lambda(birth, survive)
        iterator.set_postfix(rule=rstr, lam=f"{lam:.2f}")

        feats = characterise_rule(
            birth,
            survive,
            grid_size,
            n_steps,
            densities,
            n_samples,
            boundary,
            feat_cfg,
            rng,
        )
        all_feats.append(feats)
        rule_info.append(
            {"birth": birth, "survive": survive, "rule_str": rstr, "lambda": lam}
        )

    feature_names = list(all_feats[0].keys())
    matrix = np.array(
        [[f[k] for k in feature_names] for f in all_feats], dtype=np.float64
    )
    return feature_names, matrix, rule_info
