#!/usr/bin/env python
"""
level_3/run.py  –  Rule-Space Phase Classification driver
==========================================================

Samples the space of outer-totalistic 2-state CA rules, characterises each
rule by running a small simulation ensemble, and uses unsupervised learning
(UMAP + HDBSCAN) to classify rules into dynamical "universality classes"
corresponding to Wolfram's four CA regimes.

Usage (from project root)
-------------------------
    python level_3/run.py                   # defaults: 200 rules
    python level_3/run.py --quick           # fast test: 50 rules
    python level_3/run.py --n_rules 500     # more rules for higher resolution
"""

import argparse
import os
import sys
import time

# Allow running from the project root or from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from common.clustering import build_pipeline
from common.config import ClusteringConfig
from common.interpretation import feature_importance, print_interpretation_summary
from common.plotting import (
    ensure_dir,
    plot_feature_importance,
    plot_silhouette,
    plot_umap_clusters,
    plot_umap_embedding,
)
from level_3.rule_space import (
    langton_lambda,
    rule_sweep,
    rule_to_string,
    sample_rules_critical,
    sample_rules_lambda_stratified,
    sample_rules_random,
)

# ── Output paths ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "results")
DEFAULT_FIGURE_DIR = os.path.join(_HERE, "figures")


def parse_args():
    p = argparse.ArgumentParser(description="Level 3: Rule-space classification")
    p.add_argument("--n_rules", type=int, default=200, help="Number of rules to sample")
    p.add_argument(
        "--sampling",
        type=str,
        default="critical",
        choices=["random", "stratified", "critical"],
        help="Rule sampling strategy (default: critical-region focused)",
    )
    p.add_argument(
        "--grid_size", type=int, default=50, help="Grid side for per-rule sims"
    )
    p.add_argument("--n_steps", type=int, default=300, help="Steps per simulation")
    p.add_argument(
        "--n_samples", type=int, default=5, help="Replicates per (rule, density)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 50 rules, 40×40 grid, 150 steps, 3 samples",
    )
    return p.parse_args()


# ── Level-3 specific plots ────────────────────────────────────────────────


def plot_lambda_vs_feature(rule_info, feat_names, feat_matrix, feature_name, fig_dir):
    """Scatter: Langton λ vs a chosen feature, one point per rule."""
    ensure_dir(fig_dir)
    lambdas = np.array([r["lambda"] for r in rule_info])
    idx = feat_names.index(feature_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(lambdas, feat_matrix[:, idx], s=12, alpha=0.5)
    ax.set_xlabel("Langton $\\lambda$")
    ax.set_ylabel(feature_name.replace("_", " "))
    ax.set_title(f"Rule space: $\\lambda$ vs {feature_name.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, f"lambda_vs_{feature_name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    return fig


def plot_rule_umap_lambda(embedding, rule_info, labels, fig_dir):
    """UMAP coloured by Langton λ with cluster labels overlaid."""
    ensure_dir(fig_dir)
    lambdas = np.array([r["lambda"] for r in rule_info])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=lambdas, cmap="coolwarm", s=15, alpha=0.7
    )
    plt.colorbar(sc, ax=axes[0], label="Langton $\\lambda$")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title("Rule space coloured by $\\lambda$")

    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        colour = "lightgrey" if lab < 0 else cmap(i)
        name = "extinct" if lab == -2 else ("noise" if lab == -1 else f"class {lab}")
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colour],
            s=15,
            alpha=0.7,
            label=name,
        )
    axes[1].legend(fontsize=7, markerscale=2)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].set_title("Rule classes (HDBSCAN)")

    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "rule_umap_lambda_clusters.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    return fig


def plot_rule_class_summary(feat_names, feat_matrix, labels, rule_info, fig_dir):
    """Bar charts of mean λ, density, activity, and settling per class."""
    ensure_dir(fig_dir)
    unique = sorted([l for l in set(labels) if l >= 0])
    if not unique:
        return None

    lambdas = np.array([r["lambda"] for r in rule_info])
    metrics = {"Langton $\\lambda$": lambdas}
    for name in ["rho_final_mean", "activity", "settling_fraction"]:
        if name in feat_names:
            metrics[name.replace("_", " ")] = feat_matrix[:, feat_names.index(name)]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.colormaps.get_cmap("tab10").resampled(len(unique))

    for ax, (mname, mvals) in zip(axes, metrics.items()):
        means, stds, xlabels = [], [], []
        for lab in unique:
            mask = labels == lab
            means.append(mvals[mask].mean())
            stds.append(mvals[mask].std())
            xlabels.append(f"C{lab}")
        ax.bar(
            xlabels,
            means,
            yerr=stds,
            capsize=4,
            color=[colors(i) for i in range(len(unique))],
            alpha=0.7,
        )
        ax.set_ylabel(mname)
        ax.set_title(mname)

    fig.suptitle("Rule class summary", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "rule_class_summary.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    return fig


def plot_summary_panel(
    embedding, feat_names, feat_matrix, labels, rule_info, lambdas, fig_dir
):
    """Publication-quality 2×3 summary panel."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    unique_labels = sorted(set(labels))
    cmap_class = plt.colormaps.get_cmap("tab10").resampled(max(len(unique_labels), 1))
    label_colors = []
    for lab in labels:
        if lab == -2:
            label_colors.append("lightgrey")
        elif lab == -1:
            label_colors.append("silver")
        else:
            label_colors.append(cmap_class(unique_labels.index(lab)))

    sc = axes[0, 0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=lambdas,
        cmap="coolwarm",
        s=18,
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=axes[0, 0], label="$\\lambda$", shrink=0.8)
    axes[0, 0].set_xlabel("UMAP-1")
    axes[0, 0].set_ylabel("UMAP-2")
    axes[0, 0].set_title("(a) UMAP — Langton $\\lambda$")

    for lab in unique_labels:
        mask = labels == lab
        colour = "lightgrey" if lab < 0 else cmap_class(unique_labels.index(lab))
        name = "extinct" if lab == -2 else ("noise" if lab == -1 else f"class {lab}")
        axes[0, 1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colour],
            s=18,
            alpha=0.7,
            label=name,
            edgecolors="none",
        )
    axes[0, 1].legend(fontsize=7, markerscale=2, loc="best")
    axes[0, 1].set_xlabel("UMAP-1")
    axes[0, 1].set_ylabel("UMAP-2")
    axes[0, 1].set_title("(b) UMAP — rule classes")

    def _scatter_panel(ax, feat_name, ylabel, title, panel_label):
        vals = (
            feat_matrix[:, feat_names.index(feat_name)]
            if feat_name in feat_names
            else np.zeros(len(lambdas))
        )
        ax.scatter(lambdas, vals, c=label_colors, s=18, alpha=0.7, edgecolors="none")
        ax.set_xlabel("Langton $\\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({panel_label}) {title}")

    _scatter_panel(
        axes[0, 2],
        "damage_spreading_rate",
        "spreading rate",
        "$\\lambda$ vs damage spreading",
        "c",
    )
    _scatter_panel(
        axes[1, 0],
        "spatial_entropy",
        "block entropy",
        "$\\lambda$ vs spatial entropy",
        "d",
    )
    _scatter_panel(
        axes[1, 1],
        "rho_final_mean",
        "$\\rho_{\\mathrm{final}}$",
        "$\\lambda$ vs final density",
        "e",
    )
    _scatter_panel(
        axes[1, 2],
        "damage_saturation",
        "saturation",
        "$\\lambda$ vs damage saturation",
        "f",
    )

    fig.suptitle("Rule-Space Phase Classification — Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "summary_panel.png"), dpi=200, bbox_inches="tight"
    )
    fig.savefig(os.path.join(fig_dir, "summary_panel.pdf"), bbox_inches="tight")
    plt.close(fig)
    return fig


def assign_wolfram_regimes(labels, feat_names, feat_matrix, n_extinct=0):
    """
    Assign each ML cluster to a Wolfram-like dynamical regime.

    Returns regime_labels (0=I, 1=II, 2=III, 3=IV) and regime_names dict.
    """
    regime_labels = np.full_like(labels, -1)
    _get = lambda name: feat_names.index(name) if name in feat_names else None

    idx_dmg = _get("damage_spreading_rate")
    idx_per = _get("period")
    idx_rho = _get("rho_final_mean")

    regime_names = {
        -1: "unclassified",
        0: "I  (dead / frozen)",
        1: "II (ordered / static)",
        2: "III (chaotic)",
        3: "IV (complex / periodic)",
    }

    for lab in sorted(set(labels)):
        mask = labels == lab
        if lab == -2:
            regime_labels[mask] = 0
            continue
        if lab == -1:
            regime_labels[mask] = -1
            continue

        m_dmg = feat_matrix[mask, idx_dmg].mean() if idx_dmg is not None else 0
        m_per = feat_matrix[mask, idx_per].mean() if idx_per is not None else 0
        m_rho = feat_matrix[mask, idx_rho].mean() if idx_rho is not None else 0

        if m_rho < 0.05:
            regime_labels[mask] = 0
        elif m_rho > 0.9 and m_dmg < 0.05:
            regime_labels[mask] = 0
        elif m_per > 2:
            regime_labels[mask] = 3
        elif m_dmg < 0.08 and m_per >= 1:
            regime_labels[mask] = 1
        elif m_dmg > 0.12 and m_per == 0:
            regime_labels[mask] = 2
        elif m_dmg > 0.08:
            regime_labels[mask] = 2
        else:
            regime_labels[mask] = 1

    return regime_labels, regime_names


def plot_wolfram_panel(
    embedding,
    feat_names,
    feat_matrix,
    regime_labels,
    regime_names,
    rule_info,
    lambdas,
    fig_dir,
):
    """Publication-quality 2×2 Wolfram regime panel."""
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    regime_cmap = {
        0: "#1f77b4",
        1: "#2ca02c",
        2: "#d62728",
        3: "#ff7f0e",
        -1: "lightgrey",
    }
    colors = [regime_cmap.get(r, "grey") for r in regime_labels]

    for rid in sorted(set(regime_labels)):
        mask = regime_labels == rid
        axes[0, 0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=regime_cmap.get(rid, "grey"),
            s=22,
            alpha=0.7,
            label=regime_names.get(rid, "?"),
            edgecolors="none",
        )
    axes[0, 0].legend(fontsize=8, markerscale=2, loc="best")
    axes[0, 0].set_xlabel("UMAP-1")
    axes[0, 0].set_ylabel("UMAP-2")
    axes[0, 0].set_title("(a) Wolfram dynamical regimes")

    idx_dmg = (
        feat_names.index("damage_spreading_rate")
        if "damage_spreading_rate" in feat_names
        else None
    )
    if idx_dmg is not None:
        axes[0, 1].scatter(
            lambdas,
            feat_matrix[:, idx_dmg],
            c=colors,
            s=22,
            alpha=0.7,
            edgecolors="none",
        )
    axes[0, 1].set_xlabel("Langton $\\lambda$")
    axes[0, 1].set_ylabel("Damage spreading rate")
    axes[0, 1].set_title("(b) $\\lambda$ vs Lyapunov-like exponent")
    axes[0, 1].axhline(y=0.1, color="grey", ls="--", lw=0.8, alpha=0.5)

    idx_ent = (
        feat_names.index("spatial_entropy") if "spatial_entropy" in feat_names else None
    )
    if idx_ent is not None:
        axes[1, 0].scatter(
            lambdas,
            feat_matrix[:, idx_ent],
            c=colors,
            s=22,
            alpha=0.7,
            edgecolors="none",
        )
    axes[1, 0].set_xlabel("Langton $\\lambda$")
    axes[1, 0].set_ylabel("Spatial block entropy")
    axes[1, 0].set_title("(c) $\\lambda$ vs spatial entropy")

    n_bins = 8
    lam_edges = np.linspace(lambdas.min() - 0.01, lambdas.max() + 0.01, n_bins + 1)
    regime_ids = sorted([r for r in set(regime_labels) if r >= 0])
    bottom = np.zeros(n_bins)
    lam_centers = 0.5 * (lam_edges[:-1] + lam_edges[1:])
    bar_width = (lam_edges[1] - lam_edges[0]) * 0.85

    for rid in regime_ids:
        counts = np.zeros(n_bins)
        for i in range(n_bins):
            in_bin = (lambdas >= lam_edges[i]) & (lambdas < lam_edges[i + 1])
            counts[i] = ((regime_labels == rid) & in_bin).sum()
        axes[1, 1].bar(
            lam_centers,
            counts,
            width=bar_width,
            bottom=bottom,
            color=regime_cmap[rid],
            label=regime_names[rid],
            alpha=0.8,
        )
        bottom += counts

    axes[1, 1].set_xlabel("Langton $\\lambda$")
    axes[1, 1].set_ylabel("Rule count")
    axes[1, 1].set_title("(d) Phase prevalence vs $\\lambda$")
    axes[1, 1].legend(fontsize=7, loc="upper left")

    fig.suptitle("Wolfram-Class Interpretation of Rule Space", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "wolfram_regimes.png"), dpi=200, bbox_inches="tight"
    )
    fig.savefig(os.path.join(fig_dir, "wolfram_regimes.pdf"), bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if args.quick:
        args.n_rules = 50
        args.grid_size = 40
        args.n_steps = 150
        args.n_samples = 3

    fig_dir = DEFAULT_FIGURE_DIR
    out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Level 3: Rule-Space Phase Classification")
    print(f"{'='*60}")
    print(f" Rules:        {args.n_rules}  ({args.sampling} sampling)")
    print(f" Grid:         {args.grid_size} × {args.grid_size}")
    print(f" Steps:        {args.n_steps}")
    print(f" Samples/rule: {args.n_samples} × 3 densities = {args.n_samples * 3}")
    print(f" Total sims:   {args.n_rules * args.n_samples * 3}")
    print(f"{'='*60}\n")

    # ── 1. Sample rules ───────────────────────────────────────────────
    print("  Sampling rules ...")
    if args.sampling == "critical":
        rules = sample_rules_critical(args.n_rules, seed=args.seed)
    elif args.sampling == "stratified":
        rules = sample_rules_lambda_stratified(args.n_rules, seed=args.seed)
    else:
        rules = sample_rules_random(args.n_rules, seed=args.seed)
    print(f"  Got {len(rules)} rules\n")

    # ── 2. Characterise each rule ─────────────────────────────────────
    t0 = time.time()
    feat_names, feat_matrix, rule_info = rule_sweep(
        rules,
        grid_size=args.grid_size,
        n_steps=args.n_steps,
        densities=[0.2, 0.4, 0.6],
        n_samples=args.n_samples,
        seed=args.seed,
        verbose=True,
    )
    t_sweep = time.time() - t0
    print(f"\n  Rule sweep complete in {t_sweep:.1f}s")
    print(f"  {len(feat_names)} features per rule\n")

    np.savez(
        os.path.join(out_dir, "rule_features.npz"),
        names=feat_names,
        matrix=feat_matrix,
        rule_strings=[r["rule_str"] for r in rule_info],
        lambdas=[r["lambda"] for r in rule_info],
    )
    print(f"  Saved → {out_dir}/rule_features.npz")

    # ── 3. Cluster the rule space ─────────────────────────────────────
    print("\n  Clustering rule space ...")
    exclude = [
        "density_init",
        "density_init_std",
        "rule_id",
        "rule_id_std",
        "langton_lambda",
        "langton_lambda_std",
        "n_birth",
        "n_birth_std",
        "n_survive",
        "n_survive_std",
    ]
    clust_cfg = ClusteringConfig(
        umap_n_neighbors=min(15, max(2, len(rules) // 5)),
        hdbscan_min_cluster_size=max(5, len(rules) // 25),
        hdbscan_min_samples=3,
        hdbscan_selection_method="leaf",
    )
    cl = build_pipeline(
        feat_matrix, clust_cfg, feature_names=feat_names, exclude_features=exclude
    )

    labels_hdb = cl["labels_hdbscan"]
    n_hdb = (
        len(set(labels_hdb))
        - (1 if -1 in labels_hdb else 0)
        - (1 if -2 in labels_hdb else 0)
    )
    n_extinct = cl.get("n_extinct", 0)
    km = cl["kmeans_results"]
    best_k = km["best_k"]

    labels = labels_hdb if n_hdb >= 3 else km["best_labels"]
    method_used = "HDBSCAN (leaf)" if n_hdb >= 3 else f"k-means (k={best_k})"

    print(f"  Extinct rules: {n_extinct}")
    print(f"  HDBSCAN found {n_hdb} rule classes")
    print(
        f"  k-means best k = {best_k}  "
        f"(silhouette = {max(km['silhouette_scores']):.3f})"
    )
    print(f"  → Using {method_used}\n")

    # ── 4. Interpretation ─────────────────────────────────────────────
    lambdas_arr = np.array([r["lambda"] for r in rule_info])
    imp_df = feature_importance(feat_matrix, feat_names, labels)
    meta_feats = set(exclude)
    imp_phys = imp_df[~imp_df["feature"].isin(meta_feats)].reset_index(drop=True)

    print("  Top-5 discriminating features (η²):")
    for _, row in imp_phys.head(5).iterrows():
        print(f"    {row['feature']:30s}  η² = {row['eta_squared']:.3f}")

    unique_classes = sorted([l for l in set(labels) if l >= 0])
    print(f"\n  Rule class summary:")
    for lab in unique_classes:
        mask = labels == lab
        n = mask.sum()
        lam_mean = lambdas_arr[mask].mean()
        lam_std = lambdas_arr[mask].std()
        sigs = []
        for fname in [
            "damage_spreading_rate",
            "spatial_entropy",
            "period",
            "rho_final_mean",
        ]:
            if fname in feat_names:
                val = feat_matrix[mask, feat_names.index(fname)].mean()
                sigs.append(f"{fname}={val:.3f}")
        reps = [rule_info[idx]["rule_str"] for idx in np.where(mask)[0][:3]]
        print(f"    Class {lab}  (n={n}, λ = {lam_mean:.3f} ± {lam_std:.3f})")
        print(f"      Features: {', '.join(sigs)}")
        print(f"      Examples: {', '.join(reps)}")

    # ── 5. Wolfram regime assignment ──────────────────────────────────
    print(f"\n  Assigning Wolfram dynamical regimes ...")
    regime_labels, regime_names = assign_wolfram_regimes(
        labels, feat_names, feat_matrix, n_extinct
    )
    for rid in sorted(set(regime_labels)):
        if rid < 0:
            continue
        n_r = (regime_labels == rid).sum()
        print(f"    {regime_names[rid]:30s}: {n_r} rules")

    # ── 6. Figures ────────────────────────────────────────────────────
    print(f"\n  Generating figures ...")

    plot_rule_umap_lambda(cl["embedding"], rule_info, labels, fig_dir=fig_dir)
    plot_feature_importance(imp_phys, top_n=12, fig_dir=fig_dir)
    plot_silhouette(cl["kmeans_results"], fig_dir=fig_dir)
    plot_rule_class_summary(feat_names, feat_matrix, labels, rule_info, fig_dir=fig_dir)

    for obs in [
        "rho_final_mean",
        "activity",
        "settling_fraction",
        "n_clusters",
        "spatial_entropy",
        "temporal_entropy",
        "damage_spreading_rate",
        "damage_saturation",
        "period",
    ]:
        if obs in feat_names:
            plot_lambda_vs_feature(
                rule_info, feat_names, feat_matrix, obs, fig_dir=fig_dir
            )

    plot_summary_panel(
        cl["embedding"],
        feat_names,
        feat_matrix,
        labels,
        rule_info,
        lambdas_arr,
        fig_dir=fig_dir,
    )
    plot_wolfram_panel(
        cl["embedding"],
        feat_names,
        feat_matrix,
        regime_labels,
        regime_names,
        rule_info,
        lambdas_arr,
        fig_dir=fig_dir,
    )

    print(f"  Figures saved → {fig_dir}/\n")

    n_classes = len(unique_classes)
    print(f"{'='*60}")
    print(f" Done!  Level 3 results:")
    print(f"   • {len(rules)} rules characterised  ({method_used})")
    print(f"   • {n_classes} dynamical classes found (+ {n_extinct} extinct)")
    print(
        f"   • Wolfram regimes: "
        f"{', '.join(regime_names[r] for r in sorted(set(regime_labels)) if r >= 0)}"
    )
    print(f"   • Feature matrix: {out_dir}/rule_features.npz")
    print(f"   • Figures: {fig_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
