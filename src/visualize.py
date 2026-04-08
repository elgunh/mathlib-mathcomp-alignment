"""Generate visualisations of the alignment results."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

CANDIDATES = os.path.join("outputs", "candidate_matches.csv")
ALIGNMENT = os.path.join("outputs", "alignment_matrix.npz")
MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
FIG_DIR = os.path.join("outputs", "figures")


def plot_score_distribution(top1: pd.DataFrame):
    """Histogram of top-1 combined scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(top1["combined_score"], bins=25, edgecolor="black", alpha=0.7,
            color="#4C72B0")
    ax.axvline(top1["combined_score"].median(), color="red", linestyle="--",
               label=f"median = {top1['combined_score'].median():.3f}")
    ax.axvline(0.30, color="green", linestyle=":", label="high-confidence threshold (0.30)")
    ax.set_xlabel("Combined Score (top-1 match)")
    ax.set_ylabel("Number of MathComp modules")
    ax.set_title("Distribution of Top-1 Alignment Scores")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def plot_cluster_barchart(top1: pd.DataFrame):
    """Bar chart: count of 'good' matches (score >= 0.25) per cluster."""
    threshold = 0.25
    cluster_counts = (
        top1.groupby("mathcomp_cluster")
        .apply(lambda g: pd.Series({
            "good": (g["combined_score"] >= threshold).sum(),
            "total": len(g),
        }))
        .reset_index()
    )
    cluster_counts = cluster_counts.sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cluster_counts))
    width = 0.35
    ax.bar(x - width/2, cluster_counts["total"], width, label="Total modules",
           color="#A1C9F4", edgecolor="black")
    ax.bar(x + width/2, cluster_counts["good"], width,
           label=f"Score >= {threshold}", color="#4C72B0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_counts["mathcomp_cluster"], rotation=30, ha="right")
    ax.set_ylabel("Number of modules")
    ax.set_title(f"Alignment Quality by MathComp Cluster (threshold = {threshold})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cluster_barchart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def plot_heatmap():
    """Heatmap of the alignment matrix, grouped by MathComp cluster."""
    if not os.path.exists(ALIGNMENT):
        print("[visualize] No alignment matrix found, skipping heatmap")
        return

    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)

    mat = sparse.load_npz(ALIGNMENT).toarray()

    cluster_order = mc.sort_values("cluster").index.tolist()
    mat_sorted = mat[cluster_order, :]

    top_ml_indices = set()
    for i in range(mat_sorted.shape[0]):
        top_j = np.argsort(mat_sorted[i])[-5:]
        top_ml_indices.update(top_j.tolist())
    top_ml_indices = sorted(top_ml_indices)

    if len(top_ml_indices) > 200:
        top_ml_indices = top_ml_indices[:200]

    sub_mat = mat_sorted[:, top_ml_indices]

    mc_labels = [mc["module_id"].iloc[idx] for idx in cluster_order]
    ml_labels = [ml["module_name"].iloc[idx].split(".")[-1] for idx in top_ml_indices]

    fig, ax = plt.subplots(figsize=(20, 14))
    sns.heatmap(sub_mat, ax=ax, cmap="YlOrRd", vmin=0, vmax=0.5,
                xticklabels=False,
                yticklabels=mc_labels)
    ax.set_xlabel(f"Top Mathlib modules (n={len(top_ml_indices)})")
    ax.set_ylabel("MathComp modules (grouped by cluster)")
    ax.set_title("Alignment Matrix Heatmap (MathComp x top Mathlib candidates)")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "alignment_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def plot_signal_contributions(df: pd.DataFrame):
    """Stacked bar showing signal contributions for top-1 matches."""
    top1 = df[df["rank"] == 1].copy()
    top1 = top1.sort_values("combined_score", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top1))
    w_name, w_text, w_cat, w_graph = 0.30, 0.40, 0.10, 0.20

    name_contrib = top1["name_score"] * w_name
    text_contrib = top1["text_score"] * w_text
    cat_contrib = top1["category_score"] * w_cat
    graph_contrib = top1["graph_score"] * w_graph

    ax.bar(x, name_contrib, label="Name (0.30)", color="#4C72B0")
    ax.bar(x, text_contrib, bottom=name_contrib, label="Text (0.40)", color="#55A868")
    ax.bar(x, cat_contrib, bottom=name_contrib + text_contrib,
           label="Category (0.10)", color="#C44E52")
    ax.bar(x, graph_contrib, bottom=name_contrib + text_contrib + cat_contrib,
           label="Graph (0.20)", color="#8172B2")

    ax.set_xticks(x)
    ax.set_xticklabels(top1["mathcomp_module"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Combined score (weighted)")
    ax.set_title("Signal Contributions for Top-30 Best Matches")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "signal_contributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("[visualize] Loading candidate matches...")
    df = pd.read_csv(CANDIDATES)
    top1 = df[df["rank"] == 1]

    plot_score_distribution(top1)
    plot_cluster_barchart(top1)
    plot_heatmap()
    plot_signal_contributions(df)
    print("[visualize] All figures saved.")


if __name__ == "__main__":
    main()
