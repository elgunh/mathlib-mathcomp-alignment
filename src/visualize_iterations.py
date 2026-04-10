"""Visualisations for the iterative alignment (Deliverable 2)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITER_MATCHES = os.path.join("outputs", "iterative_matches.csv")
PROP_LOG = os.path.join("outputs", "propagation_log.csv")
D1_MATCHES = os.path.join("outputs", "candidate_matches.csv")
FIG_DIR = os.path.join("outputs", "figures")


def plot_convergence(log_df):
    """Number of anchored modules per round."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(log_df["round"], log_df["total_anchors"], "o-", color="#4C72B0",
            linewidth=2, markersize=8)
    ax.fill_between(log_df["round"], 0, log_df["total_anchors"],
                    alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Round")
    ax.set_ylabel("Anchored modules (cumulative)")
    ax.set_title("Iterative Alignment Convergence")
    ax.set_xticks(log_df["round"].astype(int))
    ax.set_ylim(bottom=0, top=110)
    ax.axhline(106, color="gray", linestyle="--", alpha=0.5, label="Total (106)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis_iter] Saved {path}")


def plot_score_improvement(iter_df, d1_df):
    """Histogram comparing D1 combined_score vs D2 final_score (top-1)."""
    top1_d2 = iter_df[iter_df["rank"] == 1].copy()
    top1_d1 = d1_df[d1_df["rank"] == 1].copy()

    merged = top1_d1[["mathcomp_module", "combined_score"]].merge(
        top1_d2[["mathcomp_module", "final_score"]], on="mathcomp_module", how="inner"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(0, 0.7, 30)
    ax.hist(merged["combined_score"], bins=bins, alpha=0.6, label="D1 (with graph)",
            color="#C44E52", edgecolor="black")
    ax.hist(merged["final_score"], bins=bins, alpha=0.6, label="D2 (iterative)",
            color="#4C72B0", edgecolor="black")
    ax.set_xlabel("Top-1 Score")
    ax.set_ylabel("Module count")
    ax.set_title("Score Distribution: D1 vs D2")
    ax.legend()

    ax = axes[1]
    delta = merged["final_score"] - merged["combined_score"]
    colors = ["#55A868" if d > 0.001 else "#C44E52" if d < -0.001 else "#999999"
              for d in delta]
    order = delta.argsort()
    ax.bar(range(len(delta)), delta.iloc[order], color=[colors[i] for i in order],
           edgecolor="none", width=1.0)
    ax.set_xlabel("Modules (sorted)")
    ax.set_ylabel("Score change (D2 - D1)")
    ax.set_title("Per-Module Score Change")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "before_after.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis_iter] Saved {path}")


def plot_tier_breakdown(iter_df):
    """Pie chart of confidence tiers."""
    top1 = iter_df[iter_df["rank"] == 1]
    tiers = top1["confidence_tier"].value_counts()

    tier_order = ["HIGH", "MEDIUM", "MEDIUM-LOW", "LOW"]
    colors_map = {"HIGH": "#55A868", "MEDIUM": "#4C72B0",
                  "MEDIUM-LOW": "#DDC553", "LOW": "#C44E52"}
    sizes = [tiers.get(t, 0) for t in tier_order]
    labels = [f"{t} ({s})" for t, s in zip(tier_order, sizes)]
    colors = [colors_map[t] for t in tier_order]
    sizes_nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([x[0] for x in sizes_nonzero],
           labels=[x[1] for x in sizes_nonzero],
           colors=[x[2] for x in sizes_nonzero],
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title("Confidence Tier Breakdown (Iterative Alignment)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "tier_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis_iter] Saved {path}")


def plot_low_confidence_comparison(iter_df, d1_df):
    """Compare D1-LOW modules to their D2 scores."""
    top1_d1 = d1_df[d1_df["rank"] == 1].copy()
    top1_d2 = iter_df[iter_df["rank"] == 1].copy()

    d1_sorted = top1_d1.sort_values("combined_score")
    low_modules = d1_sorted.head(15)["mathcomp_module"].tolist()

    merged = top1_d1[top1_d1["mathcomp_module"].isin(low_modules)][
        ["mathcomp_module", "combined_score"]
    ].merge(
        top1_d2[top1_d2["mathcomp_module"].isin(low_modules)][
            ["mathcomp_module", "final_score"]
        ],
        on="mathcomp_module",
    ).sort_values("combined_score")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width/2, merged["combined_score"], width, label="D1",
           color="#C44E52", edgecolor="black")
    ax.bar(x + width/2, merged["final_score"], width, label="D2 (iterative)",
           color="#4C72B0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["mathcomp_module"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 Score")
    ax.set_title("Lowest-Confidence Modules: D1 vs D2")
    ax.legend()
    ax.axhline(0.35, color="green", linestyle=":", alpha=0.6, label="anchor threshold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "low_confidence_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis_iter] Saved {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    log_df = pd.read_csv(PROP_LOG)
    iter_df = pd.read_csv(ITER_MATCHES)

    plot_convergence(log_df)
    plot_tier_breakdown(iter_df)

    if os.path.exists(D1_MATCHES):
        d1_df = pd.read_csv(D1_MATCHES)
        plot_score_improvement(iter_df, d1_df)
        plot_low_confidence_comparison(iter_df, d1_df)
    else:
        print("[vis_iter] D1 candidate_matches.csv not found; skipping comparison plots.")

    print("[vis_iter] All iteration figures saved.")


if __name__ == "__main__":
    main()
