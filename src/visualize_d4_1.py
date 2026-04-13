"""Deliverable 4.1 figures.

1. d4_1_gold_audit.png      — Pie: FALSE_ERROR vs AMBIGUOUS vs CONFIRMED_ERROR
2. d4_1_concept_groups.png  — Top concept groups for 4 hard modules
3. d4_1_precision_comparison.png — D2 → D3.2 (v1/v2) → D4.1 bar chart
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

GOLD_V2_JSON = "data/processed/gold_standard_v2.json"
HIER_SCORES  = "data/processed/hier_group_scores.npz"
HIER_INDEX   = "data/processed/hier_group_index.json"
MC_MOD       = "data/processed/mathcomp_modules.csv"
OUT_DIR      = "outputs/figures"

os.makedirs(OUT_DIR, exist_ok=True)


def fig1_gold_audit(audit):
    verdicts = [d["verdict"] for d in audit.values()]
    counts = {
        "FALSE_ERROR\n(match correct,\ngold too narrow)": verdicts.count("FALSE_ERROR"),
        "AMBIGUOUS\n(adjacent but\nnot canonical)": verdicts.count("AMBIGUOUS"),
        "CONFIRMED_ERROR\n(genuinely wrong)": verdicts.count("CONFIRMED_ERROR"),
    }
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = list(counts.keys())
    sizes  = list(counts.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140,
        pctdistance=0.65,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.set_title("Gold standard audit of 18 D3.2 apparent errors\n"
                 "(3 false errors → gold_v2 adds 3 correct matches)", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "d4_1_gold_audit.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_d4_1] Saved {out}")


def fig2_concept_groups(mc_ids, mc_id_to_idx, hier_scores, group_keys):
    """Show top-5 concept groups for 4 hard modules (stacked bar)."""
    modules = ["bigop", "finset", "fraction", "ring_quotient"]
    gold_groups = {
        "bigop":          "Algebra.BigOperators",
        "finset":         "Data.Finset",
        "fraction":       "RingTheory.Localization",
        "ring_quotient":  "RingTheory.Ideal.Quotient",
    }
    n_top = 5

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Top-5 concept groups by aggregate base score\n"
                 "(green = gold group; red = wrong top group)", fontsize=10)

    for ax, mc_mid in zip(axes, modules):
        i = mc_id_to_idx.get(mc_mid)
        if i is None:
            ax.set_visible(False)
            continue
        row = hier_scores[i]
        top_idx = np.argsort(row)[-n_top:][::-1]
        top_groups = [(group_keys[gi], float(row[gi])) for gi in top_idx if row[gi] > 0]

        gold = gold_groups.get(mc_mid, "")
        labels = [g.replace(".", ".\n") for g, _ in top_groups]
        scores = [s for _, s in top_groups]
        colors = ["#2ecc71" if g == gold else "#e74c3c" if gi == 0 else "#85c1e9"
                  for gi, (g, _) in enumerate(top_groups)]

        bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Group score", fontsize=8)
        ax.set_title(f"`{mc_mid}`\n(gold: {gold.split('.')[-1]})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, sc in zip(bars, scores):
            ax.text(sc + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{sc:.3f}", va="center", fontsize=7)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2ecc71", label="Gold group"),
                        Patch(color="#e74c3c", label="Wrong top group"),
                        Patch(color="#85c1e9", label="Other top groups")],
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "d4_1_concept_groups.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz_d4_1] Saved {out}")


def fig3_precision_comparison():
    # D2 / D3.2 gold_v1 / D3.2 gold_v2 (D4.1 = same as D3.2)
    systems = [
        "D2\n(iterative)",
        "D3.2\n(gold v1)",
        "D3.2\n(gold v2\naudit-corrected)",
    ]
    p1 = [67.7, 71.0, 75.8]
    p5 = [79.0, 79.0, 80.6]

    x = np.arange(len(systems))
    w = 0.32
    c_p1 = ["#3498db", "#e67e22", "#27ae60"]
    c_p5 = ["#85c1e9", "#f0b27a", "#a9dfbf"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, p1, w, color=c_p1, label="P@1", edgecolor="white")
    ax.bar(x + w/2, p5, w, color=c_p5, label="P@5", edgecolor="white")

    for i, (v1, v5) in enumerate(zip(p1, p5)):
        ax.text(i - w/2, v1 + 0.4, f"{v1:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, v5 + 0.4, f"{v5:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.annotate("", xy=(1.5, 75.8), xytext=(0.9, 71.0),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(1.55, 73.5, "+4.8pp\n(3 audit fixes)", color="green", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("Precision (%)", fontsize=10)
    ax.set_title("Precision across deliverables (62-pair gold standard)\n"
                 "D4.1 hierarchical matching = same as D3.2 (no improvement from hier signal)",
                 fontsize=9)
    ax.set_ylim(58, 88)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "d4_1_precision_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_d4_1] Saved {out}")


def main():
    with open(GOLD_V2_JSON, encoding="utf-8") as f:
        gdata = json.load(f)
    audit = gdata["audit_decisions"]

    mc_mod_df   = pd.read_csv(MC_MOD)
    mc_ids      = list(mc_mod_df["module_id"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}

    hier_scores = sparse.load_npz(HIER_SCORES).toarray()
    with open(HIER_INDEX, encoding="utf-8") as f:
        hidx = json.load(f)
    group_keys = hidx["group_keys"]

    fig1_gold_audit(audit)
    fig2_concept_groups(mc_ids, mc_id_to_idx, hier_scores, group_keys)
    fig3_precision_comparison()
    print("[viz_d4_1] All figures done.")


if __name__ == "__main__":
    main()
