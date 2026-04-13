"""Deliverable 4 figures.

1. d4_declaration_overlap_examples.png
   Bar chart showing decl-overlap score vs token count for 8 illustrative
   modules: 4 where the signal correctly identifies the gold match in the
   top candidate set, and 4 hard cases where it does not.

2. d4_precision_comparison.png
   Grouped bars: D2 → D3.2 → D4 for P@1 and P@5.

3. d4_idf_token_distribution.png
   Histogram of IDF values for the full 6,959-token vocabulary, with
   example labels for discriminative vs generic tokens.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from declaration_similarity import (
    make_token_set, build_idf, GENERIC_TOKENS
)

DECL_NPZ  = os.path.join("data", "processed", "decl_sim.npz")
MC_DECL   = os.path.join("data", "processed", "mathcomp_declarations.csv")
ML_DOCS   = os.path.join("data", "processed", "mathlib_docstrings.csv")
MC_MOD    = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MOD    = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_DIR   = os.path.join("outputs", "figures")

os.makedirs(OUT_DIR, exist_ok=True)

GOLD_PAIRS = {
    "sylow":      "Mathlib.GroupTheory.Sylow",
    "character":  "Mathlib.RepresentationTheory.Character",
    "poly":       "Mathlib.RingTheory.Polynomial",
    "ssrnat":     "Mathlib.Data.Nat",
    "bigop":      "Mathlib.Algebra.BigOperators",
    "eqtype":     "Mathlib.Logic.Equiv",
    "fingraph":   "Mathlib.Combinatorics.SimpleGraph",
    "path":       "Mathlib.Combinatorics.SimpleGraph",
    "cyclotomic": "Mathlib.RingTheory.Polynomial.Cyclotomic",
}


def load_tokens():
    mc_decl  = pd.read_csv(MC_DECL)
    ml_docs  = pd.read_csv(ML_DOCS)
    mc_mod   = pd.read_csv(MC_MOD)
    ml_mod   = pd.read_csv(ML_MOD)
    mc_ids   = list(mc_mod["module_id"])
    ml_names = list(ml_mod["module_name"])

    mc_decl_map = {r["module_id"]: str(r.get("raw_declarations", ""))
                   for _, r in mc_decl.iterrows()}
    ml_decl_map = {r["module_name"]: str(r.get("declaration_names", ""))
                   for _, r in ml_docs.iterrows()}

    mc_token_sets = [make_token_set(mc_decl_map.get(mid, "")) for mid in mc_ids]
    ml_token_sets = [make_token_set(ml_decl_map.get(n, "")) for n in ml_names]
    idf = build_idf(mc_token_sets + ml_token_sets)
    return mc_ids, ml_names, mc_token_sets, ml_token_sets, idf


def fig1_overlap_examples(mc_ids, ml_names, mc_token_sets, ml_token_sets, idf):
    """Show decl-overlap scores for 9 selected modules."""
    scores, shared_counts, labels, is_gold_found = [], [], [], []

    decl_mat = sparse.load_npz(DECL_NPZ).toarray()

    for mc_mid, gold_prefix in GOLD_PAIRS.items():
        if mc_mid not in mc_ids:
            scores.append(0); shared_counts.append(0)
            labels.append(mc_mid); is_gold_found.append(False)
            continue

        i = mc_ids.index(mc_mid)
        mc_ts = mc_token_sets[i]
        mc_w  = sum(idf.get(t, 1.0) for t in mc_ts)

        # Find best matching ML module by gold prefix
        best_score = 0.0
        best_shared = 0
        found = False
        for j, ml_name in enumerate(ml_names):
            if not ml_name.startswith(gold_prefix):
                continue
            ml_ts = ml_token_sets[j]
            inter = mc_ts & ml_ts
            if len(inter) < 2:
                continue
            ml_w = sum(idf.get(t, 1.0) for t in ml_ts)
            sc = sum(idf.get(t, 1.0) for t in inter) / min(mc_w, ml_w)
            if sc > best_score:
                best_score = sc
                best_shared = len(inter)
                found = True

        scores.append(best_score)
        shared_counts.append(best_shared)
        labels.append(mc_mid)
        is_gold_found.append(found and best_score > 0)

    colors = ["#2ecc71" if ok else "#e74c3c" for ok in is_gold_found]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate with shared token counts
    for bar, cnt in zip(bars, shared_counts):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{cnt}t", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best decl-overlap score\n(IDF-weighted Szymkiewicz-Simpson)", fontsize=9)
    ax.set_title("Declaration-name overlap: best score with gold-target module\n"
                 "(green = some overlap found; red = too sparse / no matching tokens)", fontsize=10)
    ax.set_ylim(0, max(scores + [0.1]) * 1.2)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ecc71", label="Overlap found"),
                        Patch(color="#e74c3c", label="No/weak overlap")],
              fontsize=9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "d4_declaration_overlap_examples.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_d4] Saved {out}")


def fig2_precision_comparison():
    systems = ["D2\n(iterative)", "D3.2\n(synonym+rerank)", "D4\n(+decl signal)"]
    p1 = [67.7, 71.0, 71.0]
    p5 = [79.0, 79.0, 79.0]

    x = np.arange(len(systems))
    w = 0.32
    colors_p1 = ["#3498db", "#e67e22", "#9b59b6"]
    colors_p5 = ["#85c1e9", "#f0b27a", "#d2b4de"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, p1, w, color=colors_p1, label="P@1", edgecolor="white")
    ax.bar(x + w/2, p5, w, color=colors_p5, label="P@5", edgecolor="white")

    for i, (v1, v5) in enumerate(zip(p1, p5)):
        ax.text(i - w/2, v1 + 0.4, f"{v1:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, v5 + 0.4, f"{v5:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Precision (%)", fontsize=10)
    ax.set_title("Precision comparison: D2 → D3.2 → D4 (62-pair gold standard)", fontsize=10)
    ax.set_ylim(55, 88)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=71.0, color="#e67e22", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "d4_precision_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_d4] Saved {out}")


def fig3_idf_distribution(idf):
    """Histogram of IDF values with token annotations."""
    values = np.array(list(idf.values()))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=50, color="#5dade2", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("IDF value", fontsize=10)
    ax.set_ylabel("Number of tokens", fontsize=10)
    ax.set_title("Distribution of IDF values across 6,959-token declaration vocabulary\n"
                 "(low IDF = generic; high IDF = discriminative)", fontsize=10)

    # Annotate extremes
    sorted_idf = sorted(idf.items(), key=lambda x: x[1])
    low5  = [t for t, _ in sorted_idf[:5]]
    high5 = [t for t, _ in sorted_idf[-5:]]

    lo_val = sorted_idf[4][1]
    hi_val = sorted_idf[-5][1]
    ax.axvline(lo_val, color="#e74c3c", linestyle="--", alpha=0.6,
               label=f"Generic: {', '.join(low5)}")
    ax.axvline(hi_val, color="#2ecc71", linestyle="--", alpha=0.6,
               label=f"Discriminative: {', '.join(high5[:3])}…")

    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "d4_idf_token_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_d4] Saved {out}")


def main():
    mc_ids, ml_names, mc_token_sets, ml_token_sets, idf = load_tokens()
    fig1_overlap_examples(mc_ids, ml_names, mc_token_sets, ml_token_sets, idf)
    fig2_precision_comparison()
    fig3_idf_distribution(idf)
    print("[viz_d4] All figures done.")


if __name__ == "__main__":
    main()
