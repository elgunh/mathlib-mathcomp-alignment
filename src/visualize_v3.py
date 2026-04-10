"""Deliverable 3 figures.

1. d1_d2_d3_comparison.png  — grouped bar chart P@1 and P@5
2. text_signal_scatter.png  — old vs new text score (62 gold pairs)
3. docstring_coverage.png   — pie: with-docstring / decl-only / path-only
4. rank_improvement.png     — histogram of rank changes for gold pairs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

FIGURES_DIR = os.path.join("outputs", "figures")

OLD_NPZ = os.path.join("data", "processed", "text_sim.npz")
NEW_NPZ = os.path.join("data", "processed", "text_sim_v2.npz")
ML_DOCS = os.path.join("data", "processed", "mathlib_docstrings.csv")
MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
TEXT_COMP = os.path.join("outputs", "text_signal_comparison.csv")

GOLD_PAIRS = {
    "sylow":          ["Mathlib.GroupTheory.Sylow"],
    "nilpotent":      ["Mathlib.GroupTheory.Nilpotent"],
    "cyclic":         ["Mathlib.GroupTheory.SpecificGroups.Cyclic"],
    "perm":           ["Mathlib.GroupTheory.Perm"],
    "abelian":        ["Mathlib.GroupTheory.Abelianization"],
    "commutator":     ["Mathlib.GroupTheory.Commutator"],
    "center":         ["Mathlib.GroupTheory.Subgroup.Center"],
    "quotient":       ["Mathlib.GroupTheory.QuotientGroup"],
    "action":         ["Mathlib.GroupTheory.GroupAction"],
    "fingroup":       ["Mathlib.GroupTheory"],
    "morphism":       ["Mathlib.GroupTheory.Subgroup"],
    "automorphism":   ["Mathlib.GroupTheory.Aut"],
    "pgroup":         ["Mathlib.GroupTheory.PGroup"],
    "gproduct":       ["Mathlib.GroupTheory.SemidirectProduct"],
    "gseries":        ["Mathlib.GroupTheory.Subgroup"],
    "hall":           ["Mathlib.GroupTheory.Solvable"],
    "alt":            ["Mathlib.GroupTheory.SpecificGroups.Alternating"],
    "presentation":   ["Mathlib.GroupTheory.PresentedGroup"],
    "jordanholder":   ["Mathlib.Order.JordanHolder"],
    "burnside_app":   ["Mathlib.GroupTheory.GroupAction"],
    "ssralg":         ["Mathlib.Algebra.Ring"],
    "matrix":         ["Mathlib.LinearAlgebra.Matrix"],
    "poly":           ["Mathlib.RingTheory.Polynomial"],
    "ring_quotient":  ["Mathlib.RingTheory.Ideal.Quotient"],
    "intdiv":         ["Mathlib.Data.Int"],
    "bigop":          ["Mathlib.Algebra.BigOperators"],
    "ssrnat":         ["Mathlib.Data.Nat"],
    "ssrint":         ["Mathlib.Data.Int"],
    "rat":            ["Mathlib.Data.Rat"],
    "prime":          ["Mathlib.Data.Nat.Prime"],
    "zmodp":          ["Mathlib.Data.ZMod"],
    "fraction":       ["Mathlib.RingTheory.Localization"],
    "binomial":       ["Mathlib.Data.Nat.Choose"],
    "mxpoly":         ["Mathlib.LinearAlgebra.Matrix.Polynomial"],
    "mxalgebra":      ["Mathlib.LinearAlgebra.Matrix"],
    "vector":         ["Mathlib.LinearAlgebra"],
    "sesquilinear":   ["Mathlib.LinearAlgebra.SesquilinearForm"],
    "mxrepresentation": ["Mathlib.RepresentationTheory"],
    "character":      ["Mathlib.RepresentationTheory.Character"],
    "vcharacter":     ["Mathlib.RepresentationTheory.Character"],
    "separable":      ["Mathlib.FieldTheory.Separable"],
    "galois":         ["Mathlib.FieldTheory.Galois"],
    "algC":           ["Mathlib.FieldTheory.IsAlgClosed"],
    "cyclotomic":     ["Mathlib.NumberTheory.Cyclotomic"],
    "fieldext":       ["Mathlib.FieldTheory"],
    "finfield":       ["Mathlib.FieldTheory.Finite"],
    "order":          ["Mathlib.Order"],
    "preorder":       ["Mathlib.Order"],
    "archimedean":    ["Mathlib.Algebra.Order.Archimedean"],
    "seq":            ["Mathlib.Data.List"],
    "fintype":        ["Mathlib.Data.Fintype"],
    "finset":         ["Mathlib.Data.Finset"],
    "tuple":          ["Mathlib.Data.Fin"],
    "eqtype":         ["Mathlib.Logic.Equiv"],
    "choice":         ["Mathlib.Logic.Classical"],
    "path":           ["Mathlib.Combinatorics.SimpleGraph"],
    "fingraph":       ["Mathlib.Combinatorics.SimpleGraph"],
    "div":            ["Mathlib.Data.Nat.Div"],
    "finfun":         ["Mathlib.Data.Fin"],
    "ssrbool":        ["Mathlib.Data.Bool"],
    "ssrfun":         ["Mathlib.Logic.Function"],
    "classfun":       ["Mathlib.RepresentationTheory"],
}


def precision_at_k(df, k, score_col, gold):
    hits, total = 0, 0
    for mc_mod, prefixes in gold.items():
        sub = df[df["mathcomp_module"] == mc_mod]
        if sub.empty:
            continue
        total += 1
        top_k = sub.sort_values(score_col).tail(k)
        for _, r in top_k.iterrows():
            if any(r["mathlib_module"].startswith(p) for p in prefixes):
                hits += 1
                break
    return hits / max(total, 1)


def plot_comparison(d1_path, d2_path, d3_path):
    if not all(os.path.exists(p) for p in [d1_path, d2_path, d3_path]):
        print("[viz_v3] Skipping comparison chart — missing files")
        return
    d1 = pd.read_csv(d1_path)
    d2 = pd.read_csv(d2_path)
    d3 = pd.read_csv(d3_path)

    d1_p1 = precision_at_k(d1, 1, "combined_score", GOLD_PAIRS)
    d1_p5 = precision_at_k(d1, 5, "combined_score", GOLD_PAIRS)
    d2_p1 = precision_at_k(d2, 1, "final_score", GOLD_PAIRS)
    d2_p5 = precision_at_k(d2, 5, "final_score", GOLD_PAIRS)
    d3_p1 = precision_at_k(d3, 1, "final_score", GOLD_PAIRS)
    d3_p5 = precision_at_k(d3, 5, "final_score", GOLD_PAIRS)

    labels = ["D1\n(baseline)", "D2\n(iterative)", "D3\n(+ docstrings)"]
    p1_vals = [d1_p1 * 100, d2_p1 * 100, d3_p1 * 100]
    p5_vals = [d1_p5 * 100, d2_p5 * 100, d3_p5 * 100]

    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, p1_vals, width, label="P@1", color="#3b5fc0", alpha=0.85)
    bars2 = ax.bar(x + width/2, p5_vals, width, label="P@5", color="#e07b39", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Precision (%)")
    ax.set_title("D1 / D2 / D3 — Precision@1 and Precision@5\n(62-pair gold standard)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "d1_d2_d3_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_v3] Saved {out}")


def plot_text_scatter():
    if not os.path.exists(TEXT_COMP):
        print("[viz_v3] Skipping text scatter — missing text_signal_comparison.csv")
        return
    df = pd.read_csv(TEXT_COMP)
    hit_mask = df["delta_rank"] > 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df.loc[~hit_mask, "old_text"], df.loc[~hit_mask, "new_text"],
               alpha=0.6, s=40, c="#888888", label="no change / regression")
    ax.scatter(df.loc[hit_mask, "old_text"], df.loc[hit_mask, "new_text"],
               alpha=0.8, s=50, c="#3b5fc0", label="rank improved")
    lim_max = max(df["old_text"].max(), df["new_text"].max()) * 1.05 + 0.01
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("Old text score (path tokens)")
    ax.set_ylabel("New text score (docstrings + declarations)")
    ax.set_title("Text Signal: Old vs New\nfor 62 gold pairs")
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "text_signal_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_v3] Saved {out}")


def plot_docstring_coverage():
    if not os.path.exists(ML_DOCS):
        print("[viz_v3] Skipping coverage chart — missing mathlib_docstrings.csv")
        return
    df = pd.read_csv(ML_DOCS)
    counts = df["text_source"].value_counts()
    labels_map = {
        "docstring": "Docstring present",
        "declarations": "Declarations only",
        "path_only": "Path tokens only",
        "skipped": "Skipped",
        "empty": "Empty / error",
    }
    labels = [labels_map.get(k, k) for k in counts.index]
    colors = ["#3b5fc0", "#e07b39", "#aaaaaa", "#cccccc", "#eeeeee"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colors[:len(labels)], startangle=140,
        pctdistance=0.78, labeldistance=1.12)
    for t in texts:
        t.set_fontsize(10)
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(f"Mathlib Docstring Coverage\n({len(df)} modules)")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "docstring_coverage.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_v3] Saved {out}")


def plot_rank_improvement():
    if not os.path.exists(TEXT_COMP):
        print("[viz_v3] Skipping rank histogram — missing text_signal_comparison.csv")
        return
    df = pd.read_csv(TEXT_COMP)
    deltas = df["delta_rank"].values
    cap = 500
    clipped = np.clip(deltas, -cap, cap)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-cap, cap, 41)
    pos_mask = clipped > 0
    neg_mask = clipped < 0
    zero_mask = clipped == 0
    ax.hist(clipped[pos_mask], bins=bins, color="#3b5fc0", alpha=0.8, label="Improved")
    ax.hist(clipped[neg_mask], bins=bins, color="#e07b39", alpha=0.8, label="Regressed")
    ax.hist(clipped[zero_mask], bins=bins, color="#888888", alpha=0.6, label="Unchanged")
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Rank change (old_rank − new_rank), clipped to ±500")
    ax.set_ylabel("Number of gold pairs")
    ax.set_title("Rank Change in Gold Pairs: New vs Old Text Signal")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "rank_improvement.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[viz_v3] Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_comparison(
        os.path.join("outputs", "candidate_matches.csv"),
        os.path.join("outputs", "iterative_matches.csv"),
        os.path.join("outputs", "iterative_matches_v3.csv"),
    )
    plot_text_scatter()
    plot_docstring_coverage()
    plot_rank_improvement()
    print("[viz_v3] Done.")


if __name__ == "__main__":
    main()
