"""Deliverable 3.2 visualizations.

Generates:
  outputs/figures/d3_vs_d3_2_comparison.png   — P@1/P@5 grouped bar chart
  outputs/figures/hard_case_rank_changes.png  — rank-change bar chart for hard cases
  outputs/figures/synonym_impact.png          — modules that improved via synonym expansion
  outputs/figures/token_source_contribution.png — Mathlib ML text source breakdown
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join("outputs", "figures")

D2_MATCHES  = os.path.join("outputs", "iterative_matches.csv")
D3_MATCHES  = os.path.join("outputs", "iterative_matches_v3.csv")
D32_TV3     = os.path.join("outputs", "iterative_matches_v3_textv3.csv")
D32_RERANK  = os.path.join("outputs", "iterative_matches_v3_reranked.csv")
BREAKDOWN   = os.path.join("outputs", "rerank_feature_breakdown.csv")
DIAG_CSV    = os.path.join("data", "processed", "text_v3_diagnostics.csv")
TEXT_V2_DIAG = os.path.join("data", "processed", "text_v3_diagnostics.csv")

INFRA_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)

GOLD_PAIRS = {
    "sylow": ["Mathlib.GroupTheory.Sylow"], "nilpotent": ["Mathlib.GroupTheory.Nilpotent"],
    "cyclic": ["Mathlib.GroupTheory.SpecificGroups.Cyclic"], "perm": ["Mathlib.GroupTheory.Perm"],
    "abelian": ["Mathlib.GroupTheory.Abelianization", "Mathlib.GroupTheory.AbelianGroup"],
    "commutator": ["Mathlib.GroupTheory.Commutator"],
    "center": ["Mathlib.GroupTheory.Subgroup.Center"],
    "quotient": ["Mathlib.GroupTheory.QuotientGroup", "Mathlib.GroupTheory.Coset"],
    "action": ["Mathlib.GroupTheory.GroupAction"],
    "fingroup": ["Mathlib.GroupTheory"],
    "morphism": ["Mathlib.GroupTheory.GroupHom", "Mathlib.GroupTheory.Subgroup", "Mathlib.GroupTheory.Hom"],
    "automorphism": ["Mathlib.GroupTheory.Aut", "Mathlib.GroupTheory.Subgroup"],
    "pgroup": ["Mathlib.GroupTheory.PGroup"],
    "gproduct": ["Mathlib.GroupTheory.SemidirectProduct", "Mathlib.GroupTheory.DirectProduct"],
    "gseries": ["Mathlib.GroupTheory.Subgroup", "Mathlib.GroupTheory.Series"],
    "hall": ["Mathlib.GroupTheory.Complement", "Mathlib.GroupTheory.Solvable"],
    "alt": ["Mathlib.GroupTheory.SpecificGroups.Alternating"],
    "presentation": ["Mathlib.GroupTheory.PresentedGroup", "Mathlib.GroupTheory.FreeGroup"],
    "jordanholder": ["Mathlib.Order.JordanHolder", "Mathlib.GroupTheory.CompositionSeries"],
    "burnside_app": ["Mathlib.GroupTheory.GroupAction", "Mathlib.GroupTheory.Burnside"],
    "ssralg": ["Mathlib.Algebra.Ring", "Mathlib.Algebra.Group"],
    "matrix": ["Mathlib.LinearAlgebra.Matrix"],
    "poly": ["Mathlib.RingTheory.Polynomial", "Mathlib.Algebra.Polynomial"],
    "ring_quotient": ["Mathlib.RingTheory.Ideal.Quotient"],
    "intdiv": ["Mathlib.Data.Int.Div", "Mathlib.Data.Int"],
    "bigop": ["Mathlib.Algebra.BigOperators"],
    "ssrnat": ["Mathlib.Data.Nat"], "ssrint": ["Mathlib.Data.Int"],
    "rat": ["Mathlib.Data.Rat"], "prime": ["Mathlib.Data.Nat.Prime"],
    "zmodp": ["Mathlib.Data.ZMod"],
    "fraction": ["Mathlib.RingTheory.Localization.FractionRing"],
    "binomial": ["Mathlib.Data.Nat.Choose"],
    "mxpoly": ["Mathlib.LinearAlgebra.Matrix.Polynomial", "Mathlib.LinearAlgebra.Matrix.Charpoly"],
    "mxalgebra": ["Mathlib.LinearAlgebra.Matrix"],
    "vector": ["Mathlib.LinearAlgebra"],
    "sesquilinear": ["Mathlib.LinearAlgebra.SesquilinearForm"],
    "mxrepresentation": ["Mathlib.RepresentationTheory"],
    "character": ["Mathlib.RepresentationTheory.Character"],
    "vcharacter": ["Mathlib.RepresentationTheory.Character"],
    "separable": ["Mathlib.FieldTheory.Separable"],
    "galois": ["Mathlib.FieldTheory.Galois", "Mathlib.FieldTheory.Finite.GaloisField"],
    "algC": ["Mathlib.FieldTheory.IsAlgClosed"],
    "cyclotomic": ["Mathlib.NumberTheory.Cyclotomic", "Mathlib.RingTheory.Polynomial.Cyclotomic"],
    "fieldext": ["Mathlib.FieldTheory.Extension"],
    "finfield": ["Mathlib.FieldTheory.Finite", "Mathlib.FieldTheory.Galois"],
    "order": ["Mathlib.Order.Lattice", "Mathlib.Order"],
    "preorder": ["Mathlib.Order.Preorder", "Mathlib.Order"],
    "archimedean": ["Mathlib.Algebra.Order.Archimedean"],
    "seq": ["Mathlib.Data.Seq", "Mathlib.Data.List"],
    "fintype": ["Mathlib.Data.Fintype"], "finset": ["Mathlib.Data.Finset"],
    "tuple": ["Mathlib.Data.Vector", "Mathlib.Data.Fin.Tuple"],
    "eqtype": ["Mathlib.Logic.Equiv", "Mathlib.Data.Subtype"],
    "choice": ["Mathlib.Logic.Classical", "Mathlib.Order.Zorn"],
    "path": ["Mathlib.Combinatorics.SimpleGraph", "Mathlib.Topology.Path"],
    "fingraph": ["Mathlib.Combinatorics.SimpleGraph"],
    "div": ["Mathlib.Data.Nat.Div", "Mathlib.Data.Int.Div"],
    "finfun": ["Mathlib.Data.PiFin", "Mathlib.Data.Fin"],
    "ssrbool": ["Mathlib.Data.Bool"],
    "ssrfun": ["Mathlib.Logic.Function"],
    "classfun": ["Mathlib.RepresentationTheory"],
}

HARD_CASES = [
    "burnside_app", "commutator", "quotient", "galois", "cyclotomic",
    "eqtype", "path", "presentation", "mxalgebra", "bigop",
    "ssrnat", "finfun", "fingraph", "ssralg", "ssrbool",
]


def gold_match(prefixes, ml_module):
    return any(ml_module.startswith(p) for p in prefixes)


def get_top1(df, mc_mod, score_col="final_score"):
    sub = df[df["mathcomp_module"] == mc_mod]
    if sub.empty:
        return None
    return sub.sort_values(score_col, ascending=False).iloc[0]["mathlib_module"]


def get_score_col(df):
    for col in ["reranked_score", "final_score", "base_score"]:
        if col in df.columns:
            return col
    return df.columns[-1]


def get_rank(df, mc_mod, ml_prefix, score_col="final_score"):
    sub = df[df["mathcomp_module"] == mc_mod].sort_values(score_col, ascending=False)
    for r, (_, row) in enumerate(sub.iterrows(), 1):
        if gold_match([ml_prefix], row["mathlib_module"]):
            return r
    return None


def precision_at_k(df, k, score_col="final_score"):
    hits, total = 0, 0
    for mc_mod, prefixes in GOLD_PAIRS.items():
        sub = df[df["mathcomp_module"] == mc_mod]
        if sub.empty:
            continue
        total += 1
        topk = sub.sort_values(score_col, ascending=False).head(k)
        for _, r in topk.iterrows():
            if gold_match(prefixes, r["mathlib_module"]):
                hits += 1
                break
    return hits / total if total else 0.0


def load_systems():
    systems = {}
    for label, path in [("D2", D2_MATCHES), ("D3", D3_MATCHES),
                        ("D3.2_text_v3", D32_TV3), ("D3.2_rerank", D32_RERANK)]:
        if os.path.exists(path):
            systems[label] = pd.read_csv(path)
    return systems


def plot_comparison(systems):
    labels = [l for l in ["D2", "D3", "D3.2_text_v3", "D3.2_rerank"] if l in systems]
    display = {"D2": "D2", "D3": "D3\ncalibrated",
               "D3.2_text_v3": "D3.2\ntext-v3", "D3.2_rerank": "D3.2\nreranked"}
    p1s = [precision_at_k(systems[l], 1, get_score_col(systems[l])) * 100 for l in labels]
    p5s = [precision_at_k(systems[l], 5, get_score_col(systems[l])) * 100 for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, p1s, w, label="P@1", color="#3B82F6", alpha=0.85)
    bars2 = ax.bar(x + w/2, p5s, w, label="P@5", color="#10B981", alpha=0.85)

    for bar, v in zip(bars1, p1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(bars2, p5s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([display[l] for l in labels], fontsize=11)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Precision Comparison: D2 / D3 / D3.2 (62-pair gold standard)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT_DIR, "d3_vs_d3_2_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[viz_v3_2] Saved {path}")


def plot_hard_case_ranks(systems):
    """Bar chart of rank of gold module for each hard case, across systems."""
    labels = [l for l in ["D3", "D3.2_text_v3", "D3.2_rerank"] if l in systems]
    colors = {"D3": "#F59E0B", "D3.2_text_v3": "#6366F1", "D3.2_rerank": "#10B981"}

    cases = [m for m in HARD_CASES if any(systems[l][systems[l]["mathcomp_module"] == m].shape[0] > 0
                                           for l in labels)]
    # Keep only modules with a gold answer
    cases = [m for m in cases if m in GOLD_PAIRS][:12]

    x = np.arange(len(cases))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, label in enumerate(labels):
        df = systems[label]
        scol = get_score_col(df)
        ranks = []
        for m in cases:
            prefixes = GOLD_PAIRS[m]
            best_rank = None
            for pfx in prefixes:
                r = get_rank(df, m, pfx, scol)
                if r is not None:
                    best_rank = r if best_rank is None else min(best_rank, r)
            ranks.append(min(best_rank, 10) if best_rank else 11)

        offset = (i - len(labels)/2 + 0.5) * w
        bars = ax.bar(x + offset, ranks, w, label=label,
                      color=colors.get(label, "gray"), alpha=0.85)

    ax.axhline(y=1, color="green", linestyle="--", alpha=0.6, label="Rank 1")
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.4, label="Rank 5")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Rank of gold module", fontsize=12)
    ax.set_title("Gold module rank in hard cases (lower = better)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT_DIR, "hard_case_rank_changes.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[viz_v3_2] Saved {path}")


def plot_synonym_impact(systems):
    """Highlight modules where D3.2 text-v3 improved over D3."""
    if "D3" not in systems or "D3.2_text_v3" not in systems:
        return

    df_d3  = systems["D3"]
    df_tv3 = systems["D3.2_text_v3"]
    sc_d3  = get_score_col(df_d3)
    sc_tv3 = get_score_col(df_tv3)

    improved, same, regressed = [], [], []
    for mc_mod, prefixes in GOLD_PAIRS.items():
        t1_d3  = get_top1(df_d3,  mc_mod, sc_d3)
        t1_tv3 = get_top1(df_tv3, mc_mod, sc_tv3)
        if t1_d3 is None or t1_tv3 is None:
            continue
        h3  = gold_match(prefixes, t1_d3)
        htv = gold_match(prefixes, t1_tv3)
        if htv and not h3:
            improved.append(mc_mod)
        elif h3 and not htv:
            regressed.append(mc_mod)
        else:
            same.append(mc_mod)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [len(improved), len(same), len(regressed)]
    lbls = [f"Improved\n({len(improved)})", f"Unchanged\n({len(same)})",
            f"Regressed\n({len(regressed)})"]
    colors = ["#10B981", "#9CA3AF", "#EF4444"]
    wedges, texts, autotexts = ax.pie(
        vals, labels=lbls, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 11},
    )
    ax.set_title("D3 → D3.2 text-v3 changes on 62-pair gold standard",
                 fontsize=12, fontweight="bold")

    # Annotation listing improved modules
    if improved:
        note = "Improved: " + ", ".join(improved[:6])
        if len(improved) > 6:
            note += f" +{len(improved)-6} more"
        ax.text(0, -1.4, note, ha="center", fontsize=8.5, color="#065F46",
                style="italic")

    path = os.path.join(OUT_DIR, "synonym_impact.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[viz_v3_2] Saved {path}")


def plot_token_source(diag_csv):
    """Pie chart of Mathlib module text sources from diagnostics."""
    if not os.path.exists(diag_csv):
        print("[viz_v3_2] No diagnostics CSV, skipping token_source_contribution.png")
        return

    diag = pd.read_csv(diag_csv)
    if "source" not in diag.columns:
        return

    counts = diag["source"].value_counts()
    labels = [f"{s}\n({c})" for s, c in counts.items()]
    colors = ["#3B82F6", "#8B5CF6", "#F59E0B", "#9CA3AF"][:len(counts)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Mathlib module text source distribution (text_sim_v3)",
                 fontsize=12, fontweight="bold")

    path = os.path.join(OUT_DIR, "token_source_contribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[viz_v3_2] Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    systems = load_systems()
    if not systems:
        print("[viz_v3_2] No match files found — run the pipeline first.")
        return

    print(f"[viz_v3_2] Systems available: {list(systems.keys())}")
    plot_comparison(systems)
    plot_hard_case_ranks(systems)
    plot_synonym_impact(systems)
    plot_token_source(DIAG_CSV)
    print("[viz_v3_2] Done.")


if __name__ == "__main__":
    main()
