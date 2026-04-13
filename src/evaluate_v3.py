"""Deliverable 3 evaluation: three-way D1 / D2 / D3 comparison.

Produces:
  .ai/deliverable_3_eval.md  (private, per-pair details)
  outputs/deliverable_3_summary.md  (public report)
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse

D1_MATCHES = os.path.join("outputs", "candidate_matches.csv")
D2_MATCHES = os.path.join("outputs", "iterative_matches.csv")
D3_MATCHES = os.path.join("outputs", "iterative_matches_v3.csv")
PROP_LOG_D2 = os.path.join("outputs", "propagation_log.csv")
PROP_LOG_D3 = os.path.join("outputs", "propagation_log_v3.csv")
ML_DOCS = os.path.join("data", "processed", "mathlib_docstrings.csv")
TEXT_COMP = os.path.join("outputs", "text_signal_comparison.csv")

OUT_DIR = ".ai"
OUT_EVAL = os.path.join(OUT_DIR, "deliverable_3_eval.md")
OUT_SUMMARY = os.path.join("outputs", "deliverable_3_summary.md")

INFRA_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)

GOLD_PAIRS = {
    "sylow":          ["Mathlib.GroupTheory.Sylow"],
    "nilpotent":      ["Mathlib.GroupTheory.Nilpotent"],
    "cyclic":         ["Mathlib.GroupTheory.SpecificGroups.Cyclic"],
    "perm":           ["Mathlib.GroupTheory.Perm"],
    "abelian":        ["Mathlib.GroupTheory.Abelianization",
                       "Mathlib.GroupTheory.AbelianGroup"],
    "commutator":     ["Mathlib.GroupTheory.Commutator"],
    "center":         ["Mathlib.GroupTheory.Subgroup.Center"],
    "quotient":       ["Mathlib.GroupTheory.QuotientGroup",
                       "Mathlib.GroupTheory.Coset"],
    "action":         ["Mathlib.GroupTheory.GroupAction"],
    "fingroup":       ["Mathlib.GroupTheory"],
    "morphism":       ["Mathlib.GroupTheory.GroupHom",
                       "Mathlib.GroupTheory.Subgroup",
                       "Mathlib.GroupTheory.Hom"],
    "automorphism":   ["Mathlib.GroupTheory.Aut",
                       "Mathlib.GroupTheory.Subgroup"],
    "pgroup":         ["Mathlib.GroupTheory.PGroup"],
    "gproduct":       ["Mathlib.GroupTheory.SemidirectProduct",
                       "Mathlib.GroupTheory.DirectProduct"],
    "gseries":        ["Mathlib.GroupTheory.Subgroup",
                       "Mathlib.GroupTheory.Series"],
    "hall":           ["Mathlib.GroupTheory.Complement",
                       "Mathlib.GroupTheory.Solvable"],
    "alt":            ["Mathlib.GroupTheory.SpecificGroups.Alternating"],
    "presentation":   ["Mathlib.GroupTheory.PresentedGroup",
                       "Mathlib.GroupTheory.FreeGroup"],
    "jordanholder":   ["Mathlib.Order.JordanHolder",
                       "Mathlib.GroupTheory.GroupAction.Jordan",
                       "Mathlib.GroupTheory.CompositionSeries"],
    "burnside_app":   ["Mathlib.GroupTheory.GroupAction",
                       "Mathlib.GroupTheory.Burnside"],
    "ssralg":         ["Mathlib.Algebra.Ring", "Mathlib.Algebra.Group"],
    "matrix":         ["Mathlib.LinearAlgebra.Matrix"],
    "poly":           ["Mathlib.RingTheory.Polynomial",
                       "Mathlib.Algebra.Polynomial"],
    "ring_quotient":  ["Mathlib.RingTheory.Ideal.Quotient",
                       "Mathlib.RingTheory.Polynomial.Quotient"],
    "intdiv":         ["Mathlib.Data.Int.Div",
                       "Mathlib.Data.Int.ModCast",
                       "Mathlib.Data.Int.Order",
                       "Mathlib.Data.Int.GCD",
                       "Mathlib.Data.Int"],
    "bigop":          ["Mathlib.Algebra.BigOperators",
                       "Mathlib.Algebra.Order.Sum"],
    "ssrnat":         ["Mathlib.Data.Nat"],
    "ssrint":         ["Mathlib.Data.Int"],
    "rat":            ["Mathlib.Data.Rat"],
    "prime":          ["Mathlib.Data.Nat.Prime",
                       "Mathlib.Data.Nat.Factors"],
    "zmodp":          ["Mathlib.Data.ZMod"],
    "fraction":       ["Mathlib.RingTheory.Localization.FractionRing",
                       "Mathlib.RingTheory.Localization"],
    "binomial":       ["Mathlib.Data.Nat.Choose",
                       "Mathlib.Data.Nat.Factorial",
                       "Mathlib.RingTheory.Binomial"],
    "mxpoly":         ["Mathlib.LinearAlgebra.Matrix.Polynomial",
                       "Mathlib.LinearAlgebra.Matrix.Charpoly"],
    "mxalgebra":      ["Mathlib.LinearAlgebra.Matrix"],
    "vector":         ["Mathlib.LinearAlgebra"],
    "sesquilinear":   ["Mathlib.LinearAlgebra.SesquilinearForm",
                       "Mathlib.LinearAlgebra.Matrix.SesquilinearForm"],
    "mxrepresentation": ["Mathlib.RepresentationTheory"],
    "character":      ["Mathlib.RepresentationTheory.Character"],
    "vcharacter":     ["Mathlib.RepresentationTheory.Character",
                       "Mathlib.RepresentationTheory.VirtualCharacter"],
    "separable":      ["Mathlib.FieldTheory.Separable"],
    "galois":         ["Mathlib.FieldTheory.Galois",
                       "Mathlib.FieldTheory.Finite.GaloisField"],
    "algC":           ["Mathlib.FieldTheory.IsAlgClosed",
                       "Mathlib.Analysis.SpecialFunctions.Complex"],
    "cyclotomic":     ["Mathlib.NumberTheory.Cyclotomic",
                       "Mathlib.RingTheory.Polynomial.Cyclotomic"],
    "fieldext":       ["Mathlib.FieldTheory.Extension",
                       "Mathlib.FieldTheory.IntermediateField"],
    "finfield":       ["Mathlib.FieldTheory.Finite",
                       "Mathlib.FieldTheory.Galois"],
    "order":          ["Mathlib.Order.Lattice", "Mathlib.Order"],
    "preorder":       ["Mathlib.Order.Preorder", "Mathlib.Order"],
    "archimedean":    ["Mathlib.Algebra.Order.Archimedean",
                       "Mathlib.Algebra.Order.Ring.Archimedean"],
    "seq":            ["Mathlib.Data.Seq", "Mathlib.Data.List"],
    "fintype":        ["Mathlib.Data.Fintype"],
    "finset":         ["Mathlib.Data.Finset"],
    "tuple":          ["Mathlib.Data.Vector", "Mathlib.Data.Fin.Tuple",
                       "Mathlib.Order.Fin"],
    "eqtype":         ["Mathlib.Logic.Equiv", "Mathlib.Data.Subtype"],
    "choice":         ["Mathlib.Logic.Classical", "Mathlib.Order.Zorn",
                       "Mathlib.Logic.Choice"],
    "path":           ["Mathlib.Combinatorics.SimpleGraph",
                       "Mathlib.Topology.Path"],
    "fingraph":       ["Mathlib.Combinatorics.SimpleGraph",
                       "Mathlib.Order.Graph"],
    "div":            ["Mathlib.Data.Nat.Div", "Mathlib.Data.Int.Div",
                       "Mathlib.Data.Nat.GCD"],
    "finfun":         ["Mathlib.Data.PiFin", "Mathlib.Logic.Fin",
                       "Mathlib.Data.Fin"],
    "ssrbool":        ["Mathlib.Data.Bool"],
    "ssrfun":         ["Mathlib.Logic.Function", "Mathlib.Data.Function"],
    "classfun":       ["Mathlib.RepresentationTheory",
                       "Mathlib.GroupTheory.ClassEquation"],
}


def gold_match(prefixes, ml_module):
    return any(ml_module.startswith(p) for p in prefixes)


def precision_at_k(df, k, score_col):
    hits, total = 0, 0
    for mc_mod, prefixes in GOLD_PAIRS.items():
        subset = df[df["mathcomp_module"] == mc_mod]
        if subset.empty:
            continue
        total += 1
        top_k = subset.sort_values(score_col).tail(k)
        for _, r in top_k.iterrows():
            if gold_match(prefixes, r["mathlib_module"]):
                hits += 1
                break
    return hits, total


def per_cluster(df, score_col, label):
    rows = []
    clusters = df["mathcomp_cluster"].unique()
    for clust in sorted(clusters):
        cl_df = df[df["mathcomp_cluster"] == clust]
        hits = 0
        total = 0
        for mc_mod, prefixes in GOLD_PAIRS.items():
            row = cl_df[(cl_df["mathcomp_module"] == mc_mod) &
                        (cl_df["rank"] == 1)]
            if row.empty:
                continue
            total += 1
            if gold_match(prefixes, row.iloc[0]["mathlib_module"]):
                hits += 1
        if total > 0:
            rows.append({"cluster": clust, f"hits_{label}": hits,
                         "total": total,
                         f"p1_{label}": round(hits / total, 3)})
    return pd.DataFrame(rows)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    d1 = pd.read_csv(D1_MATCHES)
    d2 = pd.read_csv(D2_MATCHES)
    d3 = pd.read_csv(D3_MATCHES)

    # ---- precision ----
    d1_p1, d1_t = precision_at_k(d1, 1, "combined_score")
    d1_p5, _ = precision_at_k(d1, 5, "combined_score")
    d2_p1, _ = precision_at_k(d2, 1, "final_score")
    d2_p5, _ = precision_at_k(d2, 5, "final_score")
    d3_p1, _ = precision_at_k(d3, 1, "final_score")
    d3_p5, _ = precision_at_k(d3, 5, "final_score")

    print("[eval_v3] Three-way precision on 62 gold pairs:")
    print(f"  D1  P@1 = {d1_p1}/{d1_t} = {d1_p1/d1_t:.1%}  "
          f"P@5 = {d1_p5}/{d1_t} = {d1_p5/d1_t:.1%}")
    print(f"  D2  P@1 = {d2_p1}/{d1_t} = {d2_p1/d1_t:.1%}  "
          f"P@5 = {d2_p5}/{d1_t} = {d2_p5/d1_t:.1%}")
    print(f"  D3  P@1 = {d3_p1}/{d1_t} = {d3_p1/d1_t:.1%}  "
          f"P@5 = {d3_p5}/{d1_t} = {d3_p5/d1_t:.1%}")

    # ---- per-pair details (D2 -> D3 flips/regressions) ----
    flip_right = []   # D2 miss, D3 hit
    flip_wrong = []   # D2 hit, D3 miss
    per_pair = []

    for mc_mod, prefixes in sorted(GOLD_PAIRS.items()):
        r1 = d1[(d1["mathcomp_module"] == mc_mod) & (d1["rank"] == 1)]
        r2 = d2[(d2["mathcomp_module"] == mc_mod) & (d2["rank"] == 1)]
        r3 = d3[(d3["mathcomp_module"] == mc_mod) & (d3["rank"] == 1)]
        if r1.empty:
            continue
        m1 = r1.iloc[0]["mathlib_module"] if not r1.empty else ""
        m2 = r2.iloc[0]["mathlib_module"] if not r2.empty else ""
        m3 = r3.iloc[0]["mathlib_module"] if not r3.empty else ""
        h1 = gold_match(prefixes, m1)
        h2 = gold_match(prefixes, m2)
        h3 = gold_match(prefixes, m3)
        s1 = float(r1.iloc[0]["combined_score"]) if not r1.empty else 0.0
        s2 = float(r2.iloc[0]["final_score"]) if not r2.empty else 0.0
        s3 = float(r3.iloc[0]["final_score"]) if not r3.empty else 0.0
        per_pair.append({"module": mc_mod,
                         "d1": "HIT" if h1 else "miss",
                         "d2": "HIT" if h2 else "miss",
                         "d3": "HIT" if h3 else "miss",
                         "s1": s1, "s2": s2, "s3": s3,
                         "m1": m1, "m2": m2, "m3": m3})
        if not h2 and h3:
            flip_right.append((mc_mod, m2, m3, s2, s3))
        if h2 and not h3:
            flip_wrong.append((mc_mod, m2, m3, s2, s3))

    print(f"\n  D2->D3 improvements (miss->hit): {len(flip_right)}")
    for mc, old, new, s2, s3 in flip_right:
        print(f"    {mc:<20} {old[:40]:<42}-> {new[:40]} ({s3:.3f})")

    print(f"\n  D2->D3 regressions  (hit->miss):  {len(flip_wrong)}")
    for mc, old, new, s2, s3 in flip_wrong:
        print(f"    {mc:<20} {old[:40]:<42}-> {new[:40]} ({s3:.3f})")

    # ---- per-cluster ----
    cl_d1 = per_cluster(d1, "combined_score", "d1")
    cl_d2 = per_cluster(d2, "final_score", "d2")
    cl_d3 = per_cluster(d3, "final_score", "d3")
    cluster_df = cl_d1.merge(cl_d2[["cluster","hits_d2","p1_d2"]], on="cluster", how="outer")
    cluster_df = cluster_df.merge(cl_d3[["cluster","hits_d3","p1_d3"]], on="cluster", how="outer")

    # ---- docstring coverage stats ----
    doc_stats = {"with_doc": 0, "decl_only": 0, "path_only": 0,
                 "skipped": 0, "total": 0}
    if os.path.exists(ML_DOCS):
        ml_docs = pd.read_csv(ML_DOCS)
        doc_stats["total"] = len(ml_docs)
        for src in ml_docs.get("text_source", pd.Series(dtype=str)):
            if src == "docstring":
                doc_stats["with_doc"] += 1
            elif src == "declarations":
                doc_stats["decl_only"] += 1
            elif src == "path_only":
                doc_stats["path_only"] += 1
            else:
                doc_stats["skipped"] += 1

    # ---- text comparison summary ----
    avg_delta = 0.0
    med_delta = 0.0
    if os.path.exists(TEXT_COMP):
        tc = pd.read_csv(TEXT_COMP)
        avg_delta = tc["delta_rank"].mean()
        med_delta = tc["delta_rank"].median()
        n_improved = (tc["delta_rank"] > 0).sum()
        n_regressed = (tc["delta_rank"] < 0).sum()
    else:
        n_improved = n_regressed = 0

    # ---- tactic counts ----
    bad_d3 = d3[(d3["rank"] == 1) & d3["mathlib_module"].apply(
        lambda x: any(x.startswith(p) for p in INFRA_PREFIXES))]

    # ==== WRITE PRIVATE EVAL FILE ====
    eval_lines = [
        "# Deliverable 3 — Detailed Evaluation\n",
        f"## Three-way precision on {d1_t} gold pairs\n",
        f"| Metric | D1 | D2 | D3 |",
        "|--------|----|----|-----|",
        f"| P@1    | {d1_p1}/{d1_t} = {d1_p1/d1_t:.1%} "
        f"| {d2_p1}/{d1_t} = {d2_p1/d1_t:.1%} "
        f"| {d3_p1}/{d1_t} = {d3_p1/d1_t:.1%} |",
        f"| P@5    | {d1_p5}/{d1_t} = {d1_p5/d1_t:.1%} "
        f"| {d2_p5}/{d1_t} = {d2_p5/d1_t:.1%} "
        f"| {d3_p5}/{d1_t} = {d3_p5/d1_t:.1%} |",
        "",
        "## Per-pair detail\n",
        "| Module | D1 | D2 | D3 | D1 top-1 | D2 top-1 | D3 top-1 |",
        "|--------|----|----|----|----|----|----|",
    ]
    for p in per_pair:
        eval_lines.append(
            f"| {p['module']} | {p['d1']} | {p['d2']} | {p['d3']} "
            f"| {p['m1'].split('.')[-1]} "
            f"| {p['m2'].split('.')[-1]} "
            f"| {p['m3'].split('.')[-1]} |"
        )

    eval_lines += [
        "",
        f"## D2->D3 improvements: {len(flip_right)}\n",
    ]
    for mc, old, new, s2, s3 in flip_right:
        eval_lines.append(f"- {mc}: {old} -> {new} (score {s3:.3f})")

    eval_lines += [
        "",
        f"## D2->D3 regressions: {len(flip_wrong)}\n",
    ]
    for mc, old, new, s2, s3 in flip_wrong:
        eval_lines.append(f"- {mc}: {old} -> {new} (score {s3:.3f})")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_lines))
    print(f"\n[eval_v3] Saved {OUT_EVAL}")

    # ==== WRITE PUBLIC SUMMARY ====
    n_with_doc = doc_stats["with_doc"]
    n_decl = doc_stats["decl_only"]
    n_total = doc_stats["total"] if doc_stats["total"] > 0 else 7661
    pct_doc = n_with_doc / n_total * 100 if n_total > 0 else 0
    pct_decl = n_decl / n_total * 100 if n_total > 0 else 0

    summary = [
        "# Deliverable 3: Enhanced Semantic Alignment via Mathlib Docstrings\n",
        "**Author:** Elgün Hasanov  ",
        "**Supervisors:** Thomas Bonald (Télécom Paris), Marc Lelarge (ENS)  ",
        "**Date:** April 2026\n",
        "---\n",

        "## 1. Objective\n",
        "Deliverables 1 and 2 constructed the Mathlib text signal exclusively from "
        "tokenised module paths (e.g. `Mathlib.Algebra.Group.Basic` → "
        "*algebra group basic*). This makes the text signal nearly redundant with "
        "the name similarity signal — the two are ≥ 90 % correlated. "
        "Deliverable 3 replaces this proxy with real semantic content scraped "
        "directly from Lean source files on GitHub, then re-runs the identical "
        "D2 iterative pipeline as a controlled experiment.\n",

        "## 2. Data collection\n",
        f"Each of the {n_total} Mathlib modules corresponds to a `.lean` file at "
        "`https://raw.githubusercontent.com/leanprover-community/mathlib4/master/`.\n",
        "Files are cached on disk so the scrape can be resumed without re-fetching.\n",
        "Three text sources are extracted per file:\n",
        "- **Module docstring**: `/-! … -/` blocks, markdown headers stripped.",
        "- **Declaration bag**: names from `def`, `theorem`, `lemma`, `class`, "
        "  `structure`, `instance`, `inductive`, `abbrev` declarations, "
        "  CamelCase-split into tokens.",
        "- **Path tokens**: the existing signal, retained as a fallback.\n",
        "### Coverage\n",
        f"| Source type | Modules | Share |",
        "|-------------|---------|-------|",
        f"| Docstring present | {n_with_doc} | {pct_doc:.1f}% |",
        f"| Declarations only | {n_decl} | {pct_decl:.1f}% |",
        f"| Path tokens only  | {doc_stats['path_only']} "
        f"| {doc_stats['path_only']/n_total*100:.1f}% |",
        f"| Skipped (non-Mathlib) | {doc_stats['skipped']} | — |\n",

        "## 3. Method\n",
        "The text similarity matrix is recomputed with `TfidfVectorizer` "
        "(`min_df=2, max_df=0.7, ngram_range=(1,2), max_features=10000, "
        "sublinear_tf=True, stop_words='english'`) fitted jointly on all "
        "MathComp and Mathlib texts. "
        "MathComp texts are unchanged: scraped HTML descriptions + module name tokens. "
        "The resulting 106 × 7,661 cosine-similarity matrix replaces the old "
        "`text_sim.npz`.\n",
        "The iterative pipeline (`iterative_alignment_v3.py`) is an exact clone of "
        "the D2 pipeline with one substitution: `text_sim_v2.npz` in place of "
        "`text_sim.npz`. All weights (W_NAME=0.40, W_TEXT=0.45, W_CAT=0.15), "
        "thresholds, anchor validation, tactic mask, and propagation cap are "
        "identical — ensuring a controlled single-variable experiment.\n",

        "## 4. Results\n",
        "Evaluated on the same 62-pair gold standard used in D1 and D2 "
        "(partial prefix matching).\n",
        f"| Metric   | D1 (baseline) | D2 (iterative) | D3 (+ docstrings) |",
        "|----------|:---:|:---:|:---:|",
        f"| **P@1**  | {d1_p1}/{d1_t} = **{d1_p1/d1_t:.1%}** "
        f"| {d2_p1}/{d1_t} = **{d2_p1/d1_t:.1%}** "
        f"| {d3_p1}/{d1_t} = **{d3_p1/d1_t:.1%}** |",
        f"| **P@5**  | {d1_p5}/{d1_t} = **{d1_p5/d1_t:.1%}** "
        f"| {d2_p5}/{d1_t} = **{d2_p5/d1_t:.1%}** "
        f"| {d3_p5}/{d1_t} = **{d3_p5/d1_t:.1%}** |",
        f"| Tactic@1 | 8 | 0 | {len(bad_d3)} |\n",

        f"D2->D3 improvements (miss at D2, hit at D3): **{len(flip_right)}** modules.  ",
        f"D2->D3 regressions (hit at D2, miss at D3): **{len(flip_wrong)}** modules.\n",
    ]

    # Case studies
    if flip_right:
        summary.append("### Case studies: modules improved by docstrings\n")
        for mc, old, new, s2, s3 in flip_right[:5]:
            summary.append(
                f"- **{mc}**: D2 matched `{old.split('.')[-1]}` "
                f"(score {s2:.3f}); D3 matched `{new.split('.')[-1]}` "
                f"(score {s3:.3f})."
            )
        summary.append("")

    # Per-cluster table
    summary += [
        "## 5. Per-cluster breakdown\n",
        "| Cluster | D1 P@1 | D2 P@1 | D3 P@1 |",
        "|---------|--------|--------|--------|",
    ]
    for _, row in cluster_df.iterrows():
        def fmt(val):
            return f"{val:.0%}" if pd.notna(val) else "—"
        summary.append(
            f"| {row['cluster']} | {fmt(row.get('p1_d1'))} "
            f"| {fmt(row.get('p1_d2'))} | {fmt(row.get('p1_d3'))} |"
        )

    # Text signal comparison
    summary += [
        "",
        "## 6. Text signal comparison\n",
        "For the 62 gold pairs, the rank of the correct Mathlib module "
        "under the new text signal compared to the old path-only proxy:\n",
        f"- Average rank improvement: **{avg_delta:.0f}** positions  ",
        f"- Median rank improvement: **{med_delta:.0f}** positions  ",
        f"- Modules with improved rank: **{n_improved}**  ",
        f"- Modules with lower rank: **{n_regressed}**\n",
        "![Text signal scatter](figures/text_signal_scatter.png)\n",
        "![Rank improvement histogram](figures/rank_improvement.png)\n",
    ]

    # Docstring coverage figure
    summary += [
        "## 7. Coverage analysis\n",
        "![Docstring coverage](figures/docstring_coverage.png)\n",
        "![D1/D2/D3 comparison](figures/d1_d2_d3_comparison.png)\n",
    ]

    summary += [
        "## 8. Remaining challenges\n",
        "- **burnside_app** (and a few similar modules) remain ambiguous: "
        "multiple `GroupTheory` sub-modules score nearly identically, "
        "and the docstring does not provide enough discriminating signal.",
        "- Some modules, particularly in the `Data.*` namespace, have no "
        "docstring at all — the declaration bag helps but is weaker.",
        "- MathComp-specific naming conventions (SSReflect prefixes, "
        "abbreviations like `ssralg`, `bigop`) still do not match "
        "Mathlib's more verbose naming without synonym expansion.\n",

        "## 9. Possible directions\n",
        "- **Embedding-based retrieval**: encode docstrings and declarations "
        "with a mathematical language model; cosine search in embedding space.",
        "- **Cross-library synonym expansion**: map SSReflect abbreviations "
        "(`mx`, `seq`, `zmod`) to their Mathlib equivalents automatically.",
        "- **Manual annotation**: extend the gold standard beyond 62 pairs "
        "to provide a more robust evaluation signal.",
        "- **Weighted TF-IDF**: up-weight docstring tokens relative to "
        "declaration tokens to better exploit the semantic content.",
    ]

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"[eval_v3] Saved {OUT_SUMMARY}")


if __name__ == "__main__":
    run()
