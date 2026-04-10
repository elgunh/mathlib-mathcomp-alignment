"""Compare old (path-only) vs new (docstring-based) text similarity signal.

For each of the 62 gold pairs, reports old text score, old rank, new text
score, new rank, and the delta — proving whether docstrings helped.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse

ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
OLD_NPZ = os.path.join("data", "processed", "text_sim.npz")
NEW_NPZ = os.path.join("data", "processed", "text_sim_v2.npz")
MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")

# Same gold standard as evaluate_v2.py
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


def best_prefix_rank(sim_row: np.ndarray, ml_names: list, prefixes: list) -> tuple:
    """Find the best (highest-scoring) correct Mathlib module and its rank."""
    scored = sorted(enumerate(sim_row), key=lambda x: -x[1])
    ranked = [(j, sc) for j, sc in scored]
    # Find rank of best correct match
    for rank_1indexed, (j, sc) in enumerate(ranked, 1):
        if any(ml_names[j].startswith(p) for p in prefixes):
            return rank_1indexed, float(sc), ml_names[j]
    return len(ml_names), 0.0, ""


def compare():
    if not os.path.exists(OLD_NPZ):
        print(f"[compare] Missing {OLD_NPZ}")
        sys.exit(1)
    if not os.path.exists(NEW_NPZ):
        print(f"[compare] Missing {NEW_NPZ} — run text_similarity_v2.py first")
        sys.exit(1)

    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    ml_names = list(ml["module_name"])
    mc_ids = list(mc["module_id"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}

    old_sim = sparse.load_npz(OLD_NPZ).toarray()
    new_sim = sparse.load_npz(NEW_NPZ).toarray()

    rows = []
    for mc_mod, prefixes in GOLD_PAIRS.items():
        mc_idx = mc_id_to_idx.get(mc_mod)
        if mc_idx is None:
            continue
        old_rank, old_score, old_match = best_prefix_rank(
            old_sim[mc_idx], ml_names, prefixes)
        new_rank, new_score, new_match = best_prefix_rank(
            new_sim[mc_idx], ml_names, prefixes)
        rows.append({
            "module": mc_mod,
            "old_text": round(old_score, 3),
            "old_rank": old_rank,
            "new_text": round(new_score, 3),
            "new_rank": new_rank,
            "delta_rank": old_rank - new_rank,
            "old_match": old_match.split(".")[-1] if old_match else "",
            "new_match": new_match.split(".")[-1] if new_match else "",
        })

    df = pd.DataFrame(rows).sort_values("delta_rank", ascending=False)

    improved = (df["delta_rank"] > 0).sum()
    regressed = (df["delta_rank"] < 0).sum()
    same = (df["delta_rank"] == 0).sum()

    print("[compare_text_signals] Old vs new text signal for 62 gold pairs")
    print(f"  Improved: {improved}, Regressed: {regressed}, Unchanged: {same}\n")

    header = f"{'Module':<20} {'OldTxt':>7} {'OldRk':>6} {'NewTxt':>7} {'NewRk':>6} {'Delta':>6}"
    print(header)
    print("-" * len(header))
    for _, r in df.head(30).iterrows():
        arrow = "+" if r["delta_rank"] > 0 else ("" if r["delta_rank"] == 0 else "")
        print(f"{r['module']:<20} {r['old_text']:>7.3f} {r['old_rank']:>6} "
              f"{r['new_text']:>7.3f} {r['new_rank']:>6} "
              f"{arrow}{int(r['delta_rank']):>5}")

    # Save
    out_path = os.path.join("outputs", "text_signal_comparison.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[compare_text_signals] Saved {out_path}")

    # Also print worst regressions
    reg = df[df["delta_rank"] < 0].sort_values("delta_rank")
    if not reg.empty:
        print("\n  Regressions (new signal ranks correct match lower):")
        for _, r in reg.iterrows():
            print(f"    {r['module']:<20} old_rank={r['old_rank']} "
                  f"new_rank={r['new_rank']} delta={int(r['delta_rank'])}")

    return df


if __name__ == "__main__":
    compare()
