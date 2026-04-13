"""Part A: Gold Standard Audit for D4.1.

Inspects the 18 modules wrong in D3.2 and decides for each:
  CONFIRMED_ERROR  — the match is genuinely wrong
  FALSE_ERROR      — the match is actually correct; gold prefix needs expansion
  AMBIGUOUS        — the match is defensible but not ideal

Produces:
  outputs/gold_audit_results.md
  data/processed/gold_standard_v2.json
"""

import os
import json
import pandas as pd

D32_RERANKED  = "outputs/iterative_matches_v3_reranked.csv"
D32_BASE      = "outputs/iterative_matches_v3_textv3.csv"
MC_DESC       = "data/processed/mathcomp_descriptions.csv"
ML_DOCS       = "data/processed/mathlib_docstrings.csv"
OUT_REPORT    = "outputs/gold_audit_results.md"
OUT_GOLD_V2   = "data/processed/gold_standard_v2.json"

# ── Gold v1 ───────────────────────────────────────────────────────────────
GOLD_V1 = {
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
                       "Mathlib.GroupTheory.Subgroup"],
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
                       "Mathlib.GroupTheory.CompositionSeries"],
    "burnside_app":   ["Mathlib.GroupTheory.GroupAction",
                       "Mathlib.GroupTheory.Burnside"],
    "ssralg":         ["Mathlib.Algebra.Ring", "Mathlib.Algebra.Group"],
    "matrix":         ["Mathlib.LinearAlgebra.Matrix"],
    "poly":           ["Mathlib.RingTheory.Polynomial",
                       "Mathlib.Algebra.Polynomial"],
    "ring_quotient":  ["Mathlib.RingTheory.Ideal.Quotient"],
    "intdiv":         ["Mathlib.Data.Int.Div", "Mathlib.Data.Int"],
    "bigop":          ["Mathlib.Algebra.BigOperators"],
    "ssrnat":         ["Mathlib.Data.Nat"],
    "ssrint":         ["Mathlib.Data.Int"],
    "rat":            ["Mathlib.Data.Rat"],
    "prime":          ["Mathlib.Data.Nat.Prime"],
    "zmodp":          ["Mathlib.Data.ZMod"],
    "fraction":       ["Mathlib.RingTheory.Localization.FractionRing"],
    "binomial":       ["Mathlib.Data.Nat.Choose"],
    "mxpoly":         ["Mathlib.LinearAlgebra.Matrix.Polynomial",
                       "Mathlib.LinearAlgebra.Matrix.Charpoly"],
    "mxalgebra":      ["Mathlib.LinearAlgebra.Matrix"],
    "vector":         ["Mathlib.LinearAlgebra"],
    "sesquilinear":   ["Mathlib.LinearAlgebra.SesquilinearForm"],
    "mxrepresentation": ["Mathlib.RepresentationTheory"],
    "character":      ["Mathlib.RepresentationTheory.Character"],
    "vcharacter":     ["Mathlib.RepresentationTheory.Character"],
    "separable":      ["Mathlib.FieldTheory.Separable"],
    "galois":         ["Mathlib.FieldTheory.Galois",
                       "Mathlib.FieldTheory.Finite.GaloisField"],
    "algC":           ["Mathlib.FieldTheory.IsAlgClosed"],
    "cyclotomic":     ["Mathlib.NumberTheory.Cyclotomic",
                       "Mathlib.RingTheory.Polynomial.Cyclotomic"],
    "fieldext":       ["Mathlib.FieldTheory.Extension"],
    "finfield":       ["Mathlib.FieldTheory.Finite",
                       "Mathlib.FieldTheory.Galois"],
    "order":          ["Mathlib.Order.Lattice", "Mathlib.Order"],
    "preorder":       ["Mathlib.Order.Preorder", "Mathlib.Order"],
    "archimedean":    ["Mathlib.Algebra.Order.Archimedean"],
    "seq":            ["Mathlib.Data.Seq", "Mathlib.Data.List"],
    "fintype":        ["Mathlib.Data.Fintype"],
    "finset":         ["Mathlib.Data.Finset"],
    "tuple":          ["Mathlib.Data.Vector", "Mathlib.Data.Fin.Tuple"],
    "eqtype":         ["Mathlib.Logic.Equiv", "Mathlib.Data.Subtype"],
    "choice":         ["Mathlib.Logic.Classical", "Mathlib.Order.Zorn"],
    "path":           ["Mathlib.Combinatorics.SimpleGraph",
                       "Mathlib.Topology.Path"],
    "fingraph":       ["Mathlib.Combinatorics.SimpleGraph"],
    "div":            ["Mathlib.Data.Nat.Div", "Mathlib.Data.Int.Div"],
    "finfun":         ["Mathlib.Data.PiFin", "Mathlib.Data.Fin"],
    "ssrbool":        ["Mathlib.Data.Bool"],
    "ssrfun":         ["Mathlib.Logic.Function"],
    "classfun":       ["Mathlib.RepresentationTheory"],
}

# ── Audit decisions (manually derived after inspecting docstrings) ─────────
#
# FALSE_ERROR = the match IS correct, gold prefix was too narrow
# AMBIGUOUS   = match is defensible but not clearly the primary concept
# CONFIRMED_ERROR = match is genuinely wrong
#
AUDIT_DECISIONS = {
    "abelian": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.GroupTheory.FreeAbelianGroup",
        "reasoning": (
            "FreeAbelianGroup is about the FREE abelian group on a type, defined as "
            "the abelianisation of the free group. MathComp's `abelian` module "
            "concerns general abelianisation theory and properties of abelian groups "
            "in finite group theory. These are related but the free abelian group is "
            "a specific construction distinct from the general abelian group theory."
        ),
    },
    "automorphism": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.GroupTheory.Coset.Basic",
        "reasoning": (
            "Coset.Basic covers left and right cosets. MathComp's `automorphism` "
            "module is about group automorphisms. No conceptual connection."
        ),
    },
    "bigop": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Data.Finset.Sum",
        "reasoning": (
            "Data.Finset.Sum is one specific operation over finsets. MathComp's "
            "`bigop` is the complete general-purpose big-operator library covering "
            "∑, ∏, bigmax, bigmin over arbitrary commutative structures. The primary "
            "Mathlib analogue is Mathlib.Algebra.BigOperators, which is absent from "
            "the top-10 candidates because bigop's name/text signal overlaps more "
            "with finset operations than with the BigOperators namespace."
        ),
    },
    "div": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Data.Nat.Digits.Div",
        "reasoning": (
            "Digits.Div covers divisibility tests BASED ON DIGIT REPRESENTATIONS "
            "(e.g., a number is divisible by 9 iff its digit sum is). MathComp's "
            "`div` is the general Euclidean division algorithm for naturals. "
            "Conceptually unrelated despite sharing the word 'div'."
        ),
    },
    "eqtype": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Data.TypeVec",
        "reasoning": (
            "TypeVec is about type-indexed product families and is used in "
            "categorical constructions (QPF). MathComp's `eqtype` defines the "
            "decidable equality interface (eqType typeclass) used throughout SSReflect. "
            "These are completely unrelated."
        ),
    },
    "fingraph": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Order.Preorder.Finite",
        "reasoning": (
            "Order.Preorder.Finite is about finitely-many-element preorders. "
            "MathComp's `fingraph` defines finite graph connectivity (dfs, connect, "
            "the `connect` relation). The correct match is Mathlib.Combinatorics.SimpleGraph "
            "but no SimpleGraph module appears in the top-10 candidates."
        ),
    },
    "finset": {
        "verdict": "AMBIGUOUS",
        "d32_match": "Mathlib.Data.Finite.Set",
        "reasoning": (
            "Finite.Set proves lemmas relating Finset and (set-theoretic) Set types. "
            "MathComp's `finset` defines finsets as the primary finite-set data "
            "structure. Finite.Set IS about finsets but is an adapter module; the "
            "primary finset library is Data.Finset. The match is mathematically "
            "adjacent but not the canonical answer. Marked AMBIGUOUS because "
            "Finite.Set does discuss finsets explicitly."
        ),
    },
    "fraction": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.RingTheory.WittVector.FrobeniusFractionField",
        "reasoning": (
            "FrobeniusFractionField is a deep module about the Frobenius endomorphism "
            "on the fraction field of the Witt vectors W(k). MathComp's `fraction` "
            "is the general fraction field library (Localization, FractionRing). "
            "Conceptually distant; the correct match (RingTheory.Localization.FractionRing) "
            "is at rank 7 in base scoring."
        ),
    },
    "gseries": {
        "verdict": "AMBIGUOUS",
        "d32_match": "Mathlib.GroupTheory.IsSubnormal",
        "reasoning": (
            "IsSubnormal defines subnormal subgroups — a subgroup H is subnormal in G "
            "iff there exists a subnormal series from H to G. MathComp's `gseries` "
            "defines composition series, chief series, and the general gseries "
            "construction. The match is not ideal (series vs subnormality property) "
            "but is mathematically adjacent. Marked AMBIGUOUS."
        ),
    },
    "intdiv": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Algebra.MonoidAlgebra.Division",
        "reasoning": (
            "MonoidAlgebra.Division covers division of polynomials / multivariate "
            "polynomials (monoid algebras) by monomials. MathComp's `intdiv` provides "
            "Euclidean division for integers (`divz`, `modz`, `dvdz`). Entirely "
            "different content despite sharing 'division' vocabulary."
        ),
    },
    "morphism": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.GroupTheory.Coset.Card",
        "reasoning": (
            "Coset.Card is about cardinality of cosets (index of a subgroup). "
            "MathComp's `morphism` is the group morphism library. No match."
        ),
    },
    "path": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.Data.List.Cycle",
        "reasoning": (
            "Data.List.Cycle models cyclic ROTATIONS of lists. MathComp's `path` "
            "defines relational paths and cycles in directed graphs: "
            "`path p x s` is a list s such that consecutive elements satisfy "
            "relation p starting from x. The Mathlib analogue is SimpleGraph.Walk. "
            "Data.List.Cycle coincidentally overlaps on 'cycle' but is unrelated."
        ),
    },
    "prime": {
        "verdict": "FALSE_ERROR",
        "d32_match": "Mathlib.Data.List.Prime",
        "reasoning": (
            "Data.List.Prime docstring: 'Products of lists of prime elements. "
            "This file contains theorems relating products of lists of prime elements "
            "and products of squarefree numbers.' This IS about prime factorization: "
            "it defines and studies lists whose elements are prime, exactly what "
            "MathComp's `prime` module covers (primality, prime factorization, "
            "unique factorization). The gold prefix `Mathlib.Data.Nat.Prime` was "
            "too narrow; `Mathlib.Data.List.Prime` is a valid co-equal match. "
            "Adding `Mathlib.Data.List.Prime` to gold_v2."
        ),
        "gold_v2_addition": ["Mathlib.Data.List.Prime"],
    },
    "quotient": {
        "verdict": "FALSE_ERROR",
        "d32_match": "Mathlib.GroupTheory.GroupAction.Quotient",
        "reasoning": (
            "GroupAction.Quotient docstring: 'Properties of group actions involving "
            "quotient groups. This file proves properties of group actions which use "
            "the quotient group structure.' This file is EXPLICITLY about quotient "
            "groups — it is the group-action perspective on G/N. MathComp's `quotient` "
            "module defines the quotient group G/N and its properties. The match is "
            "genuine: both files are about quotient groups. Gold_v1 was too narrow "
            "by accepting only QuotientGroup.* and Coset.* but not GroupAction.Quotient. "
            "Adding `Mathlib.GroupTheory.GroupAction.Quotient` to gold_v2."
        ),
        "gold_v2_addition": ["Mathlib.GroupTheory.GroupAction.Quotient"],
    },
    "rat": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.NumberTheory.Cyclotomic.Rat",
        "reasoning": (
            "NumberTheory.Cyclotomic.Rat has no docstring (empty). Inspecting the "
            "module, it provides specific results about cyclotomic polynomials "
            "evaluated over the rationals — a highly specialized topic. MathComp's "
            "`rat` module is the rational number library (arithmetic, order, floor/ceil). "
            "Cyclotomic.Rat is not a match despite containing 'Rat' in its path."
        ),
    },
    "ring_quotient": {
        "verdict": "AMBIGUOUS",
        "d32_match": "Mathlib.RingTheory.LocalRing.Quotient",
        "reasoning": (
            "LocalRing.Quotient docstring: 'We gather results about the quotients of "
            "local rings.' MathComp's `ring_quotient` is: 'Quotient rings by an ideal'. "
            "A quotient of a local ring IS a quotient ring (by an ideal). The match "
            "is semantically adjacent but covers a MORE SPECIFIC case (local rings) "
            "rather than the general ideal quotient. Marked AMBIGUOUS; the primary "
            "canonical match remains RingTheory.Ideal.Quotient."
        ),
    },
    "ssrint": {
        "verdict": "CONFIRMED_ERROR",
        "d32_match": "Mathlib.RingTheory.Int.Basic",
        "reasoning": (
            "RingTheory.Int.Basic covers divisibility over ℤ using ring-theoretic "
            "methods. MathComp's `ssrint` is the complete integer library: arithmetic "
            "(`addz`, `mulz`, `absz`), ordering, divisibility, GCD, and the `int` "
            "type itself. The primary Mathlib match is Data.Int, which provides the "
            "integer type and basic operations. RingTheory.Int.Basic is a secondary, "
            "specialized module."
        ),
    },
    "zmodp": {
        "verdict": "FALSE_ERROR",
        "d32_match": "Mathlib.Algebra.Field.ZMod",
        "reasoning": (
            "Algebra.Field.ZMod docstring: 'is a field'. Specifically, it proves "
            "that ℤ/pℤ is a field for prime p. MathComp's `zmodp` module is "
            "defined as: 'Z/pZ is a field when p is prime'. These two modules "
            "have IDENTICAL mathematical content. Gold_v1 only accepted "
            "`Mathlib.Data.ZMod` (the type definition), missing the fact that "
            "Algebra.Field.ZMod is the precise analogue of the `zmodp` module. "
            "Adding `Mathlib.Algebra.Field.ZMod`, `Mathlib.Algebra.ZMod`, "
            "and `Mathlib.RingTheory.ZMod` to gold_v2."
        ),
        "gold_v2_addition": [
            "Mathlib.Algebra.Field.ZMod",
            "Mathlib.Algebra.ZMod",
            "Mathlib.RingTheory.ZMod",
        ],
    },
}


def build_gold_v2():
    import copy
    gv2 = copy.deepcopy(GOLD_V1)
    for mc_mod, info in AUDIT_DECISIONS.items():
        if info["verdict"] == "FALSE_ERROR" and "gold_v2_addition" in info:
            existing = gv2.get(mc_mod, [])
            for new_prefix in info["gold_v2_addition"]:
                if new_prefix not in existing:
                    existing.append(new_prefix)
            gv2[mc_mod] = existing
    return gv2


def run():
    df32 = pd.read_csv(D32_RERANKED)
    df_base = pd.read_csv(D32_BASE)

    try:
        mc_desc_df = pd.read_csv(MC_DESC)
        mc_desc_map = {r["module_id"]: str(r.get("description", ""))
                       for _, r in mc_desc_df.iterrows()}
    except Exception:
        mc_desc_map = {}

    ml_docs_df = pd.read_csv(ML_DOCS)
    ml_doc_map = {r["module_name"]: str(r.get("docstring", ""))
                  for _, r in ml_docs_df.iterrows()}

    # Find wrong modules
    wrong_mods = set()
    for mc, prefixes in GOLD_V1.items():
        sub = df32[df32["mathcomp_module"] == mc]
        if sub.empty:
            continue
        t1 = sub.sort_values("reranked_score", ascending=False).iloc[0]
        ml = t1["mathlib_module"]
        if not any(ml.startswith(p) for p in prefixes):
            wrong_mods.add(mc)

    gold_v2 = build_gold_v2()

    # Evaluate on both
    def eval_gold(gld, df, scol):
        h, t = 0, 0
        for mc, prefixes in gld.items():
            sub = df[df["mathcomp_module"] == mc]
            if sub.empty:
                continue
            t += 1
            t1 = sub.sort_values(scol, ascending=False).iloc[0]["mathlib_module"]
            if any(t1.startswith(p) for p in prefixes):
                h += 1
        return h, t

    h1_v1, n = eval_gold(GOLD_V1, df32, "reranked_score")
    h1_v2, _ = eval_gold(gold_v2, df32, "reranked_score")

    false_errors = [m for m, d in AUDIT_DECISIONS.items() if d["verdict"] == "FALSE_ERROR"]
    ambiguous    = [m for m, d in AUDIT_DECISIONS.items() if d["verdict"] == "AMBIGUOUS"]
    confirmed    = [m for m, d in AUDIT_DECISIONS.items() if d["verdict"] == "CONFIRMED_ERROR"]

    print(f"[audit] D3.2 on gold_v1: {h1_v1}/{n} = {h1_v1/n*100:.1f}%")
    print(f"[audit] D3.2 on gold_v2: {h1_v2}/{n} = {h1_v2/n*100:.1f}%")
    print(f"[audit] FALSE_ERROR ({len(false_errors)}): {false_errors}")
    print(f"[audit] AMBIGUOUS   ({len(ambiguous)}):    {ambiguous}")
    print(f"[audit] CONFIRMED   ({len(confirmed)}):  {confirmed}")

    # Save gold_v2
    os.makedirs("data/processed", exist_ok=True)
    gold_export = {
        "gold_v1": GOLD_V1,
        "gold_v2": gold_v2,
        "audit_decisions": {
            k: {"verdict": v["verdict"], "reasoning": v["reasoning"]}
            for k, v in AUDIT_DECISIONS.items()
        },
    }
    with open(OUT_GOLD_V2, "w", encoding="utf-8") as f:
        json.dump(gold_export, f, indent=2)
    print(f"[audit] Saved {OUT_GOLD_V2}")

    # Write markdown report
    false_n  = len(false_errors)
    ambig_n  = len(ambiguous)
    conf_n   = len(confirmed)

    lines = [
        "# Gold Standard Audit — D4.1\n",
        f"D3.2 P@1 on gold_v1: **{h1_v1}/{n} = {h1_v1/n*100:.1f}%**\n",
        f"D3.2 P@1 on gold_v2: **{h1_v2}/{n} = {h1_v2/n*100:.1f}%** "
        f"(+{h1_v2-h1_v1} from audit)\n\n",
        "## Summary\n\n",
        f"| Verdict | Count | Modules |\n",
        f"|---------|-------|---------|\n",
        f"| FALSE_ERROR | {false_n} | {', '.join(f'`{m}`' for m in false_errors)} |\n",
        f"| AMBIGUOUS   | {ambig_n} | {', '.join(f'`{m}`' for m in ambiguous)} |\n",
        f"| CONFIRMED_ERROR | {conf_n} | {', '.join(f'`{m}`' for m in confirmed)} |\n\n",
        "## Gold v2 additions\n\n",
    ]

    for mc_mod, info in AUDIT_DECISIONS.items():
        if info["verdict"] == "FALSE_ERROR":
            adds = info.get("gold_v2_addition", [])
            lines.append(f"- **`{mc_mod}`**: added `{', '.join(adds)}`\n")
    lines.append("\n## Per-module audit\n\n")

    for mc_mod in sorted(AUDIT_DECISIONS.keys()):
        info = AUDIT_DECISIONS[mc_mod]
        verdict = info["verdict"]
        d32_top1 = info["d32_match"]
        doc = ml_doc_map.get(d32_top1, "")
        if doc and doc != "nan":
            doc = doc[:200].replace("\n", " ")
        else:
            doc = "(no docstring)"

        sub32 = df32[df32["mathcomp_module"] == mc_mod]
        sub_base = df_base[df_base["mathcomp_module"] == mc_mod]
        top5_base = sub_base.sort_values("final_score", ascending=False).head(5)

        gold_v1 = GOLD_V1.get(mc_mod, [])
        gold_v2_pref = gold_v2.get(mc_mod, [])

        mc_d = mc_desc_map.get(mc_mod, "(no description)")
        if mc_d and len(mc_d) > 200:
            mc_d = mc_d[:200] + "…"

        emoji = {"FALSE_ERROR": "🟢", "AMBIGUOUS": "🟡", "CONFIRMED_ERROR": "🔴"}.get(verdict, "")

        lines.append(f"### `{mc_mod}` {emoji} {verdict}\n\n")
        lines.append(f"**MathComp description**: {mc_d}\n\n")
        lines.append(f"**D3.2 top-1 match**: `{d32_top1}`\n")
        lines.append(f"**Match docstring**: {doc}\n\n")
        lines.append(f"**Gold v1 prefixes**: {gold_v1}\n\n")
        if gold_v2_pref != gold_v1:
            lines.append(f"**Gold v2 prefixes**: {gold_v2_pref} ← EXPANDED\n\n")
        lines.append(f"**Reasoning**: {info['reasoning']}\n\n")

        lines.append("**Top-5 base-score candidates**:\n\n")
        lines.append("| Rank | Module | Score | Gold v1 | Gold v2 |\n")
        lines.append("|------|--------|-------|---------|----------|\n")
        for rk, (_, r) in enumerate(top5_base.iterrows(), 1):
            ml = r["mathlib_module"]
            sc = r["final_score"]
            hit_v1 = "YES" if any(ml.startswith(p) for p in gold_v1) else ""
            hit_v2 = "YES" if any(ml.startswith(p) for p in gold_v2_pref) else ""
            lines.append(f"| {rk} | `{ml}` | {sc:.3f} | {hit_v1} | {hit_v2} |\n")
        lines.append("\n---\n\n")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[audit] Saved {OUT_REPORT}")
    return gold_v2, GOLD_V1, h1_v1, h1_v2, n


if __name__ == "__main__":
    run()
