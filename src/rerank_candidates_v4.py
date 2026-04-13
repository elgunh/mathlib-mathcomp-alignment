"""Reranker v4: D3.2 features + IDF-weighted declaration-name overlap.

Extended reranking formula:

  reranked = base_score
           + 0.25 * concept_match_bonus
           + 0.08 * synonym_overlap_bonus
           + 0.10 * text_v3_score
           - 0.04 * broad_namespace_penalty
           + w_decl * declaration_overlap_score   ← NEW

Runs a weight sweep over w_decl in [0.00, 0.03, 0.05, 0.08, 0.10]
and evaluates each configuration on the 62-pair gold standard.

Outputs
-------
  outputs/iterative_matches_d4_w{w}.csv   for each w_decl value
  outputs/d4_weight_sweep.csv             sweep summary table
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from build_synonym_map import SYNONYM_MAP, MODULE_LEVEL_EXTRA, expand_module_name

MATCHES_IN   = os.path.join("outputs", "iterative_matches_v3_textv3.csv")
TEXT_V3_NPZ  = os.path.join("data", "processed", "text_sim_v3.npz")
DECL_NPZ     = os.path.join("data", "processed", "decl_sim.npz")
MC_MODULES   = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES   = os.path.join("data", "processed", "mathlib_modules.csv")

SWEEP_WEIGHTS = [0.00, 0.03, 0.05, 0.08, 0.10]

BAD_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)

GOLD_PAIRS = {
    "sylow": ["Mathlib.GroupTheory.Sylow"],
    "nilpotent": ["Mathlib.GroupTheory.Nilpotent"],
    "cyclic": ["Mathlib.GroupTheory.SpecificGroups.Cyclic"],
    "perm": ["Mathlib.GroupTheory.Perm"],
    "abelian": ["Mathlib.GroupTheory.Abelianization", "Mathlib.GroupTheory.AbelianGroup"],
    "commutator": ["Mathlib.GroupTheory.Commutator"],
    "center": ["Mathlib.GroupTheory.Subgroup.Center"],
    "quotient": ["Mathlib.GroupTheory.QuotientGroup", "Mathlib.GroupTheory.Coset"],
    "action": ["Mathlib.GroupTheory.GroupAction"],
    "fingroup": ["Mathlib.GroupTheory"],
    "morphism": ["Mathlib.GroupTheory.GroupHom", "Mathlib.GroupTheory.Subgroup"],
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
    "ssrnat": ["Mathlib.Data.Nat"],
    "ssrint": ["Mathlib.Data.Int"],
    "rat": ["Mathlib.Data.Rat"],
    "prime": ["Mathlib.Data.Nat.Prime"],
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
    "fintype": ["Mathlib.Data.Fintype"],
    "finset": ["Mathlib.Data.Finset"],
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


def camel_split(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t.lower() for t in s.split() if len(t) > 1]


def module_path_tokens(module_name: str) -> set[str]:
    parts = module_name.split(".")
    if parts and parts[0].lower() == "mathlib":
        parts = parts[1:]
    tokens = set()
    for p in parts:
        tokens.add(p.lower())
        for sub in camel_split(p):
            tokens.add(sub)
    return tokens


def gold_match(prefixes, ml):
    return any(ml.startswith(p) for p in prefixes)


def precision_at_k(df, k, scol):
    hits, total = 0, 0
    for mc, prefixes in GOLD_PAIRS.items():
        sub = df[df["mathcomp_module"] == mc]
        if sub.empty:
            continue
        total += 1
        topk = sub.sort_values(scol, ascending=False).head(k)
        if any(gold_match(prefixes, r["mathlib_module"]) for _, r in topk.iterrows()):
            hits += 1
    return hits, total


def rerank(w_decl: float, df: pd.DataFrame,
           text_v3: np.ndarray, decl_sim: np.ndarray,
           mc_ids: list, ml_names: list,
           mc_id_to_idx: dict, ml_name_to_idx: dict) -> pd.DataFrame:
    """Apply full reranker (v3 features + declaration overlap) to df."""

    result_rows = []
    for mc_mid, grp in df.groupby("mathcomp_module", sort=False):
        mc_idx = mc_id_to_idx.get(mc_mid)
        if mc_idx is None:
            result_rows.append(grp)
            continue

        mc_toks = set(expand_module_name(mc_mid, SYNONYM_MAP, MODULE_LEVEL_EXTRA))
        candidates_ml = list(grp["mathlib_module"])

        scored = []
        for _, row in grp.iterrows():
            ml_name = row["mathlib_module"]
            ml_idx  = ml_name_to_idx.get(ml_name)
            base    = float(row["final_score"])

            # ── v3 features ──────────────────────────────────────────────
            ml_toks = module_path_tokens(ml_name)

            # Feature 1: concept match
            raw_mc = set(re.split(r"[_/]", mc_mid.lower()))
            for p in list(raw_mc):
                raw_mc.update(camel_split(p))
            overlap = sum(
                2.0 if tok in raw_mc else 1.0
                for tok in mc_toks if tok in ml_toks
            )
            n_weighted = sum(2.0 if tok in raw_mc else 1.0 for tok in mc_toks)
            f_concept = overlap / max(n_weighted, 1.0)

            # Feature 2: synonym overlap (Jaccard after expansion)
            inter_syn = mc_toks & ml_toks
            union_syn = mc_toks | ml_toks
            f_syn = len(inter_syn) / max(len(union_syn), 1)

            # Feature 3: text_v3
            f_tv3 = float(text_v3[mc_idx, ml_idx]) if ml_idx is not None else 0.0

            # Feature 4: broad namespace penalty
            ns2 = ".".join(ml_name.split(".")[:3])
            broad_ns = {".".join(c.split(".")[:3])
                        for c in candidates_ml}  # check if exact-name cand exists
            raw_mc_set = set(re.split(r"[_/]", mc_mid.lower()))
            has_better = any(
                c != ml_name and any(r in module_path_tokens(c) for r in raw_mc_set)
                for c in candidates_ml
            )
            f_broad = 1.0 if (ns2 in {
                "Mathlib.GroupTheory", "Mathlib.Algebra",
                "Mathlib.Data", "Mathlib.Order", "Mathlib.LinearAlgebra",
            } and has_better) else 0.0

            # ── NEW: declaration overlap ──────────────────────────────────
            f_decl = float(decl_sim[mc_idx, ml_idx]) if ml_idx is not None else 0.0

            reranked = (base
                        + 0.25 * f_concept
                        + 0.08 * f_syn
                        + 0.10 * f_tv3
                        - 0.04 * f_broad
                        + w_decl * f_decl)

            scored.append((row, f_concept, f_syn, f_tv3, f_broad, f_decl, reranked))

        # Sort by reranked score
        scored.sort(key=lambda x: -x[-1])
        for new_rank, (row, fc, fs, ft, fb, fd, rs) in enumerate(scored, 1):
            nr = dict(row)
            nr["original_rank"]    = int(nr.get("rank", new_rank))
            nr["rank"]             = new_rank
            nr["reranked_score"]   = round(rs, 4)
            nr["concept_bonus"]    = round(fc * 0.25, 4)
            nr["synonym_bonus"]    = round(fs * 0.08, 4)
            nr["text_v3_contrib"]  = round(ft * 0.10, 4)
            nr["decl_contrib"]     = round(fd * w_decl, 4)
            result_rows.append(pd.Series(nr).to_frame().T)

    return pd.concat(result_rows, ignore_index=True)


def run_sweep(matches_in: str = MATCHES_IN):
    print("[rerank_v4] Loading matrices…")
    mc_mod  = pd.read_csv(MC_MODULES)
    ml_mod  = pd.read_csv(ML_MODULES)
    mc_ids  = list(mc_mod["module_id"])
    ml_names= list(ml_mod["module_name"])
    mc_id_to_idx  = {mid: i for i, mid in enumerate(mc_ids)}
    ml_name_to_idx= {n: i for i, n in enumerate(ml_names)}

    text_v3  = sparse.load_npz(TEXT_V3_NPZ).toarray()
    decl_sim = sparse.load_npz(DECL_NPZ).toarray()

    df = pd.read_csv(matches_in)
    print(f"[rerank_v4] Input: {matches_in}  ({len(df)} rows)")

    os.makedirs("outputs", exist_ok=True)
    sweep_rows = []

    for w_decl in SWEEP_WEIGHTS:
        print(f"\n[rerank_v4] w_decl={w_decl:.2f}…")
        df_out = rerank(w_decl, df, text_v3, decl_sim,
                        mc_ids, ml_names, mc_id_to_idx, ml_name_to_idx)

        out_path = f"outputs/iterative_matches_d4_w{w_decl:.2f}.csv"
        df_out.to_csv(out_path, index=False)

        h1, t1 = precision_at_k(df_out, 1, "reranked_score")
        h5, t5 = precision_at_k(df_out, 5, "reranked_score")
        print(f"  P@1={h1}/{t1}={h1/t1*100:.1f}%  P@5={h5}/{t5}={h5/t5*100:.1f}%")
        sweep_rows.append({"w_decl": w_decl,
                           "p1": round(h1/t1, 4), "p5": round(h5/t5, 4),
                           "h1": h1, "h5": h5, "n": t1,
                           "P@1_str": f"{h1/t1*100:.1f}% ({h1}/{t1})",
                           "P@5_str": f"{h5/t5*100:.1f}%"})

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("outputs/d4_weight_sweep.csv", index=False)
    print(f"\n[rerank_v4] Saved outputs/d4_weight_sweep.csv")
    print(sweep_df[["w_decl","P@1_str","P@5_str"]].to_string(index=False))

    # Pick best weight
    best = sweep_df.loc[sweep_df["p1"].idxmax()]
    best_w = float(best["w_decl"])
    print(f"\n[rerank_v4] Best w_decl={best_w:.2f}: "
          f"P@1={best['P@1_str']}  P@5={best['P@5_str']}")

    # Copy best to canonical output path
    best_src = f"outputs/iterative_matches_d4_w{best_w:.2f}.csv"
    import shutil
    shutil.copy(best_src, "outputs/iterative_matches_d4_best.csv")
    print(f"[rerank_v4] Best output: outputs/iterative_matches_d4_best.csv")

    return sweep_df, best_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_in", default=MATCHES_IN)
    args = parser.parse_args()
    run_sweep(args.matches_in)
