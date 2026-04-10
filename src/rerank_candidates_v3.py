"""Lightweight top-k reranker for Deliverable 3.2.

Takes the top-10 candidates from the calibrated D3 iterative pipeline and
re-scores them using interpretable additive features:

  reranked = base_score
           + α * concept_match_bonus     (key concept word in Mathlib path)
           + β * synonym_overlap_bonus   (expanded Jaccard similarity)
           + γ * text_v3_score           (richer TF-IDF from text_sim_v3)
           - δ * broad_ns_penalty        (penalise overly-generic namespace hits)

Concept-match bonus is the primary discriminator: it fixes cases where semantic
docstring overlap pushed a semantically adjacent but wrong module to rank-1
(e.g. commutator→solvable, galois→extension, cyclotomic→polynomial_roots).

Output
------
  outputs/iterative_matches_v3_reranked.csv
  outputs/rerank_feature_breakdown.csv
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from build_synonym_map import (
    SYNONYM_MAP, MODULE_LEVEL_EXTRA,
    expand_module_name, expand_tokens,
)

# ---- Inputs ----
MATCHES_IN   = os.path.join("outputs", "iterative_matches_v3_textv3.csv")
TEXT_V3_NPZ  = os.path.join("data", "processed", "text_sim_v3.npz")
TEXT_V2_NPZ  = os.path.join("data", "processed", "text_sim_v2.npz")
NAME_NPZ     = os.path.join("data", "processed", "name_sim.npz")
MC_MODULES   = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES   = os.path.join("data", "processed", "mathlib_modules.csv")

MATCHES_OUT  = os.path.join("outputs", "iterative_matches_v3_reranked.csv")
BREAKDOWN_OUT= os.path.join("outputs", "rerank_feature_breakdown.csv")

BAD_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)

# ---- Reranker weights (additive) ----
# Tuned by hand to be conservative: concept_match is strongest,
# synonym_overlap adds secondary support, text_v3 replaces text_v2.
ALPHA_CONCEPT    = 0.25   # exact concept token in Mathlib path
BETA_SYNONYM     = 0.08   # expanded Jaccard overlap
GAMMA_TEXT_V3    = 0.10   # text_sim_v3 supplement
DELTA_BROAD_NS   = 0.04   # broad-namespace penalty

# Modules whose Mathlib counterpart has a very generic top-level namespace:
# penalise a match if the candidate shares only the top-level component
# and there's a more specific matching path in the top-10.
BROAD_NAMESPACES = {"Mathlib.GroupTheory", "Mathlib.Algebra",
                    "Mathlib.Data", "Mathlib.Order", "Mathlib.LinearAlgebra"}


def camel_split(name: str) -> list[str]:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return [t.lower() for t in s.split() if len(t) > 1]


def module_path_tokens(module_name: str) -> set[str]:
    """CamelCase-split all path components of a Mathlib module name."""
    parts = module_name.split(".")
    if parts and parts[0].lower() == "mathlib":
        parts = parts[1:]
    tokens = set()
    for p in parts:
        tokens.add(p.lower())
        for sub in camel_split(p):
            tokens.add(sub)
    return tokens


def mc_concept_tokens(module_id: str) -> set[str]:
    """
    Return the expanded concept-token set for a MathComp module.
    These are the tokens we look for in the Mathlib candidate path.
    """
    expanded = expand_module_name(module_id, SYNONYM_MAP, MODULE_LEVEL_EXTRA)
    return set(t.lower() for t in expanded)


def concept_match_bonus(mc_id: str, ml_name: str,
                        mc_tokens: set, ml_tokens: set) -> float:
    """
    Fraction of MathComp concept tokens found in the Mathlib path tokens.
    High bonus when the MathComp name word literally appears in the Mathlib path.

    Example:
      mc=galois  -> concept_tokens={'galois', ...}
      ml=Mathlib.FieldTheory.Galois.Basic -> ml_tokens={'galois','fieldtheory','basic'}
      overlap = {'galois'} / mc_tokens -> bonus = 1/n_mc_toks
    """
    if not mc_tokens:
        return 0.0
    # Weight direct module-ID tokens higher (the raw name, not expansions)
    raw_parts = set(re.split(r'[_/]', mc_id.lower()))
    for p in list(raw_parts):
        raw_parts.update(camel_split(p))

    # Count overlapping tokens, with higher weight for raw-name tokens
    overlap = 0.0
    for tok in mc_tokens:
        if tok in ml_tokens:
            weight = 2.0 if tok in raw_parts else 1.0
            overlap += weight

    n_weighted = sum(2.0 if tok in raw_parts else 1.0 for tok in mc_tokens)
    return overlap / max(n_weighted, 1.0)


def synonym_overlap(mc_tokens: set, ml_tokens: set) -> float:
    """Jaccard similarity after synonym expansion."""
    intersection = len(mc_tokens & ml_tokens)
    union = len(mc_tokens | ml_tokens)
    return intersection / max(union, 1)


def broad_ns_penalty(ml_name: str, mc_id: str, candidates: list[str]) -> float:
    """
    Penalise a candidate that shares only the top-level Mathlib namespace
    when a more specific (deeper) match exists in the candidate set.

    Only fires when the candidate top-2-component prefix is among
    BROAD_NAMESPACES and ≥1 other candidate in the top-10 has a longer
    path prefix match to the MathComp concept.
    """
    ns2 = ".".join(ml_name.split(".")[:3])  # e.g. Mathlib.GroupTheory.Nilpotent
    if ns2 not in BROAD_NAMESPACES:
        return 0.0
    # Check if any other candidate in the top-10 has the MathComp module name
    # (or a synonym) in its path
    raw_mc = re.split(r'[_/]', mc_id.lower())
    for c in candidates:
        if c == ml_name:
            continue
        c_toks = module_path_tokens(c)
        if any(r in c_toks for r in raw_mc):
            # A better-named candidate exists → penalise the broad-namespace one
            return 1.0
    return 0.0


def rerank(matches_in: str = MATCHES_IN,
           out_matches: str = MATCHES_OUT,
           out_breakdown: str = BREAKDOWN_OUT):
    if not os.path.exists(matches_in):
        print(f"[rerank] Input not found: {matches_in}")
        print("  Run: python src/iterative_alignment_v3.py "
              "--text_sim data/processed/text_sim_v3.npz "
              "--out_matches outputs/iterative_matches_v3_textv3.csv")
        return

    print("[rerank] Loading data…")
    df = pd.read_csv(matches_in)
    mc_mod = pd.read_csv(MC_MODULES)
    ml_mod = pd.read_csv(ML_MODULES)

    mc_ids   = list(mc_mod["module_id"])
    ml_names = list(ml_mod["module_name"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}
    ml_name_to_idx = {n: i for i, n in enumerate(ml_names)}

    text_v3 = sparse.load_npz(TEXT_V3_NPZ).toarray()

    # Pre-compute MC concept token sets
    mc_concept_map = {mid: mc_concept_tokens(mid) for mid in mc_ids}
    # Pre-compute ML path token sets (cached)
    ml_path_map: dict = {}

    def get_ml_tokens(ml_name: str) -> set:
        if ml_name not in ml_path_map:
            ml_path_map[ml_name] = module_path_tokens(ml_name)
        return ml_path_map[ml_name]

    # Process each MathComp module
    breakdown_rows = []
    result_rows = []

    for mc_mid, grp in df.groupby("mathcomp_module", sort=False):
        mc_idx = mc_id_to_idx.get(mc_mid)
        if mc_idx is None:
            result_rows.append(grp)
            continue

        mc_toks = mc_concept_map[mc_mid]
        candidates_ml = list(grp["mathlib_module"])

        scored = []
        for _, row in grp.iterrows():
            ml_name = row["mathlib_module"]
            ml_idx  = ml_name_to_idx.get(ml_name)
            base    = float(row["final_score"])

            # Feature 1: concept match
            ml_toks = get_ml_tokens(ml_name)
            f_concept = concept_match_bonus(mc_mid, ml_name, mc_toks, ml_toks)

            # Feature 2: synonym overlap
            f_syn = synonym_overlap(mc_toks, ml_toks)

            # Feature 3: text_v3 score
            f_tv3 = float(text_v3[mc_idx, ml_idx]) if ml_idx is not None else 0.0

            # Feature 4: broad-namespace penalty
            is_bad = any(ml_name.startswith(p) for p in BAD_PREFIXES)
            f_broad = broad_ns_penalty(ml_name, mc_mid, candidates_ml) if not is_bad else 0.0

            reranked = (base
                        + ALPHA_CONCEPT * f_concept
                        + BETA_SYNONYM  * f_syn
                        + GAMMA_TEXT_V3 * f_tv3
                        - DELTA_BROAD_NS * f_broad)

            scored.append((row, f_concept, f_syn, f_tv3, f_broad, reranked))
            breakdown_rows.append({
                "mc_module":  mc_mid,
                "ml_module":  ml_name,
                "base_score": round(base, 4),
                "concept":    round(f_concept, 4),
                "synonym":    round(f_syn, 4),
                "text_v3":    round(f_tv3, 4),
                "broad_pen":  round(f_broad, 4),
                "reranked":   round(reranked, 4),
            })

        # Re-sort by reranked score
        scored.sort(key=lambda x: -x[-1])
        for new_rank, (row, fc, fs, ft, fb, rs) in enumerate(scored, 1):
            new_row = dict(row)
            new_row["original_rank"] = int(new_row["rank"])
            new_row["rank"]          = new_rank
            new_row["reranked_score"]= round(rs, 4)
            new_row["concept_bonus"] = round(fc * ALPHA_CONCEPT, 4)
            new_row["synonym_bonus"] = round(fs * BETA_SYNONYM, 4)
            new_row["text_v3_contrib"] = round(ft * GAMMA_TEXT_V3, 4)
            result_rows.append(pd.Series(new_row).to_frame().T)

    df_out = pd.concat(result_rows, ignore_index=True)
    os.makedirs("outputs", exist_ok=True)
    df_out.to_csv(out_matches, index=False)

    df_break = pd.DataFrame(breakdown_rows)
    df_break.to_csv(out_breakdown, index=False)

    # Quick stats
    top1_before = df[df["rank"] == 1]["mathlib_module"]
    top1_after  = df_out[df_out["rank"] == 1]["mathlib_module"]
    changed = (top1_before.values != top1_after.values).sum()

    print(f"[rerank] Rank-1 changes: {changed}/106")
    print(f"[rerank] Saved {out_matches}")
    print(f"[rerank] Saved {out_breakdown}")

    # Show concept-match bonus for hard cases
    hard = ["commutator", "galois", "cyclotomic", "eqtype", "path",
            "presentation", "burnside_app", "mxalgebra", "ssrnat"]
    print("\n[rerank] Concept-match bonus for hard cases:")
    for mc_mid in hard:
        row = df_break[df_break["mc_module"] == mc_mid].sort_values(
            "reranked", ascending=False).head(1)
        if not row.empty:
            r = row.iloc[0]
            print(f"  {mc_mid:20s} top: {r['ml_module'].split('.')[-1][:35]:35s} "
                  f"base={r['base_score']:.3f} concept={r['concept']:.3f} "
                  f"reranked={r['reranked']:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_in",   default=MATCHES_IN)
    parser.add_argument("--out_matches",  default=MATCHES_OUT)
    parser.add_argument("--out_breakdown",default=BREAKDOWN_OUT)
    args = parser.parse_args()
    rerank(args.matches_in, args.out_matches, args.out_breakdown)
