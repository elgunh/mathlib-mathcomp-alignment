"""Part B: Hierarchical Prefix Matching for Mathlib alignment.

Motivation
----------
MathComp has ONE module per concept (e.g. `finset`).
Mathlib has a SUBTREE of 5-50 files per concept (e.g. Data.Finset.*,
Data.Finset.Card, Data.Finset.Lattice, …).

Scoring against individual leaves means the group evidence is fragmented.
Aggregating scores over the CONCEPT GROUP (2-level Mathlib prefix) captures
"there are 10 matching finset files" rather than picking the single best one.

Algorithm
---------
1. Build concept groups: group all Mathlib modules by their 2-level prefix
   (e.g. "Data.Finset", "Algebra.BigOperators", "GroupTheory.Sylow").
2. Compute base_score matrix (106 × 7661):
     base = 0.55 * name_sim + 0.25 * text_sim_v3 + 0.20 * cat_sim
3. Aggregate per group:
     group_score(mc, g) = 0.7 * max(base[mc, members(g)])
                        + 0.3 * mean(base[mc, members(g)] > 0)
4. For each MC module, find top-K concept groups.
5. Hierarchical bonus for a candidate: proportional to the group_score of the
   concept group it belongs to.

Output
------
  data/processed/hier_group_scores.npz — (106, n_groups) group score matrix
  data/processed/hier_group_index.json — group index {prefix: [ml_indices]}
  This data is consumed by rerank_candidates_v4_1.py.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse

NAME_SIM   = os.path.join("data", "processed", "name_sim.npz")
TEXT_V3    = os.path.join("data", "processed", "text_sim_v3.npz")
CAT_SIM    = os.path.join("data", "processed", "category_sim.npz")
MC_MOD     = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MOD     = os.path.join("data", "processed", "mathlib_modules.csv")

OUT_SCORES = os.path.join("data", "processed", "hier_group_scores.npz")
OUT_INDEX  = os.path.join("data", "processed", "hier_group_index.json")

W_NAME = 0.55
W_TEXT = 0.25
W_CAT  = 0.20

INFRA_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)


def build_concept_groups(ml_names: list[str],
                         levels: tuple[int, ...] = (2, 3)) -> dict[str, list[int]]:
    """Group Mathlib modules by their 2-level and 3-level prefix (after Mathlib.)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(ml_names):
        if any(name.startswith(p) for p in INFRA_PREFIXES):
            continue
        parts = name.replace("Mathlib.", "").split(".")
        for level in levels:
            if len(parts) >= level:
                prefix = ".".join(parts[:level])
                groups[prefix].append(idx)
        if len(parts) == 1:
            groups[parts[0]].append(idx)
    return dict(groups)


def group_score_vec(base_row: np.ndarray,
                    member_indices: list[int]) -> float:
    """0.7 * max + 0.3 * mean-of-nonzero aggregate."""
    if not member_indices:
        return 0.0
    scores = base_row[member_indices]
    mx = float(np.max(scores))
    pos = scores[scores > 0]
    mean_nz = float(np.mean(pos)) if len(pos) > 0 else 0.0
    return 0.7 * mx + 0.3 * mean_nz


def compute():
    print("[hier] Loading similarity matrices…")
    mc_mod  = pd.read_csv(MC_MOD)
    ml_mod  = pd.read_csv(ML_MOD)
    mc_ids  = list(mc_mod["module_id"])
    ml_names = list(ml_mod["module_name"])
    n_mc, n_ml = len(mc_ids), len(ml_names)

    name_sim = sparse.load_npz(NAME_SIM).toarray()
    text_v3  = sparse.load_npz(TEXT_V3).toarray()
    cat_sim  = sparse.load_npz(CAT_SIM).toarray()

    # Trim matrices to (n_mc, n_ml)
    name_sim = name_sim[:n_mc, :n_ml]
    text_v3  = text_v3[:n_mc, :n_ml]
    cat_sim  = cat_sim[:n_mc, :n_ml]

    base = W_NAME * name_sim + W_TEXT * text_v3 + W_CAT * cat_sim
    print(f"[hier] Base score matrix: {base.shape}, "
          f"max={base.max():.3f}, mean={base.mean():.4f}")

    groups = build_concept_groups(ml_names)
    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)
    print(f"[hier] Concept groups: {n_groups} (2-level prefixes)")

    # Build group score matrix (n_mc × n_groups)
    g_scores = np.zeros((n_mc, n_groups), dtype=np.float32)
    for gi, prefix in enumerate(group_keys):
        members = groups[prefix]
        for i in range(n_mc):
            g_scores[i, gi] = group_score_vec(base[i], members)

    # Save as sparse
    g_sparse = sparse.csr_matrix(g_scores)
    os.makedirs(os.path.dirname(OUT_SCORES), exist_ok=True)
    sparse.save_npz(OUT_SCORES, g_sparse)

    # Save index
    index = {
        "group_keys": group_keys,
        "groups": {k: groups[k] for k in group_keys},
    }
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"[hier] Saved {OUT_SCORES}")
    print(f"[hier] Saved {OUT_INDEX}")

    # Diagnostics: top-3 groups for hard modules
    hard = ["bigop", "fingraph", "finset", "fraction", "eqtype",
            "abelian", "morphism", "quotient", "ring_quotient",
            "prime", "zmodp", "rat", "div", "ssrint"]
    gold_prefixes = {
        "bigop": "Algebra.BigOperators",
        "fingraph": "Combinatorics.SimpleGraph",
        "finset": "Data.Finset",
        "fraction": "RingTheory.Localization",
        "eqtype": "Logic.Equiv",
        "abelian": "GroupTheory.Abelian",
        "morphism": "GroupTheory.GroupHom",
        "quotient": "GroupTheory.QuotientGroup",
        "ring_quotient": "RingTheory.Ideal",
        "prime": "Data.Nat.Prime",
        "zmodp": "Data.ZMod",
        "rat": "Data.Rat",
        "div": "Data.Nat",
        "ssrint": "Data.Int",
    }

    print(f"\n[hier] Top-3 concept groups for hard modules:")
    for mc_mid in hard:
        if mc_mid not in mc_ids:
            continue
        i = mc_ids.index(mc_mid)
        row = g_scores[i]
        top3 = np.argsort(row)[-3:][::-1]
        gold = gold_prefixes.get(mc_mid, "")
        print(f"\n  {mc_mid} (gold group: {gold})")
        for gi in top3:
            mark = " <<GOLD" if group_keys[gi] == gold else ""
            print(f"    {group_keys[gi]:40s} {row[gi]:.4f} "
                  f"({len(groups[group_keys[gi]])} members){mark}")

    return group_keys, groups, g_scores, mc_ids, ml_names


if __name__ == "__main__":
    compute()
