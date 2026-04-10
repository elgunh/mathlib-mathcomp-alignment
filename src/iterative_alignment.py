"""Iterative graph-propagation alignment.

Round 0: base alignment using name + text + category (no graph signal).
Round k: propagate alignment through import graphs to discover new anchors.

Fixes applied:
- Tactic/Linter/Meta namespaces excluded from anchoring (never valid MC targets)
- Anchor validation: only propagate from high-confidence anchors
- Threshold floor at 0.25; decay with exponent 0.9 instead of 0.8
- Propagation bonus capped at 50% of base_score to prevent flipping
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
MC_EDGES = os.path.join("data", "processed", "mathcomp_edges.csv")
ML_EDGES = os.path.join("data", "processed", "mathlib_edges.csv")
NAME_SIM = os.path.join("data", "processed", "name_sim.npz")
TEXT_SIM = os.path.join("data", "processed", "text_sim.npz")
CAT_SIM = os.path.join("data", "processed", "category_sim.npz")

OUT_MATCHES = os.path.join("outputs", "iterative_matches.csv")
OUT_LOG = os.path.join("outputs", "propagation_log.csv")

W_NAME = 0.40
W_TEXT = 0.45
W_CAT = 0.15

TOP_K = 10
N_ROUNDS = 6
HIGH_THRESHOLD = 0.35
PROPAGATION_DECAY = 0.90
THRESHOLD_FLOOR = 0.25

# Namespaces that are internal Lean infrastructure — never valid MC targets
BAD_PREFIXES = (
    "Mathlib.Tactic",
    "Mathlib.Lean",
    "Mathlib.Meta",
    "Mathlib.Elab",
    "Mathlib.Linter",
    "Mathlib.Attr",
    "Mathlib.Parser",
    "Mathlib.Init",
)


def is_valid_anchor_target(ml_module: str) -> bool:
    return not any(ml_module.startswith(p) for p in BAD_PREFIXES)


def build_neighborhoods(edges_df, src_col, tgt_col):
    nbrs = defaultdict(set)
    for _, row in edges_df.iterrows():
        s, t = str(row[src_col]), str(row[tgt_col])
        nbrs[s].add(t)
        nbrs[t].add(s)
    return dict(nbrs)


def run():
    print("[iterative] Loading data...")
    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    mc_ids = list(mc["module_id"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}
    ml_names = list(ml["module_name"])

    # Validity mask: which Mathlib modules are acceptable targets
    valid_mask = np.array([is_valid_anchor_target(name) for name in ml_names],
                          dtype=bool)
    print(f"[iterative] Valid Mathlib anchor targets: "
          f"{valid_mask.sum()}/{n_ml} "
          f"({valid_mask.sum()/n_ml:.0%}; excluded Tactic/Linter/Meta etc.)")

    name_sim = sparse.load_npz(NAME_SIM).toarray()
    text_sim = sparse.load_npz(TEXT_SIM).toarray()
    cat_sim = sparse.load_npz(CAT_SIM).toarray()

    base_score = W_NAME * name_sim + W_TEXT * text_sim + W_CAT * cat_sim

    # Zero out scores for invalid (Tactic/Linter/...) targets so they
    # can never win unless truly nothing better exists
    base_score[:, ~valid_mask] *= 0.05

    print(f"[iterative] Base score range after masking: "
          f"max={base_score.max():.3f}, mean={base_score.mean():.4f}")

    mc_edges = pd.read_csv(MC_EDGES)
    ml_edges = pd.read_csv(ML_EDGES)

    mc_nbrs = build_neighborhoods(mc_edges, "source", "target")

    ml_idx_strs = ml["module_idx"].astype(str).tolist()
    ml_nbrs = defaultdict(set)
    for _, row in ml_edges.iterrows():
        s, t = str(row["source"]), str(row["target"])
        ml_nbrs[s].add(t)
        ml_nbrs[t].add(s)

    ml_str_to_pos = {s: i for i, s in enumerate(ml_idx_strs)}

    propagation_bonus = np.zeros((n_mc, n_ml), dtype=np.float64)

    # anchors[mc_idx] = (ml_idx, final_score, round)
    anchors = {}
    anchor_round = {}
    # propagation_sources: subset of anchors that are high-confidence enough
    # to safely seed propagation (Fix 1)
    propagation_sources = set()

    log_rows = []

    # --- Round 0 ---
    print(f"\n[iterative] === Round 0 (threshold={HIGH_THRESHOLD}, "
          f"no graph signal) ===")

    for i in range(n_mc):
        scores_i = base_score[i]
        sorted_j = np.argsort(scores_i)[::-1]
        best_j = sorted_j[0]
        second_j = sorted_j[1]
        score1 = scores_i[best_j]
        score2 = scores_i[second_j]
        gap = score1 - score2

        if score1 >= HIGH_THRESHOLD:
            anchors[i] = (int(best_j), float(score1), 0)
            anchor_round[i] = 0

            # Only use as propagation source if clearly confident (Fix 1)
            if score1 >= 0.40 or gap >= 0.10:
                propagation_sources.add(i)

    print(f"  Anchored round 0: {len(anchors)}/{n_mc} "
          f"({len(propagation_sources)} as propagation sources)")

    if anchors:
        sample = sorted(anchors.items(), key=lambda x: -x[1][1])[:5]
        for mc_i, (ml_j, sc, _) in sample:
            print(f"    {mc_ids[mc_i]:25s} -> "
                  f"{ml_names[ml_j]:55s} ({sc:.3f})")

    log_rows.append({
        "round": 0,
        "new_anchors": len(anchors),
        "total_anchors": len(anchors),
        "propagation_sources": len(propagation_sources),
        "avg_score": float(np.mean([v[1] for v in anchors.values()])) if anchors else 0.0,
        "modules_remaining": n_mc - len(anchors),
    })

    # --- Rounds 1..N ---
    for rnd in range(1, N_ROUNDS + 1):
        threshold = max(HIGH_THRESHOLD * (PROPAGATION_DECAY ** rnd), THRESHOLD_FLOOR)
        new_anchors_this_round = 0
        new_sources_this_round = 0
        print(f"\n[iterative] === Round {rnd} (threshold={threshold:.3f}) ===")

        # Build propagation bonuses using only validated sources
        for i in range(n_mc):
            if i in anchors:
                continue

            mid = mc_ids[i]
            mc_nb = mc_nbrs.get(mid, set())
            if not mc_nb:
                continue

            # Only look at neighbors that ARE propagation sources
            anchored_nb = []
            for nb_id in mc_nb:
                nb_idx = mc_id_to_idx.get(nb_id)
                if nb_idx is not None and nb_idx in propagation_sources:
                    anchored_nb.append((nb_idx, anchors[nb_idx]))

            if not anchored_nb:
                continue

            n_nb = len(mc_nb)
            for nb_idx, (ml_anchor_j, anchor_score, _) in anchored_nb:
                ml_anchor_str = ml_idx_strs[ml_anchor_j]
                ml_nb_set = ml_nbrs.get(ml_anchor_str, set())
                bonus_weight = (PROPAGATION_DECAY ** rnd) * anchor_score / n_nb
                for ml_nb_str in ml_nb_set:
                    ml_pos = ml_str_to_pos.get(ml_nb_str)
                    if ml_pos is not None and valid_mask[ml_pos]:
                        propagation_bonus[i, ml_pos] += bonus_weight

        # Cap bonus at 25% of base score — prevents graph from overriding strong text evidence
        effective_bonus = np.minimum(propagation_bonus, 0.25 * base_score)
        final_score = base_score + effective_bonus

        for i in range(n_mc):
            if i in anchors:
                continue
            scores_i = final_score[i]
            sorted_j = np.argsort(scores_i)[::-1]
            best_j = sorted_j[0]
            second_j = sorted_j[1]
            score1 = scores_i[best_j]
            score2 = scores_i[second_j]
            gap = score1 - score2

            if score1 >= threshold:
                anchors[i] = (int(best_j), float(score1), rnd)
                anchor_round[i] = rnd
                new_anchors_this_round += 1

                if score1 >= 0.40 or gap >= 0.10:
                    propagation_sources.add(i)
                    new_sources_this_round += 1

        print(f"  New anchors: {new_anchors_this_round} "
              f"({new_sources_this_round} become propagation sources), "
              f"total: {len(anchors)}/{n_mc}")

        log_rows.append({
            "round": rnd,
            "new_anchors": new_anchors_this_round,
            "total_anchors": len(anchors),
            "propagation_sources": len(propagation_sources),
            "avg_score": float(np.mean([v[1] for v in anchors.values()])) if anchors else 0.0,
            "modules_remaining": n_mc - len(anchors),
        })

        if new_anchors_this_round == 0:
            print("  No new anchors — stopping early.")
            break

    # --- Final pass: assign remaining unanchored modules their best score ---
    effective_bonus = np.minimum(propagation_bonus, 0.25 * base_score)
    final_score = base_score + effective_bonus

    print(f"\n[iterative] Assigning remaining {n_mc - len(anchors)} unanchored modules...")
    for i in range(n_mc):
        if i not in anchors:
            best_j = int(np.argmax(final_score[i]))
            anchors[i] = (best_j, float(final_score[i, best_j]), -1)
            anchor_round[i] = -1

    # --- Build output ---
    rows = []
    for i in range(n_mc):
        scores = final_score[i]
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        mid = mc_ids[i]
        cluster = mc["cluster"].iloc[i]

        a_round = anchor_round.get(i, -1)
        is_prop_source = i in propagation_sources
        if a_round == 0 and is_prop_source:
            tier = "HIGH"
        elif a_round == 0:
            tier = "HIGH-UNCERTAIN"
        elif 0 < a_round <= 2:
            tier = "MEDIUM"
        elif a_round > 2:
            tier = "MEDIUM-LOW"
        else:
            tier = "LOW"

        for rank, j in enumerate(top_indices, 1):
            pb = float(propagation_bonus[i, j])
            eff_pb = float(min(pb, 0.25 * base_score[i, j]))
            rows.append({
                "mathcomp_module": mid,
                "mathcomp_cluster": cluster,
                "mathlib_module": ml_names[j],
                "mathlib_category": ml["category"].iloc[j],
                "base_score": round(float(base_score[i, j]), 4),
                "propagation_bonus": round(eff_pb, 4),
                "final_score": round(float(scores[j]), 4),
                "anchor_round": a_round if rank == 1 else -1,
                "confidence_tier": tier if rank == 1 else "",
                "rank": rank,
            })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_MATCHES), exist_ok=True)
    df_out.to_csv(OUT_MATCHES, index=False)

    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(OUT_LOG, index=False)

    top1 = df_out[df_out["rank"] == 1]
    print(f"\n[iterative] === Summary ===")
    print(f"  Total anchored (non-floor): "
          f"{sum(1 for v in anchor_round.values() if v >= 0)}/{n_mc}")

    tier_counts = top1["confidence_tier"].value_counts()
    for t in ["HIGH", "HIGH-UNCERTAIN", "MEDIUM", "MEDIUM-LOW", "LOW"]:
        print(f"    {t}: {tier_counts.get(t, 0)}")

    bad = top1[top1["mathlib_module"].apply(
        lambda x: any(x.startswith(p) for p in BAD_PREFIXES))]
    print(f"\n  Tactic/Linter/Meta matches: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"    {r['mathcomp_module']:20s} -> {r['mathlib_module']}")

    improved = int((effective_bonus > 0.001).any(axis=1).sum())
    print(f"  Modules with effective propagation bonus: {improved}/{n_mc}")

    print(f"\n  Convergence:")
    for lr in log_rows:
        print(f"    Round {int(lr['round'])}: +{int(lr['new_anchors'])} anchors "
              f"({int(lr['propagation_sources'])} sources), "
              f"remaining={int(lr['modules_remaining'])}")

    print(f"\n[iterative] Saved {OUT_MATCHES}")
    print(f"[iterative] Saved {OUT_LOG}")


if __name__ == "__main__":
    run()
