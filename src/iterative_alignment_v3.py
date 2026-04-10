"""Iterative graph-propagation alignment — Deliverable 3 variant.

Uses text_sim_v2.npz (real Lean docstrings) instead of the path-token proxy.
Weights default to the calibrated values (0.55, 0.25, 0.20) found by
src/calibrate_weights.py; they can be overridden via CLI arguments for
ablation experiments.

Outputs go to *_v3.csv so D2 results are preserved for comparison.

Usage
-----
# Default calibrated weights:
  python src/iterative_alignment_v3.py

# Explicit weights (must sum to 1.0):
  python src/iterative_alignment_v3.py --w_name 0.50 --w_text 0.30 --w_cat 0.20
"""

import argparse
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
TEXT_SIM = os.path.join("data", "processed", "text_sim_v2.npz")
CAT_SIM = os.path.join("data", "processed", "category_sim.npz")

OUT_MATCHES = os.path.join("outputs", "iterative_matches_v3.csv")
OUT_LOG = os.path.join("outputs", "propagation_log_v3.csv")

# Calibrated defaults (from src/calibrate_weights.py Stage B)
W_NAME = 0.55
W_TEXT = 0.25
W_CAT  = 0.20

# Default text-sim file (may be overridden via CLI to test text_sim_v3)
DEFAULT_TEXT_SIM = TEXT_SIM

TOP_K = 10
N_ROUNDS = 6
HIGH_THRESHOLD = 0.35
PROPAGATION_DECAY = 0.90
THRESHOLD_FLOOR = 0.25

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


def run(w_name=W_NAME, w_text=W_TEXT, w_cat=W_CAT,
        text_sim_path=None, out_matches=None, out_log=None):
    text_sim_path = text_sim_path or TEXT_SIM
    out_matches   = out_matches   or OUT_MATCHES
    out_log       = out_log       or OUT_LOG
    print(f"[iterative_v3] text_sim={os.path.basename(text_sim_path)}  "
          f"w_name={w_name:.2f} w_text={w_text:.2f} w_cat={w_cat:.2f}")
    print("[iterative_v3] Loading data...")
    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    mc_ids = list(mc["module_id"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}
    ml_names = list(ml["module_name"])

    valid_mask = np.array([is_valid_anchor_target(name) for name in ml_names],
                          dtype=bool)
    print(f"[iterative_v3] Valid targets: {valid_mask.sum()}/{n_ml}")

    name_sim = sparse.load_npz(NAME_SIM).toarray()
    text_sim = sparse.load_npz(text_sim_path).toarray()
    cat_sim = sparse.load_npz(CAT_SIM).toarray()

    base_score = w_name * name_sim + w_text * text_sim + w_cat * cat_sim
    base_score[:, ~valid_mask] *= 0.05

    print(f"[iterative_v3] Base score: max={base_score.max():.3f}, "
          f"mean={base_score.mean():.4f}")

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
    anchors = {}
    anchor_round = {}
    propagation_sources = set()
    log_rows = []

    print(f"\n[iterative_v3] === Round 0 (threshold={HIGH_THRESHOLD}) ===")
    for i in range(n_mc):
        scores_i = base_score[i]
        sorted_j = np.argsort(scores_i)[::-1]
        best_j, second_j = sorted_j[0], sorted_j[1]
        score1, score2 = scores_i[best_j], scores_i[second_j]
        gap = score1 - score2
        if score1 >= HIGH_THRESHOLD:
            anchors[i] = (int(best_j), float(score1), 0)
            anchor_round[i] = 0
            if score1 >= 0.40 or gap >= 0.10:
                propagation_sources.add(i)

    print(f"  Anchored: {len(anchors)}/{n_mc} "
          f"({len(propagation_sources)} propagation sources)")
    if anchors:
        sample = sorted(anchors.items(), key=lambda x: -x[1][1])[:5]
        for mc_i, (ml_j, sc, _) in sample:
            print(f"    {mc_ids[mc_i]:25s} -> {ml_names[ml_j]:55s} ({sc:.3f})")

    log_rows.append({
        "round": 0,
        "new_anchors": len(anchors),
        "total_anchors": len(anchors),
        "propagation_sources": len(propagation_sources),
        "avg_score": float(np.mean([v[1] for v in anchors.values()])) if anchors else 0.0,
        "modules_remaining": n_mc - len(anchors),
    })

    for rnd in range(1, N_ROUNDS + 1):
        threshold = max(HIGH_THRESHOLD * (PROPAGATION_DECAY ** rnd), THRESHOLD_FLOOR)
        new_anchors_this_round = new_sources_this_round = 0
        print(f"\n[iterative_v3] === Round {rnd} (threshold={threshold:.3f}) ===")

        for i in range(n_mc):
            if i in anchors:
                continue
            mid = mc_ids[i]
            mc_nb = mc_nbrs.get(mid, set())
            if not mc_nb:
                continue
            anchored_nb = [(mc_id_to_idx[nb], anchors[mc_id_to_idx[nb]])
                           for nb in mc_nb
                           if mc_id_to_idx.get(nb) in propagation_sources]
            if not anchored_nb:
                continue
            n_nb = len(mc_nb)
            for nb_idx, (ml_anchor_j, anchor_score, _) in anchored_nb:
                ml_anchor_str = ml_idx_strs[ml_anchor_j]
                bonus_weight = (PROPAGATION_DECAY ** rnd) * anchor_score / n_nb
                for ml_nb_str in ml_nbrs.get(ml_anchor_str, set()):
                    ml_pos = ml_str_to_pos.get(ml_nb_str)
                    if ml_pos is not None and valid_mask[ml_pos]:
                        propagation_bonus[i, ml_pos] += bonus_weight

        effective_bonus = np.minimum(propagation_bonus, 0.25 * base_score)
        final_score = base_score + effective_bonus

        for i in range(n_mc):
            if i in anchors:
                continue
            scores_i = final_score[i]
            sorted_j = np.argsort(scores_i)[::-1]
            best_j, second_j = sorted_j[0], sorted_j[1]
            score1, score2 = scores_i[best_j], scores_i[second_j]
            gap = score1 - score2
            if score1 >= threshold:
                anchors[i] = (int(best_j), float(score1), rnd)
                anchor_round[i] = rnd
                new_anchors_this_round += 1
                if score1 >= 0.40 or gap >= 0.10:
                    propagation_sources.add(i)
                    new_sources_this_round += 1

        print(f"  New anchors: {new_anchors_this_round} "
              f"({new_sources_this_round} sources), total: {len(anchors)}/{n_mc}")
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

    effective_bonus = np.minimum(propagation_bonus, 0.25 * base_score)
    final_score = base_score + effective_bonus

    print(f"\n[iterative_v3] Assigning {n_mc - len(anchors)} remaining modules...")
    for i in range(n_mc):
        if i not in anchors:
            best_j = int(np.argmax(final_score[i]))
            anchors[i] = (best_j, float(final_score[i, best_j]), -1)
            anchor_round[i] = -1

    rows = []
    for i in range(n_mc):
        scores = final_score[i]
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        mid = mc_ids[i]
        cluster = mc["cluster"].iloc[i]
        a_round = anchor_round.get(i, -1)
        is_src = i in propagation_sources
        if a_round == 0 and is_src:
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
    df_out.to_csv(out_matches, index=False)
    pd.DataFrame(log_rows).to_csv(out_log, index=False)

    top1 = df_out[df_out["rank"] == 1]
    tiers = top1["confidence_tier"].value_counts()
    print(f"\n[iterative_v3] === Summary ===")
    for t in ["HIGH", "HIGH-UNCERTAIN", "MEDIUM", "MEDIUM-LOW", "LOW"]:
        print(f"  {t}: {tiers.get(t, 0)}")
    bad = top1[top1["mathlib_module"].apply(
        lambda x: any(x.startswith(p) for p in BAD_PREFIXES))]
    print(f"  Tactic/Linter/Meta: {len(bad)}")
    print(f"\n[iterative_v3] Saved {out_matches}")
    print(f"[iterative_v3] Saved {out_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative alignment v3 (docstring text)")
    parser.add_argument("--w_name",    type=float, default=W_NAME)
    parser.add_argument("--w_text",    type=float, default=W_TEXT)
    parser.add_argument("--w_cat",     type=float, default=W_CAT)
    parser.add_argument("--text_sim",  type=str,   default=TEXT_SIM,
                        help="Path to text similarity .npz (default: text_sim_v2.npz)")
    parser.add_argument("--out_matches", type=str, default=OUT_MATCHES)
    parser.add_argument("--out_log",     type=str, default=OUT_LOG)
    args = parser.parse_args()
    if abs(args.w_name + args.w_text + args.w_cat - 1.0) > 0.01:
        print(f"WARNING: weights sum to {args.w_name+args.w_text+args.w_cat:.3f}")
    run(args.w_name, args.w_text, args.w_cat,
        args.text_sim, args.out_matches, args.out_log)
