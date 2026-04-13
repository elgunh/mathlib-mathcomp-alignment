"""Reranker v4.1: D3.2 features + hierarchical concept-group bonus.

Extended formula:
  reranked = base_score
           + 0.25 * concept_match_bonus
           + 0.08 * synonym_overlap_bonus
           + 0.10 * text_v3_score
           - 0.04 * broad_namespace_penalty
           + w_hier * hierarchical_bonus   ← NEW

hierarchical_bonus(candidate, top_groups):
    For the best matching group in top-5 groups for the MathComp module,
    return that group's group_score. Else 0.

Weight sweep: w_hier in [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]

Evaluates on BOTH gold_v1 and gold_v2 for transparency.

Outputs
-------
  outputs/iterative_matches_d4_1_w{w}.csv  — reranked output per weight
  outputs/d4_1_weight_sweep.csv            — sweep summary table
  outputs/iterative_matches_d4_1_best.csv  — best w by gold_v2 P@1
"""

import os
import re
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from build_synonym_map import SYNONYM_MAP, MODULE_LEVEL_EXTRA, expand_module_name

MATCHES_IN    = os.path.join("outputs", "iterative_matches_v3_textv3.csv")
TEXT_V3_NPZ   = os.path.join("data", "processed", "text_sim_v3.npz")
HIER_SCORES   = os.path.join("data", "processed", "hier_group_scores.npz")
HIER_INDEX    = os.path.join("data", "processed", "hier_group_index.json")
MC_MODULES    = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES    = os.path.join("data", "processed", "mathlib_modules.csv")
GOLD_V2_JSON  = os.path.join("data", "processed", "gold_standard_v2.json")

SWEEP_WEIGHTS = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]
TOP_K_GROUPS  = 5   # number of top concept groups to consider for bonus


def _load_gold_standards():
    with open(GOLD_V2_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["gold_v1"], data["gold_v2"]


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


def precision_at_k(df, k, scol, gold):
    hits, total = 0, 0
    for mc, prefixes in gold.items():
        sub = df[df["mathcomp_module"] == mc]
        if sub.empty:
            continue
        total += 1
        topk = sub.sort_values(scol, ascending=False).head(k)
        if any(gold_match(prefixes, r["mathlib_module"])
               for _, r in topk.iterrows()):
            hits += 1
    return hits, total


def rerank(w_hier: float, df: pd.DataFrame,
           text_v3: np.ndarray, hier_scores: np.ndarray,
           group_keys: list[str],
           mc_ids: list[str], ml_names: list[str],
           mc_id_to_idx: dict, ml_name_to_idx: dict) -> pd.DataFrame:
    """Apply D3.2 + hierarchical bonus to df."""

    # Precompute top-K groups per MC module
    top_k_per_mc: dict[str, list[tuple[str, float]]] = {}
    for mc_mid in mc_ids:
        i = mc_id_to_idx.get(mc_mid)
        if i is None:
            top_k_per_mc[mc_mid] = []
            continue
        row = hier_scores[i]
        top_indices = np.argsort(row)[-TOP_K_GROUPS:][::-1]
        top_k_per_mc[mc_mid] = [
            (group_keys[gi], float(row[gi])) for gi in top_indices if row[gi] > 0
        ]

    result_rows = []
    for mc_mid, grp in df.groupby("mathcomp_module", sort=False):
        mc_idx = mc_id_to_idx.get(mc_mid)
        if mc_idx is None:
            result_rows.append(grp)
            continue

        mc_toks = set(expand_module_name(mc_mid, SYNONYM_MAP, MODULE_LEVEL_EXTRA))
        top_groups = top_k_per_mc.get(mc_mid, [])
        candidates_ml = list(grp["mathlib_module"])

        scored = []
        for _, row in grp.iterrows():
            ml_name = row["mathlib_module"]
            ml_idx  = ml_name_to_idx.get(ml_name)
            base    = float(row["final_score"])

            # ── v3 features ──────────────────────────────────────────────
            ml_toks = module_path_tokens(ml_name)

            raw_mc = set(re.split(r"[_/]", mc_mid.lower()))
            for p in list(raw_mc):
                raw_mc.update(camel_split(p))
            overlap = sum(
                2.0 if tok in raw_mc else 1.0
                for tok in mc_toks if tok in ml_toks
            )
            n_weighted = sum(2.0 if tok in raw_mc else 1.0 for tok in mc_toks)
            f_concept = overlap / max(n_weighted, 1.0)

            inter_syn = mc_toks & ml_toks
            union_syn = mc_toks | ml_toks
            f_syn = len(inter_syn) / max(len(union_syn), 1)

            f_tv3 = float(text_v3[mc_idx, ml_idx]) if ml_idx is not None else 0.0

            ns2 = ".".join(ml_name.split(".")[:3])
            raw_mc_set = set(re.split(r"[_/]", mc_mid.lower()))
            has_better = any(
                c != ml_name and any(r2 in module_path_tokens(c) for r2 in raw_mc_set)
                for c in candidates_ml
            )
            f_broad = 1.0 if (ns2 in {
                "Mathlib.GroupTheory", "Mathlib.Algebra",
                "Mathlib.Data", "Mathlib.Order", "Mathlib.LinearAlgebra",
            } and has_better) else 0.0

            # ── NEW: hierarchical group bonus ─────────────────────────────
            # Find the highest group_score for a group this candidate belongs to
            f_hier = 0.0
            ml_rel = ml_name.replace("Mathlib.", "")  # remove "Mathlib." prefix
            for group_prefix, g_score in top_groups:
                if ml_rel.startswith(group_prefix):
                    f_hier = max(f_hier, g_score)
            # f_hier is the raw group_score; w_hier scales the contribution

            reranked = (base
                        + 0.25 * f_concept
                        + 0.08 * f_syn
                        + 0.10 * f_tv3
                        - 0.04 * f_broad
                        + w_hier * f_hier)

            scored.append((row, f_concept, f_syn, f_tv3, f_broad, f_hier, reranked))

        scored.sort(key=lambda x: -x[-1])
        for new_rank, (row, fc, fs, ft, fb, fh, rs) in enumerate(scored, 1):
            nr = dict(row)
            nr["original_rank"]    = int(nr.get("rank", new_rank))
            nr["rank"]             = new_rank
            nr["reranked_score"]   = round(rs, 4)
            nr["hier_bonus"]       = round(fh * w_hier, 4)
            result_rows.append(pd.Series(nr).to_frame().T)

    return pd.concat(result_rows, ignore_index=True)


def run_sweep(matches_in: str = MATCHES_IN):
    gold_v1, gold_v2 = _load_gold_standards()

    print("[rerank_v4_1] Loading matrices…")
    mc_mod   = pd.read_csv(MC_MODULES)
    ml_mod   = pd.read_csv(ML_MODULES)
    mc_ids   = list(mc_mod["module_id"])
    ml_names = list(ml_mod["module_name"])
    mc_id_to_idx  = {mid: i for i, mid in enumerate(mc_ids)}
    ml_name_to_idx = {n: i for i, n in enumerate(ml_names)}

    text_v3     = sparse.load_npz(TEXT_V3_NPZ).toarray()
    hier_scores = sparse.load_npz(HIER_SCORES).toarray()

    with open(HIER_INDEX, encoding="utf-8") as f:
        hier_idx = json.load(f)
    group_keys = hier_idx["group_keys"]

    df = pd.read_csv(matches_in)
    print(f"[rerank_v4_1] Input: {len(df)} candidates")

    os.makedirs("outputs", exist_ok=True)
    sweep_rows = []

    for w_hier in SWEEP_WEIGHTS:
        print(f"\n[rerank_v4_1] w_hier={w_hier:.2f}…")
        df_out = rerank(w_hier, df, text_v3, hier_scores, group_keys,
                        mc_ids, ml_names, mc_id_to_idx, ml_name_to_idx)

        out_path = f"outputs/iterative_matches_d4_1_w{w_hier:.2f}.csv"
        df_out.to_csv(out_path, index=False)

        h1v1, n = precision_at_k(df_out, 1, "reranked_score", gold_v1)
        h5v1, _ = precision_at_k(df_out, 5, "reranked_score", gold_v1)
        h1v2, _ = precision_at_k(df_out, 1, "reranked_score", gold_v2)
        h5v2, _ = precision_at_k(df_out, 5, "reranked_score", gold_v2)
        print(f"  gold_v1: P@1={h1v1}/{n}={h1v1/n*100:.1f}%  P@5={h5v1/n*100:.1f}%")
        print(f"  gold_v2: P@1={h1v2}/{n}={h1v2/n*100:.1f}%  P@5={h5v2/n*100:.1f}%")

        sweep_rows.append({
            "w_hier": w_hier,
            "p1_v1": round(h1v1/n, 4), "h1_v1": h1v1,
            "p5_v1": round(h5v1/n, 4), "h5_v1": h5v1,
            "p1_v2": round(h1v2/n, 4), "h1_v2": h1v2,
            "p5_v2": round(h5v2/n, 4), "h5_v2": h5v2,
            "n": n,
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("outputs/d4_1_weight_sweep.csv", index=False)
    print(f"\n[rerank_v4_1] Sweep table (gold_v2):")
    for _, r in sweep_df.iterrows():
        print(f"  w={r['w_hier']:.2f}  "
              f"gold_v1 P@1={r['h1_v1']}/{r['n']}={r['p1_v1']*100:.1f}%  "
              f"gold_v2 P@1={r['h1_v2']}/{r['n']}={r['p1_v2']*100:.1f}%")

    # Best by gold_v2 P@1
    best = sweep_df.loc[sweep_df["p1_v2"].idxmax()]
    best_w = float(best["w_hier"])
    print(f"\n[rerank_v4_1] Best w_hier={best_w:.2f}: "
          f"gold_v1={best['h1_v1']}/{best['n']}={best['p1_v1']*100:.1f}%  "
          f"gold_v2={best['h1_v2']}/{best['n']}={best['p1_v2']*100:.1f}%")

    shutil.copy(f"outputs/iterative_matches_d4_1_w{best_w:.2f}.csv",
                "outputs/iterative_matches_d4_1_best.csv")
    print(f"[rerank_v4_1] Best output → outputs/iterative_matches_d4_1_best.csv")

    return sweep_df, best_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_in", default=MATCHES_IN)
    args = parser.parse_args()
    run_sweep(args.matches_in)
