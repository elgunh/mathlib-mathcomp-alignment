"""Combine the four similarity signals and produce top-K candidate matches."""

import os
import numpy as np
import pandas as pd
from scipy import sparse

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
MC_DESC = os.path.join("data", "processed", "mathcomp_descriptions.csv")
NAME_SIM = os.path.join("data", "processed", "name_sim.npz")
TEXT_SIM = os.path.join("data", "processed", "text_sim.npz")
CAT_SIM = os.path.join("data", "processed", "category_sim.npz")
GRAPH_SIM = os.path.join("data", "processed", "graph_sim.npz")
OUT_CSV = os.path.join("outputs", "candidate_matches.csv")
OUT_NPZ = os.path.join("outputs", "alignment_matrix.npz")

TOP_K = 10

W_NAME_PRIMARY = 0.30
W_TEXT_PRIMARY = 0.40
W_CAT_PRIMARY = 0.10
W_GRAPH_PRIMARY = 0.20

W_NAME_FALLBACK = 0.50
W_TEXT_FALLBACK = 0.15
W_CAT_FALLBACK = 0.15
W_GRAPH_FALLBACK = 0.20


def load_sparse_dense(path: str, shape: tuple) -> np.ndarray:
    mat = sparse.load_npz(path)
    return mat.toarray()


def compute():
    print("[combine_signals] Loading modules...")
    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    print(f"[combine_signals] {n_mc} x {n_ml}")

    shape = (n_mc, n_ml)
    name_sim = load_sparse_dense(NAME_SIM, shape)
    text_sim = load_sparse_dense(TEXT_SIM, shape)
    cat_sim = load_sparse_dense(CAT_SIM, shape)
    graph_sim = load_sparse_dense(GRAPH_SIM, shape)

    mc_desc = None
    has_desc_count = 0
    if os.path.exists(MC_DESC):
        mc_desc = pd.read_csv(MC_DESC)
        desc_map = {}
        for _, r in mc_desc.iterrows():
            d = str(r.get("clean_description", "")).strip()
            desc_map[r["module_id"]] = d
        has_desc_count = sum(1 for d in desc_map.values()
                            if d and d != "nan")

    desc_frac = has_desc_count / n_mc if n_mc > 0 else 0
    if desc_frac > 0.5:
        w_name, w_text, w_cat, w_graph = (
            W_NAME_PRIMARY, W_TEXT_PRIMARY, W_CAT_PRIMARY, W_GRAPH_PRIMARY)
        regime = "primary"
    else:
        w_name, w_text, w_cat, w_graph = (
            W_NAME_FALLBACK, W_TEXT_FALLBACK, W_CAT_FALLBACK, W_GRAPH_FALLBACK)
        regime = "fallback"

    print(f"[combine_signals] Weight regime: {regime} "
          f"(desc coverage: {has_desc_count}/{n_mc} = {desc_frac:.0%})")
    print(f"  w_name={w_name}, w_text={w_text}, w_cat={w_cat}, w_graph={w_graph}")

    combined = (w_name * name_sim
                + w_text * text_sim
                + w_cat * cat_sim
                + w_graph * graph_sim)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, sparse.csr_matrix(combined))

    rows = []
    for i in range(n_mc):
        scores = combined[i]
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        mid = mc["module_id"].iloc[i]
        cluster = mc["cluster"].iloc[i]
        for rank, j in enumerate(top_indices, 1):
            rows.append({
                "mathcomp_module": mid,
                "mathcomp_cluster": cluster,
                "mathlib_module": ml["module_name"].iloc[j],
                "mathlib_category": ml["category"].iloc[j],
                "name_score": round(float(name_sim[i, j]), 4),
                "text_score": round(float(text_sim[i, j]), 4),
                "category_score": round(float(cat_sim[i, j]), 4),
                "graph_score": round(float(graph_sim[i, j]), 4),
                "combined_score": round(float(scores[j]), 4),
                "rank": rank,
            })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    top1 = df_out[df_out["rank"] == 1]
    print(f"\n[combine_signals] Output: {len(df_out)} rows ({n_mc} x {TOP_K})")
    print(f"[combine_signals] Top-1 combined score stats:")
    print(f"  min={top1['combined_score'].min():.3f}, "
          f"mean={top1['combined_score'].mean():.3f}, "
          f"max={top1['combined_score'].max():.3f}")

    print("\n[combine_signals] Sample top-1 matches:")
    for _, r in top1.head(15).iterrows():
        print(f"  {r['mathcomp_module']:30s} -> {r['mathlib_module']:50s} "
              f"({r['combined_score']:.3f})")

    print(f"\n[combine_signals] Saved {OUT_CSV}")
    print(f"[combine_signals] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
