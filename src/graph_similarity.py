"""Signal 4: Graph neighbourhood similarity via translated-neighbour Jaccard."""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
MC_EDGES = os.path.join("data", "processed", "mathcomp_edges.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
ML_EDGES = os.path.join("data", "processed", "mathlib_edges.csv")
NAME_SIM = os.path.join("data", "processed", "name_sim.npz")
OUT_NPZ = os.path.join("data", "processed", "graph_sim.npz")


def build_neighborhoods(edges_df: pd.DataFrame, id_col_src: str,
                        id_col_tgt: str) -> dict[str, set[str]]:
    """Build undirected 1-hop neighbourhood for each node."""
    nbrs = defaultdict(set)
    for _, row in edges_df.iterrows():
        s, t = str(row[id_col_src]), str(row[id_col_tgt])
        nbrs[s].add(t)
        nbrs[t].add(s)
    return dict(nbrs)


def compute():
    print("[graph_similarity] Loading data...")
    mc_mod = pd.read_csv(MC_MODULES)
    ml_mod = pd.read_csv(ML_MODULES)
    mc_edges = pd.read_csv(MC_EDGES)
    ml_edges = pd.read_csv(ML_EDGES)

    n_mc = len(mc_mod)
    n_ml = len(ml_mod)

    mc_ids = list(mc_mod["module_id"])
    ml_names = list(ml_mod["module_name"])
    ml_idx_map = {name: i for i, name in enumerate(ml_names)}
    ml_idxs = list(ml_mod["module_idx"].astype(str))
    ml_idx_to_pos = {idx: i for i, idx in enumerate(ml_idxs)}

    print("[graph_similarity] Building MathComp neighbourhoods...")
    mc_nbrs = build_neighborhoods(mc_edges, "source", "target")

    print("[graph_similarity] Building Mathlib neighbourhoods...")
    ml_nbrs_raw = defaultdict(set)
    for _, row in ml_edges.iterrows():
        s, t = str(row["source"]), str(row["target"])
        ml_nbrs_raw[s].add(t)
        ml_nbrs_raw[t].add(s)

    print("[graph_similarity] Loading name similarity for neighbour translation...")
    name_sim = sparse.load_npz(NAME_SIM)

    mc_top1_ml = {}
    for i, mid in enumerate(mc_ids):
        row = name_sim.getrow(i).toarray().flatten()
        best_j = int(np.argmax(row))
        if row[best_j] > 0:
            mc_top1_ml[mid] = best_j

    print("[graph_similarity] Computing graph similarity...")
    rows, cols, vals = [], [], []

    for i, mid in enumerate(mc_ids):
        if (i + 1) % 20 == 0:
            print(f"  MathComp module {i+1}/{n_mc}...")

        mc_nb = mc_nbrs.get(mid, set())
        if not mc_nb:
            continue

        translated = set()
        for nb in mc_nb:
            if nb in mc_top1_ml:
                ml_pos = mc_top1_ml[nb]
                ml_module_idx = str(ml_mod["module_idx"].iloc[ml_pos])
                translated.add(ml_module_idx)

        if not translated:
            continue

        for j in range(n_ml):
            ml_module_idx = str(ml_mod["module_idx"].iloc[j])
            ml_nb = ml_nbrs_raw.get(ml_module_idx, set())
            if not ml_nb:
                continue

            inter = len(translated & ml_nb)
            if inter == 0:
                continue
            union = len(translated | ml_nb)
            score = inter / union
            rows.append(i)
            cols.append(j)
            vals.append(score)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_mc, n_ml))

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, mat)

    print(f"[graph_similarity] Matrix: {mat.shape}, {mat.nnz} nonzero entries")
    if mat.nnz > 0:
        print(f"[graph_similarity] Max={mat.max():.3f}, "
              f"Mean of nonzero={np.mean(mat.data):.4f}")
    print(f"[graph_similarity] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
