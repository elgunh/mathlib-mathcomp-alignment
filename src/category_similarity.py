"""Signal 3: Category/cluster alignment bonus."""

import os
import numpy as np
import pandas as pd
from scipy import sparse

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_NPZ = os.path.join("data", "processed", "category_sim.npz")

CLUSTER_CATEGORY_MAP = {
    "boot":      {"Data", "Logic", "Init", "Order", "Tactic"},
    "algebra":   {"Algebra", "RingTheory", "LinearAlgebra", "NumberTheory"},
    "field":     {"FieldTheory", "NumberTheory", "Algebra"},
    "fingroup":  {"GroupTheory", "Combinatorics"},
    "solvable":  {"GroupTheory"},
    "character": {"RepresentationTheory", "GroupTheory"},
    "order":     {"Order"},
    "ssreflect": {"Logic", "Tactic", "Data"},
    "all":       set(),
}


def compute():
    print("[category_similarity] Loading modules...")
    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    print(f"[category_similarity] {n_mc} MathComp x {n_ml} Mathlib")

    ml_categories = ml["category"].values

    rows, cols, vals = [], [], []
    for i, cluster in enumerate(mc["cluster"]):
        allowed = CLUSTER_CATEGORY_MAP.get(cluster, set())
        if not allowed:
            continue
        for j in range(n_ml):
            if ml_categories[j] in allowed:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_mc, n_ml))

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, mat)

    total_ones = len(vals)
    total_cells = n_mc * n_ml
    print(f"[category_similarity] {total_ones}/{total_cells} cells = 1.0 "
          f"({100*total_ones/total_cells:.1f}%)")
    print(f"[category_similarity] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
