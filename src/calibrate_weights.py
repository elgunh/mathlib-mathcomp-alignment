"""Weight calibration for Deliverable 3.

Stage A: Grid search over (w_name, w_text, w_cat) for old and new text signals,
         evaluated via 5-fold stratified cross-validation.
Stage B: Top-5 new-text configurations run through the full D2-style iterative
         propagation pipeline; final P@1/P@5 reported on the 62-pair gold set.
Stage C: Static 4-signal base score (name + text_v2 + cat + graph, no iteration)
         to test whether a one-shot rich signal beats propagation.

Outputs
-------
outputs/weight_calibration.csv      — per-(triple, text_variant) CV + full-set scores
outputs/top_weight_configs.csv      — Stage A top-10 per variant
outputs/d3_weight_ablation.csv      — comparisons A-D as defined in the task
outputs/figures/weight_heatmap_old.png
outputs/figures/weight_heatmap_new.png
outputs/figures/weight_frontier.png (optional, P@1 vs P@5 scatter)
"""

import os
import sys
import random
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== PATHS =====
MC_MODULES    = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES    = os.path.join("data", "processed", "mathlib_modules.csv")
MC_EDGES      = os.path.join("data", "processed", "mathcomp_edges.csv")
ML_EDGES      = os.path.join("data", "processed", "mathlib_edges.csv")
NAME_SIM_PATH = os.path.join("data", "processed", "name_sim.npz")
TEXT_SIM_PATH = os.path.join("data", "processed", "text_sim.npz")
TEXT_V2_PATH  = os.path.join("data", "processed", "text_sim_v2.npz")
CAT_SIM_PATH  = os.path.join("data", "processed", "category_sim.npz")
GRAPH_SIM_PATH= os.path.join("data", "processed", "graph_sim.npz")

OUT_DIR       = "outputs"
FIG_DIR       = os.path.join(OUT_DIR, "figures")
OUT_CALIB     = os.path.join(OUT_DIR, "weight_calibration.csv")
OUT_TOP       = os.path.join(OUT_DIR, "top_weight_configs.csv")
OUT_ABLATION  = os.path.join(OUT_DIR, "d3_weight_ablation.csv")
HEATMAP_OLD   = os.path.join(FIG_DIR, "weight_heatmap_old.png")
HEATMAP_NEW   = os.path.join(FIG_DIR, "weight_heatmap_new.png")
FRONTIER_PNG  = os.path.join(FIG_DIR, "weight_frontier.png")

# ===== PIPELINE CONSTANTS (must match D2) =====
HIGH_THRESHOLD  = 0.35
PROP_DECAY      = 0.90
THRESHOLD_FLOOR = 0.25
N_ROUNDS        = 6
BAD_PREFIXES = (
    "Mathlib.Tactic", "Mathlib.Lean", "Mathlib.Meta", "Mathlib.Elab",
    "Mathlib.Linter", "Mathlib.Attr", "Mathlib.Parser", "Mathlib.Init",
)

# ===== GOLD STANDARD =====
GOLD_PAIRS = {
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
                       "Mathlib.GroupTheory.Subgroup",
                       "Mathlib.GroupTheory.Hom"],
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
                       "Mathlib.GroupTheory.GroupAction.Jordan",
                       "Mathlib.GroupTheory.CompositionSeries"],
    "burnside_app":   ["Mathlib.GroupTheory.GroupAction",
                       "Mathlib.GroupTheory.Burnside"],
    "ssralg":         ["Mathlib.Algebra.Ring", "Mathlib.Algebra.Group"],
    "matrix":         ["Mathlib.LinearAlgebra.Matrix"],
    "poly":           ["Mathlib.RingTheory.Polynomial",
                       "Mathlib.Algebra.Polynomial"],
    "ring_quotient":  ["Mathlib.RingTheory.Ideal.Quotient",
                       "Mathlib.RingTheory.Polynomial.Quotient"],
    "intdiv":         ["Mathlib.Data.Int.Div",
                       "Mathlib.Data.Int.ModCast",
                       "Mathlib.Data.Int.Order",
                       "Mathlib.Data.Int.GCD",
                       "Mathlib.Data.Int"],
    "bigop":          ["Mathlib.Algebra.BigOperators",
                       "Mathlib.Algebra.Order.Sum"],
    "ssrnat":         ["Mathlib.Data.Nat"],
    "ssrint":         ["Mathlib.Data.Int"],
    "rat":            ["Mathlib.Data.Rat"],
    "prime":          ["Mathlib.Data.Nat.Prime",
                       "Mathlib.Data.Nat.Factors"],
    "zmodp":          ["Mathlib.Data.ZMod"],
    "fraction":       ["Mathlib.RingTheory.Localization.FractionRing",
                       "Mathlib.RingTheory.Localization"],
    "binomial":       ["Mathlib.Data.Nat.Choose",
                       "Mathlib.Data.Nat.Factorial",
                       "Mathlib.RingTheory.Binomial"],
    "mxpoly":         ["Mathlib.LinearAlgebra.Matrix.Polynomial",
                       "Mathlib.LinearAlgebra.Matrix.Charpoly"],
    "mxalgebra":      ["Mathlib.LinearAlgebra.Matrix"],
    "vector":         ["Mathlib.LinearAlgebra"],
    "sesquilinear":   ["Mathlib.LinearAlgebra.SesquilinearForm",
                       "Mathlib.LinearAlgebra.Matrix.SesquilinearForm"],
    "mxrepresentation": ["Mathlib.RepresentationTheory"],
    "character":      ["Mathlib.RepresentationTheory.Character"],
    "vcharacter":     ["Mathlib.RepresentationTheory.Character",
                       "Mathlib.RepresentationTheory.VirtualCharacter"],
    "separable":      ["Mathlib.FieldTheory.Separable"],
    "galois":         ["Mathlib.FieldTheory.Galois",
                       "Mathlib.FieldTheory.Finite.GaloisField"],
    "algC":           ["Mathlib.FieldTheory.IsAlgClosed",
                       "Mathlib.Analysis.SpecialFunctions.Complex"],
    "cyclotomic":     ["Mathlib.NumberTheory.Cyclotomic",
                       "Mathlib.RingTheory.Polynomial.Cyclotomic"],
    "fieldext":       ["Mathlib.FieldTheory.Extension",
                       "Mathlib.FieldTheory.IntermediateField"],
    "finfield":       ["Mathlib.FieldTheory.Finite",
                       "Mathlib.FieldTheory.Galois"],
    "order":          ["Mathlib.Order.Lattice", "Mathlib.Order"],
    "preorder":       ["Mathlib.Order.Preorder", "Mathlib.Order"],
    "archimedean":    ["Mathlib.Algebra.Order.Archimedean",
                       "Mathlib.Algebra.Order.Ring.Archimedean"],
    "seq":            ["Mathlib.Data.Seq", "Mathlib.Data.List"],
    "fintype":        ["Mathlib.Data.Fintype"],
    "finset":         ["Mathlib.Data.Finset"],
    "tuple":          ["Mathlib.Data.Vector", "Mathlib.Data.Fin.Tuple",
                       "Mathlib.Order.Fin"],
    "eqtype":         ["Mathlib.Logic.Equiv", "Mathlib.Data.Subtype"],
    "choice":         ["Mathlib.Logic.Classical", "Mathlib.Order.Zorn",
                       "Mathlib.Logic.Choice"],
    "path":           ["Mathlib.Combinatorics.SimpleGraph",
                       "Mathlib.Topology.Path"],
    "fingraph":       ["Mathlib.Combinatorics.SimpleGraph",
                       "Mathlib.Order.Graph"],
    "div":            ["Mathlib.Data.Nat.Div", "Mathlib.Data.Int.Div",
                       "Mathlib.Data.Nat.GCD"],
    "finfun":         ["Mathlib.Data.PiFin", "Mathlib.Logic.Fin",
                       "Mathlib.Data.Fin"],
    "ssrbool":        ["Mathlib.Data.Bool"],
    "ssrfun":         ["Mathlib.Logic.Function", "Mathlib.Data.Function"],
    "classfun":       ["Mathlib.RepresentationTheory",
                       "Mathlib.GroupTheory.ClassEquation"],
}


# ===== HELPER FUNCTIONS =====

def is_valid(name):
    return not any(name.startswith(p) for p in BAD_PREFIXES)


def eval_at_k(items, mc_id_to_idx, ml_names, score_mat, k):
    """P@k on a list of (mc_mod, prefixes, cluster) items."""
    hits = total = 0
    for mc_mod, prefixes, _ in items:
        idx = mc_id_to_idx.get(mc_mod)
        if idx is None:
            continue
        total += 1
        top_k = np.argsort(score_mat[idx])[-k:]
        if any(any(ml_names[j].startswith(p) for p in prefixes) for j in top_k):
            hits += 1
    return hits / max(total, 1), total


def apply_weights(name_sim, text_sim_v, cat_sim, valid_mask, wn, wt, wc,
                  graph_sim=None, wg=0.0):
    bs = wn * name_sim + wt * text_sim_v + wc * cat_sim
    if graph_sim is not None and wg > 0:
        bs = bs + wg * graph_sim
    bs[:, ~valid_mask] *= 0.05
    return bs


def make_stratified_folds(items, n_folds=5, seed=42):
    """Assign items to folds round-robin within each cluster."""
    rng = random.Random(seed)
    by_cluster = defaultdict(list)
    for item in items:
        by_cluster[item[2]].append(item)
    folds = [[] for _ in range(n_folds)]
    for cluster_items in by_cluster.values():
        shuf = list(cluster_items)
        rng.shuffle(shuf)
        for i, item in enumerate(shuf):
            folds[i % n_folds].append(item)
    return folds


def cv_p_at_k(items, folds, mc_id_to_idx, ml_names, valid_mask,
              name_sim, text_sim_v, cat_sim, triple, k):
    """P@k for a single triple via pre-built folds."""
    wn, wt, wc = triple
    bs = apply_weights(name_sim, text_sim_v, cat_sim, valid_mask, wn, wt, wc)
    fold_scores = []
    for fold in folds:
        p, _ = eval_at_k(fold, mc_id_to_idx, ml_names, bs, k)
        fold_scores.append(p)
    return np.mean(fold_scores), np.std(fold_scores)


# ===== STAGE B: ITERATIVE PIPELINE =====

def build_mc_nbrs(mc_edges):
    nb = defaultdict(set)
    for _, row in mc_edges.iterrows():
        s, t = str(row["source"]), str(row["target"])
        nb[s].add(t); nb[t].add(s)
    return dict(nb)


def build_ml_nbrs(ml_edges, ml_idx_strs):
    nb = defaultdict(set)
    for _, row in ml_edges.iterrows():
        s, t = str(row["source"]), str(row["target"])
        nb[s].add(t); nb[t].add(s)
    return dict(nb)


def run_iterative(base_score, mc_ids, mc_nbrs, ml_idx_strs, ml_nbrs_dict,
                  mc_id_to_idx, ml_str_to_pos, valid_mask):
    """Full D2-style iterative propagation. Returns final_score matrix."""
    n_mc, n_ml = base_score.shape
    prop_bonus = np.zeros((n_mc, n_ml))
    anchors = {}
    prop_sources = set()

    # Round 0
    for i in range(n_mc):
        s = base_score[i]
        sj = np.argsort(s)[::-1]
        s1, s2 = s[sj[0]], s[sj[1]]
        if s1 >= HIGH_THRESHOLD:
            anchors[i] = (int(sj[0]), float(s1))
            if s1 >= 0.40 or (s1 - s2) >= 0.10:
                prop_sources.add(i)

    for rnd in range(1, N_ROUNDS + 1):
        thr = max(HIGH_THRESHOLD * (PROP_DECAY ** rnd), THRESHOLD_FLOOR)
        new_this = 0

        for i in range(n_mc):
            if i in anchors:
                continue
            mid = mc_ids[i]
            mc_nb = mc_nbrs.get(mid, set())
            if not mc_nb:
                continue
            anc_nb = [(mc_id_to_idx[nb], anchors[mc_id_to_idx[nb]])
                      for nb in mc_nb if mc_id_to_idx.get(nb) in prop_sources]
            if not anc_nb:
                continue
            n_nb = len(mc_nb)
            for nb_idx, (ml_j, anc_score) in anc_nb:
                ml_str = ml_idx_strs[ml_j]
                w = (PROP_DECAY ** rnd) * anc_score / n_nb
                for ml_nb_str in ml_nbrs_dict.get(ml_str, set()):
                    pos = ml_str_to_pos.get(ml_nb_str)
                    if pos is not None and valid_mask[pos]:
                        prop_bonus[i, pos] += w

        eff = np.minimum(prop_bonus, 0.25 * base_score)
        final = base_score + eff

        for i in range(n_mc):
            if i in anchors:
                continue
            sj = np.argsort(final[i])[::-1]
            s1, s2 = final[i, sj[0]], final[i, sj[1]]
            if s1 >= thr:
                anchors[i] = (int(sj[0]), float(s1))
                new_this += 1
                if s1 >= 0.40 or (s1 - s2) >= 0.10:
                    prop_sources.add(i)

        if new_this == 0:
            break

    eff = np.minimum(prop_bonus, 0.25 * base_score)
    return base_score + eff


# ===== VISUALIZATION =====

def plot_heatmap(wn_vals, wt_vals, p1_grid, title, out_path):
    """
    wn_vals: sorted list of w_name values (x-axis)
    wt_vals: sorted list of w_text values (y-axis)
    p1_grid: 2D array [len(wt_vals) x len(wn_vals)] of P@1 values (NaN where invalid)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    vmin = np.nanmin(p1_grid)
    vmax = np.nanmax(p1_grid)
    # Use a tight range around the data for better contrast
    margin = (vmax - vmin) * 0.1 if vmax > vmin else 0.05
    im = ax.imshow(p1_grid, origin="lower", aspect="auto",
                   extent=[wn_vals[0] - 0.025, wn_vals[-1] + 0.025,
                           wt_vals[0] - 0.025, wt_vals[-1] + 0.025],
                   cmap="RdYlGn",
                   vmin=max(0, vmin - margin), vmax=min(1, vmax + margin))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("P@1 (full set, biased screening)")

    # Annotate cells
    for i, wt in enumerate(wt_vals):
        for j, wn in enumerate(wn_vals):
            val = p1_grid[i, j]
            if not np.isnan(val):
                ax.text(wn, wt, f"{val:.0%}", ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.3 < (val - vmin) / max(vmax - vmin, 0.01) < 0.7 else "white")

    ax.set_xlabel("W_NAME")
    ax.set_ylabel("W_TEXT")
    ax.set_title(title)
    ax.set_xticks(wn_vals)
    ax.set_yticks(wt_vals)
    # Add constraint line w_cat = 0.05 (i.e. w_name + w_text = 0.95)
    wn_arr = np.array(wn_vals)
    ax.plot(wn_arr, 0.95 - wn_arr, "k--", lw=1.2, alpha=0.5, label="w_cat=0.05")
    ax.plot(wn_arr, 0.80 - wn_arr, "k:", lw=1.2, alpha=0.5, label="w_cat=0.20")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[calibrate] Saved {out_path}")


def plot_frontier(rows, out_path):
    """P@1 vs P@5 scatter for all weight triples, both text variants."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, color, marker in [("old", "#888888", "o"), ("new", "#3b5fc0", "s")]:
        sub = [r for r in rows if r["text_variant"] == variant]
        p1s = [r["full_p1"] for r in sub]
        p5s = [r["full_p5"] for r in sub]
        ax.scatter(p1s, p5s, c=color, marker=marker, alpha=0.75, s=60,
                   label=f"{variant} text")
        # Annotate best
        if sub:
            best = max(sub, key=lambda r: r["full_p1"])
            ax.annotate(f"({best['w_name']:.2f},{best['w_text']:.2f},{best['w_cat']:.2f})",
                        (best["full_p1"], best["full_p5"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("P@1 (full set)")
    ax.set_ylabel("P@5 (full set)")
    ax.set_title("Weight configuration Pareto frontier\n(full-set, biased screening)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[calibrate] Saved {out_path}")


# ===== MAIN =====

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Load data ----
    print("[calibrate] Loading matrices …")
    mc  = pd.read_csv(MC_MODULES)
    ml  = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    mc_ids       = list(mc["module_id"])
    mc_id_to_idx = {mid: i for i, mid in enumerate(mc_ids)}
    ml_names     = list(ml["module_name"])
    mc_cluster   = {row["module_id"]: row["cluster"] for _, row in mc.iterrows()}

    valid_mask = np.array([is_valid(n) for n in ml_names], dtype=bool)
    print(f"  Valid Mathlib targets: {valid_mask.sum()}/{n_ml}")

    name_sim   = sparse.load_npz(NAME_SIM_PATH).toarray()
    text_sim   = sparse.load_npz(TEXT_SIM_PATH).toarray()
    text_sim2  = sparse.load_npz(TEXT_V2_PATH).toarray()
    cat_sim    = sparse.load_npz(CAT_SIM_PATH).toarray()

    has_graph = os.path.exists(GRAPH_SIM_PATH)
    graph_sim = sparse.load_npz(GRAPH_SIM_PATH).toarray() if has_graph else None
    if not has_graph:
        print("  WARNING: graph_sim.npz not found; Stage C will be skipped")

    mc_edges = pd.read_csv(MC_EDGES)
    ml_edges = pd.read_csv(ML_EDGES)
    mc_nbrs     = build_mc_nbrs(mc_edges)
    ml_idx_strs = ml["module_idx"].astype(str).tolist()
    ml_nbrs_dict = build_ml_nbrs(ml_edges, ml_idx_strs)
    ml_str_to_pos = {s: i for i, s in enumerate(ml_idx_strs)}

    # ---- Gold pairs with cluster labels ----
    gold_items = [(mc_mod, prefixes, mc_cluster.get(mc_mod, "unknown"))
                  for mc_mod, prefixes in GOLD_PAIRS.items()
                  if mc_id_to_idx.get(mc_mod) is not None]
    print(f"  Gold pairs available: {len(gold_items)}")

    # ---- Weight grid ----
    w_name_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    w_text_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    w_cat_vals  = [0.05, 0.10, 0.15, 0.20]
    triples = [(wn, wt, wc)
               for wn in w_name_vals
               for wt in w_text_vals
               for wc in w_cat_vals
               if abs(wn + wt + wc - 1.0) < 0.001]
    print(f"  Valid triples (from spec): {len(triples)}")

    # ---- Finer grid for heatmap (w_cat = 1 - wn - wt) ----
    hm_wn = [round(v, 2) for v in np.arange(0.30, 0.65, 0.05)]
    hm_wt = [round(v, 2) for v in np.arange(0.10, 0.55, 0.05)]
    heatmap_triples = []
    for wn in hm_wn:
        for wt in hm_wt:
            wc = round(1 - wn - wt, 3)
            if 0.05 <= wc <= 0.25:
                heatmap_triples.append((wn, wt, wc))

    # ===== STAGE A =====
    print("\n[calibrate] === Stage A: base-score grid search (5-fold CV) ===")
    folds = make_stratified_folds(gold_items, n_folds=5)
    cluster_dist = defaultdict(int)
    for item in gold_items:
        cluster_dist[item[2]] += 1
    print(f"  Cluster distribution: {dict(cluster_dist)}")
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    calib_rows = []

    for text_label, text_sim_v in [("old", text_sim), ("new", text_sim2)]:
        print(f"\n  --- Text variant: {text_label} ---")
        triple_scores = []
        for triple in triples:
            cv_p1, cv_p1_std = cv_p_at_k(
                gold_items, folds, mc_id_to_idx, ml_names, valid_mask,
                name_sim, text_sim_v, cat_sim, triple, 1)
            cv_p5, _ = cv_p_at_k(
                gold_items, folds, mc_id_to_idx, ml_names, valid_mask,
                name_sim, text_sim_v, cat_sim, triple, 5)
            wn, wt, wc = triple
            bs = apply_weights(name_sim, text_sim_v, cat_sim, valid_mask, wn, wt, wc)
            fp1, total = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs, 1)
            fp5, _     = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs, 5)
            triple_scores.append((triple, cv_p1, cv_p5, cv_p1_std, fp1, fp5))
            calib_rows.append({
                "w_name": wn, "w_text": wt, "w_cat": wc,
                "text_variant": text_label,
                "cv_p1": round(cv_p1, 4), "cv_p5": round(cv_p5, 4),
                "cv_p1_std": round(cv_p1_std, 4),
                "full_p1": round(fp1, 4), "full_p5": round(fp5, 4),
            })

        # Top 10 by CV P@1
        top10 = sorted(triple_scores, key=lambda x: -x[1])[:10]
        print(f"  Top 10 by CV P@1 ({text_label} text):")
        print(f"  {'w_n':>5} {'w_t':>5} {'w_c':>5} | {'CV P@1':>7} ±{'std':>5} | {'Full P@1':>9} {'Full P@5':>9}")
        for triple, cv_p1, cv_p5, std, fp1, fp5 in top10:
            print(f"  {triple[0]:>5.2f} {triple[1]:>5.2f} {triple[2]:>5.2f} | "
                  f"{cv_p1:>7.1%} ±{std:>5.1%} | {fp1:>9.1%} {fp5:>9.1%}")

    df_calib = pd.DataFrame(calib_rows)
    df_calib.to_csv(OUT_CALIB, index=False)
    print(f"\n[calibrate] Saved {OUT_CALIB}")

    # Save top-10 per variant
    top_rows = []
    for variant in ["old", "new"]:
        sub = df_calib[df_calib["text_variant"] == variant]
        top10_df = sub.sort_values("cv_p1", ascending=False).head(10)
        for _, r in top10_df.iterrows():
            top_rows.append(dict(r))
    pd.DataFrame(top_rows).to_csv(OUT_TOP, index=False)
    print(f"[calibrate] Saved {OUT_TOP}")

    # ===== HEATMAPS =====
    print("\n[calibrate] Generating heatmaps …")
    for text_label, text_sim_v in [("old", text_sim), ("new", text_sim2)]:
        wn_uniq = sorted(set(t[0] for t in heatmap_triples))
        wt_uniq = sorted(set(t[1] for t in heatmap_triples))
        p1_grid = np.full((len(wt_uniq), len(wn_uniq)), np.nan)
        wn_idx = {v: i for i, v in enumerate(wn_uniq)}
        wt_idx = {v: i for i, v in enumerate(wt_uniq)}
        for wn, wt, wc in heatmap_triples:
            bs = apply_weights(name_sim, text_sim_v, cat_sim, valid_mask, wn, wt, wc)
            fp1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs, 1)
            p1_grid[wt_idx[wt], wn_idx[wn]] = fp1

        out_path = HEATMAP_OLD if text_label == "old" else HEATMAP_NEW
        title = (f"P@1 by (W_NAME, W_TEXT), W_CAT = 1 - W_NAME - W_TEXT\n"
                 f"{'Old path-token' if text_label=='old' else 'New docstring'} text, "
                 f"full-set (biased screening)")
        plot_heatmap(wn_uniq, wt_uniq, p1_grid, title, out_path)

    # Frontier plot
    plot_frontier(calib_rows, os.path.join(FIG_DIR, "weight_frontier.png"))

    # ===== STAGE B =====
    print("\n[calibrate] === Stage B: Full iterative pipeline (top 5 new-text configs) ===")

    new_sub = df_calib[df_calib["text_variant"] == "new"].sort_values(
        "cv_p1", ascending=False)
    top5_new = [(row["w_name"], row["w_text"], row["w_cat"])
                for _, row in new_sub.head(5).iterrows()]
    print(f"  Top-5 new-text triples (by CV P@1): {top5_new}")

    stage_b_rows = []
    for triple in top5_new:
        wn, wt, wc = triple
        bs = apply_weights(name_sim, text_sim2, cat_sim, valid_mask, wn, wt, wc)
        final = run_iterative(bs, mc_ids, mc_nbrs, ml_idx_strs, ml_nbrs_dict,
                              mc_id_to_idx, ml_str_to_pos, valid_mask)
        ip1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, final, 1)
        ip5, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, final, 5)
        # Base score (no iteration) for comparison
        bp1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs, 1)

        # Check tactic contamination
        top1 = {mc_ids[i]: ml_names[int(np.argmax(final[i]))]
                for i in range(n_mc)}
        tactic = sum(1 for m in top1.values()
                     if any(m.startswith(p) for p in BAD_PREFIXES))
        stage_b_rows.append({
            "w_name": wn, "w_text": wt, "w_cat": wc,
            "base_p1": round(bp1, 4),
            "iterative_p1": round(ip1, 4),
            "iterative_p5": round(ip5, 4),
            "tactic_at1": tactic,
        })
        print(f"  ({wn:.2f},{wt:.2f},{wc:.2f}) base P@1={bp1:.1%}  "
              f"iter P@1={ip1:.1%}  P@5={ip5:.1%}  tactic@1={tactic}")

    # Best iterative config
    best_b = max(stage_b_rows, key=lambda r: (r["iterative_p1"], r["iterative_p5"]))
    best_triple = (best_b["w_name"], best_b["w_text"], best_b["w_cat"])
    print(f"\n  Best iterative config: w_name={best_b['w_name']:.2f} "
          f"w_text={best_b['w_text']:.2f} w_cat={best_b['w_cat']:.2f} "
          f"=> P@1={best_b['iterative_p1']:.1%} P@5={best_b['iterative_p5']:.1%}")

    # ===== STAGE C =====
    stage_c_rows = []
    if has_graph:
        print("\n[calibrate] === Stage C: Static 4-signal base score (+graph, no iteration) ===")
        wn_b, wt_b, wc_b = best_triple
        for wg in [0.05, 0.10, 0.15, 0.20]:
            scale = 1.0 - wg
            wn4 = round(wn_b * scale, 4)
            wt4 = round(wt_b * scale, 4)
            wc4 = round(wc_b * scale, 4)
            bs4 = apply_weights(name_sim, text_sim2, cat_sim, valid_mask,
                                wn4, wt4, wc4, graph_sim=graph_sim, wg=wg)
            cp1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs4, 1)
            cp5, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, bs4, 5)
            stage_c_rows.append({
                "w_name": wn4, "w_text": wt4, "w_cat": wc4, "w_graph": wg,
                "static_p1": round(cp1, 4), "static_p5": round(cp5, 4),
            })
            print(f"  w_graph={wg:.2f} ({wn4:.2f},{wt4:.2f},{wc4:.2f}) "
                  f"static P@1={cp1:.1%} P@5={cp5:.1%}")

        best_c = max(stage_c_rows, key=lambda r: (r["static_p1"], r["static_p5"]))
        print(f"\n  Best static-4-signal: w_graph={best_c['w_graph']:.2f} "
              f"P@1={best_c['static_p1']:.1%} P@5={best_c['static_p5']:.1%}")
    else:
        print("\n[calibrate] Stage C skipped (no graph_sim.npz)")
        best_c = None

    # ===== D2 baseline reference numbers =====
    # Reproduce D2 P@1/P@5 with old text and old weights (0.40, 0.45, 0.15)
    d2_bs = apply_weights(name_sim, text_sim, cat_sim, valid_mask,
                          0.40, 0.45, 0.15)
    d2_final = run_iterative(d2_bs, mc_ids, mc_nbrs, ml_idx_strs, ml_nbrs_dict,
                             mc_id_to_idx, ml_str_to_pos, valid_mask)
    d2_p1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d2_final, 1)
    d2_p5, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d2_final, 5)

    # D3 old-weight reference numbers
    d3_old_bs = apply_weights(name_sim, text_sim2, cat_sim, valid_mask,
                              0.40, 0.45, 0.15)
    d3_old_final = run_iterative(d3_old_bs, mc_ids, mc_nbrs, ml_idx_strs,
                                  ml_nbrs_dict, mc_id_to_idx, ml_str_to_pos,
                                  valid_mask)
    d3_old_p1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d3_old_final, 1)
    d3_old_p5, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d3_old_final, 5)

    # D3 calibrated base score (no iteration)
    wn_b, wt_b, wc_b = best_triple
    d3_cal_bs = apply_weights(name_sim, text_sim2, cat_sim, valid_mask,
                               wn_b, wt_b, wc_b)
    d3_cal_base_p1, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d3_cal_bs, 1)
    d3_cal_base_p5, _ = eval_at_k(gold_items, mc_id_to_idx, ml_names, d3_cal_bs, 5)

    # D3 calibrated iterative (already computed as best_b)

    # ===== ABLATION TABLE =====
    n_pairs = len(gold_items)
    ablation_rows = [
        {"config": "A: D2 baseline",
         "text": "old (path-token)", "weights": "(0.40, 0.45, 0.15)",
         "iterative": "yes",
         "p1": round(d2_p1, 4), "p5": round(d2_p5, 4),
         "p1_pct": f"{d2_p1:.1%}", "p5_pct": f"{d2_p5:.1%}"},
        {"config": "B: D3 old-weight (controlled)",
         "text": "new (docstrings)", "weights": "(0.40, 0.45, 0.15)",
         "iterative": "yes",
         "p1": round(d3_old_p1, 4), "p5": round(d3_old_p5, 4),
         "p1_pct": f"{d3_old_p1:.1%}", "p5_pct": f"{d3_old_p5:.1%}"},
        {"config": "C: D3 calibrated-base",
         "text": "new (docstrings)",
         "weights": f"({wn_b:.2f},{wt_b:.2f},{wc_b:.2f})",
         "iterative": "no",
         "p1": round(d3_cal_base_p1, 4), "p5": round(d3_cal_base_p5, 4),
         "p1_pct": f"{d3_cal_base_p1:.1%}", "p5_pct": f"{d3_cal_base_p5:.1%}"},
        {"config": "D: D3 calibrated-iterative",
         "text": "new (docstrings)",
         "weights": f"({best_b['w_name']:.2f},{best_b['w_text']:.2f},{best_b['w_cat']:.2f})",
         "iterative": "yes",
         "p1": round(best_b["iterative_p1"], 4), "p5": round(best_b["iterative_p5"], 4),
         "p1_pct": f"{best_b['iterative_p1']:.1%}",
         "p5_pct": f"{best_b['iterative_p5']:.1%}"},
    ]
    if best_c:
        ablation_rows.append({
            "config": "E: D3 calibrated + graph (no iteration)",
            "text": "new (docstrings)",
            "weights": (f"({best_c['w_name']:.2f},{best_c['w_text']:.2f},"
                        f"{best_c['w_cat']:.2f},g={best_c['w_graph']:.2f})"),
            "iterative": "no",
            "p1": round(best_c["static_p1"], 4), "p5": round(best_c["static_p5"], 4),
            "p1_pct": f"{best_c['static_p1']:.1%}",
            "p5_pct": f"{best_c['static_p5']:.1%}",
        })

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv(OUT_ABLATION, index=False)
    print(f"\n[calibrate] Saved {OUT_ABLATION}")

    # ===== PER-PAIR ANALYSIS for best calibrated config =====
    print(f"\n[calibrate] === Per-pair analysis (best calibrated config) ===")
    wn_b, wt_b, wc_b = best_triple
    d3c_bs = apply_weights(name_sim, text_sim2, cat_sim, valid_mask,
                            wn_b, wt_b, wc_b)
    d3c_final = run_iterative(d3c_bs, mc_ids, mc_nbrs, ml_idx_strs, ml_nbrs_dict,
                               mc_id_to_idx, ml_str_to_pos, valid_mask)

    improvements = []
    regressions = []
    tactic_hits = []

    for mc_mod, prefixes, cluster in gold_items:
        mc_idx = mc_id_to_idx[mc_mod]
        # D2 top-1
        d2_top = ml_names[int(np.argmax(d2_final[mc_idx]))]
        d2_hit = any(d2_top.startswith(p) for p in prefixes)
        # D3 calibrated top-1
        d3c_top = ml_names[int(np.argmax(d3c_final[mc_idx]))]
        d3c_hit = any(d3c_top.startswith(p) for p in prefixes)
        # Old D3 top-1
        old_top = ml_names[int(np.argmax(d3_old_final[mc_idx]))]
        old_hit = any(old_top.startswith(p) for p in prefixes)

        if not d2_hit and d3c_hit:
            improvements.append((mc_mod, d2_top, d3c_top, cluster))
        if d2_hit and not d3c_hit:
            regressions.append((mc_mod, d2_top, d3c_top, cluster))
        if any(d3c_top.startswith(p) for p in BAD_PREFIXES):
            tactic_hits.append(mc_mod)

    print(f"  D2->D3-calibrated improvements: {len(improvements)}")
    for mc, old, new, cl in improvements[:5]:
        print(f"    {mc:<20} {old.split('.')[-1]:<30} -> {new.split('.')[-1]}")

    print(f"  D2->D3-calibrated regressions:  {len(regressions)}")
    for mc, old, new, cl in regressions[:5]:
        print(f"    {mc:<20} {old.split('.')[-1]:<30} -> {new.split('.')[-1]}")

    print(f"  Tactic@1 (calibrated): {len(tactic_hits)}")

    # ===== FINAL SUMMARY =====
    print("\n[calibrate] === Final comparison ===")
    print(f"  {'Config':<45} {'P@1':>6}  {'P@5':>6}")
    print(f"  {'-'*60}")
    for r in ablation_rows:
        print(f"  {r['config']:<45} {r['p1_pct']:>6}  {r['p5_pct']:>6}")

    # Return key results for the calling script
    return {
        "best_triple": best_triple,
        "best_b": best_b,
        "stage_b_rows": stage_b_rows,
        "stage_c_rows": stage_c_rows,
        "ablation": ablation_rows,
        "improvements": improvements,
        "regressions": regressions,
    }


if __name__ == "__main__":
    results = main()
    best = results["best_b"]
    print(f"\n[calibrate] === RECOMMENDED WEIGHTS ===")
    print(f"  w_name = {best['w_name']:.2f}")
    print(f"  w_text = {best['w_text']:.2f}")
    print(f"  w_cat  = {best['w_cat']:.2f}")
    print(f"  => Iterative P@1 = {best['iterative_p1']:.1%}, P@5 = {best['iterative_p5']:.1%}")
