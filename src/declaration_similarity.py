"""Signal 5: IDF-weighted asymmetric overlap on declaration-name token bags.

Motivation
----------
MathComp and Mathlib mathematicians name theorems similarly regardless of
proof assistant (e.g. 'mul_comm', 'add_assoc', 'card_Sylow').  Token bags
from declaration names provide a direct cross-library lexical bridge that
bypasses both module-path and docstring vocabulary.

Method
------
1. Build IDF over all MathComp + Mathlib token bags combined.
2. Filter tokens: length >= 4, not in GENERIC_TOKENS.
3. Score each (mc_module, ml_module) pair:

       score = sum(idf(t) for t in mc_tokens ∩ ml_tokens)
               / min(sum(idf(t) for t in mc_tokens),
                     sum(idf(t) for t in ml_tokens))

   (IDF-weighted Szymkiewicz-Simpson coefficient)

Outputs
-------
  data/processed/decl_sim.npz     — sparse 106 × 7661 matrix
  outputs/decl_sim_diagnostics.txt — diagnostics for report
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter, defaultdict

MC_DECL   = os.path.join("data", "processed", "mathcomp_declarations.csv")
ML_DOCS   = os.path.join("data", "processed", "mathlib_docstrings.csv")
MC_MOD    = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MOD    = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_NPZ   = os.path.join("data", "processed", "decl_sim.npz")
OUT_DIAG  = os.path.join("outputs", "decl_sim_diagnostics.txt")

MIN_TOKEN_LEN = 4
# Require at least this many tokens to overlap for a nonzero score.
# Prevents 1-token matches (e.g. both have 'ring') from scoring 1.0.
MIN_INTER_SIZE = 2

GENERIC_TOKENS = {
    # Structural / boilerplate
    "basic", "defs", "init", "main", "core", "misc", "extra",
    "helper", "auxiliary", "internal", "impl",
    "left", "right", "cast", "lift", "comp", "apply",
    "proof", "have", "show", "this", "with", "from",
    "unnamed", "factory", "definition", "record", "context",
    "import", "theory", "plain", "dummy", "notation",
    "section", "module", "class", "structure", "inductive",
    "canonical", "coercion", "fixpoint", "variant",
    # Near-universal math vocabulary — too common to discriminate
    "prop", "type", "bool", "true", "false",
    "zero", "succ", "pred", "plus", "times",
    "ring", "prod", "mono", "refl", "symm", "trans",
    "finite", "linear", "order", "equiv", "subset",
    "inductive", "unit", "comm", "empty", "cont",
    # Very common MathComp-specific
    "injective", "surjective", "bijective",
    "morphism", "homomorphism", "isomorphism",
    # Very frequent 4-letter tokens that add no discrimination
    "smul", "inst", "self", "elem", "calc", "cast",
}


# ── Token processing ──────────────────────────────────────────────────────

def camel_split(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t.lower() for t in s.split() if t]


def tokenise(name: str) -> list[str]:
    parts = re.split(r"[_.\s']+", name)
    tokens = []
    for p in parts:
        tokens.extend(camel_split(p) or [p.lower()])
    return [t for t in tokens
            if len(t) >= MIN_TOKEN_LEN and t not in GENERIC_TOKENS]


def make_token_set(decl_string: str) -> set[str]:
    if not decl_string or str(decl_string).lower() == "nan":
        return set()
    tokens = set()
    for name in str(decl_string).split():
        tokens.update(tokenise(name))
    return tokens


# ── IDF ───────────────────────────────────────────────────────────────────

def build_idf(all_token_sets: list[set]) -> dict[str, float]:
    n_docs = len(all_token_sets)
    df = Counter()
    for ts in all_token_sets:
        for t in ts:
            df[t] += 1
    return {t: float(np.log(n_docs / (1.0 + cnt)))
            for t, cnt in df.items()}


# ── Scoring ───────────────────────────────────────────────────────────────

def idf_weighted_overlap(mc_tokens: set, ml_tokens: set,
                         idf: dict) -> float:
    """IDF-weighted Szymkiewicz-Simpson coefficient.

    Requires at least MIN_INTER_SIZE tokens to overlap to avoid
    noise from single-token coincidences.
    """
    if not mc_tokens or not ml_tokens:
        return 0.0
    inter = mc_tokens & ml_tokens
    if len(inter) < MIN_INTER_SIZE:
        return 0.0
    inter_w = sum(idf.get(t, 1.0) for t in inter)
    mc_w    = sum(idf.get(t, 1.0) for t in mc_tokens)
    ml_w    = sum(idf.get(t, 1.0) for t in ml_tokens)
    return inter_w / min(mc_w, ml_w)


# ── Main ──────────────────────────────────────────────────────────────────

def compute():
    print("[decl_sim] Loading data…")
    mc_decl = pd.read_csv(MC_DECL)
    ml_docs = pd.read_csv(ML_DOCS)
    mc_mod  = pd.read_csv(MC_MOD)
    ml_mod  = pd.read_csv(ML_MOD)

    mc_ids    = list(mc_mod["module_id"])
    ml_names  = list(ml_mod["module_name"])
    n_mc, n_ml = len(mc_ids), len(ml_names)

    # Build MC token sets
    mc_decl_map = {r["module_id"]: str(r.get("raw_declarations", ""))
                   for _, r in mc_decl.iterrows()}
    mc_token_sets = [make_token_set(mc_decl_map.get(mid, ""))
                     for mid in mc_ids]

    # Build ML token sets (from declaration_names column in mathlib_docstrings.csv)
    ml_decl_map = {r["module_name"]: str(r.get("declaration_names", ""))
                   for _, r in ml_docs.iterrows()}
    ml_token_sets = [make_token_set(ml_decl_map.get(name, ""))
                     for name in ml_names]

    print(f"[decl_sim] MC modules with declarations: "
          f"{sum(1 for ts in mc_token_sets if ts)}/{n_mc}")
    print(f"[decl_sim] ML modules with declarations: "
          f"{sum(1 for ts in ml_token_sets if ts)}/{n_ml}")

    # Build IDF over both libraries combined
    idf = build_idf(mc_token_sets + ml_token_sets)
    print(f"[decl_sim] IDF vocabulary: {len(idf)} tokens")

    # Top IDF tokens (most discriminative)
    top_idf = sorted(idf.items(), key=lambda x: -x[1])[:20]
    print(f"[decl_sim] Top-20 discriminative tokens (high IDF):")
    for t, v in top_idf:
        print(f"    {t:20s} {v:.3f}")

    # Low IDF tokens (most generic)
    low_idf = sorted(idf.items(), key=lambda x: x[1])[:20]
    print(f"[decl_sim] Top-20 generic tokens (low IDF):")
    for t, v in low_idf:
        print(f"    {t:20s} {v:.3f}")

    # Compute sparse similarity matrix
    print(f"\n[decl_sim] Computing {n_mc}×{n_ml} declaration similarity…")
    rows, cols, vals = [], [], []

    for i, mc_ts in enumerate(mc_token_sets):
        if not mc_ts:
            continue
        mc_w_total = sum(idf.get(t, 1.0) for t in mc_ts)

        for j, ml_ts in enumerate(ml_token_sets):
            if not ml_ts:
                continue
            inter = mc_ts & ml_ts
            if len(inter) < MIN_INTER_SIZE:
                continue
            inter_w = sum(idf.get(t, 1.0) for t in inter)
            ml_w    = sum(idf.get(t, 1.0) for t in ml_ts)
            score   = inter_w / min(mc_w_total, ml_w)
            if score > 0:
                rows.append(i)
                cols.append(j)
                vals.append(score)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_mc, n_ml))
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, mat)

    # ── Diagnostics ───────────────────────────────────────────────────────
    data = mat.data
    modules_with_support = int((mat.getnnz(axis=1) > 0).sum())
    print(f"\n[decl_sim] === Matrix diagnostics ===")
    print(f"  Shape:           {mat.shape}")
    print(f"  Nonzero entries: {mat.nnz:,}")
    print(f"  MC modules with any support: {modules_with_support}/{n_mc}")
    if len(data):
        print(f"  Max:    {data.max():.4f}")
        print(f"  Mean:   {data.mean():.4f}")
        print(f"  Median: {np.median(data):.4f}")

    # Top-10 pairs (sanity check)
    print(f"\n[decl_sim] Top-10 highest-scoring pairs:")
    top_idx = np.argsort(vals)[-10:][::-1]
    for idx in top_idx:
        i, j, s = rows[idx], cols[idx], vals[idx]
        mc_toks = mc_token_sets[i] & ml_token_sets[j]
        print(f"  {mc_ids[i]:20s} ↔ {ml_names[j]:50s} {s:.4f}  "
              f"shared: {sorted(mc_toks)[:5]}")

    # Hard modules: check top-5 for the known wrong ones
    # Modules currently wrong in D3.2 (approximate list)
    hard = {
        "bigop":    ["Mathlib.Algebra.BigOperators"],
        "eqtype":   ["Mathlib.Logic.Equiv"],
        "fingraph": ["Mathlib.Combinatorics.SimpleGraph"],
        "path":     ["Mathlib.Combinatorics.SimpleGraph"],
        "ssrbool":  ["Mathlib.Data.Bool"],
        "ssrnat":   ["Mathlib.Data.Nat"],
        "cyclotomic": ["Mathlib.NumberTheory.Cyclotomic",
                       "Mathlib.RingTheory.Polynomial.Cyclotomic"],
    }

    print(f"\n[decl_sim] Hard-module top-5 declaration matches:")
    for mid, gold_prefixes in hard.items():
        if mid not in mc_ids:
            continue
        i = mc_ids.index(mid)
        row_arr = mat.getrow(i).toarray().flatten()
        top5 = np.argsort(row_arr)[-5:][::-1]
        print(f"\n  {mid} (gold: {gold_prefixes[0].split('.')[-1]})")
        mc_ts = mc_token_sets[i]
        for j in top5:
            if row_arr[j] == 0:
                continue
            shared = mc_ts & ml_token_sets[j]
            gold_hit = any(ml_names[j].startswith(p) for p in gold_prefixes)
            mark = "GOLD" if gold_hit else "    "
            print(f"    [{mark}] {ml_names[j]:55s} {row_arr[j]:.4f}  "
                  f"tokens: {sorted(shared)[:6]}")

    # Save diagnostics for report
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        f.write(f"Declaration similarity diagnostics\n")
        f.write(f"Shape: {mat.shape}, nnz={mat.nnz:,}\n")
        f.write(f"MC with support: {modules_with_support}/{n_mc}\n")
        if len(data):
            f.write(f"Max={data.max():.4f} Mean={data.mean():.4f} "
                    f"Median={np.median(data):.4f}\n\n")
        f.write("Top-20 discriminative (high IDF):\n")
        for t, v in top_idf:
            f.write(f"  {t:20s} {v:.3f}\n")
        f.write("\nTop-20 generic (low IDF):\n")
        for t, v in low_idf:
            f.write(f"  {t:20s} {v:.3f}\n")

    print(f"\n[decl_sim] Saved {OUT_NPZ}")
    print(f"[decl_sim] Saved {OUT_DIAG}")
    return mat, mc_token_sets, ml_token_sets, idf, mc_ids, ml_names


if __name__ == "__main__":
    compute()
