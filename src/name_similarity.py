"""Signal 1: Name-based Jaccard similarity with synonym expansion."""

import os
import re
import numpy as np
import pandas as pd
from scipy import sparse

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_NPZ = os.path.join("data", "processed", "name_sim.npz")

SYNONYMS = {
    "ssralg":    {"algebra", "ring", "module", "field", "semiring"},
    "ssrnat":    {"nat", "natural", "arithmetic"},
    "ssrint":    {"int", "integer"},
    "ssrbool":   {"bool", "boolean", "decidable"},
    "ssrfun":    {"function", "fun"},
    "ssrnum":    {"num", "number", "int", "rat"},
    "ssrAC":     {"ac", "associativity", "commutativity"},
    "fintype":   {"finite", "type", "fintype"},
    "finset":    {"finite", "set", "finset"},
    "fingroup":  {"finite", "group", "subgroup"},
    "finfun":    {"finite", "function", "fun"},
    "fingraph":  {"finite", "graph"},
    "finalg":    {"finite", "algebra"},
    "finfield":  {"finite", "field"},
    "finmodule": {"finite", "module"},
    "bigop":     {"big", "operator", "sum", "prod", "finset"},
    "eqtype":    {"eq", "equality", "decidable", "type"},
    "poly":      {"polynomial"},
    "polydiv":   {"polynomial", "division", "euclidean"},
    "polyXY":    {"polynomial", "bivariate"},
    "qpoly":     {"polynomial", "quotient"},
    "qfpoly":    {"polynomial", "quotient", "field"},
    "mxpoly":    {"matrix", "polynomial", "characteristic"},
    "mxalgebra": {"matrix", "algebra", "rank", "row", "space"},
    "mxrepresentation": {"matrix", "representation"},
    "mxred":     {"matrix", "diagonalization", "reduction"},
    "mxabelem":  {"matrix", "abelian", "elementary"},
    "zmodp":     {"zmod", "modular", "prime"},
    "perm":      {"permutation", "equiv", "symmetric"},
    "galois":    {"galois", "field", "extension"},
    "algC":      {"algebraic", "complex", "closed"},
    "algnum":    {"algebraic", "number"},
    "character": {"character", "representation"},
    "vcharacter": {"virtual", "character"},
    "classfun":  {"class", "function"},
    "presentation": {"presentation", "generators", "relations"},
    "morphism":  {"morphism", "homomorphism", "group"},
    "automorphism": {"automorphism", "group"},
    "quotient":  {"quotient", "coset", "group"},
    "gproduct":  {"group", "product", "semidirect", "direct"},
    "action":    {"action", "orbit", "stabiliser", "group"},
    "gseries":   {"group", "series", "subnormal"},
    "gfunctor":  {"group", "functor", "characteristic"},
    "pgroup":    {"group", "prime", "decomposition"},
    "nilpotent": {"nilpotent", "solvable", "group"},
    "cyclic":    {"cyclic", "group"},
    "sylow":     {"sylow", "group", "prime"},
    "center":    {"center", "group"},
    "commutator": {"commutator", "group"},
    "abelian":   {"abelian", "group"},
    "hall":      {"hall", "group", "complement"},
    "frobenius": {"frobenius", "group", "semiregular"},
    "jordanholder": {"jordan", "holder", "composition", "series"},
    "extremal":  {"extremal", "group", "prime"},
    "extraspecial": {"extraspecial", "group", "prime"},
    "maximal":   {"maximal", "subgroup"},
    "burnside_app": {"burnside", "coloring", "counting"},
    "primitive_action": {"primitive", "action", "transitive"},
    "alt":       {"alternating", "symmetric", "group"},
    "separable": {"separable", "field", "extension"},
    "cyclotomic": {"cyclotomic", "polynomial"},
    "falgebra":  {"algebra", "finite", "dimensional"},
    "fieldext":  {"field", "extension"},
    "closed_field": {"algebraically", "closed", "field"},
    "algebraics_fundamentals": {"algebraic", "number", "fundamentals"},
    "rat":       {"rational", "number", "rat"},
    "fraction":  {"fraction", "field", "integral", "domain"},
    "intdiv":    {"integer", "division", "divisibility"},
    "ssrint":    {"integer", "int", "signed"},
    "archimedean": {"archimedean"},
    "countalg":  {"countable", "algebra"},
    "interval":  {"interval", "order"},
    "interval_inference": {"interval", "inference", "number"},
    "ring_quotient": {"ring", "quotient"},
    "sesquilinear": {"sesquilinear", "form", "bilinear"},
    "spectral":  {"spectral", "gram", "schmidt", "orthogonal"},
    "vector":    {"vector", "space", "finite", "dimensional"},
    "matrix":    {"matrix", "linear", "algebra", "determinant"},
    "order":     {"order", "lattice", "partial"},
    "preorder":  {"preorder", "order"},
    "seq":       {"sequence", "list"},
    "tuple":     {"tuple", "sequence", "fixed", "length"},
    "path":      {"path", "graph", "cycle"},
    "choice":    {"choice", "type"},
    "prime":     {"prime", "number"},
    "binomial":  {"binomial", "coefficient", "combinatorics"},
    "div":       {"division", "nat", "modulo"},
    "monoid":    {"monoid", "group"},
    "nmodule":   {"module", "additive", "group"},
    "generic_quotient": {"quotient", "type"},
    "inertia":   {"inertia", "group", "character"},
    "integral_char": {"integral", "character"},
    "ssreflect": {"reflection", "tactic"},
    "ssrnotations": {"notation"},
    "ssrmatching": {"matching", "tactic"},
    "num_theory/numfield": {"number", "field", "num"},
    "num_theory/ssrnum": {"number", "num", "structure"},
    "num_theory/numdomain": {"number", "domain", "num"},
    "num_theory/orderedzmod": {"ordered", "zmod", "num"},
}


def camel_split(name: str) -> list[str]:
    tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
    return [t.lower() for t in tokens.split() if len(t) > 1]


def tokenize_mathcomp(module_id: str) -> set[str]:
    """Tokenize a MathComp module id into a set of lowercase terms."""
    parts = re.split(r'[_/]', module_id)
    base_tokens = set()
    for p in parts:
        base_tokens.update(camel_split(p) if p else [])
        if p.lower():
            base_tokens.add(p.lower())

    expanded = set(base_tokens)
    if module_id in SYNONYMS:
        expanded.update(SYNONYMS[module_id])
    for token in list(base_tokens):
        if token in SYNONYMS:
            expanded.update(SYNONYMS[token])

    expanded.discard("")
    return expanded


def tokenize_mathlib(module_name: str) -> set[str]:
    """Tokenize a Mathlib module name into a set of lowercase terms."""
    parts = module_name.split(".")
    if parts and parts[0].lower() == "mathlib":
        parts = parts[1:]
    tokens = set()
    for part in parts:
        tokens.update(camel_split(part))
        if part.lower() and len(part.lower()) > 1:
            tokens.add(part.lower())
    tokens.discard("mathlib")
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def compute():
    print("[name_similarity] Loading modules...")
    mc = pd.read_csv(MC_MODULES)
    ml = pd.read_csv(ML_MODULES)
    n_mc, n_ml = len(mc), len(ml)
    print(f"[name_similarity] {n_mc} MathComp x {n_ml} Mathlib modules")

    mc_tokens = [tokenize_mathcomp(mid) for mid in mc["module_id"]]
    ml_tokens = [tokenize_mathlib(name) for name in ml["module_name"]]

    print("[name_similarity] Computing Jaccard similarities...")
    rows, cols, vals = [], [], []
    for i in range(n_mc):
        if (i + 1) % 20 == 0:
            print(f"  MathComp module {i+1}/{n_mc}...")
        for j in range(n_ml):
            score = jaccard(mc_tokens[i], ml_tokens[j])
            if score > 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(score)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_mc, n_ml))

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, mat)

    print(f"[name_similarity] Matrix: {mat.shape}, {mat.nnz} nonzero entries")
    print(f"[name_similarity] Max={mat.max():.3f}, "
          f"Mean of nonzero={np.mean(mat.data):.4f}")

    for i in range(min(5, n_mc)):
        row = mat.getrow(i).toarray().flatten()
        top_j = np.argsort(row)[-3:][::-1]
        mid = mc["module_id"].iloc[i]
        matches = [(ml["module_name"].iloc[j], row[j]) for j in top_j if row[j] > 0]
        print(f"  {mid}: {matches}")

    print(f"[name_similarity] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
