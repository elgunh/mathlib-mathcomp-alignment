"""Build and persist the synonym/expansion map for MathComp/SSReflect vocabulary.

The map is small and auditable: every entry has a clear mathematical rationale.
Tokens are preserved and expansions are *added* (not replacing), so no
original signal is lost.

Saves to outputs/synonym_map_used.json for reproducibility / inspection.
"""

import json
import os

OUTPUTS_DIR = "outputs"

# ===== SYNONYM MAP =================================================================
# key:   a MathComp/SSReflect token (lowercase)
# value: list of Mathlib-aligned expansion tokens to ADD
#
# Design rules:
#  1. Only add tokens that a Mathlib module name, docstring, or declaration
#     would plausibly contain.
#  2. Prefer single-word forms that appear in Mathlib paths.
#  3. Keep entries small — max ~4 expansions per key.
#  4. Do not add cross-domain expansions (no "nat" -> "category").
SYNONYM_MAP: dict[str, list[str]] = {
    # ---- SSReflect library prefixes -----------------------------------------------
    "ssr":          ["ssreflect"],
    "ssrnat":       ["natural", "nat", "arithmetic", "number"],
    "ssralg":       ["algebra", "ring", "algebraic", "algebraicstructure"],
    "ssrint":       ["integer", "int"],
    "ssrfun":       ["function", "basicfunctions"],
    "ssrbool":      ["bool", "boolean", "decidable"],
    "ssrnum":       ["numeric", "ordered", "archimedean", "number"],

    # ---- Matrix / linear algebra --------------------------------------------------
    "mx":           ["matrix"],
    "mxalgebra":    ["matrix", "algebra", "rank", "rowspace", "linearalgebra"],
    "mxpoly":       ["matrix", "polynomial", "charpoly", "minimal"],
    "mxrepresentation": ["matrix", "representation", "module"],

    # ---- Combinatorics / finset operators -----------------------------------------
    "bigop":        ["bigoperators", "operator", "finset", "sum", "product"],

    # ---- Finite structures --------------------------------------------------------
    "fingroup":     ["finite", "group", "finitegroup"],
    "fintype":      ["finite", "type", "finitetype"],
    "finset":       ["finite", "set", "finset"],
    "finfun":       ["finite", "function", "pifin"],
    "fingraph":     ["finite", "graph", "simplegraph"],
    "finfield":     ["finite", "field", "galoisfield"],

    # ---- Type-theory machinery ----------------------------------------------------
    "eqtype":       ["equality", "decidable", "equiv", "subtype"],
    "choice":       ["choice", "classical", "zorn", "axiomofchoice"],

    # ---- Number-theory shorthand --------------------------------------------------
    "zmodp":        ["zmod", "modular", "quotient", "integers", "integers_mod"],
    "zmod":         ["modular", "quotient", "integers"],

    # ---- Common abbreviated components (applied per camelCase token) ---------------
    "alg":          ["algebra"],
    "nat":          ["natural", "number"],
    "int":          ["integer"],
    "rat":          ["rational"],
    "poly":         ["polynomial"],
    "fin":          ["finite"],
    "perm":         ["permutation"],
    "quot":         ["quotient"],
    "comm":         ["commutator", "commutative"],
    "seq":          ["sequence", "list"],
    "eq":           ["equality", "equiv"],
    "repr":         ["representation"],
    "char":         ["character", "charpoly"],
    "cyc":          ["cyclic"],
    "aut":          ["automorphism"],
    "hom":          ["homomorphism"],
    "iso":          ["isomorphism"],
    "grp":          ["group"],
    "ord":          ["order"],
    "arith":        ["arithmetic"],
    "num":          ["number", "numeric"],
    "op":           ["operator"],
    "div":          ["divisibility", "division"],
    "gcd":          ["greatestcommondivisor", "divisibility"],
    "lcm":          ["leastcommonmultiple"],
    "exp":          ["exponentiation", "power"],
    "mod":          ["modular", "modulo"],
    "abs":          ["absolute", "norm"],
    "trunc":        ["truncation"],
    "lim":          ["limit", "convergence"],
    "top":          ["topology", "topological"],
    "bij":          ["bijective", "bijection"],
    "inj":          ["injective", "injection"],
    "surj":         ["surjective", "surjection"],
    "pred":         ["predicate"],
    "rel":          ["relation"],
}

# ===== MODULE-LEVEL OVERRIDES =====================================================
# For specific MathComp module IDs, override or augment the token set.
# These are added ON TOP of the per-token synonym expansion.
MODULE_LEVEL_EXTRA: dict[str, list[str]] = {
    "burnside_app":   ["burnside", "groupaction", "orbit", "stabilizer"],
    "commutator":     ["commutator", "hallwitt", "derivedsubgroup"],
    "jordanholder":   ["jordanholder", "compositionseries", "subnormal"],
    "gseries":        ["subnormal", "series", "filtration"],
    "mxrepresentation": ["linearrepresentation", "maschke", "semisimple"],
    "vcharacter":     ["virtualcharacter", "character", "frobenius"],
    "classfun":       ["classfunction", "conjugacy", "character"],
    "alt":            ["alternating", "evenpermutation"],
    "gproduct":       ["semidirect", "product", "directproduct"],
    "presentation":   ["presentedgroup", "generator", "relation", "freegroup"],
    "zmodp":          ["zmod", "integers_mod", "cyclicgroup"],
    "ssrAC":          ["associative", "commutative", "rewriting"],
}


def expand_tokens(tokens: list[str], synonym_map: dict) -> list[str]:
    """Add synonym expansions to a token list (preserving originals)."""
    result = list(tokens)
    for tok in tokens:
        exps = synonym_map.get(tok.lower(), [])
        result.extend(exps)
    return result


def expand_module_name(module_id: str, synonym_map: dict,
                       module_extras: dict) -> list[str]:
    """
    Split a MathComp module ID into tokens and apply synonym expansion.

    E.g.:  "mxalgebra" -> ["mxalgebra", "matrix", "algebra", "rank", ...]
           "ssrnat"    -> ["ssrnat", "natural", "nat", "arithmetic", "number"]
    """
    import re

    # Split on underscores / slashes first, then camelCase
    def camel_split(s: str) -> list[str]:
        tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
        tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
        return [t.lower() for t in tokens.split() if len(t) > 1]

    raw_parts = re.split(r'[_/]', module_id)
    base_tokens = []
    for p in raw_parts:
        # Try exact match first
        base_tokens.append(p.lower())
        # Then per-camelCase subtokens
        for sub in camel_split(p):
            if sub != p.lower():
                base_tokens.append(sub)

    expanded = expand_tokens(base_tokens, synonym_map)

    # Add module-level extras
    extras = module_extras.get(module_id, [])
    expanded.extend(extras)

    # Deduplicate preserving order
    seen = set()
    out = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def save_map(synonym_map: dict, module_extras: dict, out_dir: str = OUTPUTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "synonym_map": synonym_map,
        "module_level_extra": module_extras,
    }
    path = os.path.join(out_dir, "synonym_map_used.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[synonym_map] Saved {path}")
    return path


def load_map(path: str = None):
    """Load from file if it exists, otherwise return defaults."""
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "synonym_map_used.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["synonym_map"], data["module_level_extra"]
    return SYNONYM_MAP, MODULE_LEVEL_EXTRA


if __name__ == "__main__":
    save_map(SYNONYM_MAP, MODULE_LEVEL_EXTRA)
    # Demo
    for mod in ["ssrnat", "mxalgebra", "eqtype", "burnside_app", "commutator"]:
        toks = expand_module_name(mod, SYNONYM_MAP, MODULE_LEVEL_EXTRA)
        print(f"  {mod:20s} -> {toks[:8]}")
