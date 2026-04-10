"""Signal 2 (v3): synonym-aware, field-weighted TF-IDF cosine similarity.

Improvements over text_sim_v2:
  1. MathComp module names are expanded with synonym tokens before vectorisation.
  2. Mathlib text is split into three fields (path, declarations, docstring),
     each weighted separately via token repetition.
  3. Custom stopword list filters structural Lean/library words that add noise
     without discriminating between mathematical concepts.
  4. Diagnostics CSV records token-source coverage per module.

Field weights (path > declarations > docstring) reflect the principle that
the module path is the most precise semantic signal:
  path_repeats = 4
  decl_repeats = 2
  doc_repeats  = 1
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- local import ----
import sys
sys.path.insert(0, os.path.dirname(__file__))
from build_synonym_map import (
    SYNONYM_MAP, MODULE_LEVEL_EXTRA,
    expand_module_name, expand_tokens, save_map,
)

MC_MODULES  = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES  = os.path.join("data", "processed", "mathlib_modules.csv")
MC_DESC     = os.path.join("data", "processed", "mathcomp_descriptions.csv")
ML_DOCS     = os.path.join("data", "processed", "mathlib_docstrings.csv")
OUT_NPZ     = os.path.join("data", "processed", "text_sim_v3.npz")
OUT_DIAG    = os.path.join("data", "processed", "text_v3_diagnostics.csv")

# Field repetition weights
PATH_REPEATS = 4
DECL_REPEATS = 2
DOC_REPEATS  = 1

# ---- Custom stopwords ----
# Structural Lean/library words that appear in many modules without
# discriminating mathematical content.
LEAN_STOPWORDS = {
    # Lean syntax keywords
    "theorem", "lemma", "def", "definition", "define",
    "instance", "structure", "inductive", "abbrev",
    "class", "variable", "notation", "syntax", "command",
    "tactic", "linter", "meta", "elab", "macro",
    # Documentation boilerplate
    "section", "namespace", "file", "module", "library",
    "basic", "helper", "auxiliary", "implementation",
    "import", "export", "result", "proof", "show",
    # Over-general mathematical words (low IDF, high noise)
    # Only suppress when they would drown out specific tokens
    "nonterminal", "introduce", "denote", "represent",
    "following", "following", "given", "using", "used",
    "note", "notes", "remark", "remarks", "see", "also",
    "analogous", "corresponding", "version", "variant",
    "generalize", "generalization", "generalized",
    "simple", "simply", "trivial", "direct",
}

# Build the full stopword list (English + Lean structural)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
FULL_STOPWORDS = list(ENGLISH_STOP_WORDS) + list(LEAN_STOPWORDS)


def camel_split(name: str) -> list[str]:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return [t.lower() for t in s.split() if len(t) > 1]


def clean_text(text: str) -> str:
    text = str(text).lower()
    # Remove Lean code fences
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`\n]+`', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    # Keep letters, digits, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_mathlib_text_v3(module_name: str, docstring: str,
                          declaration_names: str) -> tuple[str, str]:
    """
    Returns (combined_text, source_label).
    source_label: 'docstring' | 'declarations' | 'path_only'
    """
    parts = []
    source = "path_only"

    # --- Field 1: Path tokens (PATH_REPEATS × weight) ---
    path_parts = module_name.split(".")
    if path_parts and path_parts[0].lower() == "mathlib":
        path_parts = path_parts[1:]
    path_tokens = []
    for p in path_parts:
        path_tokens.extend(camel_split(p))
    path_text = " ".join(path_tokens)
    for _ in range(PATH_REPEATS):
        parts.append(path_text)

    # --- Field 2: Declaration names (DECL_REPEATS × weight) ---
    decl = str(declaration_names) if declaration_names else ""
    if decl and decl.lower() != "nan":
        decl_tokens = []
        for name in decl.split():
            decl_tokens.extend(camel_split(name))
        decl_text = " ".join(decl_tokens)
        if decl_text.strip():
            for _ in range(DECL_REPEATS):
                parts.append(decl_text)
            if source == "path_only":
                source = "declarations"

    # --- Field 3: Filtered docstring (DOC_REPEATS × weight) ---
    doc = str(docstring) if docstring else ""
    if doc and doc.lower() != "nan":
        doc_clean = clean_text(doc)
        if doc_clean.strip():
            for _ in range(DOC_REPEATS):
                parts.append(doc_clean)
            source = "docstring"

    return " ".join(parts), source


def build_mc_text_v3(module_id: str, description: str) -> tuple[str, str]:
    """
    Returns (combined_text, source_label).
    Applies synonym expansion to the module name tokens.
    """
    parts = []

    # --- Field 1: Expanded module name (PATH_REPEATS × weight) ---
    expanded_tokens = expand_module_name(module_id, SYNONYM_MAP, MODULE_LEVEL_EXTRA)
    name_text = " ".join(expanded_tokens)
    for _ in range(PATH_REPEATS):
        parts.append(name_text)

    # --- Field 2: Scraped description ---
    desc = str(description) if description else ""
    if desc and desc.lower() != "nan":
        desc_clean = clean_text(desc)
        if desc_clean.strip():
            parts.append(desc_clean)
            source = "description"
        else:
            source = "name_only"
    else:
        source = "name_only"

    return " ".join(parts), source


def compute():
    print("[text_sim_v3] Loading data…")
    mc_mod = pd.read_csv(MC_MODULES)
    ml_mod = pd.read_csv(ML_MODULES)

    # MathComp descriptions
    desc_map: dict = {}
    if os.path.exists(MC_DESC):
        mc_desc = pd.read_csv(MC_DESC)
        desc_map = {r["module_id"]: str(r.get("clean_description", ""))
                    for _, r in mc_desc.iterrows()}

    # Mathlib docstrings
    if not os.path.exists(ML_DOCS):
        raise FileNotFoundError(f"Missing {ML_DOCS} — run scrape_mathlib_docs.py first")
    ml_docs = pd.read_csv(ML_DOCS)
    doc_map = {r["module_name"]: (
        str(r.get("docstring", "")),
        str(r.get("declaration_names", "")),
    ) for _, r in ml_docs.iterrows()}

    # Save synonym map artifact
    save_map(SYNONYM_MAP, MODULE_LEVEL_EXTRA)

    # Build texts
    mc_texts, mc_sources = [], []
    for _, row in mc_mod.iterrows():
        mid = row["module_id"]
        desc = desc_map.get(mid, "")
        txt, src = build_mc_text_v3(mid, desc)
        mc_texts.append(txt)
        mc_sources.append(src)

    ml_texts, ml_sources = [], []
    for _, row in ml_mod.iterrows():
        name = row["module_name"]
        doc, decl = doc_map.get(name, ("", ""))
        txt, src = build_mathlib_text_v3(name, doc, decl)
        ml_texts.append(txt)
        ml_sources.append(src)

    n_mc = len(mc_texts)
    all_texts = mc_texts + ml_texts

    print(f"[text_sim_v3] Total documents: {len(all_texts)}")
    print(f"  MC sources: {pd.Series(mc_sources).value_counts().to_dict()}")
    print(f"  ML sources: {pd.Series(ml_sources).value_counts().to_dict()}")

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.70,
        ngram_range=(1, 2),
        max_features=12000,   # slightly larger to absorb synonym tokens
        sublinear_tf=True,
        stop_words=FULL_STOPWORDS,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    print(f"[text_sim_v3] Vocabulary size: {len(vectorizer.vocabulary_)}")

    mc_tfidf = tfidf_matrix[:n_mc]
    ml_tfidf = tfidf_matrix[n_mc:]

    print("[text_sim_v3] Computing cosine similarity (106 × 7661)…")
    sim_dense = cosine_similarity(mc_tfidf, ml_tfidf)
    sim_sparse = sparse.csr_matrix(sim_dense)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, sim_sparse)
    print(f"[text_sim_v3] Shape: {sim_sparse.shape}, nonzero: {sim_sparse.nnz}")
    print(f"[text_sim_v3] Max={sim_dense.max():.3f}, Mean={sim_dense.mean():.4f}")

    # Diagnostics
    ml_names = list(ml_mod["module_name"])
    diag_rows = []
    for i, (mid, src) in enumerate(zip(mc_mod["module_id"], mc_sources)):
        row = sim_dense[i]
        top_j = int(np.argmax(row))
        diag_rows.append({
            "module_id": mid,
            "source": src,
            "max_sim": round(float(row[top_j]), 4),
            "top_match": ml_names[top_j],
            "mean_sim": round(float(row.mean()), 5),
        })
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_DIAG, index=False)

    # Sample top matches
    mc_ids = list(mc_mod["module_id"])
    hard = ["ssrnat", "mxalgebra", "commutator", "galois", "cyclotomic",
            "eqtype", "burnside_app", "presentation"]
    for mod in hard:
        if mod in mc_ids:
            i = mc_ids.index(mod)
            top3 = np.argsort(sim_dense[i])[-3:][::-1]
            matches = [(ml_names[j], round(float(sim_dense[i, j]), 3)) for j in top3]
            print(f"  {mod:20s}: {matches}")

    print(f"[text_sim_v3] Saved {OUT_NPZ}")
    print(f"[text_sim_v3] Saved {OUT_DIAG}")


if __name__ == "__main__":
    compute()
