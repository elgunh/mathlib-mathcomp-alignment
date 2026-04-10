"""Signal 2 (v2): TF-IDF cosine similarity using real Mathlib docstrings.

Reads mathlib_docstrings.csv (scraped Lean source) instead of the path-only
proxy used in v1. MathComp side is unchanged (scraped HTML descriptions).
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MC_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
ML_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
MC_DESC = os.path.join("data", "processed", "mathcomp_descriptions.csv")
ML_DOCS = os.path.join("data", "processed", "mathlib_docstrings.csv")
OUT_NPZ = os.path.join("data", "processed", "text_sim_v2.npz")


def camel_split(name: str) -> list[str]:
    tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
    return [t.lower() for t in tokens.split() if len(t) > 1]


def expand_decl_names(decl_str: str) -> str:
    """Split CamelCase declaration names into space-separated tokens."""
    if not decl_str or str(decl_str) == "nan":
        return ""
    expanded = []
    for name in decl_str.split():
        tokens = camel_split(name)
        expanded.extend(tokens)
    return " ".join(expanded)


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_mathlib_text(module_name: str, docstring: str,
                       declaration_names: str) -> str:
    """Combine path tokens + docstring + declaration names."""
    parts = []

    # 1. Module path tokens (same as v1)
    path_parts = module_name.split(".")
    if path_parts and path_parts[0].lower() == "mathlib":
        path_parts = path_parts[1:]
    path_tokens = []
    for p in path_parts:
        path_tokens.extend(camel_split(p))
    if path_tokens:
        parts.append(" ".join(path_tokens))

    # 2. Docstring (new)
    doc = str(docstring) if docstring and str(docstring) != "nan" else ""
    if doc:
        parts.append(clean(doc))

    # 3. Declaration names expanded (new)
    decl = expand_decl_names(str(declaration_names)
                              if declaration_names
                              and str(declaration_names) != "nan" else "")
    if decl:
        parts.append(decl)

    return " ".join(parts)


def build_mc_text(module_id: str, description: str) -> str:
    """MathComp: scraped HTML description + module name tokens (unchanged)."""
    name_tokens = " ".join(re.split(r'[_/]', module_id))
    desc = str(description) if description and str(description) != "nan" else ""
    combined = f"{name_tokens} {desc}"
    return clean(combined)


def compute():
    print("[text_similarity_v2] Loading data...")
    mc_mod = pd.read_csv(MC_MODULES)
    ml_mod = pd.read_csv(ML_MODULES)

    # MathComp descriptions
    mc_desc = None
    if os.path.exists(MC_DESC):
        mc_desc = pd.read_csv(MC_DESC)
        desc_map = {r["module_id"]: str(r.get("clean_description", ""))
                    for _, r in mc_desc.iterrows()}
        n_mc_desc = sum(1 for d in desc_map.values()
                        if d and d != "nan")
        print(f"[text_similarity_v2] MathComp descriptions: {n_mc_desc}/{len(mc_mod)}")
    else:
        desc_map = {}

    # Mathlib docstrings (new)
    ml_docs = None
    if os.path.exists(ML_DOCS):
        ml_docs = pd.read_csv(ML_DOCS)
        doc_map = {r["module_name"]: (
            str(r.get("docstring", "")),
            str(r.get("declaration_names", "")),
            str(r.get("text_source", "path_only")),
        ) for _, r in ml_docs.iterrows()}
        n_ml_doc = sum(1 for v in doc_map.values() if v[0] and v[0] != "nan")
        n_ml_decl = sum(1 for v in doc_map.values()
                        if (not v[0] or v[0] == "nan")
                        and v[1] and v[1] != "nan")
        print(f"[text_similarity_v2] Mathlib with docstring: {n_ml_doc}/{len(ml_mod)}")
        print(f"[text_similarity_v2] Mathlib decl-only:      {n_ml_decl}/{len(ml_mod)}")
    else:
        raise FileNotFoundError(
            f"{ML_DOCS} not found — run src/scrape_mathlib_docs.py first")

    # Build text vectors
    mc_texts = []
    for _, row in mc_mod.iterrows():
        mid = row["module_id"]
        desc = desc_map.get(mid, "")
        mc_texts.append(build_mc_text(mid, desc))

    ml_texts = []
    for _, row in ml_mod.iterrows():
        name = row["module_name"]
        entry = doc_map.get(name, ("", "", "path_only"))
        ml_texts.append(build_mathlib_text(name, entry[0], entry[1]))

    n_mc = len(mc_texts)
    all_texts = mc_texts + ml_texts
    print(f"[text_similarity_v2] Building TF-IDF on {len(all_texts)} documents...")

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.70,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[text_similarity_v2] Vocabulary size: {vocab_size}")

    mc_tfidf = tfidf_matrix[:n_mc]
    ml_tfidf = tfidf_matrix[n_mc:]

    print("[text_similarity_v2] Computing cosine similarity (106 x 7661)...")
    sim_dense = cosine_similarity(mc_tfidf, ml_tfidf)
    sim_sparse = sparse.csr_matrix(sim_dense)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, sim_sparse)

    print(f"[text_similarity_v2] Shape: {sim_sparse.shape}, "
          f"nonzero: {sim_sparse.nnz}")
    print(f"[text_similarity_v2] Max={sim_dense.max():.3f}, "
          f"Mean={sim_dense.mean():.4f}")

    # Sample top matches
    for i in range(min(5, n_mc)):
        row = sim_dense[i]
        top_j = np.argsort(row)[-3:][::-1]
        mid = mc_mod["module_id"].iloc[i]
        matches = [(ml_mod["module_name"].iloc[j], round(float(row[j]), 3))
                   for j in top_j if row[j] > 0]
        print(f"  {mid}: {matches}")

    print(f"[text_similarity_v2] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
