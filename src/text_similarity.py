"""Signal 2: TF-IDF cosine similarity on descriptions + name tokens."""

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
ML_DESC = os.path.join("data", "processed", "mathlib_descriptions.csv")
OUT_NPZ = os.path.join("data", "processed", "text_sim.npz")

STOP_WORDS = {
    "the", "of", "a", "is", "in", "for", "this", "file", "we", "from",
    "that", "it", "an", "are", "and", "to", "with", "on", "as", "by",
    "be", "or", "its", "has", "have", "see", "also", "at", "was",
    "md", "nb", "hb", "contributing", "introduction", "concepts",
    "commands", "conventions", "rocq", "coq", "lean", "mathlib",
}


def camel_split(name: str) -> list[str]:
    tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
    return [t.lower() for t in tokens.split() if len(t) > 1]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


def build_mc_texts(modules_df: pd.DataFrame, desc_df: pd.DataFrame) -> list[str]:
    desc_map = {}
    if desc_df is not None:
        for _, row in desc_df.iterrows():
            desc_map[row["module_id"]] = str(row.get("clean_description", ""))

    texts = []
    for _, row in modules_df.iterrows():
        mid = row["module_id"]
        name_tokens = " ".join(re.split(r'[_/]', mid))
        desc = desc_map.get(mid, "")
        combined = f"{name_tokens} {desc}"
        texts.append(clean_text(combined))
    return texts


def build_ml_texts(modules_df: pd.DataFrame, desc_df: pd.DataFrame) -> list[str]:
    desc_map = {}
    if desc_df is not None:
        for _, row in desc_df.iterrows():
            desc_map[int(row["module_idx"])] = str(row.get("description", ""))

    texts = []
    for _, row in modules_df.iterrows():
        idx = int(row["module_idx"])
        name = row["module_name"]
        parts = name.split(".")
        if parts and parts[0].lower() == "mathlib":
            parts = parts[1:]
        name_tokens = []
        for p in parts:
            name_tokens.extend(camel_split(p))
        desc = desc_map.get(idx, "")
        combined = " ".join(name_tokens) + " " + desc
        texts.append(clean_text(combined))
    return texts


def compute():
    print("[text_similarity] Loading data...")
    mc_mod = pd.read_csv(MC_MODULES)
    ml_mod = pd.read_csv(ML_MODULES)

    mc_desc = None
    if os.path.exists(MC_DESC):
        mc_desc = pd.read_csv(MC_DESC)
        n_desc = sum(1 for _, r in mc_desc.iterrows()
                     if str(r.get("clean_description", "")).strip()
                     and str(r.get("clean_description", "")) != "nan")
        print(f"[text_similarity] MathComp descriptions: {n_desc}/{len(mc_desc)}")

    ml_desc = None
    if os.path.exists(ML_DESC):
        ml_desc = pd.read_csv(ML_DESC)
        print(f"[text_similarity] Mathlib descriptions: {len(ml_desc)}")

    mc_texts = build_mc_texts(mc_mod, mc_desc)
    ml_texts = build_ml_texts(ml_mod, ml_desc)

    n_mc = len(mc_texts)
    all_texts = mc_texts + ml_texts

    print(f"[text_similarity] Building TF-IDF on {len(all_texts)} documents...")
    vectorizer = TfidfVectorizer(
        min_df=1, max_df=0.8, ngram_range=(1, 2), max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    mc_tfidf = tfidf_matrix[:n_mc]
    ml_tfidf = tfidf_matrix[n_mc:]

    print("[text_similarity] Computing cosine similarity...")
    sim_dense = cosine_similarity(mc_tfidf, ml_tfidf)
    sim_sparse = sparse.csr_matrix(sim_dense)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    sparse.save_npz(OUT_NPZ, sim_sparse)

    print(f"[text_similarity] Matrix: {sim_sparse.shape}, "
          f"nonzero: {sim_sparse.nnz}")
    print(f"[text_similarity] Max={sim_dense.max():.3f}, "
          f"Mean={sim_dense.mean():.4f}")

    for i in range(min(5, n_mc)):
        row = sim_dense[i]
        top_j = np.argsort(row)[-3:][::-1]
        mid = mc_mod["module_id"].iloc[i]
        matches = [(ml_mod["module_name"].iloc[j], row[j])
                    for j in top_j if row[j] > 0]
        print(f"  {mid}: {matches}")

    print(f"[text_similarity] Saved {OUT_NPZ}")


if __name__ == "__main__":
    compute()
