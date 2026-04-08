"""Basic sanity checks for the alignment pipeline outputs."""

import os
import pytest
import pandas as pd
import numpy as np
from scipy import sparse

DATA_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "outputs"


def test_mathcomp_modules():
    path = os.path.join(DATA_DIR, "mathcomp_modules.csv")
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 106, f"Expected 106 MathComp modules, got {len(df)}"
    assert "module_id" in df.columns
    assert "cluster" in df.columns
    assert df["cluster"].nunique() == 9


def test_mathcomp_edges():
    path = os.path.join(DATA_DIR, "mathcomp_edges.csv")
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) >= 180, f"Expected ~185 edges, got {len(df)}"


def test_mathlib_modules():
    path = os.path.join(DATA_DIR, "mathlib_modules.csv")
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 7661, f"Expected 7661 Mathlib modules, got {len(df)}"
    assert "module_name" in df.columns
    assert "category" in df.columns
    assert df["category"].nunique() == 33


def test_mathcomp_descriptions():
    path = os.path.join(DATA_DIR, "mathcomp_descriptions.csv")
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 106
    n_desc = sum(1 for _, r in df.iterrows()
                 if str(r.get("clean_description", "")).strip())
    assert n_desc >= 70, f"Expected >=70 descriptions, got {n_desc}"


def test_similarity_matrices():
    for name in ["name_sim.npz", "text_sim.npz", "category_sim.npz", "graph_sim.npz"]:
        path = os.path.join(DATA_DIR, name)
        assert os.path.exists(path), f"Missing {path}"
        mat = sparse.load_npz(path)
        assert mat.shape == (106, 7661), f"{name}: expected (106, 7661), got {mat.shape}"
        assert mat.max() <= 1.0 + 1e-6, f"{name}: max value > 1.0"
        assert mat.min() >= -1e-6, f"{name}: min value < 0.0"


def test_candidate_matches():
    path = os.path.join(OUTPUT_DIR, "candidate_matches.csv")
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 1060, f"Expected 1060 rows (106 x 10), got {len(df)}"
    assert set(df.columns) == {
        "mathcomp_module", "mathcomp_cluster", "mathlib_module",
        "mathlib_category", "name_score", "text_score", "category_score",
        "graph_score", "combined_score", "rank",
    }
    assert df["rank"].min() == 1
    assert df["rank"].max() == 10
    assert len(df[df["rank"] == 1]) == 106


def test_alignment_matrix():
    path = os.path.join(OUTPUT_DIR, "alignment_matrix.npz")
    assert os.path.exists(path), f"Missing {path}"
    mat = sparse.load_npz(path)
    assert mat.shape == (106, 7661)




def test_output_figures():
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    expected = ["score_distribution.png", "cluster_barchart.png",
                "alignment_heatmap.png", "signal_contributions.png"]
    for name in expected:
        path = os.path.join(fig_dir, name)
        assert os.path.exists(path), f"Missing figure {path}"
        assert os.path.getsize(path) > 1000, f"{name} is too small"


def test_known_good_matches():
    """Check that known-correct alignments appear in top-10."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "candidate_matches.csv"))

    known = {
        "sylow": "Sylow",
        "nilpotent": "Nilpotent",
        "pgroup": "PGroup",
        "separable": "Separable",
        "cyclic": "Cyclic",
        "character": "Character",
    }

    for mc_mod, ml_keyword in known.items():
        subset = df[df["mathcomp_module"] == mc_mod]
        matched = any(ml_keyword in str(r["mathlib_module"])
                      for _, r in subset.iterrows())
        assert matched, (
            f"Expected '{ml_keyword}' in top-10 for '{mc_mod}', "
            f"got: {list(subset['mathlib_module'])}"
        )
