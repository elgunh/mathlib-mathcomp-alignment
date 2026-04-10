# Mathlib–MathComp Module Alignment

Bipartite alignment between two formal mathematics libraries:

- **Mathlib** (Lean 4): 7,661 modules, 21,178 import edges, 33 categories
- **MathComp** (Rocq/Coq): 106 modules, 194 import edges, 9 clusters

For each (MathComp module, Mathlib module) pair, computes a similarity score
in [0, 1] combining four signals: name similarity, text similarity,
category/cluster alignment, and import-graph neighbourhood overlap.

**Deliverable 2** adds an iterative graph-propagation step: high-confidence
matches (text-only signals) are anchored first, then their graph neighbourhoods
are used to discover and verify additional alignments over multiple rounds.

## Quick Start

```bash
pip install -r requirements.txt

# Linux / macOS
bash run.sh

# Windows (PowerShell)
.\run.ps1
```

## Pipeline Overview

| Step          | Script(s)                                                                                                   | Output                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1. Parse      | `src/parse_mathcomp.py`, `src/parse_mathlib.py`                                                             | `data/processed/*.csv`                                          |
| 2. Scrape     | `src/scrape_mathcomp.py`, `src/scrape_mathlib.py`                                                           | `data/processed/*_descriptions.csv`                             |
| 3. Similarity | `src/name_similarity.py`, `src/text_similarity.py`, `src/category_similarity.py`, `src/graph_similarity.py` | `data/processed/*_sim.npz`                                      |
| 4. Combine    | `src/combine_signals.py`                                                                                    | `outputs/candidate_matches.csv`, `outputs/alignment_matrix.npz` |
| 5. Visualize  | `src/visualize.py`                                                                                          | `outputs/figures/*.png`                                         |
| 6. Iterative  | `src/iterative_alignment.py`                                                                                | `outputs/iterative_matches.csv`, `outputs/propagation_log.csv`  |
| 7. Iter. Viz  | `src/visualize_iterations.py`                                                                               | `outputs/figures/convergence.png`, `outputs/figures/before_after.png` |

## Data Sources

- Mathlib dependency graph from the [Netset collection](https://netset.telecom-paris.fr)
- MathComp module graph in Cytoscape-style JSON
- MathComp HTML documentation from [math-comp.github.io](https://math-comp.github.io/htmldoc_2_5_0/)
- Mathlib documentation from [leanprover-community.github.io](https://leanprover-community.github.io/mathlib4_docs/)

## Tests

```bash
python -m pytest tests/
```
