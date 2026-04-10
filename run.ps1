$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Parse raw data ==="
python src/parse_mathcomp.py
python src/parse_mathlib.py

Write-Host "=== Step 2: Scrape descriptions ==="
python src/scrape_mathcomp.py
python src/scrape_mathlib.py

Write-Host "=== Step 3: Compute similarities ==="
python src/name_similarity.py
python src/text_similarity.py
python src/category_similarity.py
python src/graph_similarity.py

Write-Host "=== Step 4: Combine and produce outputs ==="
python src/combine_signals.py
python src/visualize.py

Write-Host "=== Step 5: Iterative graph-propagation alignment ==="
python src/iterative_alignment.py
python src/visualize_iterations.py

Write-Host "=== Step 6: Deliverable 3 — docstring-enhanced alignment ==="
python src/scrape_mathlib_docs.py
python src/text_similarity_v2.py
python src/compare_text_signals.py
python src/iterative_alignment_v3.py
python src/visualize_v3.py
python .ai/evaluate_v3.py

Write-Host "=== Step 7: Deliverable 3.2 — synonym-aware text and reranking ==="
python src/build_synonym_map.py
python src/text_similarity_v3.py
python src/iterative_alignment_v3.py --text_sim data/processed/text_sim_v3.npz --out_matches outputs/iterative_matches_v3_textv3.csv --out_log outputs/propagation_log_v3_textv3.csv
python src/rerank_candidates_v3.py
python src/visualize_v3_2.py
python .ai/evaluate_v3_2.py

Write-Host "=== Done! Check outputs/ ==="
