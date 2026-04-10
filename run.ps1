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

Write-Host "=== Done! Check outputs/ ==="
