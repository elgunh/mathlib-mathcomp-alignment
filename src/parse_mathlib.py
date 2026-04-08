"""Parse the Mathlib TSV files into a structured CSV."""

import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_MODULES = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_EDGES = os.path.join("data", "processed", "mathlib_edges.csv")


def parse():
    names_path = os.path.join(RAW_DIR, "names.tsv")
    labels_path = os.path.join(RAW_DIR, "labels.tsv")
    names_labels_path = os.path.join(RAW_DIR, "names_labels.tsv")
    adj_path = os.path.join(RAW_DIR, "adjacency.tsv")

    print("[parse_mathlib] Reading module names from", names_path)
    with open(names_path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    print("[parse_mathlib] Reading labels from", labels_path)
    with open(labels_path, encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    print("[parse_mathlib] Reading category names from", names_labels_path)
    with open(names_labels_path, encoding="utf-8") as f:
        category_names = [line.strip() for line in f if line.strip()]

    assert len(names) == len(labels), (
        f"names ({len(names)}) and labels ({len(labels)}) length mismatch"
    )

    categories = [category_names[l] if l < len(category_names) else "Unknown"
                  for l in labels]

    df_mod = pd.DataFrame({
        "module_idx": range(len(names)),
        "module_name": names,
        "label_idx": labels,
        "category": categories,
    })

    print("[parse_mathlib] Reading adjacency from", adj_path)
    df_adj = pd.read_csv(adj_path, sep="\t", header=None,
                         names=["source", "target", "weight"])

    os.makedirs(os.path.dirname(OUT_MODULES), exist_ok=True)
    df_mod.to_csv(OUT_MODULES, index=False)
    df_adj.to_csv(OUT_EDGES, index=False)

    print(f"[parse_mathlib] {len(df_mod)} modules, {len(df_adj)} edges, "
          f"{len(category_names)} categories")
    print(f"[parse_mathlib] Categories: {category_names}")
    print(f"[parse_mathlib] Saved {OUT_MODULES} and {OUT_EDGES}")


if __name__ == "__main__":
    parse()
