"""Parse the MathComp Cytoscape-style JSON into structured CSV files."""

import json
import os
import re
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "mathcomp.json")
OUT_MODULES = os.path.join("data", "processed", "mathcomp_modules.csv")
OUT_EDGES = os.path.join("data", "processed", "mathcomp_edges.csv")


def fix_json(raw: str) -> str:
    """Convert JS-style object notation to valid JSON."""
    fixed = re.sub(r'(\w+)\s*:', r'"\1":', raw)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    return fixed


def parse():
    print("[parse_mathcomp] Reading", RAW_PATH)
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = f.read()

    fixed = fix_json(raw)
    items = json.loads(fixed)
    print(f"[parse_mathcomp] Parsed {len(items)} items from JSON")

    modules = []
    edges = []
    clusters = {}

    for item in items:
        d = item["data"]
        node_id = d["id"]

        if node_id.startswith("edge"):
            edges.append({
                "edge_id": node_id,
                "source": d["source"],
                "target": d["target"],
            })
            continue

        if node_id.startswith("cluster_"):
            if not node_id.endswith("_plus"):
                clusters[node_id] = d.get("name", node_id)
            continue

        if node_id.endswith("_plus"):
            continue

        parent_raw = d.get("parent", "")
        cluster_name = parent_raw.replace("cluster_", "") if parent_raw else ""

        modules.append({
            "module_id": node_id,
            "name": d.get("name", node_id),
            "cluster": cluster_name,
            "released": d.get("released", ""),
        })

    module_ids = {m["module_id"] for m in modules}
    real_edges = [e for e in edges if e["source"] in module_ids and e["target"] in module_ids]

    df_mod = pd.DataFrame(modules)
    df_edge = pd.DataFrame(real_edges)

    os.makedirs(os.path.dirname(OUT_MODULES), exist_ok=True)
    df_mod.to_csv(OUT_MODULES, index=False)
    df_edge.to_csv(OUT_EDGES, index=False)

    print(f"[parse_mathcomp] {len(df_mod)} modules, {len(df_edge)} edges, "
          f"{len(clusters)} clusters")
    print(f"[parse_mathcomp] Clusters: {sorted(clusters.values())}")
    print(f"[parse_mathcomp] Saved {OUT_MODULES} and {OUT_EDGES}")


if __name__ == "__main__":
    parse()
