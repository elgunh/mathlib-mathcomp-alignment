"""Generate text descriptions for Mathlib modules from their hierarchical names.

Scraping all 7,661 Mathlib pages is impractical. Instead we tokenise each
module's dotted path into natural-language tokens and use those as a text
proxy.  The category label provides an additional coarse signal.
"""

import os
import re
import pandas as pd

MODULES_CSV = os.path.join("data", "processed", "mathlib_modules.csv")
OUT_CSV = os.path.join("data", "processed", "mathlib_descriptions.csv")


def camel_split(name: str) -> list[str]:
    """Split a CamelCase or mixedCase string into lowercase tokens."""
    tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
    return [t.lower() for t in tokens.split() if len(t) > 1]


def tokenize_module_name(name: str) -> list[str]:
    """Turn 'Mathlib.Algebra.Group.Basic' into ['algebra','group','basic']."""
    parts = name.split(".")
    if parts and parts[0].lower() == "mathlib":
        parts = parts[1:]
    tokens = []
    for part in parts:
        tokens.extend(camel_split(part))
    return tokens


def generate():
    print("[scrape_mathlib] Loading modules from", MODULES_CSV)
    df = pd.read_csv(MODULES_CSV)
    print(f"[scrape_mathlib] {len(df)} modules loaded")

    results = []
    for _, row in df.iterrows():
        name = row["module_name"]
        cat = row["category"]
        tokens = tokenize_module_name(name)
        text = " ".join(tokens)

        cat_tokens = camel_split(cat)
        full_text = " ".join(cat_tokens) + " " + text

        results.append({
            "module_idx": row["module_idx"],
            "module_name": name,
            "category": cat,
            "source": "name_tokens",
            "description": full_text.strip(),
        })

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"[scrape_mathlib] Generated text for {len(df_out)} modules")
    print(f"[scrape_mathlib] Sample: {results[1]['module_name']} -> "
          f"'{results[1]['description']}'")
    print(f"[scrape_mathlib] Saved {OUT_CSV}")


if __name__ == "__main__":
    generate()
