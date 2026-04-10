"""Scrape Lean source files from GitHub to extract module docstrings and
declaration names for all 7,661 Mathlib modules.

Files are cached under data/raw/mathlib_lean_cache/ so interrupted runs
can resume without re-fetching.  Add that directory to .gitignore.
"""

import os
import re
import time
import requests
import pandas as pd

MODULES_CSV = os.path.join("data", "processed", "mathlib_modules.csv")
CACHE_DIR = os.path.join("data", "raw", "mathlib_lean_cache")
OUT_CSV = os.path.join("data", "processed", "mathlib_docstrings.csv")

BASE_URL = ("https://raw.githubusercontent.com/"
            "leanprover-community/mathlib4/master/")

SKIP_PREFIXES = ("Lean.", "Init.", "Std.", "Aesop.")

# Declarations we care about for semantic signal
DECL_RE = re.compile(
    r'^(?:private\s+|protected\s+|@\[.*?\]\s+)*'
    r'(?:noncomputable\s+|unsafe\s+|partial\s+)*'
    r'(def|theorem|lemma|instance|class|structure|inductive|abbrev)\s+'
    r'([A-Za-z_\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9_\u03b1-\u03c9\u0391-\u03a9\']*)',
    re.MULTILINE
)

# /-! ... -/  module docstring blocks
DOC_BLOCK_RE = re.compile(r'/\-!\s*([\s\S]*?)\s*-/', re.DOTALL)

# copyright / license header  /-  ... -/  (NOT docstring)
COPY_RE = re.compile(r'^/\-\s*\n.*?-/', re.DOTALL)


def module_to_path(module_name: str) -> str | None:
    """Convert 'Mathlib.Algebra.Group.Basic' → 'Mathlib/Algebra/Group/Basic.lean'."""
    if module_name == "Mathlib":
        return None
    if any(module_name.startswith(p) for p in SKIP_PREFIXES):
        return None
    return module_name.replace(".", "/") + ".lean"


def cache_path(lean_path: str) -> str:
    return os.path.join(CACHE_DIR, lean_path)


def fetch_lean(lean_path: str, session: requests.Session) -> tuple[str | None, str]:
    """Return (content, status). Uses disk cache when available."""
    fpath = cache_path(lean_path)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "cached"

    url = BASE_URL + lean_path
    backoff = 2
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                content = resp.text
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(content)
                time.sleep(0.1)
                return content, "ok"
            elif resp.status_code == 404:
                return None, "404"
            elif resp.status_code in (429, 403, 503):
                print(f"  [rate-limit {resp.status_code}] sleeping {backoff}s …")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                return None, f"http_{resp.status_code}"
        except requests.RequestException as exc:
            print(f"  [network error] {exc} — sleeping {backoff}s …")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return None, "error"


def extract_docstring(content: str) -> str:
    """Extract and concatenate all /-! ... -/ blocks, stripping headers."""
    blocks = DOC_BLOCK_RE.findall(content)
    if not blocks:
        return ""
    parts = []
    for block in blocks:
        # Remove markdown headers, code fences, inline code
        cleaned = re.sub(r'```[\s\S]*?```', ' ', block)
        cleaned = re.sub(r'`[^`\n]+`', ' ', cleaned)
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def extract_declarations(content: str) -> str:
    """Extract declaration names joined by spaces."""
    names = []
    for m in DECL_RE.finditer(content):
        name = m.group(2)
        # Skip private/internal names (single letter, or starting with _)
        if len(name) > 1 and not name.startswith("_"):
            names.append(name)
    return " ".join(names)


def scrape():
    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_csv(MODULES_CSV)
    print(f"[scrape_mathlib_docs] {len(df)} modules to process")

    session = requests.Session()
    session.headers.update({"User-Agent": "mathlib-alignment-research/1.0"})

    results = []
    n_ok = n_404 = n_error = n_skip = n_cached = 0
    n_with_doc = n_decl_only = n_empty = 0

    for i, row in df.iterrows():
        name = row["module_name"]
        lean_path = module_to_path(name)

        if lean_path is None:
            n_skip += 1
            results.append({
                "module_name": name,
                "docstring": "",
                "declaration_names": "",
                "text_source": "skipped",
                "docstring_length": 0,
                "declaration_count": 0,
                "fetch_status": "skipped",
            })
            continue

        content, status = fetch_lean(lean_path, session)

        if status == "cached":
            n_cached += 1
        elif status == "ok":
            n_ok += 1
        elif status == "404":
            n_404 += 1
        else:
            n_error += 1

        if content:
            docstring = extract_docstring(content)
            decl_names = extract_declarations(content)
            doc_len = len(docstring)
            decl_count = len(decl_names.split()) if decl_names else 0

            if docstring:
                text_source = "docstring"
                n_with_doc += 1
            elif decl_names:
                text_source = "declarations"
                n_decl_only += 1
            else:
                text_source = "path_only"
                n_empty += 1
        else:
            docstring, decl_names = "", ""
            doc_len, decl_count = 0, 0
            text_source = "empty"
            n_empty += 1

        results.append({
            "module_name": name,
            "docstring": docstring,
            "declaration_names": decl_names,
            "text_source": text_source,
            "docstring_length": doc_len,
            "declaration_count": decl_count,
            "fetch_status": status,
        })

        if (i + 1) % 100 == 0:
            fetched = n_ok + n_cached
            print(f"  [{i+1}/{len(df)}] fetched={fetched} cached={n_cached} "
                  f"404={n_404} err={n_error} skip={n_skip}")

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    fetched = n_ok + n_cached
    print(f"\n[scrape_mathlib_docs] === Summary ===")
    print(f"  Total modules:       {len(df)}")
    print(f"  Skipped (non-ML):    {n_skip}")
    print(f"  Successfully fetched:{fetched}  (fresh={n_ok}, cached={n_cached})")
    print(f"  404:                 {n_404}")
    print(f"  Errors:              {n_error}")
    print(f"  With docstring:      {n_with_doc}")
    print(f"  Declarations only:   {n_decl_only}")
    print(f"  Path-only/empty:     {n_empty}")
    avg_doc = df_out[df_out["docstring_length"] > 0]["docstring_length"].mean()
    print(f"  Avg docstring len:   {avg_doc:.0f} chars")
    print(f"[scrape_mathlib_docs] Saved {OUT_CSV}")


if __name__ == "__main__":
    scrape()
