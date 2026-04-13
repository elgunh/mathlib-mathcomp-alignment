"""Scrape declaration names from MathComp source files.

Primary method: fetch raw .v files from GitHub
  https://raw.githubusercontent.com/math-comp/math-comp/master/mathcomp/{cluster}/{module}.v

Fallback: parse declaration names from HTML anchor IDs in
  data/raw/mathcomp_html/{module}.html

Output: data/processed/mathcomp_declarations.csv
  columns: module_id, cluster, raw_declarations, declaration_count,
           token_bag, source
"""

import os
import re
import sys
import time
import requests
import pandas as pd
from collections import Counter

MC_MODULES  = os.path.join("data", "processed", "mathcomp_modules.csv")
CACHE_DIR   = os.path.join("data", "raw", "mathcomp_v_cache")
HTML_DIR    = os.path.join("data", "raw", "mathcomp_html")
OUT_CSV     = os.path.join("data", "processed", "mathcomp_declarations.csv")

GITHUB_RAW  = ("https://raw.githubusercontent.com/"
               "math-comp/math-comp/master/{cluster}/{module}.v")

REQUEST_DELAY = 0.5   # seconds between GitHub requests

# ── Lean-style keywords used in .v files ─────────────────────────────────
DECL_RE = re.compile(
    r'^(?:Lemma|Theorem|Definition|Fixpoint|Canonical|Coercion|'
    r'Inductive|Record|Structure|Variant|Fact|Remark|Proposition|'
    r'Let|Notation|Section|Module|Class|Instance|HB\.mixin|HB\.structure|'
    r'HB\.factory|HB\.builders?)\s+(\w[\w\']*)',
    re.MULTILINE,
)

# ── Token helpers ─────────────────────────────────────────────────────────
def camel_split(name: str) -> list[str]:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return [t.lower() for t in s.split() if len(t) > 1]


def tokenise(name: str) -> list[str]:
    """Split a declaration name into lowercase tokens (camelCase + snake_case)."""
    parts = re.split(r'[_.\s]+', name)
    tokens = []
    for p in parts:
        tokens.extend(camel_split(p) or [p.lower()])
    return [t for t in tokens if len(t) >= 2]


def make_token_bag(names: list[str]) -> list[str]:
    tokens = []
    for n in names:
        tokens.extend(tokenise(n))
    return tokens


# ── Module-to-path mapping ────────────────────────────────────────────────
# Some module_ids contain slashes (e.g. num_theory/ssrnum) — the part before
# the slash is a subdirectory inside the cluster directory on GitHub.
def module_to_url(module_id: str, cluster: str) -> str:
    # Handle modules like "num_theory/ssrnum" → cluster/num_theory/ssrnum.v
    # The cluster already encodes the top-level dir
    return GITHUB_RAW.format(cluster=cluster, module=module_id)


def cache_path(module_id: str) -> str:
    safe = module_id.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.v")


# ── Fetching ─────────────────────────────────────────────────────────────
def fetch_v_file(module_id: str, cluster: str,
                 session: requests.Session) -> tuple[str | None, str]:
    """Return (content, source_label). Source is 'v_file' or 'failed'."""
    cp = cache_path(module_id)
    if os.path.exists(cp):
        return open(cp, encoding="utf-8", errors="replace").read(), "v_file"

    url = module_to_url(module_id, cluster)
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                content = resp.text
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(content)
                time.sleep(REQUEST_DELAY)
                return content, "v_file"
            elif resp.status_code == 404:
                # Try without cluster subdirectory prefix
                return None, "failed"
            else:
                wait = 2 ** attempt
                time.sleep(wait)
        except Exception:
            time.sleep(2 ** attempt)
    return None, "failed"


def extract_from_v(content: str) -> list[str]:
    return DECL_RE.findall(content)


# ── HTML fallback ─────────────────────────────────────────────────────────
_HASH_RE  = re.compile(r'^[0-9a-f]{20,}$')          # MD5/SHA content hashes
_LAB_RE   = re.compile(r'^lab\d+$')                  # labN placeholders
_NUM_RE   = re.compile(r'^\d+$')                     # pure numbers
_SHORT_RE = re.compile(r'^.{1,2}$')                  # 1-2 char stubs

def _is_junk_id(aid: str) -> bool:
    return bool(
        ":" in aid or "." in aid
        or _HASH_RE.match(aid)
        or _LAB_RE.match(aid)
        or _NUM_RE.match(aid)
        or _SHORT_RE.match(aid)
    )


def extract_from_html(module_id: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    safe = module_id.replace("/", "_")
    html_path = os.path.join(HTML_DIR, f"{safe}.html")
    if not os.path.exists(html_path):
        return []
    with open(html_path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    names = []
    for a in soup.find_all("a", id=True):
        aid = a.get("id", "")
        if not _is_junk_id(aid):
            names.append(aid)
    return names


# ── Main ──────────────────────────────────────────────────────────────────
def scrape():
    print(f"[scrape_mc_decl] Loading modules from {MC_MODULES}")
    mc = pd.read_csv(MC_MODULES)
    os.makedirs(CACHE_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "mathcomp-alignment-research/1.0",
        "Accept":     "text/plain",
    })

    records = []
    v_success = v_fail = html_success = 0

    for idx, row in mc.iterrows():
        mid     = row["module_id"]
        cluster = row["cluster"]

        # Skip meta-modules like all_*, all_solvable etc.
        if mid.startswith("all_") or mid == "all":
            names, source = [], "skipped"
        else:
            content, source = fetch_v_file(mid, cluster, session)
            if content:
                names = extract_from_v(content)
                v_success += 1
            else:
                # Try HTML fallback
                names = extract_from_html(mid)
                source = "html_fallback" if names else "failed"
                if names:
                    html_success += 1
                else:
                    v_fail += 1

        token_bag = make_token_bag(names)
        records.append({
            "module_id":         mid,
            "cluster":           cluster,
            "raw_declarations":  " ".join(names),
            "declaration_count": len(names),
            "token_bag":         " ".join(token_bag),
            "source":            source,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(mc)}] {mid} ({cluster}): "
                  f"{len(names)} decls from {source}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── Diagnostics ───────────────────────────────────────────────────────
    print(f"\n[scrape_mc_decl] === Coverage ===")
    print(f"  .v file success:   {v_success}")
    print(f"  HTML fallback:     {html_success}")
    print(f"  Failed / skipped:  {v_fail + (df['source'] == 'skipped').sum()}")

    has_decls = df[df["declaration_count"] > 0]
    print(f"\n  Modules with >=10 decls: {(df['declaration_count'] >= 10).sum()}")
    print(f"  Modules with <5 decls:   {(df['declaration_count'] < 5).sum()}")

    # Token frequency
    all_tokens = []
    for bag in df["token_bag"].dropna():
        all_tokens.extend(bag.split())
    from collections import Counter
    freq = Counter(all_tokens)
    print(f"\n  Total token occurrences: {len(all_tokens)}")
    print(f"  Unique tokens: {len(freq)}")
    print(f"\n  Top-20 tokens (likely generic):")
    for tok, cnt in freq.most_common(20):
        print(f"    {tok:20s} {cnt}")

    print(f"\n[scrape_mc_decl] Saved {OUT_CSV}")

    # Sample: hard modules
    hard = ["bigop", "eqtype", "fingraph", "path", "ssrnat", "ssrbool", "ssralg"]
    print(f"\n[scrape_mc_decl] Sample declarations from hard modules:")
    for mid in hard:
        row = df[df["module_id"] == mid]
        if not row.empty:
            raw = str(row.iloc[0]["raw_declarations"])[:120]
            cnt = row.iloc[0]["declaration_count"]
            src = row.iloc[0]["source"]
            print(f"  {mid:20s} ({src}, {cnt} decls): {raw}")

    return df


if __name__ == "__main__":
    scrape()
