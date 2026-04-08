"""Scrape MathComp module descriptions from the HTML documentation."""

import os
import re
import time
import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_NETWORK = True
except ImportError:
    HAS_NETWORK = False

MODULES_CSV = os.path.join("data", "processed", "mathcomp_modules.csv")
CACHE_DIR = os.path.join("data", "raw", "mathcomp_html")
OUT_CSV = os.path.join("data", "processed", "mathcomp_descriptions.csv")

BASE_URL = "https://math-comp.github.io/htmldoc_2_5_0"

CLUSTER_TO_PACKAGES = {
    "boot":      ["boot", "ssreflect"],
    "ssreflect": ["ssreflect", "boot"],
    "algebra":   ["algebra"],
    "field":     ["field"],
    "fingroup":  ["fingroup"],
    "solvable":  ["solvable"],
    "character": ["character"],
    "order":     ["ssreflect", "order", "boot"],
    "all":       ["all"],
}

STOP_KEYWORDS = re.compile(
    r'^\s*(Set |Unset |From |Require |Import |Export |Section |Module |'
    r'Definition |Lemma |Theorem |Variable |Notation |Coercion |'
    r'Canonical |Record |Structure |Inductive |Fixpoint |'
    r'Local |Global |Implicit |Open |Close |Declare |'
    r'HB\.instance|HB\.mixin|HB\.structure|HB\.factory)',
    re.MULTILINE
)


def build_urls(module_id: str, cluster: str) -> list[str]:
    """Build candidate URLs for a module, trying multiple package paths."""
    mod_name = module_id.replace("/", ".")
    packages = CLUSTER_TO_PACKAGES.get(cluster, [cluster])
    urls = []
    for pkg in packages:
        urls.append(f"{BASE_URL}/mathcomp.{pkg}.{mod_name}.html")
    if cluster not in ("boot", "ssreflect"):
        urls.append(f"{BASE_URL}/mathcomp.ssreflect.{mod_name}.html")
    urls.append(f"{BASE_URL}/mathcomp.{mod_name}.html")
    return urls


def extract_description(html: str) -> str:
    """Extract the header description from a MathComp HTML doc page."""
    soup = BeautifulSoup(html, "html.parser")

    doc_div = soup.find("div", {"id": "doc_content"}) or soup.find("div", class_="doc")
    text_source = doc_div if doc_div else soup

    for tag in text_source.find_all(["script", "style", "nav", "header"]):
        tag.decompose()

    full_text = text_source.get_text(separator="\n")
    lines = full_text.split("\n")

    desc_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                desc_lines.append("")
            continue

        if re.match(r'^\s*From\s+', stripped) or re.match(r'^\s*Require\s+', stripped):
            started = True
            continue
        if re.match(r'^\s*Import\s+', stripped) or re.match(r'^\s*Export\s+', stripped):
            started = True
            continue

        if STOP_KEYWORDS.match(stripped):
            if desc_lines:
                break
            continue

        if re.match(r'^\(\*', stripped):
            content = re.sub(r'^\(\*\s*', '', stripped)
            content = re.sub(r'\s*\*\)$', '', content)
            if content.strip():
                desc_lines.append(content.strip())
                started = True
            continue

        started = True
        desc_lines.append(stripped)

        if len(desc_lines) > 40:
            break

    text = "\n".join(desc_lines).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    paras = text.split("\n\n")
    if len(paras) > 3:
        text = "\n\n".join(paras[:3])

    return text.strip()


def fetch_page(module_id: str, cluster: str) -> tuple[str, str, str]:
    """Fetch a module's HTML page. Returns (url_tried, status, html)."""
    cache_file = os.path.join(CACHE_DIR, f"{module_id.replace('/', '_')}.html")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            html = f.read()
        return ("cached", "ok", html)

    if not HAS_NETWORK:
        return ("", "no_network", "")

    urls = build_urls(module_id, cluster)
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                return (url, "ok", resp.text)
            elif resp.status_code == 404:
                continue
            else:
                print(f"  [WARN] {url} returned {resp.status_code}")
        except Exception as exc:
            print(f"  [WARN] {url} error: {exc}")
        time.sleep(1)

    return (urls[0] if urls else "", "not_found", "")


def scrape():
    print("[scrape_mathcomp] Loading modules from", MODULES_CSV)
    df = pd.read_csv(MODULES_CSV)
    print(f"[scrape_mathcomp] {len(df)} modules loaded")

    results = []
    skip_count = 0

    for _, row in df.iterrows():
        mid = row["module_id"]
        cluster = row["cluster"]

        if mid.startswith("all_") or mid == "all":
            skip_count += 1
            results.append({
                "module_id": mid,
                "cluster": cluster,
                "url_tried": "",
                "status": "skipped_umbrella",
                "raw_description": "",
                "clean_description": "",
            })
            continue

        print(f"  Fetching {mid} (cluster={cluster})...", end=" ")
        url_tried, status, html = fetch_page(mid, cluster)

        raw_desc = ""
        clean_desc = ""
        if status == "ok" and html:
            raw_desc = extract_description(html)
            clean_desc = raw_desc.replace("\n", " ").strip()
            clean_desc = re.sub(r'\s+', ' ', clean_desc)

        tag = "OK" if clean_desc else status.upper()
        preview = clean_desc[:80] + "..." if len(clean_desc) > 80 else clean_desc
        print(f"[{tag}] {preview}")

        results.append({
            "module_id": mid,
            "cluster": cluster,
            "url_tried": url_tried,
            "status": status,
            "raw_description": raw_desc,
            "clean_description": clean_desc,
        })

        if status != "cached":
            time.sleep(1.5)

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    ok_count = sum(1 for r in results if r["clean_description"])
    print(f"\n[scrape_mathcomp] Done. {ok_count}/{len(results)} modules have descriptions "
          f"({skip_count} umbrella skipped)")
    print(f"[scrape_mathcomp] Saved {OUT_CSV}")


if __name__ == "__main__":
    scrape()
