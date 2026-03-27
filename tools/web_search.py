"""
Fallback web search using DuckDuckGo — no API key required.
Install:  pip install duckduckgo-search

Domain filtering is enforced client-side since DuckDuckGo has no
server-side include_domains param. Two strategies are combined:
  1. Query prefix  – "site:uscis.gov OR site:dhs.gov OR ..." prepended
                     to the query so DDG biases toward those domains
  2. Post-filter   – results whose URL doesn't match any allowed domain
                     are dropped before returning
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

ALLOWED_DOMAINS_RAW = os.getenv(
    "ALLOWED_DOMAINS",
    "uscis.gov,travel.state.gov,dhs.gov,state.gov,ice.gov,cbp.gov",
)
ALLOWED_DOMAINS = [d.strip() for d in ALLOWED_DOMAINS_RAW.split(",")]

# Build a site-restricted query prefix once at import time
# e.g.  "site:uscis.gov OR site:dhs.gov OR site:travel.state.gov"
SITE_FILTER = " OR ".join(f"site:{d}" for d in ALLOWED_DOMAINS)


def _domain_allowed(url: str) -> bool:
    """Client-side guard — double-check every returned URL."""
    return any(domain in url for domain in ALLOWED_DOMAINS)


def web_search(query: str, max_results: int = 6) -> list[dict]:
    """
    Search approved government websites via DuckDuckGo.

    Strategy:
      - Prepend site: operators so DDG prioritises gov domains
      - Post-filter to drop any non-gov results that slip through
      - Retry once on rate-limit (RatelimitException)

    Returns list of {url, title, content} dicts.
    Returns [] if duckduckgo-search is not installed.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print(
            "[WebSearch] 'duckduckgo-search' not installed. "
            "Run: pip install duckduckgo-search"
        )
        return []

    restricted_query = f"({SITE_FILTER}) {query}"
    print(f"[WebSearch] DDG query: {restricted_query}")

    for attempt in range(2):          # retry once on rate-limit
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(
                    restricted_query,
                    max_results=max_results * 2,  # fetch extra; some will be filtered
                    safesearch="moderate",
                )

            results = []
            for r in (raw or []):
                url = r.get("href", "")
                if not _domain_allowed(url):      # drop non-gov results
                    continue
                results.append({
                    "url":     url,
                    "title":   r.get("title", ""),
                    "content": r.get("body", ""),
                })
                if len(results) >= max_results:
                    break

            print(f"[WebSearch] {len(results)} gov results returned.")
            return results

        except Exception as e:
            error_name = type(e).__name__
            if "Ratelimit" in error_name and attempt == 0:
                print("[WebSearch] Rate-limited by DDG — waiting 3s and retrying…")
                time.sleep(3)
                continue
            print(f"[WebSearch] Error ({error_name}): {e}")
            return []

    return []


def format_web_results(results: list[dict]) -> str:
    """Render web results into a context string for the LLM prompt."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Source {i}: {r['title']} | {r['url']}]\n{r['content']}"
        )
    return "\n\n---\n\n".join(parts)