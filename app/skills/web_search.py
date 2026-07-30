"""Web search integration skill.

Uses DuckDuckGo HTML search (no API key required).
Includes proper timeout handling and error recovery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Any

import requests


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str


class WebSearchProvider:
    """Web search provider using DuckDuckGo HTML search."""

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    DEFAULT_TIMEOUT = 10

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search DuckDuckGo and return parsed results.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        try:
            resp = requests.post(
                self.SEARCH_URL,
                data={"q": query, "b": ""},
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return self._parse_results(resp.text, num_results)
        except requests.Timeout:
            return []
        except requests.RequestException:
            return []

    def _parse_results(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML response into SearchResult objects."""
        results: list[SearchResult] = []

        # Find result links
        link_pattern = r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>'

        links = re.findall(link_pattern, html, re.DOTALL)
        snippets = re.findall(snippet_pattern, html, re.DOTALL)

        for i, (url, title) in enumerate(links[:max_results]):
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            clean_title = unescape(clean_title)

            clean_url = unescape(url)
            # DuckDuckGo wraps URLs in redirects
            url_match = re.search(r"uddg=([^&]+)", clean_url)
            if url_match:
                from urllib.parse import unquote
                clean_url = unquote(url_match.group(1))

            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                snippet = unescape(snippet)

            if clean_title and clean_url:
                results.append(SearchResult(
                    title=clean_title,
                    url=clean_url,
                    snippet=snippet,
                ))

        return results


def format_web_results(results: list[SearchResult]) -> str:
    """Format search results for inclusion in answer context.

    Args:
        results: List of SearchResult objects.

    Returns:
        Formatted string with numbered results.
    """
    if not results:
        return "No web results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        if r.snippet:
            lines.append(f"    {r.snippet}")
        lines.append("")

    return "\n".join(lines)


def search_and_format(query: str, num_results: int = 5) -> dict[str, Any]:
    """Convenience function: search and return structured results.

    Args:
        query: Search query.
        num_results: Max results.

    Returns:
        Dict with 'results' list and 'formatted' text.
    """
    provider = WebSearchProvider()
    results = provider.search(query, num_results)
    return {
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ],
        "formatted": format_web_results(results),
        "query": query,
        "count": len(results),
    }
