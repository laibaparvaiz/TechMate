"""
File: TechMate/app/tools/web_search.py

Description:
Web search fallback used by the CRAG pipeline in study_agent.py.

Used when the user's uploaded materials do not (fully) answer a
question. Returns LangChain Documents so the results can go through
the same format_context() as uploaded-PDF chunks.

Install ONE of:
    pip install ddgs
    pip install duckduckgo-search

No API key needed. Never raises: returns [] on any failure so the
caller can degrade gracefully.
"""

from typing import List

from langchain_core.documents import Document


def web_search_fallback(
    query: str,
    max_results: int = 5,
) -> List[Document]:
    """Search the web and return results as Documents ([] on failure)."""

    DDGS = None

    try:
        from ddgs import DDGS  # newer package name
    except ImportError:
        try:
            from ddgs import DDGS
        except ImportError:
            print("No web search package installed. Run: pip install ddgs")
            return []

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, max_results=max_results)
            )
    except Exception as e:
        print("Web search error:", str(e))
        return []

    docs: List[Document] = []

    for result in results:

        body = (result.get("body") or "").strip()

        if not body:
            continue

        title = (result.get("title") or "").strip()
        url = (result.get("href") or "").strip()

        docs.append(
            Document(
                page_content=f"{title}\n{body}" if title else body,
                metadata={
                    "source": url or title or "web",
                    "title": title,
                },
            )
        )

    return docs