"""HTML -> markdown via trafilatura.

Trafilatura handles boilerplate stripping (nav/footer/cookie banners),
language-aware extraction, and markdown serialisation in one call. We
expose a thin async wrapper that runs the call in a worker thread.
"""

import asyncio

import trafilatura


def _extract_sync(html: str, *, strip_boilerplate: bool) -> str:
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_images=False,
        include_tables=True,
        favor_recall=not strip_boilerplate,
    )
    return extracted or ""


async def html_to_markdown(
    html: str,
    *,
    strip_boilerplate: bool = True,
) -> str:
    """Convert an HTML document into markdown.

    Args:
        html: Raw HTML source as a string. Pass already-decoded text — the
            fetch layer hands back ``bytes`` so callers should ``decode()``
            first using whatever charset is appropriate.
        strip_boilerplate: When ``True`` (default), trafilatura aggressively
            removes nav/footer/aside content (``favor_recall=False``). Set
            to ``False`` to keep more peripheral content at the cost of
            noise.

    Returns:
        Markdown string. Returns an empty string if trafilatura yielded
        nothing extractable (e.g. the page is a redirect or login wall).
    """
    if not html.strip():
        return ""
    return await asyncio.to_thread(
        _extract_sync, html, strip_boilerplate=strip_boilerplate
    )
