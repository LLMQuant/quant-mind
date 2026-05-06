"""HTTP fetch helper.

Pulls bytes from an arbitrary URL with a sane default User-Agent, a body
size cap, and a request timeout. Returns ``Fetched`` so the format layer
can dispatch on ``content_type`` without re-inspecting headers.
"""

import httpx

from quantmind.preprocess.fetch._types import Fetched

DEFAULT_USER_AGENT = (
    "QuantMind/0.2 (+https://github.com/LLMQuant/quant-mind) "
    "preprocess.fetch.http"
)

_CAPTURED_HEADERS: tuple[str, ...] = (
    "content-type",
    "content-length",
    "etag",
    "last-modified",
    "content-disposition",
)


async def fetch_url(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 50_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Fetched:
    """GET ``url`` and return the body as ``Fetched``.

    Streams the response so we can short-circuit on payloads larger than
    ``max_bytes`` without buffering the whole thing into memory.

    Args:
        url: Absolute http(s) URL.
        timeout: Per-request timeout in seconds.
        max_bytes: Hard ceiling on response body size. Hitting it raises
            ``ValueError``.
        user_agent: Optional override; defaults to a QuantMind-branded UA.

    Returns:
        ``Fetched`` with body bytes, ``content_type`` (lower-cased,
        parameter-stripped — e.g. ``"text/html"`` not
        ``"text/html; charset=utf-8"``), and a curated subset of response
        headers.

    Raises:
        ValueError: If the response body exceeds ``max_bytes``.
        httpx.HTTPError: For network / status / timeout failures.
    """
    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError(
                        f"response body exceeded max_bytes={max_bytes} "
                        f"(received >= {received})"
                    )
                chunks.append(chunk)

            raw_content_type = response.headers.get(
                "content-type", "application/octet-stream"
            )
            content_type = raw_content_type.split(";", 1)[0].strip().lower()
            captured = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in _CAPTURED_HEADERS
            }

    return Fetched(
        bytes=b"".join(chunks),
        content_type=content_type,
        source_url=url,
        headers=captured,
    )
