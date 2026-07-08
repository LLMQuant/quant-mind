"""FXMacroData fetch helpers for macroeconomic context."""

from __future__ import annotations

from typing import Any, Optional

import httpx

DEFAULT_FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


async def fetch_fxmacrodata_calendar(
    currency: str = "usd",
    *,
    limit: int = 50,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_FXMACRODATA_BASE_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch official release-calendar rows from FXMacroData.

    The function returns the parsed JSON payload so callers can preserve
    FXMacroData metadata such as data quality, source names, and confirmed
    announcement timestamps when building knowledge items.
    """

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    if api_key:
        params["api_key"] = api_key

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}"
    headers = {"User-Agent": "QuantMind/0.2 fxmacrodata-fetch"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    if isinstance(payload.get("data"), list):
        payload["data"] = payload["data"][:limit_count]

    return payload
