"""FXMacroData fetch helpers for macroeconomic context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

DEFAULT_FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


@dataclass(frozen=True, slots=True)
class CalendarRelease:
    """One scheduled macroeconomic or central-bank release."""

    release: str
    name: str
    announcement_datetime_utc: str | None
    announcement_datetime_local: str | None
    release_date_confirmed: bool
    event_importance: str | None
    market_tier: int | None
    source: str | None
    source_url: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawCalendar:
    """Release-calendar payload for a single currency.

    ``releases`` holds the parsed rows; ``metadata`` preserves the FXMacroData
    envelope (currency, timezone, data quality) so the format layer can build
    knowledge items without a second request.
    """

    currency: str
    timezone: str | None
    url: str
    releases: tuple[CalendarRelease, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


async def fetch_fxmacrodata_calendar(
    currency: str = "usd",
    *,
    limit: int = 50,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_FXMACRODATA_BASE_URL,
    timeout: float = 30.0,
) -> RawCalendar:
    """Fetch official release-calendar rows from FXMacroData.

    Returns a frozen :class:`RawCalendar` in line with the fetch layer
    contract. No parsing beyond splitting rows from the response envelope --
    interpreting the rows is the format layer's job.
    """
    limit_count = max(1, min(int(limit), 100))
    currency_code = currency.lower()
    params: dict[str, str] = {"limit": str(limit_count)}

    url = f"{base_url.rstrip('/')}/calendar/{currency_code}"
    headers = {"User-Agent": "QuantMind/0.2 fxmacrodata-fetch"}
    api_key = (api_key or "").strip()
    if api_key:
        if any(ch.isspace() or not ch.isprintable() for ch in api_key):
            raise ValueError(
                "FXMacroData api_key must not contain whitespace or control characters"
            )
        # Sent as a header so the key is never captured in request logs or
        # proxy access logs the way a query parameter would be.
        headers["X-API-Key"] = api_key

    # Redirects are not followed so the key header is never replayed to
    # another host; a 3xx surfaces as an HTTPStatusError instead.
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False
        ) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.is_redirect:
                raise httpx.HTTPStatusError(
                    "FXMacroData returned an unexpected redirect",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                raise ValueError(
                    "FXMacroData returned a non-JSON response"
                ) from None
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            "FXMacroData request failed (HTTP %s)" % exc.response.status_code,
            request=exc.request,
            response=exc.response,
        ) from None
    except httpx.RequestError as exc:
        raise httpx.RequestError(
            "FXMacroData request failed", request=exc.request
        ) from None

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("data"), list)
        or not all(isinstance(row, dict) for row in payload["data"])
        or payload.get("error")
        or payload.get("detail")
    ):
        raise ValueError("FXMacroData returned an unexpected response")

    rows = payload.get("data")
    rows = rows[:limit_count] if isinstance(rows, list) else []
    releases = tuple(
        CalendarRelease(
            release=str(row.get("release", "")),
            name=str(row.get("name", "")),
            announcement_datetime_utc=row.get("announcement_datetime_utc"),
            announcement_datetime_local=row.get("announcement_datetime_local"),
            release_date_confirmed=row.get("release_date_confirmed") is True,
            event_importance=row.get("event_importance"),
            market_tier=row.get("market_tier"),
            source=row.get("source"),
            source_url=row.get("source_url"),
            raw=row,
        )
        for row in rows
        if isinstance(row, dict)
    )
    metadata = {key: value for key, value in payload.items() if key != "data"}
    return RawCalendar(
        currency=str(payload.get("currency", currency_code)).upper(),
        timezone=payload.get("timezone"),
        url=url,
        releases=releases,
        metadata=metadata,
    )
