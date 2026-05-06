"""Time helpers used across preprocess + downstream flows.

Pure synchronous functions. ``zoneinfo`` is stdlib (Python 3.9+) so we
avoid pulling in pendulum/arrow as a dependency. All datetimes are
expected to be aware (tz-attached); naive ones are treated as UTC.
"""

from datetime import date, datetime, timezone

# ISO 8601 + the common journal/news layouts. Order matters: more specific
# patterns first so ``2024-04-15T10:30:00Z`` doesn't get partial-matched
# as "2024-04-15".
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def to_utc(dt: datetime) -> datetime:
    """Return a UTC-aware datetime equivalent to ``dt``.

    Naive inputs are *interpreted* as UTC (we do not guess local time).
    Aware inputs are converted into the UTC zone.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_filing_date(value: str) -> datetime:
    """Parse a date string (filing dates, news timestamps) into UTC datetime.

    Accepts ISO-8601 with or without time component, ``YYYY-MM-DD``,
    ``YYYY/MM/DD``, journal long form (``Apr 15, 2024``), and a few
    related variants.

    Args:
        value: Date or datetime string.

    Returns:
        Aware UTC datetime.

    Raises:
        ValueError: If none of the accepted formats match.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty date string")

    last_error: Exception | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError as exc:
            last_error = exc
            continue
        return to_utc(parsed)

    raise ValueError(
        f"could not parse date {value!r}; tried {len(_DATE_FORMATS)} formats"
    ) from last_error


def business_days_between(a: date, b: date) -> int:
    """Count weekdays (Mon-Fri) strictly between ``a`` and ``b``.

    The count is **inclusive of both endpoints** when they fall on
    weekdays, and direction-insensitive (``a > b`` returns the same value
    as the swap). No holiday calendar — that arrives in a follow-up issue.

    Examples:
        Mon -> Mon (same day) -> 1
        Mon -> Fri (same week) -> 5
        Fri -> Mon (over weekend) -> 2
    """
    if a > b:
        a, b = b, a
    total_days = (b - a).days + 1
    full_weeks, remainder = divmod(total_days, 7)
    weekdays = full_weeks * 5

    # Walk the leftover days, starting from a's weekday.
    start_dow = a.weekday()
    for offset in range(remainder):
        if (start_dow + offset) % 7 < 5:  # 0..4 are Mon..Fri
            weekdays += 1
    return weekdays
