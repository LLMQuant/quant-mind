"""Reusable HTTP fetching with optional retry and host politeness policy."""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from urllib.parse import urlsplit

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
    "retry-after",
)


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Generic retry and per-host politeness settings for HTTP requests."""

    max_attempts: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    jitter_seconds: float = 0.1
    max_concurrency_per_host: int = 2
    min_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate policy values before requests start."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must be >= 0")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError(
                "backoff_max_seconds must be >= backoff_base_seconds"
            )
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be >= 0")
        if self.max_concurrency_per_host < 1:
            raise ValueError("max_concurrency_per_host must be >= 1")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be >= 0")


class FetchAttemptsExhausted(httpx.HTTPError):
    """A retryable HTTP failure that exhausted its configured attempts."""

    def __init__(
        self,
        *,
        url: str,
        attempts: int,
        last_error: httpx.HTTPError,
    ) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"request to {url!r} failed after {attempts} attempts: {last_error}"
        )


@dataclass(slots=True)
class _HostState:
    semaphore: asyncio.Semaphore
    spacing_lock: asyncio.Lock
    next_request_at: float = 0.0


class HttpFetcher:
    """Stateful HTTP fetcher sharing a client and per-host rate state."""

    def __init__(
        self,
        *,
        policy: FetchPolicy | None = None,
        timeout: float = 30.0,
        max_bytes: int = 50_000_000,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.policy = policy
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None
        self._host_states: dict[str, _HostState] = {}

    async def __aenter__(self) -> "HttpFetcher":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_url(
        self,
        url: str,
        *,
        timeout: float | None = None,
        max_bytes: int | None = None,
        user_agent: str | None = None,
    ) -> Fetched:
        """GET one URL using this fetcher's shared policy and host state."""
        if self._client is None:
            raise RuntimeError("HttpFetcher must be used as an async context")

        policy = self.policy
        attempts = policy.max_attempts if policy is not None else 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._request_once(
                    url,
                    timeout=self.timeout if timeout is None else timeout,
                    max_bytes=self.max_bytes
                    if max_bytes is None
                    else max_bytes,
                    user_agent=user_agent or self.user_agent,
                )
            except httpx.HTTPError as exc:
                if policy is None or not _is_retryable(exc):
                    raise
                if attempt == attempts:
                    raise FetchAttemptsExhausted(
                        url=url,
                        attempts=attempts,
                        last_error=exc,
                    ) from exc
                delay = min(
                    policy.backoff_max_seconds,
                    policy.backoff_base_seconds * (2 ** (attempt - 1)),
                )
                delay += random.uniform(0.0, policy.jitter_seconds)
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                await asyncio.sleep(delay)

        raise AssertionError("unreachable retry loop")

    async def _request_once(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int,
        user_agent: str,
    ) -> Fetched:
        policy = self.policy
        if policy is None:
            return await self._stream_get(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                user_agent=user_agent,
            )

        host = urlsplit(url).netloc.lower()
        state = self._host_states.get(host)
        if state is None:
            state = _HostState(
                semaphore=asyncio.Semaphore(policy.max_concurrency_per_host),
                spacing_lock=asyncio.Lock(),
            )
            self._host_states[host] = state

        async with state.semaphore:
            async with state.spacing_lock:
                loop = asyncio.get_running_loop()
                delay = state.next_request_at - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                state.next_request_at = (
                    loop.time() + policy.min_interval_seconds
                )
            return await self._stream_get(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                user_agent=user_agent,
            )

    async def _stream_get(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int,
        user_agent: str,
    ) -> Fetched:
        if self._client is None:
            raise RuntimeError("HttpFetcher must be used as an async context")

        headers = {"User-Agent": user_agent}
        async with self._client.stream(
            "GET",
            url,
            headers=headers,
            timeout=timeout,
        ) as response:
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
                key: value
                for key, value in response.headers.items()
                if key.lower() in _CAPTURED_HEADERS
            }
            return Fetched(
                bytes=b"".join(chunks),
                content_type=content_type,
                source_url=url,
                headers=captured,
                status_code=response.status_code,
                resolved_url=str(response.url),
                fetched_at=datetime.now(timezone.utc),
            )


async def fetch_url(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 50_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
    fetch_policy: FetchPolicy | None = None,
) -> Fetched:
    """GET ``url`` with optional retry/backoff and host politeness policy.

    Omitting ``fetch_policy`` preserves the original one-shot request
    behavior. Reuse ``HttpFetcher`` directly when several calls must share
    per-host rate state and one connection pool.
    """
    async with HttpFetcher(
        policy=fetch_policy,
        timeout=timeout,
        max_bytes=max_bytes,
        user_agent=user_agent,
    ) as fetcher:
        return await fetcher.fetch_url(url)


def _is_retryable(error: httpx.HTTPError) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in {408, 429} or status >= 500
    return False


def _retry_after_seconds(error: httpx.HTTPError) -> float | None:
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    if error.response.status_code not in {429, 503}:
        return None
    value = error.response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (
                parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds(),
        )
