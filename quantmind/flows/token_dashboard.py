"""Session token dashboard utilities built on Agents SDK run hooks.

`SessionTokenDashboard` can be attached to any flow call through
``extra_run_hooks``. It accumulates token usage on every LLM response and
renders a compact text dashboard showing total usage and remaining tokens from
an optional session budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import RunHooks
from agents.items import ModelResponse
from agents.usage import Usage


@dataclass(frozen=True, slots=True)
class TokenDashboardSnapshot:
    """Immutable snapshot of session token usage."""

    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    session_token_budget: int | None
    remaining_tokens: int | None
    used_percent: float | None


class SessionTokenDashboard(RunHooks[Any]):
    """Track session token usage and render a terminal-friendly dashboard."""

    def __init__(self, session_token_budget: int | None = None) -> None:
        self._session_token_budget = _validate_budget(session_token_budget)
        self._usage = Usage()

    @property
    def session_token_budget(self) -> int | None:
        """Configured budget used to compute remaining tokens."""
        return self._session_token_budget

    def set_session_token_budget(
        self, session_token_budget: int | None
    ) -> None:
        """Update the session token budget used for remaining-token math."""
        self._session_token_budget = _validate_budget(session_token_budget)

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: ModelResponse,
    ) -> None:
        """Accumulate usage emitted by the SDK at the end of each LLM call."""
        del context, agent
        self._usage.add(response.usage)

    def reset(self) -> None:
        """Reset all usage counters for a new session window."""
        self._usage = Usage()

    def snapshot(self) -> TokenDashboardSnapshot:
        """Return an immutable snapshot of current token usage."""
        remaining_tokens: int | None = None
        used_percent: float | None = None
        if self._session_token_budget is not None:
            used_percent = (
                self._usage.total_tokens / self._session_token_budget
            ) * 100.0
            remaining_tokens = max(
                self._session_token_budget - self._usage.total_tokens,
                0,
            )
        return TokenDashboardSnapshot(
            requests=self._usage.requests,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            total_tokens=self._usage.total_tokens,
            session_token_budget=self._session_token_budget,
            remaining_tokens=remaining_tokens,
            used_percent=used_percent,
        )

    def as_dict(self) -> dict[str, int | float | None]:
        """Return the dashboard snapshot as a serializable dictionary."""
        snapshot = self.snapshot()
        return {
            "requests": snapshot.requests,
            "input_tokens": snapshot.input_tokens,
            "output_tokens": snapshot.output_tokens,
            "total_tokens": snapshot.total_tokens,
            "session_token_budget": snapshot.session_token_budget,
            "remaining_tokens": snapshot.remaining_tokens,
            "used_percent": snapshot.used_percent,
        }

    def render(self, *, bar_width: int = 24) -> str:
        """Render a compact text dashboard for terminal output."""
        if bar_width < 1:
            raise ValueError(f"bar_width must be >= 1, got {bar_width}")
        snapshot = self.snapshot()
        lines = [
            "Session Token Dashboard",
            f"Requests: {snapshot.requests}",
            f"Input tokens: {snapshot.input_tokens}",
            f"Output tokens: {snapshot.output_tokens}",
            f"Total used: {snapshot.total_tokens}",
        ]
        if snapshot.session_token_budget is None:
            lines.append("Remaining: n/a (set session_token_budget)")
            return "\n".join(lines)

        used_percent = snapshot.used_percent or 0.0
        filled = min(
            bar_width,
            int(round((min(used_percent, 100.0) / 100.0) * bar_width)),
        )
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.extend(
            [
                f"Budget: {snapshot.session_token_budget}",
                f"Remaining: {snapshot.remaining_tokens}",
                (
                    f"Used/Budget: {snapshot.total_tokens}"
                    f"/{snapshot.session_token_budget}"
                ),
                f"[{bar}] {used_percent:.1f}% used",
            ]
        )
        return "\n".join(lines)


def _validate_budget(session_token_budget: int | None) -> int | None:
    """Validate and normalize the session budget."""
    if session_token_budget is None:
        return None
    if session_token_budget <= 0:
        raise ValueError("session_token_budget must be > 0 when provided")
    return session_token_budget
