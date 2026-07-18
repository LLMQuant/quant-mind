"""Tests for GrpoSuitabilityScorer (qm_mcp.grpo_suitability)."""

from __future__ import annotations

import pytest

from qm_mcp.grpo_suitability import GrpoSuitabilityScorer

SCORER = GrpoSuitabilityScorer()

# ── fixtures ──────────────────────────────────────────────────────────────────

CODE_BLOCK = "```python\nx = 1\n```"
SHORT_TEXT = "Short text."
LONG_TEXT = "word " * 5_000  # 25 000 chars


def _make_entry(
    source_type: str = "arxiv",
    source: str = "2606.25996",
    markdown: str | None = None,
    markdown_chars: int | None = None,
) -> dict:
    entry: dict = {"source_type": source_type, "source": source}
    if markdown is not None:
        entry["markdown"] = markdown
    if markdown_chars is not None:
        entry["markdown_chars"] = markdown_chars
    return entry


# ── heuristic correctness ─────────────────────────────────────────────────────


class TestHighSuitability:
    def test_long_arxiv_with_code_is_high(self) -> None:
        entry = _make_entry(
            source_type="arxiv",
            markdown=LONG_TEXT + "\n" + CODE_BLOCK,
            markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) == "high"

    def test_long_local_pdf_with_code_is_high(self) -> None:
        # local source_type maps to "arxiv" domain band (academic PDFs)
        entry = _make_entry(
            source_type="local",
            source="/home/user/paper.pdf",
            markdown=LONG_TEXT + "\n" + CODE_BLOCK,
            markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) == "high"

    def test_long_arxiv_without_code_is_medium(self) -> None:
        entry = _make_entry(
            source_type="arxiv",
            markdown=LONG_TEXT,
            markdown_chars=len(LONG_TEXT),
        )
        assert SCORER.score_entry(entry) == "medium"


class TestLowSuitability:
    def test_short_news_no_code_is_low(self) -> None:
        entry = _make_entry(
            source_type="url",
            source="https://reuters.com/article/xyz",
            markdown=SHORT_TEXT,
            markdown_chars=len(SHORT_TEXT),
        )
        assert SCORER.score_entry(entry) == "low"

    def test_short_text_ingest_no_code_is_low(self) -> None:
        entry = _make_entry(
            source_type="text",
            source="text:abcdef123456",
            markdown=SHORT_TEXT,
            markdown_chars=len(SHORT_TEXT),
        )
        assert SCORER.score_entry(entry) == "low"

    def test_short_news_with_code_is_not_low(self) -> None:
        # code_present breaks the low rule → bumps to medium
        entry = _make_entry(
            source_type="url",
            source="https://reuters.com/article/xyz",
            markdown=SHORT_TEXT + "\n" + CODE_BLOCK,
            markdown_chars=len(SHORT_TEXT) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) != "low"


class TestMediumSuitability:
    def test_medium_length_arxiv_with_code_is_medium(self) -> None:
        medium_text = "word " * 2_500  # 12 500 chars
        entry = _make_entry(
            source_type="arxiv",
            markdown=medium_text + CODE_BLOCK,
            markdown_chars=len(medium_text) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) == "medium"

    def test_ssrn_url_is_medium_not_high(self) -> None:
        entry = _make_entry(
            source_type="url",
            source="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1234",
            markdown=LONG_TEXT + CODE_BLOCK,
            markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
        )
        # ssrn domain band is not "arxiv" → can't be high
        assert SCORER.score_entry(entry) == "medium"

    def test_substack_url_is_medium(self) -> None:
        entry = _make_entry(
            source_type="url",
            source="https://example.substack.com/p/post",
            markdown=LONG_TEXT + CODE_BLOCK,
            markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) == "medium"


# ── domain band edge cases ────────────────────────────────────────────────────


class TestDomainBand:
    def test_arxiv_url_resolves_to_arxiv_band(self) -> None:
        entry = _make_entry(
            source_type="url",
            source="https://arxiv.org/abs/2606.25996",
            markdown=LONG_TEXT + CODE_BLOCK,
            markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
        )
        assert SCORER.score_entry(entry) == "high"

    def test_unknown_url_defaults_to_news_band(self) -> None:
        entry = _make_entry(
            source_type="url",
            source="https://somesite.example.com/article",
            markdown=SHORT_TEXT,
            markdown_chars=len(SHORT_TEXT),
        )
        assert SCORER.score_entry(entry) == "low"


# ── code-presence detection ───────────────────────────────────────────────────


class TestCodePresent:
    def test_fenced_code_block_detected(self) -> None:
        entry = _make_entry(
            source_type="arxiv", markdown=LONG_TEXT + "\n```\nx\n```"
        )
        assert SCORER._code_present(entry) is True

    def test_math_block_detected(self) -> None:
        entry = _make_entry(
            source_type="arxiv", markdown=LONG_TEXT + "\n$$E=mc^2$$"
        )
        assert SCORER._code_present(entry) is True

    def test_no_code_block(self) -> None:
        entry = _make_entry(source_type="arxiv", markdown=LONG_TEXT)
        assert SCORER._code_present(entry) is False

    def test_short_inline_code_ignored(self) -> None:
        # inline code < 4 chars does not count
        entry = _make_entry(source_type="arxiv", markdown=LONG_TEXT + " `x` ")
        assert SCORER._code_present(entry) is False


# ── backward compatibility ────────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_entry_without_markdown_chars_field(self) -> None:
        """Entries without markdown_chars fall back to len(markdown)."""
        entry = {
            "source_type": "arxiv",
            "source": "1234.5678",
            "markdown": LONG_TEXT,
        }
        result = SCORER.score_entry(entry)
        assert result in {"high", "medium", "low"}

    def test_minimal_entry_only_source_type(self) -> None:
        """score_entry must not raise on a minimal entry with no markdown."""
        entry = {"source_type": "arxiv"}
        result = SCORER.score_entry(entry)
        assert result in {"high", "medium", "low"}

    def test_entry_without_grpo_field_does_not_raise(self) -> None:
        """Existing corpus items that lack grpo_suitability are not broken."""
        entry = {
            "id": "abc123",
            "source_type": "arxiv",
            "source": "2606.25996",
            "title": "Some Paper",
            # no grpo_suitability key — backward-compatible
        }
        result = SCORER.score_entry(entry)
        assert result in {"high", "medium", "low"}

    def test_full_context_fallback_when_no_markdown(self) -> None:
        """Scorer falls back to full_context if markdown is absent."""
        entry = {
            "source_type": "arxiv",
            "full_context": LONG_TEXT + CODE_BLOCK,
            "markdown_chars": len(LONG_TEXT) + len(CODE_BLOCK),
        }
        assert SCORER.score_entry(entry) == "high"


# ── idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    @pytest.mark.parametrize(
        "entry",
        [
            _make_entry(
                source_type="arxiv",
                markdown=LONG_TEXT + CODE_BLOCK,
                markdown_chars=len(LONG_TEXT) + len(CODE_BLOCK),
            ),
            _make_entry(
                source_type="url",
                source="https://reuters.com/x",
                markdown=SHORT_TEXT,
                markdown_chars=len(SHORT_TEXT),
            ),
            _make_entry(
                source_type="url",
                source="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=9",
                markdown=LONG_TEXT,
                markdown_chars=len(LONG_TEXT),
            ),
        ],
    )
    def test_same_entry_scored_twice_gives_same_result(
        self, entry: dict
    ) -> None:
        first = SCORER.score_entry(entry)
        second = SCORER.score_entry(entry)
        assert first == second
