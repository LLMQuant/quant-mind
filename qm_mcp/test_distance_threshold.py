"""Tests for VECTOR_DISTANCE_THRESHOLD filtering in qm_mcp.query."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qm_mcp.query import query

_DUMMY_VEC = np.zeros(384, dtype=np.float32)

_DUMMY_RECORD = {
    "title": "Test Paper",
    "source_type": "arxiv",
    "source": "1234.5678",
    "authors": ["Author A"],
    "summary": "A test summary.",
}


def _make_store(hits: list[tuple[str, float]]) -> MagicMock:
    """Mock CorpusStore returning the given (id, cosine-similarity) pairs."""
    store = MagicMock()
    store.__len__ = MagicMock(return_value=len(hits))
    store.search = MagicMock(return_value=hits)
    store.get = MagicMock(return_value=_DUMMY_RECORD)
    return store


@pytest.fixture(autouse=True)
def _patch_io():
    """Suppress network/disk I/O for all tests in this module."""
    with (
        patch("qm_mcp.query.embed_text", return_value=_DUMMY_VEC),
        patch(
            "qm_mcp.query.synthesize_answer", return_value="Synthesized answer."
        ),
        patch("qm_mcp.query.load_secrets"),
    ):
        yield


class TestHighQualityMatch:
    """High-quality match (cosine distance < 0.7) is returned."""

    async def test_close_match_passes_default_threshold(self) -> None:
        # score=0.85 → distance=0.15 < 0.7 → kept
        store = _make_store([("abc1", 0.85)])
        result = await query("VWAP strategy", store=store)
        assert len(result["sources"]) == 1
        assert result["sources"][0]["score"] == 0.85

    async def test_score_just_above_min_boundary_passes(self) -> None:
        # score=0.31 → distance=0.69 < 0.7 → kept
        # (0.30 == 1.0-0.7 fails float equality; use 0.31 to avoid fp edge)
        store = _make_store([("edge1", 0.31)])
        result = await query("Kelly criterion", store=store)
        assert len(result["sources"]) == 1


class TestPoorMatchFiltered:
    """Poor match (cosine distance > 0.7) is filtered out."""

    async def test_distant_match_removed(self) -> None:
        # score=0.25 → distance=0.75 > 0.7 → filtered
        store = _make_store([("xyz9", 0.25)])
        result = await query("Kelly criterion", store=store)
        assert result["sources"] == []

    async def test_multiple_poor_matches_all_filtered(self) -> None:
        store = _make_store([("a1", 0.10), ("a2", 0.20)])
        result = await query("Stoikov γ parameter", store=store)
        assert result["sources"] == []


class TestAllCandidatesFiltered:
    """All candidates below threshold → empty result + log, not noise."""

    async def test_returns_empty_sources(self) -> None:
        store = _make_store([("a1", 0.05), ("a2", 0.15)])
        result = await query("obscure topic", store=store)
        assert result["sources"] == []
        assert result["answer"] is None
        assert result["question"] == "obscure topic"

    async def test_logs_no_candidates_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _make_store([("a1", 0.10)])
        with caplog.at_level(logging.INFO, logger="qm_mcp.query"):
            await query("gamma parameter Stoikov", store=store)
        assert "no candidates above threshold" in caplog.text


class TestThresholdOverride:
    """Custom threshold value is respected."""

    async def test_strict_threshold_filters_borderline(self) -> None:
        # threshold=0.3 → min_score=0.7; score=0.80 passes, score=0.60 filtered
        store = _make_store([("good", 0.80), ("borderline", 0.60)])
        result = await query(
            "Lyapunov stability", store=store, distance_threshold=0.3
        )
        assert len(result["sources"]) == 1
        assert result["sources"][0]["id"] == "good"

    async def test_lenient_threshold_keeps_poor_match(self) -> None:
        # threshold=0.9 → min_score=0.1; score=0.15 passes
        store = _make_store([("marginal", 0.15)])
        result = await query("noisy query", store=store, distance_threshold=0.9)
        assert len(result["sources"]) == 1
