"""Offline tests for source-first ``paper_flow`` behavior."""

import asyncio
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agents import ModelSettings

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    RawText,
)
from quantmind.flows._paper_summary import (
    PaperSummaryBudgetExceeded,
    PaperSummaryCitationDraft,
    PaperSummaryDraft,
    _bounded_model_settings,
    _SummaryBudget,
)
from quantmind.flows.paper import (
    PaperCitationValidationError,
    UnsupportedContentTypeError,
    _build_global_summary,
    _fetch_paper_source,
    paper_flow,
)
from quantmind.preprocess.fetch import Fetched, RawPaper
from tests.paper_helpers import build_paper_result

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "paper"
    / "golden"
    / "paper.pdf"
)
_WHEN = datetime(2017, 12, 6, tzinfo=timezone.utc)


class _FakeSummaryProvider:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls = []

    async def summarize(self, source, chunk_set, *, cfg):
        self.calls.append((source, chunk_set, cfg))
        if self.fail is not None:
            raise self.fail
        selected = []
        pages: set[int] = set()
        for chunk in chunk_set.chunks:
            page = chunk.source_spans[0].page_number
            if page not in pages or len(selected) < cfg.min_summary_citations:
                selected.append((chunk, page))
                pages.add(page)
            if (
                len(selected) >= cfg.min_summary_citations
                and len(pages) >= cfg.min_summary_pages
            ):
                break
        return PaperSummaryDraft(
            summary=(
                "The paper studies a source-first test methodology and reports "
                "deterministic evidence across physical pages."
            ),
            citations=tuple(
                PaperSummaryCitationDraft(
                    chunk_index=chunk.position,
                    page_number=page,
                )
                for chunk, page in selected
            ),
        )


class PaperFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_pdf_builds_chunks_before_summary(self) -> None:
        provider = _FakeSummaryProvider()
        cfg = PaperFlowCfg(chunk_size=256, chunk_overlap=32)

        result = await paper_flow(
            LocalFilePath(path=_FIXTURE),
            cfg=cfg,
            _summary_provider=provider,
        )

        self.assertEqual(len(provider.calls), 1)
        source_seen, chunk_set_seen, _ = provider.calls[0]
        self.assertIs(source_seen, result.source_revision)
        self.assertIs(chunk_set_seen, result.chunk_set)
        self.assertEqual(len(result.source_revision.parsed.pages), 4)
        self.assertTrue(result.chunk_set.chunks)
        self.assertTrue(result.global_summary.summary)
        self.assertGreaterEqual(len(result.global_summary.citations), 3)
        self.assertGreaterEqual(
            len(
                {
                    citation.page_number
                    for citation in result.global_summary.citations
                }
            ),
            2,
        )
        self.assertTrue(
            all(
                chunk.source_revision_id == result.source_revision.id
                and chunk.source_spans
                for chunk in result.chunk_set.chunks
            )
        )

    async def test_exact_arxiv_source_facts_are_code_owned(self) -> None:
        raw = RawPaper(
            bytes=_FIXTURE.read_bytes(),
            content_type="application/pdf",
            source_url="https://arxiv.org/pdf/1706.03762v7.pdf",
            resolved_url="https://arxiv.org/pdf/1706.03762v7.pdf",
            fetched_at=_WHEN,
            arxiv_id="1706.03762v7",
            title="Attention Is All You Need",
            authors=("Ashish Vaswani",),
            published_at=_WHEN,
            updated_at=_WHEN,
        )
        with patch(
            "quantmind.flows.paper.fetch_arxiv",
            new=AsyncMock(return_value=raw),
        ):
            result = await paper_flow(
                ArxivIdentifier(id="1706.03762v7"),
                cfg=PaperFlowCfg(chunk_size=256, chunk_overlap=32),
                _summary_provider=_FakeSummaryProvider(),
            )

        self.assertEqual(result.source_revision.arxiv_id, "1706.03762v7")
        self.assertEqual(result.source_revision.source.kind, "arxiv")
        self.assertEqual(
            result.source_revision.title, "Attention Is All You Need"
        )
        self.assertEqual(
            result.source_revision.source.content_hash,
            result.source_revision.parsed.source_hash,
        )
        self.assertFalse(hasattr(result.global_summary, "root_node_id"))

    async def test_summary_failure_prevents_flow_success(self) -> None:
        provider = _FakeSummaryProvider(fail=RuntimeError("summary failed"))

        with self.assertRaisesRegex(RuntimeError, "summary failed"):
            await paper_flow(
                LocalFilePath(path=_FIXTURE),
                cfg=PaperFlowCfg(chunk_size=256, chunk_overlap=32),
                _summary_provider=provider,
            )

    async def test_same_pdf_and_configs_have_idempotent_ids(self) -> None:
        cfg = PaperFlowCfg(chunk_size=256, chunk_overlap=32)
        first = await paper_flow(
            LocalFilePath(path=_FIXTURE),
            cfg=cfg,
            _summary_provider=_FakeSummaryProvider(),
        )
        second = await paper_flow(
            LocalFilePath(path=_FIXTURE),
            cfg=cfg,
            _summary_provider=_FakeSummaryProvider(),
        )

        self.assertEqual(first.source_revision.id, second.source_revision.id)
        self.assertEqual(first.chunk_set.id, second.chunk_set.id)
        self.assertEqual(first.global_summary.id, second.global_summary.id)
        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunk_set.chunks],
            [chunk.chunk_id for chunk in second.chunk_set.chunks],
        )


class SourceDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_pdf_preserves_resolved_url_and_fetch_time(self) -> None:
        raw = Fetched(
            bytes=b"%PDF",
            content_type="application/pdf",
            source_url="https://example.test/input.pdf",
            resolved_url="https://cdn.example.test/exact.pdf",
            fetched_at=_WHEN,
        )
        with patch(
            "quantmind.flows.paper.fetch_url",
            new=AsyncMock(return_value=raw),
        ):
            fetched = await _fetch_paper_source(
                HttpUrl(url="https://example.test/input.pdf")
            )

        self.assertEqual(fetched.uri, "https://cdn.example.test/exact.pdf")
        self.assertEqual(fetched.fetched_at, _WHEN)

    async def test_non_pdf_and_raw_text_are_rejected_before_summary(
        self,
    ) -> None:
        html = Fetched(bytes=b"<html/>", content_type="text/html")
        with patch(
            "quantmind.flows.paper.fetch_url",
            new=AsyncMock(return_value=html),
        ):
            with self.assertRaises(UnsupportedContentTypeError):
                await _fetch_paper_source(HttpUrl(url="https://example.test"))
        with self.assertRaises(UnsupportedContentTypeError):
            await _fetch_paper_source(RawText(text="no physical pages"))

    async def test_doi_requires_an_exact_pdf_resolver(self) -> None:
        with self.assertRaises(NotImplementedError):
            await _fetch_paper_source(DoiIdentifier(doi="10.1000/test"))


class CitationValidationTests(unittest.TestCase):
    def test_unknown_chunk_page_and_quote_are_rejected(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(
            min_summary_citations=1,
            min_summary_pages=1,
        )
        invalid_drafts = (
            PaperSummaryDraft(
                summary="x",
                citations=(
                    PaperSummaryCitationDraft(
                        chunk_index=99,
                        page_number=1,
                    ),
                ),
            ),
            PaperSummaryDraft(
                summary="x",
                citations=(
                    PaperSummaryCitationDraft(
                        chunk_index=0,
                        page_number=2,
                    ),
                ),
            ),
            PaperSummaryDraft(
                summary="x",
                citations=(
                    PaperSummaryCitationDraft(
                        chunk_index=0,
                        page_number=1,
                        quote="not in source chunk",
                    ),
                ),
            ),
        )

        for draft in invalid_drafts:
            with self.subTest(draft=draft):
                with self.assertRaises(PaperCitationValidationError):
                    _build_global_summary(
                        result.source_revision,
                        result.chunk_set,
                        draft,
                        cfg,
                    )

    def test_configured_citation_and_page_coverage_is_enforced(self) -> None:
        result = build_paper_result()
        draft = PaperSummaryDraft(
            summary="x",
            citations=(
                PaperSummaryCitationDraft(chunk_index=0, page_number=1),
            ),
        )

        with self.assertRaisesRegex(
            PaperCitationValidationError,
            "fewer citations",
        ):
            _build_global_summary(
                result.source_revision,
                result.chunk_set,
                draft,
                PaperFlowCfg(),
            )


class SummaryBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_and_input_budgets_are_enforced(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(
            max_summary_tool_calls=1,
            max_summary_input_tokens=10_000,
        )
        budget = _SummaryBudget(cfg, "manifest")

        await budget.read(result.chunk_set, start=0, count=1)
        with self.assertRaisesRegex(
            PaperSummaryBudgetExceeded,
            "tool_calls",
        ):
            await budget.read(result.chunk_set, start=1, count=1)

    async def test_concurrent_reads_remain_bounded_and_valid(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(
            max_summary_tool_calls=3,
            max_summary_concurrency=1,
            max_summary_input_tokens=10_000,
        )
        budget = _SummaryBudget(cfg, "manifest")

        values = await asyncio.gather(
            budget.read(result.chunk_set, start=0, count=1),
            budget.read(result.chunk_set, start=1, count=1),
            budget.read(result.chunk_set, start=2, count=1),
        )
        self.assertEqual(len(values), 3)

    def test_model_output_tokens_are_capped(self) -> None:
        cfg = PaperFlowCfg(
            max_summary_output_tokens=256,
            model_settings=ModelSettings(max_tokens=1024),
        )

        settings = _bounded_model_settings(cfg)

        self.assertEqual(settings.max_tokens, 256)
        self.assertTrue(settings.parallel_tool_calls)


if __name__ == "__main__":
    unittest.main()
