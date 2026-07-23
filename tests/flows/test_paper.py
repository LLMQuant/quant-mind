"""Offline tests for source-first ``paper_flow`` behavior."""

import asyncio
import json
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
    PaperResearchCitationDraft,
    PaperResearchDraft,
    PaperResearchFindingDraft,
    PaperSummaryCitationDraft,
    PaperSummaryDraft,
    PaperSummaryError,
    _AgentsPaperSummaryProvider,
    _chunk_groups,
    _ChunkGroup,
    _summary_model_settings,
    _validate_research_draft,
)
from quantmind.flows.paper import (
    PaperFlow,
    UnsupportedContentTypeError,
    _build_summary,
    _fetch_paper_source,
    paper_flow,
)
from quantmind.knowledge import PaperCitationValidationError
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

        result = await PaperFlow(
            cfg,
            _summary_provider=provider,
        ).build(LocalFilePath(path=_FIXTURE))

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
            result = await PaperFlow(
                PaperFlowCfg(chunk_size=256, chunk_overlap=32),
                _summary_provider=_FakeSummaryProvider(),
            ).build(ArxivIdentifier(id="1706.03762v7"))

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
            await PaperFlow(
                PaperFlowCfg(chunk_size=256, chunk_overlap=32),
                _summary_provider=provider,
            ).build(LocalFilePath(path=_FIXTURE))

    async def test_same_pdf_and_configs_have_idempotent_ids(self) -> None:
        cfg = PaperFlowCfg(chunk_size=256, chunk_overlap=32)
        first = await PaperFlow(
            cfg,
            _summary_provider=_FakeSummaryProvider(),
        ).build(LocalFilePath(path=_FIXTURE))
        second = await PaperFlow(
            cfg,
            _summary_provider=_FakeSummaryProvider(),
        ).build(LocalFilePath(path=_FIXTURE))

        self.assertEqual(first.source_revision.id, second.source_revision.id)
        self.assertEqual(first.chunk_set.id, second.chunk_set.id)
        self.assertEqual(first.global_summary.id, second.global_summary.id)
        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunk_set.chunks],
            [chunk.chunk_id for chunk in second.chunk_set.chunks],
        )


class PaperFlowDeprecationTests(unittest.IsolatedAsyncioTestCase):
    async def test_paper_flow_warns_and_delegates(self) -> None:
        # The grandfathered ``paper_flow`` wrapper must emit a
        # DeprecationWarning and delegate to ``PaperFlow.build`` for the same
        # source-first result, forwarding the summary-provider seam.
        provider = _FakeSummaryProvider()

        with self.assertWarns(DeprecationWarning):
            result = await paper_flow(
                LocalFilePath(path=_FIXTURE),
                cfg=PaperFlowCfg(chunk_size=256, chunk_overlap=32),
                _summary_provider=provider,
            )

        self.assertEqual(len(provider.calls), 1)
        _, chunk_set_seen, _ = provider.calls[0]
        self.assertIs(chunk_set_seen, result.chunk_set)
        self.assertIs(result.source_revision, provider.calls[0][0])
        self.assertTrue(result.global_summary.summary)
        self.assertEqual(len(result.source_revision.parsed.pages), 4)

    async def test_paper_flow_defaults_cfg_when_omitted(self) -> None:
        # Omitting cfg still delegates under a default ``PaperFlowCfg``.
        provider = _FakeSummaryProvider()

        with self.assertWarns(DeprecationWarning):
            result = await paper_flow(
                LocalFilePath(path=_FIXTURE),
                _summary_provider=provider,
            )

        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(result.chunk_set.chunks)


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
                    _build_summary(
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
            _build_summary(
                result.chunk_set,
                draft,
                PaperFlowCfg(),
            )


def _fake_research_run(cfg, source_pages):
    """A run_with_observability stub: real worker drafts, one reducer draft."""

    async def fake_run(agent, payload, *, cfg, memory, extra_run_hooks):
        data = json.loads(payload)
        if agent.name == "paper_chunk_researcher":
            return PaperResearchDraft(
                scope_summary="reviewed the assigned chunk group",
                findings=tuple(
                    PaperResearchFindingDraft(
                        kind="method",
                        claim=f"finding for chunk {chunk['chunk_index']}",
                        citation=PaperResearchCitationDraft(
                            chunk_index=chunk["chunk_index"],
                            page_number=chunk["pages"][0],
                        ),
                    )
                    for chunk in data["chunks"]
                ),
            )
        return PaperSummaryDraft(
            summary="synthesized global summary across physical pages",
            citations=(
                PaperSummaryCitationDraft(chunk_index=0, page_number=1),
            ),
        )

    return fake_run


class SummaryMapReduceTests(unittest.IsolatedAsyncioTestCase):
    def test_chunk_groups_tile_every_chunk_exactly_once(self) -> None:
        groups = _chunk_groups(7, 3)
        self.assertEqual(
            [(group.start, group.count) for group in groups],
            [(0, 3), (3, 3), (6, 1)],
        )
        covered = [
            index
            for group in groups
            for index in range(group.start, group.start + group.count)
        ]
        self.assertEqual(covered, list(range(7)))

    def test_research_finding_outside_its_group_is_rejected(self) -> None:
        result = build_paper_result()
        draft = PaperResearchDraft(
            scope_summary="reviewed the first chunk only",
            findings=(
                PaperResearchFindingDraft(
                    kind="method",
                    claim="a finding pointing outside the assigned group",
                    citation=PaperResearchCitationDraft(
                        chunk_index=2,
                        page_number=2,
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "outside its group"):
            _validate_research_draft(
                result.chunk_set,
                _ChunkGroup(start=0, count=1),
                draft,
            )

    def test_worker_and_reducer_output_is_capped(self) -> None:
        capped = _summary_model_settings(
            PaperFlowCfg(
                max_summary_output_tokens=256,
                model_settings=ModelSettings(max_tokens=1024),
            )
        )
        self.assertEqual(capped.max_tokens, 256)
        lower = _summary_model_settings(
            PaperFlowCfg(
                max_summary_output_tokens=256,
                model_settings=ModelSettings(max_tokens=128),
            )
        )
        self.assertEqual(lower.max_tokens, 128)

    async def test_map_reduce_fans_out_one_worker_per_group(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(
            summary_research_group_size=1,
            summary_concurrency=2,
            min_summary_citations=1,
            min_summary_pages=1,
        )
        run_mock = AsyncMock(
            side_effect=_fake_research_run(
                cfg, result.source_revision.parsed.pages
            )
        )
        with patch(
            "quantmind.flows._paper_summary.run_with_observability",
            new=run_mock,
        ):
            draft = await _AgentsPaperSummaryProvider().summarize(
                result.source_revision,
                result.chunk_set,
                cfg=cfg,
            )

        self.assertIsInstance(draft, PaperSummaryDraft)
        worker_calls = [
            call.args[0].name
            for call in run_mock.await_args_list
            if call.args[0].name == "paper_chunk_researcher"
        ]
        self.assertEqual(len(worker_calls), len(result.chunk_set.chunks))
        self.assertEqual(run_mock.await_count, len(result.chunk_set.chunks) + 1)

    async def test_summary_timeout_raises(self) -> None:
        result = build_paper_result()
        cfg = PaperFlowCfg(
            summary_research_group_size=8,
            timeout_seconds=0.01,
            min_summary_citations=1,
            min_summary_pages=1,
        )

        async def fake_run(agent, payload, *, cfg, memory, extra_run_hooks):
            data = json.loads(payload)
            if agent.name == "paper_chunk_researcher":
                return PaperResearchDraft(
                    scope_summary="reviewed the assigned chunk group",
                    findings=tuple(
                        PaperResearchFindingDraft(
                            kind="method",
                            claim=f"finding for chunk {chunk['chunk_index']}",
                            citation=PaperResearchCitationDraft(
                                chunk_index=chunk["chunk_index"],
                                page_number=chunk["pages"][0],
                            ),
                        )
                        for chunk in data["chunks"]
                    ),
                )
            await asyncio.sleep(10)

        with patch(
            "quantmind.flows._paper_summary.run_with_observability",
            new=AsyncMock(side_effect=fake_run),
        ):
            with self.assertRaises(PaperSummaryError):
                await _AgentsPaperSummaryProvider().summarize(
                    result.source_revision,
                    result.chunk_set,
                    cfg=cfg,
                )


if __name__ == "__main__":
    unittest.main()
