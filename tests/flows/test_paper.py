"""Tests for ``quantmind.flows.paper``."""

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agents import AgentOutputSchema, RunHooks

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    RawText,
)
from quantmind.flows._paper_draft import DraftPaper
from quantmind.flows.paper import (
    UnsupportedContentTypeError,
    _as_of,
    _compose_instructions,
    _fetch_and_format,
    _format_by_content_type,
    _format_input,
    _source_id,
    _source_ref,
    paper_flow,
)
from quantmind.knowledge import Paper
from quantmind.preprocess.fetch import Fetched, RawPaper


def _stub_draft() -> DraftPaper:
    return DraftPaper(title="root", summary="stub")


def _patch_runner(return_value: Any) -> Any:
    return patch(
        "quantmind.flows.paper.run_with_observability",
        new=AsyncMock(return_value=return_value),
    )


class FormatByContentTypeTests(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_dispatches_to_pdf_to_markdown(self) -> None:
        raw = Fetched(bytes=b"%PDF-x", content_type="application/pdf")
        with patch(
            "quantmind.flows.paper.pdf_to_markdown",
            new=AsyncMock(return_value="MD"),
        ) as pdf_mock:
            md = await _format_by_content_type(raw)
        pdf_mock.assert_awaited_once_with(b"%PDF-x")
        self.assertEqual(md, "MD")

    async def test_html_dispatches_to_html_to_markdown(self) -> None:
        raw = Fetched(
            bytes="<html>hi</html>".encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )
        with patch(
            "quantmind.flows.paper.html_to_markdown",
            new=AsyncMock(return_value="HTML-MD"),
        ) as html_mock:
            md = await _format_by_content_type(raw)
        html_mock.assert_awaited_once_with("<html>hi</html>")
        self.assertEqual(md, "HTML-MD")

    async def test_markdown_passes_through(self) -> None:
        raw = Fetched(
            bytes=b"# heading\n\nbody",
            content_type="text/markdown",
        )
        md = await _format_by_content_type(raw)
        self.assertEqual(md, "# heading\n\nbody")

    async def test_plain_text_passes_through(self) -> None:
        raw = Fetched(bytes=b"plain", content_type="text/plain")
        md = await _format_by_content_type(raw)
        self.assertEqual(md, "plain")

    async def test_unsupported_content_type_raises(self) -> None:
        raw = Fetched(bytes=b"\x00\x00", content_type="application/zip")
        with self.assertRaises(UnsupportedContentTypeError):
            await _format_by_content_type(raw)


class FetchAndFormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_arxiv_branch(self) -> None:
        raw_paper = RawPaper(
            bytes=b"%PDF",
            content_type="application/pdf",
            source_url="http://arxiv.org/pdf/2604.12345.pdf",
            arxiv_id="2604.12345",
            title="Momentum",
            authors=("Alice", "Bob"),
        )
        with (
            patch(
                "quantmind.flows.paper.fetch_arxiv",
                new=AsyncMock(return_value=raw_paper),
            ) as fetch_mock,
            patch(
                "quantmind.flows.paper.pdf_to_markdown",
                new=AsyncMock(return_value="MARKDOWN"),
            ) as fmt_mock,
        ):
            md, meta = await _fetch_and_format(ArxivIdentifier(id="2604.12345"))
        fetch_mock.assert_awaited_once_with("2604.12345")
        fmt_mock.assert_awaited_once_with(b"%PDF")
        self.assertEqual(md, "MARKDOWN")
        self.assertEqual(meta["source"], "arxiv")
        self.assertEqual(meta["arxiv_id"], "2604.12345")
        self.assertEqual(meta["title"], "Momentum")
        self.assertEqual(meta["authors"], ["Alice", "Bob"])

    async def test_http_pdf_branch(self) -> None:
        raw = Fetched(
            bytes=b"%PDF",
            content_type="application/pdf",
            source_url="http://example/x.pdf",
        )
        with (
            patch(
                "quantmind.flows.paper.fetch_url",
                new=AsyncMock(return_value=raw),
            ) as fetch_mock,
            patch(
                "quantmind.flows.paper.pdf_to_markdown",
                new=AsyncMock(return_value="PDFMD"),
            ),
        ):
            md, meta = await _fetch_and_format(
                HttpUrl(url="http://example/x.pdf")
            )
        fetch_mock.assert_awaited_once_with("http://example/x.pdf")
        self.assertEqual(md, "PDFMD")
        self.assertEqual(meta["source"], "web")
        self.assertEqual(meta["content_type"], "application/pdf")

    async def test_local_file_branch(self) -> None:
        raw = Fetched(
            bytes=b"## body",
            content_type="text/markdown",
            source_url="file:///tmp/p.md",
        )
        with patch(
            "quantmind.flows.paper.read_local_file",
            new=AsyncMock(return_value=raw),
        ) as read_mock:
            md, meta = await _fetch_and_format(
                LocalFilePath(path=Path("/tmp/p.md"))
            )
        read_mock.assert_awaited_once_with(Path("/tmp/p.md"))
        self.assertEqual(md, "## body")
        self.assertEqual(meta["source"], "local")
        self.assertEqual(meta["path"], "/tmp/p.md")
        self.assertEqual(meta["content_type"], "text/markdown")

    async def test_raw_text_branch(self) -> None:
        md, meta = await _fetch_and_format(RawText(text="hello"))
        self.assertEqual(md, "hello")
        self.assertEqual(meta, {"source": "inline"})

    async def test_doi_branch_raises_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError) as ctx:
            await _fetch_and_format(DoiIdentifier(doi="10.1234/abcd"))
        self.assertIn("DOI", str(ctx.exception))


class ComposeInstructionsTests(unittest.TestCase):
    def test_default_renders_cfg_flags(self) -> None:
        cfg = PaperFlowCfg(
            extract_methodology=False,
            extract_limitations=True,
            asset_class_hint="equities",
        )
        out = _compose_instructions(
            "go {extract_methodology} {extract_limitations} "
            "{asset_class_hint!r}",
            None,
            cfg,
        )
        self.assertEqual(out, "go False True 'equities'")

    def test_extra_appended(self) -> None:
        cfg = PaperFlowCfg()
        out = _compose_instructions("BASE", "USER", cfg)
        self.assertIn("BASE", out)
        self.assertIn("Additional instructions:", out)
        self.assertIn("USER", out)


class FormatInputTests(unittest.TestCase):
    def test_tuple_authors_join_as_csv(self) -> None:
        out = _format_input(
            "BODY",
            {"authors": ("Alice", "Bob"), "title": "x"},
        )
        self.assertIn("authors: Alice, Bob", out)
        self.assertIn("title: x", out)
        self.assertIn("BODY", out)

    def test_none_values_skipped(self) -> None:
        out = _format_input("BODY", {"a": "1", "b": None})
        self.assertIn("a: 1", out)
        self.assertNotIn("b:", out)


class ProvenanceHelperTests(unittest.TestCase):
    def test_source_ref_web_maps_to_http(self) -> None:
        ref = _source_ref({"source": "web", "url": "https://x/y"})
        self.assertEqual(ref.kind, "http")
        self.assertEqual(ref.uri, "https://x/y")

    def test_source_ref_local(self) -> None:
        ref = _source_ref({"source": "local", "path": "/tmp/p.pdf"})
        self.assertEqual(ref.kind, "local")
        self.assertEqual(ref.uri, "/tmp/p.pdf")

    def test_source_ref_inline_fallback(self) -> None:
        ref = _source_ref({"source": "inline"})
        self.assertEqual(ref.kind, "manual")
        self.assertIsNone(ref.uri)

    def test_source_id_local_and_web(self) -> None:
        self.assertEqual(
            _source_id({"source": "local", "path": "/tmp/p.pdf"}), "/tmp/p.pdf"
        )
        self.assertEqual(
            _source_id({"source": "web", "url": "https://x"}), "https://x"
        )

    def test_as_of_uses_published_date(self) -> None:
        draft = DraftPaper(
            title="T", summary="s", published_date=date(2023, 12, 7)
        )
        self.assertEqual(
            _as_of(draft), datetime(2023, 12, 7, tzinfo=timezone.utc)
        )


class PaperFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_arxiv(self) -> None:
        raw_paper = RawPaper(
            bytes=b"%PDF",
            content_type="application/pdf",
            arxiv_id="2604.12345",
        )
        stub = _stub_draft()
        with (
            patch(
                "quantmind.flows.paper.fetch_arxiv",
                new=AsyncMock(return_value=raw_paper),
            ),
            patch(
                "quantmind.flows.paper.pdf_to_markdown",
                new=AsyncMock(return_value="MD"),
            ),
            _patch_runner(stub) as runner,
        ):
            out = await paper_flow(ArxivIdentifier(id="2604.12345"))
        self.assertIsInstance(out, Paper)
        self.assertEqual(out.root().title, "root")
        runner.assert_awaited_once()

    async def test_extra_instructions_passed_to_agent(self) -> None:
        seen: dict[str, Any] = {}

        def _capture_agent(*_a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return MagicMock(name="agent", _name="paper_extractor")

        stub = _stub_draft()
        with (
            patch("quantmind.flows.paper.Agent", side_effect=_capture_agent),
            _patch_runner(stub),
        ):
            await paper_flow(
                RawText(text="hello"),
                extra_instructions="EXTRA-USER-DIRECTIVE",
            )
        self.assertIn("EXTRA-USER-DIRECTIVE", seen["instructions"])
        self.assertIn("nested draft tree", seen["instructions"])

    async def test_output_type_override_propagated(self) -> None:
        class MyPaper(Paper):
            pass

        with (
            patch("quantmind.flows.paper.Agent", return_value=MagicMock()),
            _patch_runner(_stub_draft()),
        ):
            out = await paper_flow(RawText(text="x"), output_type=MyPaper)
        self.assertIsInstance(out, MyPaper)

    async def test_extra_tools_and_guardrails_forwarded(self) -> None:
        seen: dict[str, Any] = {}

        def _capture_agent(*_a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return MagicMock()

        sentinel_tool = MagicMock(name="tool")
        in_g = MagicMock()
        out_g = MagicMock()
        with (
            patch("quantmind.flows.paper.Agent", side_effect=_capture_agent),
            _patch_runner(_stub_draft()),
        ):
            await paper_flow(
                RawText(text="x"),
                extra_tools=[sentinel_tool],
                extra_input_guardrails=[in_g],
                extra_output_guardrails=[out_g],
            )
        self.assertEqual(seen["tools"], [sentinel_tool])
        self.assertEqual(seen["input_guardrails"], [in_g])
        self.assertEqual(seen["output_guardrails"], [out_g])

    async def test_memory_accepted_as_no_op(self) -> None:
        with (
            patch(
                "quantmind.flows.paper.Agent",
                return_value=MagicMock(),
            ),
            _patch_runner(_stub_draft()) as runner,
        ):
            await paper_flow(RawText(text="x"), memory=object())
        # The runner sees the memory placeholder forwarded.
        self.assertIsNotNone(runner.await_args.kwargs["memory"])

    async def test_extra_run_hooks_forwarded(self) -> None:
        class _H(RunHooks[Any]):
            pass

        hook = _H()
        with (
            patch(
                "quantmind.flows.paper.Agent",
                return_value=MagicMock(),
            ),
            _patch_runner(_stub_draft()) as runner,
        ):
            await paper_flow(RawText(text="x"), extra_run_hooks=[hook])
        self.assertEqual(runner.await_args.kwargs["extra_run_hooks"], [hook])

    async def test_model_settings_forwarded_when_set(self) -> None:
        seen: dict[str, Any] = {}

        def _capture_agent(*_a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return MagicMock()

        from agents import ModelSettings

        ms = ModelSettings(temperature=0.42)
        cfg = PaperFlowCfg(model_settings=ms)
        with (
            patch("quantmind.flows.paper.Agent", side_effect=_capture_agent),
            _patch_runner(_stub_draft()),
        ):
            await paper_flow(RawText(text="x"), cfg=cfg)
        self.assertIs(seen["model_settings"], ms)

    async def test_model_settings_omitted_when_none(self) -> None:
        seen: dict[str, Any] = {}

        def _capture_agent(*_a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return MagicMock()

        with (
            patch("quantmind.flows.paper.Agent", side_effect=_capture_agent),
            _patch_runner(_stub_draft()),
        ):
            await paper_flow(RawText(text="x"))
        self.assertNotIn("model_settings", seen)

    async def test_agent_output_type_is_nonstrict_draftpaper(self) -> None:
        seen: dict[str, Any] = {}

        def _capture_agent(*_a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return MagicMock()

        with (
            patch("quantmind.flows.paper.Agent", side_effect=_capture_agent),
            _patch_runner(_stub_draft()),
        ):
            await paper_flow(RawText(text="x"))
        schema = seen["output_type"]
        self.assertIsInstance(schema, AgentOutputSchema)
        self.assertIs(schema.output_type, DraftPaper)
        self.assertFalse(schema.is_strict_json_schema())
