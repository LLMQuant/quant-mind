import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from scripts import verify_pdf_rag_e2e


def _snapshot(**overrides):
    value = {
        "arxiv_id": "1706.03762v7",
        "page_count": 15,
        "text_page_count": 15,
        "asset_count": 16,
        "screenshot_pages": list(range(1, 16)),
        "chunk_asset_reference_count": 42,
        "chunk_count": 42,
        "summary": (
            "An attention-only encoder-decoder removes recurrence and "
            "convolution, uses multi-head attention, and improves translation "
            "with greater training efficiency."
        ),
        "summary_orchestration": "manager-research-agents-v1",
        "citation_count": 3,
        "citation_pages": [2, 4],
        "first_summary_scores": [0.9],
        "first_chunk_scores": [0.8],
        "second_summary_scores": [0.9],
        "second_chunk_scores": [0.8],
        "first_summary_resolved": True,
        "second_summary_resolved": True,
        "first_multi_head_pages": [5],
        "second_multi_head_pages": [5],
        "restored": True,
        "embedding_model": "text-embedding-3-small",
    }
    value.update(overrides)
    return value


class VerifyPaperFlowE2ETests(unittest.IsolatedAsyncioTestCase):
    def test_summary_coverage_accepts_exclusive_self_attention_wording(self):
        summary = (
            "The encoder-decoder relies exclusively on self-attention instead "
            "of recurrent and convolutional networks. Multi-head attention "
            "improves translation with less training time."
        )

        self.assertTrue(
            verify_pdf_rag_e2e._summary_has_required_coverage(summary)
        )

    def test_summary_coverage_accepts_pure_attention_wording(self):
        summary = (
            "The encoder-decoder relies purely on attention, eschewing "
            "recurrence and convolutions. It uses multi-head attention and "
            "improves translation with less training time."
        )

        self.assertTrue(
            verify_pdf_rag_e2e._summary_has_required_coverage(summary)
        )

    async def test_main_passes_complete_vertical_slice(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test"}),
            patch.object(
                verify_pdf_rag_e2e,
                "_run_vertical_slice",
                new=AsyncMock(return_value=_snapshot()),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] paper-flow-v1", output.getvalue())
        self.assertIn("multi_head_pages=[5]", output.getvalue())

    async def test_main_reports_upstream_failure(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test"}),
            patch.object(
                verify_pdf_rag_e2e,
                "_run_vertical_slice",
                new=AsyncMock(side_effect=TimeoutError("bounded timeout")),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] paper-flow-v1: TimeoutError", output.getvalue())

    async def test_main_rejects_missing_reopen_or_summary_coverage(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test"}),
            patch.object(
                verify_pdf_rag_e2e,
                "_run_vertical_slice",
                new=AsyncMock(
                    return_value=_snapshot(
                        restored=False,
                        summary="An unrelated summary.",
                    )
                ),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = await verify_pdf_rag_e2e.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] paper-flow-v1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
