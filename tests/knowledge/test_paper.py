"""Tests for source-first paper revisions and independent artifacts."""

import unittest
from uuid import uuid4

from pydantic import ValidationError

from quantmind.knowledge import (
    PaperCitation,
    PaperSemanticResult,
    PaperSourceRevision,
    PaperSourceSpan,
)
from quantmind.knowledge.paper import (
    _paper_chunk_id,
    _paper_chunk_set_content_hash,
    _paper_summary_content_hash,
)
from tests.paper_helpers import build_paper_result


class PaperArtifactTests(unittest.TestCase):
    def test_result_has_one_shared_source_and_independent_artifacts(
        self,
    ) -> None:
        result = build_paper_result()

        self.assertEqual(
            result.chunk_set.source_revision_id,
            result.source_revision.id,
        )
        self.assertEqual(
            result.global_summary.source_revision_id,
            result.source_revision.id,
        )
        self.assertNotEqual(result.chunk_set.id, result.global_summary.id)
        self.assertEqual(len(result.chunk_set.chunks), 3)
        self.assertEqual(len(result.global_summary.citations), 3)

    def test_unchanged_producer_configs_create_stable_ids(self) -> None:
        first = build_paper_result()
        second = build_paper_result()

        self.assertEqual(first.source_revision.id, second.source_revision.id)
        self.assertEqual(first.chunk_set.id, second.chunk_set.id)
        self.assertEqual(first.global_summary.id, second.global_summary.id)
        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunk_set.chunks],
            [chunk.chunk_id for chunk in second.chunk_set.chunks],
        )

    def test_splitter_and_summary_configs_version_independently(self) -> None:
        original = build_paper_result()
        new_chunks = build_paper_result(chunk_size=256)
        new_summary = build_paper_result(summary_model="another-model")

        self.assertEqual(
            original.source_revision.id, new_chunks.source_revision.id
        )
        self.assertNotEqual(original.chunk_set.id, new_chunks.chunk_set.id)
        self.assertNotEqual(
            original.global_summary.id, new_chunks.global_summary.id
        )
        self.assertEqual(original.chunk_set.id, new_summary.chunk_set.id)
        self.assertNotEqual(
            original.global_summary.id, new_summary.global_summary.id
        )

    def test_canonical_models_have_no_embedding_vectors_or_text_method(
        self,
    ) -> None:
        result = build_paper_result()

        for value in (
            result.source_revision,
            result.chunk_set,
            result.global_summary,
            *result.chunk_set.chunks,
        ):
            self.assertNotIn("embedding", value.model_dump())
            self.assertFalse(hasattr(value, "embedding_text"))

    def test_source_json_excludes_blobs_but_preserves_manifest(self) -> None:
        source = build_paper_result().source_revision
        revived = PaperSourceRevision.model_validate_json(
            source.model_dump_json()
        )

        self.assertEqual(revived.id, source.id)
        self.assertEqual(revived.parsed, source.parsed)
        self.assertEqual(revived.blobs, {})
        with self.assertRaisesRegex(RuntimeError, "bytes are not loaded"):
            revived.blob_for(revived.raw_asset_id)

    def test_result_rejects_unknown_model_owned_citation_identity(self) -> None:
        result = build_paper_result()
        citation = result.global_summary.citations[0]
        invalid = PaperCitation(
            chunk_set_id=result.chunk_set.id,
            chunk_id=uuid4(),
            page_number=citation.page_number,
        )
        invalid_citations = (
            invalid,
            *result.global_summary.citations[1:],
        )
        summary = result.global_summary.model_copy(
            update={
                "citations": invalid_citations,
                "content_hash": _paper_summary_content_hash(
                    result.global_summary.summary,
                    invalid_citations,
                ),
            }
        )

        with self.assertRaisesRegex(ValidationError, "unknown chunk"):
            PaperSemanticResult(
                source_revision=result.source_revision,
                chunk_set=result.chunk_set,
                global_summary=summary,
            )

    def test_source_revision_rejects_content_owned_id_override(self) -> None:
        source = build_paper_result().source_revision
        payload = source.model_dump()
        payload["id"] = uuid4()
        payload["blobs"] = source.blobs

        with self.assertRaisesRegex(ValidationError, "ID does not match"):
            PaperSourceRevision.model_validate(payload)

    def test_result_rejects_chunk_spans_outside_source_manifest(self) -> None:
        result = build_paper_result()
        chunk = result.chunk_set.chunks[0]
        invalid_span = PaperSourceSpan(
            page_number=1,
            start_char=0,
            end_char=len(result.source_revision.parsed.pages[0].text) + 1,
        )
        invalid_chunk = chunk.model_copy(
            update={
                "chunk_id": _paper_chunk_id(
                    result.chunk_set.id,
                    position=chunk.position,
                    content_hash=chunk.content_hash,
                    spans=(invalid_span,),
                ),
                "source_spans": (invalid_span,),
            }
        )
        invalid_chunks = (
            invalid_chunk,
            *result.chunk_set.chunks[1:],
        )
        invalid_chunk_set = result.chunk_set.model_copy(
            update={
                "chunks": invalid_chunks,
                "content_hash": _paper_chunk_set_content_hash(invalid_chunks),
            }
        )

        with self.assertRaisesRegex(ValidationError, "exceeds its source page"):
            PaperSemanticResult(
                source_revision=result.source_revision,
                chunk_set=invalid_chunk_set,
                global_summary=result.global_summary,
            )


if __name__ == "__main__":
    unittest.main()
