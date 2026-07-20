import unittest
from datetime import datetime

from pydantic import ValidationError

import quantmind.library as library
from quantmind.knowledge import PaperArtifactKind
from quantmind.library import SemanticQuery


class PublicTypeTests(unittest.TestCase):
    def test_package_exports_only_domain_retrieval_types(self):
        self.assertEqual(
            set(library.__all__),
            {
                "LocalKnowledgeLibrary",
                "SearchProjection",
                "SemanticQuery",
                "SemanticHit",
            },
        )

    def test_query_rejects_blank_text(self):
        with self.assertRaises(ValidationError):
            SemanticQuery(text="  ")

    def test_query_rejects_naive_financial_cutoff(self):
        with self.assertRaisesRegex(ValidationError, "timezone-aware"):
            SemanticQuery(
                text="rates",
                available_at_before=datetime(2026, 7, 16),
            )

    def test_query_rejects_non_positive_top_k(self):
        with self.assertRaises(ValidationError):
            SemanticQuery(text="rates", top_k=0)

    def test_query_parses_typed_paper_artifact_kinds(self):
        query = SemanticQuery(
            text="attention",
            artifact_kinds=["paper_summary"],
        )

        self.assertEqual(
            query.artifact_kinds,
            [PaperArtifactKind.GLOBAL_SUMMARY],
        )
        with self.assertRaises(ValidationError):
            SemanticQuery(
                text="attention",
                artifact_kinds=["unknown_artifact"],
            )


if __name__ == "__main__":
    unittest.main()
