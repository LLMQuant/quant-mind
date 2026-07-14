"""Tests for generic news collection result contracts."""

import unittest
from dataclasses import FrozenInstanceError

from quantmind.preprocess import NewsArtifact, NewsBatch, NewsFailure


class NewsResultContractTests(unittest.TestCase):
    def test_batch_counts_documents_and_failures(self) -> None:
        failure = NewsFailure(
            source="pr-newswire",
            stage="article_fetch",
            source_url="https://example.test/release",
            item_id="release-1",
            error_type="timeout",
            message="request timed out",
        )
        batch = NewsBatch(failures=(failure,), observed_count=1, complete=True)

        self.assertEqual(batch.success_count, 0)
        self.assertEqual(batch.failure_count, 1)
        self.assertEqual(batch.observed_count, 1)
        self.assertTrue(batch.complete)

    def test_artifact_can_retain_hash_without_bytes(self) -> None:
        artifact = NewsArtifact(
            bytes=None,
            content_hash="abc123",
            content_type="text/html",
            source_url="https://example.test/release",
            resolved_url="https://example.test/release",
            status_code=200,
        )

        self.assertIsNone(artifact.bytes)
        self.assertEqual(artifact.content_hash, "abc123")
        with self.assertRaises(FrozenInstanceError):
            setattr(artifact, "bytes", b"changed")


if __name__ == "__main__":
    unittest.main()
