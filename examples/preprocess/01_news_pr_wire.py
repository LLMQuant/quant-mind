"""Minimal PR-wire news preprocessing example."""

from datetime import datetime, timezone

from quantmind.preprocess.news import RawNewsDocument, preprocess_news_document

raw = RawNewsDocument(
    title="NVIDIA Announces Results",
    body_text=(
        "NVIDIA Corporation (NASDAQ: NVDA) today reported record quarterly "
        "revenue and highlighted demand for accelerated computing."
    ),
    source_type="press_release",
    source_url="https://example.com/news/nvidia-results",
    publisher="Example Newswire",
    published_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    payload_id="example-wire-123",
)

candidate = preprocess_news_document(raw)

print(candidate.identity)
print(candidate.content_hash)
print([hint.symbol for hint in candidate.ticker_hints])
