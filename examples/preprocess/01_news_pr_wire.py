"""Minimal PR-wire news preprocessing example."""

from datetime import datetime, timezone

from quantmind.preprocess.news import RawNewsDocument, preprocess_news_document

raw = RawNewsDocument(
    title="Carnival Announces Results",
    body_text=(
        "Carnival Corporation & plc (NYSE: [CCL](#financial-modal)) today "
        "reported record quarterly revenue."
    ),
    source_type="press_release",
    source_url="https://example.com/news/carnival-results",
    publisher="Example Newswire",
    published_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    payload_id="example-wire-123",
)

candidate = preprocess_news_document(raw)

print(candidate.identity)
print(candidate.content_hash)
print(candidate.body_text)
print([hint.symbol for hint in candidate.ticker_hints])
