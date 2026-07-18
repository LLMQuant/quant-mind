"""Source revisions and independently versioned paper artifacts."""

import hashlib
import json
import re
from datetime import datetime
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from quantmind.knowledge._base import SourceRef
from quantmind.knowledge._tree import TreeKnowledge

PaperArtifactKind = Literal["paper_chunk_set", "paper_summary"]


def _stable_hash(value: object) -> str:
    """Return a deterministic SHA-256 hash for one JSON-compatible value."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _paper_source_id(content_hash: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"quantmind:paper-source:{content_hash}")


def _paper_artifact_id(
    source_revision_id: UUID,
    artifact_kind: PaperArtifactKind,
    producer_config_hash: str,
) -> UUID:
    return uuid5(
        source_revision_id,
        f"quantmind:{artifact_kind}:{producer_config_hash}",
    )


def _paper_asset_id(
    source_revision_id: UUID,
    *,
    kind: str,
    page_number: int | None,
    content_hash: str,
) -> UUID:
    page = page_number if page_number is not None else "document"
    return uuid5(
        source_revision_id,
        f"quantmind:paper-asset:{kind}:{page}:{content_hash}",
    )


class PaperBoundingBox(BaseModel):
    """One source rectangle in top-left-origin page coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _coordinates_are_ordered(self) -> "PaperBoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bounding-box coordinates must be ordered")
        return self


class PaperAssetRef(BaseModel):
    """Content-addressed source, screenshot, or extracted-image reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: UUID
    kind: Literal["raw", "screenshot", "image"]
    media_type: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)


class PaperParsedBlock(BaseModel):
    """One parser-owned text block in the durable source manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    bbox: PaperBoundingBox
    font_name: str | None = None
    font_size: float | None = None
    confidence: float | None = None


class PaperParsedPage(BaseModel):
    """One physical page and its content-addressed visual references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    text: str
    blocks: tuple[PaperParsedBlock, ...] = ()
    screenshot_asset_id: UUID | None = None
    image_asset_ids: tuple[UUID, ...] = ()


class PaperParsedManifest(BaseModel):
    """Complete page-aware parser output for one exact source revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_name: str
    parser_version: str
    cleanup_version: str
    pages: tuple[PaperParsedPage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _pages_are_contiguous(self) -> "PaperParsedManifest":
        expected = tuple(range(1, len(self.pages) + 1))
        actual = tuple(page.page_number for page in self.pages)
        if actual != expected:
            raise ValueError("parsed pages must be contiguous and 1-based")
        return self


class PaperSourceRevision(BaseModel):
    """Immutable source revision anchored to the exact fetched bytes.

    ``blobs`` carries exact bytes across the flow-to-library boundary. It is
    excluded from canonical JSON; the local library persists every referenced
    blob in a transactionally linked content-addressed table.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    schema_version: Literal["1.0"] = "1.0"
    source: SourceRef
    as_of: datetime
    available_at: datetime
    published_at: datetime | None = None
    arxiv_id: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    parsed: PaperParsedManifest
    raw_asset_id: UUID
    assets: tuple[PaperAssetRef, ...] = Field(min_length=1)
    blobs: dict[str, bytes] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )

    @field_validator("as_of", "available_at", "published_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("paper source timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_revision(self) -> "PaperSourceRevision":
        content_hash = self.source.content_hash
        if content_hash is None:
            raise ValueError("paper source content_hash is required")
        if self.id != _paper_source_id(content_hash):
            raise ValueError("paper source revision ID does not match content")
        if self.parsed.source_hash != content_hash:
            raise ValueError("parsed manifest does not match source content")
        if self.source.kind == "arxiv" and (
            self.arxiv_id is None or re.search(r"v\d+$", self.arxiv_id) is None
        ):
            raise ValueError("arXiv paper sources require an exact revision")

        assets = {asset.asset_id: asset for asset in self.assets}
        if len(assets) != len(self.assets):
            raise ValueError("paper source asset IDs must be unique")
        raw = assets.get(self.raw_asset_id)
        if raw is None or raw.kind != "raw" or raw.content_hash != content_hash:
            raise ValueError("raw paper asset does not match source content")
        for asset in self.assets:
            expected_id = _paper_asset_id(
                self.id,
                kind=asset.kind,
                page_number=asset.page_number,
                content_hash=asset.content_hash,
            )
            if asset.asset_id != expected_id:
                raise ValueError("paper asset ID does not match its content")

        for page in self.parsed.pages:
            referenced = (
                (page.screenshot_asset_id,)
                if page.screenshot_asset_id is not None
                else ()
            ) + page.image_asset_ids
            for asset_id in referenced:
                asset = assets.get(asset_id)
                if asset is None or asset.page_number != page.page_number:
                    raise ValueError("parsed page references an unknown asset")

        if self.blobs:
            for content_hash_value, blob in self.blobs.items():
                if hashlib.sha256(blob).hexdigest() != content_hash_value:
                    raise ValueError("paper source blob hash mismatch")
            for asset in self.assets:
                blob = self.blobs.get(asset.content_hash)
                if blob is None or len(blob) != asset.size_bytes:
                    raise ValueError("paper source is missing an asset blob")
        return self

    def blob_for(self, asset_id: UUID) -> bytes:
        """Return exact bytes for one referenced source asset."""
        asset = next(
            (item for item in self.assets if item.asset_id == asset_id),
            None,
        )
        if asset is None:
            raise KeyError(f"Paper asset '{asset_id}' not found")
        try:
            return self.blobs[asset.content_hash]
        except KeyError as exc:
            raise RuntimeError(
                f"Paper asset '{asset_id}' bytes are not loaded"
            ) from exc


class PaperSourceSpan(BaseModel):
    """Exact character span and page evidence retained by one chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    block_boxes: tuple[PaperBoundingBox, ...] = ()
    asset_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _span_is_ordered(self) -> "PaperSourceSpan":
        if self.end_char <= self.start_char:
            raise ValueError("paper source span must be non-empty")
        return self


def _paper_chunk_id(
    chunk_set_id: UUID,
    *,
    position: int,
    content_hash: str,
    spans: tuple[PaperSourceSpan, ...],
) -> UUID:
    span_hash = _stable_hash([span.model_dump(mode="json") for span in spans])
    return uuid5(
        chunk_set_id,
        f"quantmind:paper-chunk:{position}:{content_hash}:{span_hash}",
    )


class PaperChunk(BaseModel):
    """Directly addressable page-aware member of a chunk-set artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: UUID
    chunk_set_id: UUID
    source_revision_id: UUID
    position: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spans: tuple[PaperSourceSpan, ...] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paper chunk text must not be blank")
        return value

    @model_validator(mode="after")
    def _identity_matches_content(self) -> "PaperChunk":
        if self.content_hash != _text_hash(self.text):
            raise ValueError("paper chunk content hash mismatch")
        expected = _paper_chunk_id(
            self.chunk_set_id,
            position=self.position,
            content_hash=self.content_hash,
            spans=self.source_spans,
        )
        if self.chunk_id != expected:
            raise ValueError("paper chunk ID does not match its content")
        return self


class PaperChunkingConfig(BaseModel):
    """Exact LlamaIndex splitter identity and configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    splitter: Literal["llama-index-sentence-splitter"] = (
        "llama-index-sentence-splitter"
    )
    splitter_version: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_is_smaller_than_chunk(self) -> "PaperChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def _paper_chunk_set_content_hash(chunks: tuple[PaperChunk, ...]) -> str:
    return _stable_hash(
        [
            {
                "position": chunk.position,
                "content_hash": chunk.content_hash,
                "source_spans": [
                    span.model_dump(mode="json") for span in chunk.source_spans
                ],
            }
            for chunk in chunks
        ]
    )


class PaperChunkSet(BaseModel):
    """Versioned ordered chunks produced from one source revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    artifact_kind: Literal["paper_chunk_set"] = "paper_chunk_set"
    schema_version: Literal["1.0"] = "1.0"
    source_revision_id: UUID
    producer: PaperChunkingConfig
    producer_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[PaperChunk, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_artifact(self) -> "PaperChunkSet":
        expected_config_hash = _stable_hash(
            self.producer.model_dump(mode="json")
        )
        if self.producer_config_hash != expected_config_hash:
            raise ValueError("paper chunk-set producer hash mismatch")
        expected_id = _paper_artifact_id(
            self.source_revision_id,
            self.artifact_kind,
            self.producer_config_hash,
        )
        if self.id != expected_id:
            raise ValueError("paper chunk-set ID does not match its producer")
        if tuple(chunk.position for chunk in self.chunks) != tuple(
            range(len(self.chunks))
        ):
            raise ValueError("paper chunks must have contiguous positions")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("paper chunk IDs must be unique")
        if any(
            chunk.chunk_set_id != self.id
            or chunk.source_revision_id != self.source_revision_id
            for chunk in self.chunks
        ):
            raise ValueError("paper chunk membership does not match chunk set")
        if self.content_hash != _paper_chunk_set_content_hash(self.chunks):
            raise ValueError("paper chunk-set content hash mismatch")
        return self


def _validate_chunk_set_source(
    source: PaperSourceRevision,
    chunk_set: PaperChunkSet,
) -> None:
    """Validate every chunk span against its exact source manifest."""
    if chunk_set.source_revision_id != source.id:
        raise ValueError("paper chunk set belongs to another source")
    pages = {page.page_number: page for page in source.parsed.pages}
    assets = {asset.asset_id: asset for asset in source.assets}
    for chunk in chunk_set.chunks:
        for span in chunk.source_spans:
            page = pages.get(span.page_number)
            if page is None:
                raise ValueError("paper chunk span references an unknown page")
            if span.end_char > len(page.text):
                raise ValueError("paper chunk span exceeds its source page")
            for asset_id in span.asset_ids:
                asset = assets.get(asset_id)
                if (
                    asset is None
                    or asset.page_number != span.page_number
                    or asset.kind == "raw"
                ):
                    raise ValueError(
                        "paper chunk span references an unknown page asset"
                    )


class ArtifactLocator(BaseModel):
    """Stable address for a paper artifact or one of its members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_revision_id: UUID | None
    artifact_id: UUID
    artifact_kind: str = Field(min_length=1)
    member_id: UUID | None = None


class PaperCitation(BaseModel):
    """Code-resolved citation from a summary to one chunk and source page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_set_id: UUID
    chunk_id: UUID
    page_number: int = Field(ge=1)
    quote: str | None = Field(default=None, max_length=500)


class PaperSummaryProducer(BaseModel):
    """Exact model, prompt, and output-bound identity for a summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    prompt_version: str
    input_chunk_set_id: UUID
    instructions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_output_tokens: int = Field(gt=0)


def _paper_summary_content_hash(
    summary: str,
    citations: tuple[PaperCitation, ...],
) -> str:
    return _stable_hash(
        {
            "summary": summary,
            "citations": [
                citation.model_dump(mode="json") for citation in citations
            ],
        }
    )


class PaperGlobalSummary(BaseModel):
    """One independently versioned cited global-summary artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    artifact_kind: Literal["paper_summary"] = "paper_summary"
    schema_version: Literal["1.0"] = "1.0"
    source_revision_id: UUID
    producer: PaperSummaryProducer
    producer_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1)
    citations: tuple[PaperCitation, ...] = Field(min_length=1)
    derived_from: tuple[ArtifactLocator, ...] = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def _summary_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paper global summary must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_artifact(self) -> "PaperGlobalSummary":
        expected_config_hash = _stable_hash(
            self.producer.model_dump(mode="json")
        )
        if self.producer_config_hash != expected_config_hash:
            raise ValueError("paper summary producer hash mismatch")
        expected_id = _paper_artifact_id(
            self.source_revision_id,
            self.artifact_kind,
            self.producer_config_hash,
        )
        if self.id != expected_id:
            raise ValueError("paper summary ID does not match its producer")
        if self.content_hash != _paper_summary_content_hash(
            self.summary,
            self.citations,
        ):
            raise ValueError("paper summary content hash mismatch")
        if any(
            locator.source_revision_id != self.source_revision_id
            or locator.member_id is not None
            for locator in self.derived_from
        ):
            raise ValueError("paper summary lineage has an invalid locator")
        if not any(
            locator.artifact_kind == "paper_chunk_set"
            and locator.artifact_id == self.producer.input_chunk_set_id
            for locator in self.derived_from
        ):
            raise ValueError(
                "paper summary producer input is missing from lineage"
            )
        return self


class PaperFlowResult(BaseModel):
    """Validated V1 result containing source, chunks, and cited summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_revision: PaperSourceRevision
    chunk_set: PaperChunkSet
    global_summary: PaperGlobalSummary

    @model_validator(mode="after")
    def _validate_cross_artifact_links(self) -> "PaperFlowResult":
        source_id = self.source_revision.id
        if (
            self.chunk_set.source_revision_id != source_id
            or self.global_summary.source_revision_id != source_id
        ):
            raise ValueError("paper artifacts do not share their source")
        _validate_chunk_set_source(self.source_revision, self.chunk_set)
        chunk_set_locator = ArtifactLocator(
            source_revision_id=source_id,
            artifact_id=self.chunk_set.id,
            artifact_kind="paper_chunk_set",
        )
        if chunk_set_locator not in self.global_summary.derived_from:
            raise ValueError("paper summary is missing chunk-set lineage")

        chunks = {chunk.chunk_id: chunk for chunk in self.chunk_set.chunks}
        for citation in self.global_summary.citations:
            chunk = chunks.get(citation.chunk_id)
            if chunk is None or citation.chunk_set_id != self.chunk_set.id:
                raise ValueError("paper summary cites an unknown chunk")
            pages = {span.page_number for span in chunk.source_spans}
            if citation.page_number not in pages:
                raise ValueError("paper summary citation page is not in chunk")
            if citation.quote and citation.quote not in chunk.text:
                raise ValueError("paper summary citation quote is not in chunk")
        return self

    @property
    def source(self) -> PaperSourceRevision:
        """Return the exact source revision using a concise compatibility name."""
        return self.source_revision

    @property
    def summary(self) -> PaperGlobalSummary:
        """Return the global-summary artifact using a concise name."""
        return self.global_summary


class LegacyPaper(TreeKnowledge):
    """Pre-V1 paper tree retained only for explicit database compatibility."""

    item_type: Literal["paper"] = "paper"
    arxiv_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    asset_classes: list[str] = Field(default_factory=list)


PaperArtifact = PaperChunkSet | PaperGlobalSummary
ResolvedPaperArtifact = PaperArtifact | PaperChunk
