"""Source revisions and independently versioned paper artifacts."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from quantmind.knowledge._base import Citation, SourceRef
from quantmind.knowledge._tree import (
    StructureTree,
    StructureTreeValidationError,
    TreeKnowledge,
    TreeNode,
)


class PaperArtifactKind(str, Enum):
    """Closed paper-artifact discriminator accepted by Pydantic and JSON."""

    CHUNK_SET = "paper_chunk_set"
    GLOBAL_SUMMARY = "paper_summary"
    STRUCTURE_TREE = "paper_structure_tree"


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
    artifact_kind: PaperArtifactKind | str,
    producer_config_hash: str,
) -> UUID:
    kind = PaperArtifactKind(artifact_kind)
    return uuid5(
        source_revision_id,
        f"quantmind:{kind.value}:{producer_config_hash}",
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


class PaperCitationValidationError(ValueError):
    """A generated summary did not provide valid source coverage."""


@dataclass(frozen=True)
class PaperSourceFacts:
    """Code-owned source facts normalized by the flow before construction.

    Everything here comes from fetching and IO rather than the parser or the
    model, so identity construction can stay inside the knowledge layer while
    fetching stays in the flow.
    """

    kind: Literal["arxiv", "http", "local"]
    uri: str
    media_type: str
    raw_bytes: bytes
    fetched_at: datetime
    available_at: datetime
    published_at: datetime | None = None
    arxiv_id: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperAssetInput:
    """Raw bytes for one page-level visual asset, pre-read by the flow."""

    kind: Literal["screenshot", "image"]
    content: bytes
    media_type: str


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


@dataclass(frozen=True)
class PaperChunkInput:
    """One splitter-produced chunk in knowledge-native form."""

    page_number: int
    start_char: int
    end_char: int
    block_boxes: tuple[PaperBoundingBox, ...]
    text: str


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


@dataclass(frozen=True)
class PaperPageInput:
    """One parsed page plus pre-read visual-asset bytes for construction."""

    page_number: int
    width: float
    height: float
    text: str
    blocks: tuple[PaperParsedBlock, ...] = ()
    screenshot: PaperAssetInput | None = None
    images: tuple[PaperAssetInput, ...] = ()


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

    @classmethod
    def from_parsed(
        cls,
        *,
        facts: PaperSourceFacts,
        source_hash: str,
        parser_name: str,
        parser_version: str,
        cleanup_version: str,
        pages: Sequence[PaperPageInput],
    ) -> "PaperSourceRevision":
        """Build a source revision, minting every content-addressed ID.

        The flow supplies normalized ``facts`` and pre-read page bytes; this
        constructor owns all identity (source ID, asset IDs) so callers never
        compute a paper ID themselves.
        """
        source_id = _paper_source_id(source_hash)
        blobs: dict[str, bytes] = {source_hash: facts.raw_bytes}
        raw = PaperAssetRef(
            asset_id=_paper_asset_id(
                source_id,
                kind="raw",
                page_number=None,
                content_hash=source_hash,
            ),
            kind="raw",
            media_type=facts.media_type,
            content_hash=source_hash,
            size_bytes=len(facts.raw_bytes),
        )
        assets: dict[UUID, PaperAssetRef] = {raw.asset_id: raw}
        parsed_pages: list[PaperParsedPage] = []
        for page in pages:
            screenshot_id: UUID | None = None
            if page.screenshot is not None:
                screenshot_id = cls._register_asset(
                    source_id, page.page_number, page.screenshot, assets, blobs
                ).asset_id
            image_ids = tuple(
                cls._register_asset(
                    source_id, page.page_number, image, assets, blobs
                ).asset_id
                for image in page.images
            )
            parsed_pages.append(
                PaperParsedPage(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    text=page.text,
                    blocks=page.blocks,
                    screenshot_asset_id=screenshot_id,
                    image_asset_ids=image_ids,
                )
            )
        return cls(
            id=source_id,
            source=SourceRef(
                kind=facts.kind,
                uri=facts.uri,
                fetched_at=facts.fetched_at,
                content_hash=source_hash,
            ),
            as_of=facts.published_at or facts.available_at,
            available_at=facts.available_at,
            published_at=facts.published_at,
            arxiv_id=facts.arxiv_id,
            title=facts.title,
            authors=facts.authors,
            parsed=PaperParsedManifest(
                source_hash=source_hash,
                parser_name=parser_name,
                parser_version=parser_version,
                cleanup_version=cleanup_version,
                pages=tuple(parsed_pages),
            ),
            raw_asset_id=raw.asset_id,
            assets=tuple(assets.values()),
            blobs=blobs,
        )

    @staticmethod
    def _register_asset(
        source_id: UUID,
        page_number: int,
        asset: PaperAssetInput,
        assets: dict[UUID, PaperAssetRef],
        blobs: dict[str, bytes],
    ) -> PaperAssetRef:
        content_hash = hashlib.sha256(asset.content).hexdigest()
        ref = PaperAssetRef(
            asset_id=_paper_asset_id(
                source_id,
                kind=asset.kind,
                page_number=page_number,
                content_hash=content_hash,
            ),
            kind=asset.kind,
            media_type=asset.media_type,
            content_hash=content_hash,
            size_bytes=len(asset.content),
            page_number=page_number,
        )
        assets[ref.asset_id] = ref
        blobs[content_hash] = asset.content
        return ref


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
    artifact_kind: Literal[PaperArtifactKind.CHUNK_SET] = (
        PaperArtifactKind.CHUNK_SET
    )
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

    @classmethod
    def from_parsed_chunks(
        cls,
        source: PaperSourceRevision,
        chunks: Sequence[PaperChunkInput],
        *,
        producer: PaperChunkingConfig,
    ) -> "PaperChunkSet":
        """Build a chunk-set artifact, minting the artifact and chunk IDs.

        The flow supplies knowledge-native chunk inputs; this constructor owns
        the artifact ID, per-chunk content hashes, chunk IDs, and the set
        content hash.
        """
        producer_hash = _stable_hash(producer.model_dump(mode="json"))
        artifact_id = _paper_artifact_id(
            source.id, PaperArtifactKind.CHUNK_SET, producer_hash
        )
        page_assets = {
            page.page_number: (
                (
                    (page.screenshot_asset_id,)
                    if page.screenshot_asset_id is not None
                    else ()
                )
                + page.image_asset_ids
            )
            for page in source.parsed.pages
        }
        built: list[PaperChunk] = []
        for position, chunk in enumerate(chunks):
            span = PaperSourceSpan(
                page_number=chunk.page_number,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                block_boxes=chunk.block_boxes,
                asset_ids=page_assets.get(chunk.page_number, ()),
            )
            content_hash = _text_hash(chunk.text)
            built.append(
                PaperChunk(
                    chunk_id=_paper_chunk_id(
                        artifact_id,
                        position=position,
                        content_hash=content_hash,
                        spans=(span,),
                    ),
                    chunk_set_id=artifact_id,
                    source_revision_id=source.id,
                    position=position,
                    text=chunk.text,
                    content_hash=content_hash,
                    source_spans=(span,),
                )
            )
        if not built:
            raise ValueError("paper source produced no non-empty chunks")
        chunk_tuple = tuple(built)
        return cls(
            id=artifact_id,
            source_revision_id=source.id,
            producer=producer,
            producer_config_hash=producer_hash,
            content_hash=_paper_chunk_set_content_hash(chunk_tuple),
            chunks=chunk_tuple,
        )


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


class PaperStructureNodeDraft(BaseModel):
    """Model-proposed hierarchy node without canonical identity or links."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    children: tuple["PaperStructureNodeDraft", ...] = ()

    @field_validator("title", "summary")
    @classmethod
    def _text_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("paper structure draft text must not be blank")
        return stripped

    @model_validator(mode="after")
    def _page_span_is_ordered(self) -> "PaperStructureNodeDraft":
        if self.end_page < self.start_page:
            raise ValueError("paper structure draft page span must be ordered")
        return self


class PaperStructureTreeDraft(BaseModel):
    """Bounded model output consumed by code-owned tree construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: PaperStructureNodeDraft
    quality: Literal["low", "medium", "high"] = "high"


class PaperStructureProducer(BaseModel):
    """Exact model, prompt, page-input, and bounds used to structure a paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    prompt_version: str
    orchestration: Literal["single-pass-v1"] = "single-pass-v1"
    instructions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_text_chars: int = Field(ge=80)
    max_output_tokens: int = Field(gt=0)
    max_depth: int = Field(ge=1)
    max_nodes: int = Field(ge=1)


class ArtifactLocator(BaseModel):
    """Stable address for a paper artifact or one of its members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_revision_id: UUID | None
    artifact_id: UUID
    artifact_kind: str = Field(min_length=1)
    member_id: UUID | None = None


def _paper_structure_content_hash(
    root_node_id: UUID,
    nodes: dict[UUID, TreeNode],
) -> str:
    return _stable_hash(
        {
            "root_node_id": str(root_node_id),
            "nodes": {
                str(node_id): node.model_dump(mode="json")
                for node_id, node in sorted(
                    nodes.items(), key=lambda pair: str(pair[0])
                )
            },
        }
    )


class PaperStructureTree(StructureTree):
    """Page-preserving structure derived from one exact source revision."""

    id: UUID
    artifact_kind: Literal[PaperArtifactKind.STRUCTURE_TREE] = (
        PaperArtifactKind.STRUCTURE_TREE
    )
    schema_version: Literal["1.0"] = "1.0"
    source_revision_id: UUID
    producer: PaperStructureProducer
    producer_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_artifact(self) -> "PaperStructureTree":
        expected_config_hash = _stable_hash(
            self.producer.model_dump(mode="json")
        )
        if self.producer_config_hash != expected_config_hash:
            raise ValueError("paper structure-tree producer hash mismatch")
        expected_id = _paper_artifact_id(
            self.source_revision_id,
            self.artifact_kind,
            self.producer_config_hash,
        )
        if self.id != expected_id:
            raise ValueError(
                "paper structure-tree ID does not match its producer"
            )
        if self.content_hash != _paper_structure_content_hash(
            self.root_node_id,
            self.nodes,
        ):
            raise ValueError("paper structure-tree content hash mismatch")
        for node in self.nodes.values():
            if node.children_ids:
                if node.content is not None:
                    raise ValueError(
                        "paper structure-tree internal nodes must not carry "
                        "content"
                    )
            elif not node.content:
                raise ValueError(
                    "paper structure-tree leaf nodes require content"
                )
        if any(not node.citations for node in self.nodes.values()):
            raise ValueError("paper structure-tree nodes require citations")
        self.validate()
        return self

    @classmethod
    def from_draft(
        cls,
        source: PaperSourceRevision,
        *,
        producer: PaperStructureProducer,
        draft: PaperStructureTreeDraft,
    ) -> "PaperStructureTree":
        """Mint canonical identity, links, and page citations from a draft.

        A low-quality draft is replaced with a bounded, deterministic flat tree
        over physical source pages. Otherwise draft positions and page spans
        are retained, while every UUID, link, and citation is owned by code.
        """
        producer_hash = _stable_hash(producer.model_dump(mode="json"))
        artifact_id = _paper_artifact_id(
            source.id,
            PaperArtifactKind.STRUCTURE_TREE,
            producer_hash,
        )
        root_draft = (
            cls._flat_fallback(source, draft.root, max_nodes=producer.max_nodes)
            if draft.quality == "low"
            else draft.root
        )
        page_texts: dict[int, str] = {
            page.page_number: page.text for page in source.parsed.pages
        }
        nodes: dict[UUID, TreeNode] = {}
        node_count = 0

        def build(
            value: PaperStructureNodeDraft,
            *,
            parent_id: UUID | None,
            position: int,
            path: tuple[int, ...],
            depth: int,
        ) -> UUID:
            nonlocal node_count
            if depth > producer.max_depth:
                raise StructureTreeValidationError(
                    "paper structure draft exceeds max_depth"
                )
            if node_count >= producer.max_nodes:
                raise StructureTreeValidationError(
                    "paper structure draft exceeds max_nodes"
                )
            node_count += 1
            node_id = uuid5(
                artifact_id,
                "quantmind:paper-structure-node:"
                + ".".join(str(part) for part in path),
            )
            children_ids = [
                build(
                    child,
                    parent_id=node_id,
                    position=child_position,
                    path=(*path, child_position),
                    depth=depth + 1,
                )
                for child_position, child in enumerate(value.children)
            ]
            citations = cls._resolve_structure_citations(source, value)
            content = (
                None
                if children_ids
                else "\n\n".join(
                    page_texts[citation.page]
                    for citation in citations
                    if citation.page is not None
                )
            )
            nodes[node_id] = TreeNode(
                node_id=node_id,
                parent_id=parent_id,
                position=position,
                title=value.title,
                summary=value.summary,
                content=content,
                citations=citations,
                children_ids=children_ids,
            )
            return node_id

        root_node_id = build(
            root_draft,
            parent_id=None,
            position=0,
            path=(0,),
            depth=1,
        )
        tree = cls(
            id=artifact_id,
            source_revision_id=source.id,
            producer=producer,
            producer_config_hash=producer_hash,
            content_hash=_paper_structure_content_hash(root_node_id, nodes),
            root_node_id=root_node_id,
            nodes=nodes,
        )
        _validate_structure_tree_source(tree, source)
        return tree

    @staticmethod
    def _resolve_structure_citations(
        source: PaperSourceRevision,
        draft: PaperStructureNodeDraft,
    ) -> list[Citation]:
        source_pages = {page.page_number for page in source.parsed.pages}
        pages = set(range(draft.start_page, draft.end_page + 1))
        if not pages.issubset(source_pages):
            raise StructureTreeValidationError(
                "paper structure draft cites a page outside its source"
            )
        return [
            Citation(source_id=str(source.id), page=page)
            for page in sorted(pages)
        ]

    @staticmethod
    def _flat_fallback(
        source: PaperSourceRevision,
        proposed_root: PaperStructureNodeDraft,
        *,
        max_nodes: int,
    ) -> PaperStructureNodeDraft:
        pages = tuple(page.page_number for page in source.parsed.pages)
        leaf_count = min(len(pages), max(0, max_nodes - 1))
        leaves: list[PaperStructureNodeDraft] = []
        if leaf_count:
            group_size = (len(pages) + leaf_count - 1) // leaf_count
            for offset in range(0, len(pages), group_size):
                group = pages[offset : offset + group_size]
                leaves.append(
                    PaperStructureNodeDraft(
                        title=(
                            f"Page {group[0]}"
                            if len(group) == 1
                            else f"Pages {group[0]}-{group[-1]}"
                        ),
                        summary="Source pages retained in canonical storage.",
                        start_page=group[0],
                        end_page=group[-1],
                    )
                )
        return PaperStructureNodeDraft(
            title=proposed_root.title,
            summary=(
                "Flat source-order structure used because hierarchy quality "
                "was low."
            ),
            start_page=pages[0],
            end_page=pages[-1],
            children=tuple(leaves),
        )


def _validate_structure_tree_source(
    tree: PaperStructureTree,
    source: PaperSourceRevision,
) -> None:
    """Validate every tree page citation against its exact source revision."""
    if tree.source_revision_id != source.id:
        raise ValueError("paper structure tree belongs to another source")
    source_pages = {page.page_number for page in source.parsed.pages}
    tree.validate(source_pages=source_pages)
    for node in tree.nodes.values():
        for citation in node.citations:
            if (
                citation.source_id != str(source.id)
                or citation.page not in source_pages
                or citation.tree_id is not None
                or citation.node_id is not None
            ):
                raise ValueError(
                    "paper structure-tree citation does not resolve to its source"
                )
    if {citation.page for citation in tree.root().citations} != source_pages:
        raise ValueError(
            "paper structure-tree root must cover every source page"
        )


class PaperCitation(BaseModel):
    """Code-resolved citation from a summary to one chunk and source page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_set_id: UUID
    chunk_id: UUID
    page_number: int = Field(ge=1)
    quote: str | None = Field(default=None, max_length=500)


@dataclass(frozen=True)
class PaperCitationDraft:
    """Model-proposed citation coordinates before code resolves canonical IDs."""

    chunk_index: int
    page_number: int
    quote: str | None = None


class PaperSummaryProducer(BaseModel):
    """Exact model, prompt, and output-bound identity for a summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    prompt_version: str
    orchestration: Literal["map-reduce-v1"] = "map-reduce-v1"
    input_chunk_set_id: UUID
    instructions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_output_tokens: int = Field(gt=0)
    research_group_size: int = Field(ge=1)


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
    artifact_kind: Literal[PaperArtifactKind.GLOBAL_SUMMARY] = (
        PaperArtifactKind.GLOBAL_SUMMARY
    )
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

    @classmethod
    def from_draft(
        cls,
        chunk_set: PaperChunkSet,
        *,
        producer: PaperSummaryProducer,
        summary: str,
        citations: Sequence[PaperCitationDraft],
        min_citations: int,
        min_pages: int,
    ) -> "PaperGlobalSummary":
        """Resolve model-proposed citations and mint the summary artifact.

        The model returns only prose and ``(chunk_index, page, quote)``
        coordinates. This constructor resolves canonical chunk IDs, validates
        every citation against the chunk set, enforces the configured coverage
        policy, and mints the artifact identity and lineage.
        """
        if producer.input_chunk_set_id != chunk_set.id:
            raise ValueError("summary producer input does not match chunk set")
        resolved: list[PaperCitation] = []
        seen: set[tuple[int, int, str | None]] = set()
        for draft in citations:
            try:
                chunk = chunk_set.chunks[draft.chunk_index]
            except IndexError as exc:
                raise PaperCitationValidationError(
                    "paper summary cites an unknown chunk index"
                ) from exc
            pages = {span.page_number for span in chunk.source_spans}
            if draft.page_number not in pages:
                raise PaperCitationValidationError(
                    "paper summary citation page is not owned by its chunk"
                )
            if draft.quote is not None and draft.quote not in chunk.text:
                raise PaperCitationValidationError(
                    "paper summary citation quote is not present in its chunk"
                )
            key = (draft.chunk_index, draft.page_number, draft.quote)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(
                PaperCitation(
                    chunk_set_id=chunk_set.id,
                    chunk_id=chunk.chunk_id,
                    page_number=draft.page_number,
                    quote=draft.quote,
                )
            )
        if len(resolved) < min_citations:
            raise PaperCitationValidationError(
                "paper summary has fewer citations than min_summary_citations"
            )
        if len({citation.page_number for citation in resolved}) < min_pages:
            raise PaperCitationValidationError(
                "paper summary has fewer source pages than min_summary_pages"
            )
        producer_hash = _stable_hash(producer.model_dump(mode="json"))
        artifact_id = _paper_artifact_id(
            chunk_set.source_revision_id,
            PaperArtifactKind.GLOBAL_SUMMARY,
            producer_hash,
        )
        summary_text = summary.strip()
        citation_tuple = tuple(resolved)
        return cls(
            id=artifact_id,
            source_revision_id=chunk_set.source_revision_id,
            producer=producer,
            producer_config_hash=producer_hash,
            content_hash=_paper_summary_content_hash(
                summary_text, citation_tuple
            ),
            summary=summary_text,
            citations=citation_tuple,
            derived_from=(
                ArtifactLocator(
                    source_revision_id=chunk_set.source_revision_id,
                    artifact_id=chunk_set.id,
                    artifact_kind="paper_chunk_set",
                ),
            ),
        )


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


PaperArtifact = PaperChunkSet | PaperGlobalSummary | PaperStructureTree
ResolvedPaperArtifact = PaperArtifact | PaperChunk | TreeNode
