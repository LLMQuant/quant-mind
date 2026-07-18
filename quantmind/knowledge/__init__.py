"""quantmind.knowledge — data standard for extracted financial knowledge.

The standard defines three shapes that share `BaseKnowledge`:

- `FlattenKnowledge` — atomic cards (`News`, `Earnings`, `Factor`, `Thesis`).
- `TreeKnowledge` — hierarchical artifacts such as future navigation trees.
- `GraphKnowledge` — cross-item edges (placeholder, not implemented).

Every concrete subclass is frozen Pydantic v2 with ``extra="forbid"``,
suitable for ``Agent(output_type=...)`` and round-tripping through JSON.
Paper sources and artifacts are separate immutable models. Search projection
text and vectors remain rebuildable library-owned data.
"""

from quantmind.knowledge._base import (
    BaseKnowledge,
    Citation,
    ExtractionRef,
    SourceRef,
)
from quantmind.knowledge._flatten import FlattenKnowledge
from quantmind.knowledge._graph import GraphKnowledge
from quantmind.knowledge._tree import TreeKnowledge, TreeNode
from quantmind.knowledge.earnings import Earnings
from quantmind.knowledge.factor import Factor
from quantmind.knowledge.news import News
from quantmind.knowledge.paper import (
    ArtifactLocator,
    LegacyPaper,
    PaperArtifact,
    PaperArtifactKind,
    PaperAssetInput,
    PaperAssetRef,
    PaperBoundingBox,
    PaperChunk,
    PaperChunkingConfig,
    PaperChunkInput,
    PaperChunkSet,
    PaperCitation,
    PaperCitationDraft,
    PaperCitationValidationError,
    PaperFlowResult,
    PaperGlobalSummary,
    PaperPageInput,
    PaperParsedBlock,
    PaperParsedManifest,
    PaperParsedPage,
    PaperSourceFacts,
    PaperSourceRevision,
    PaperSourceSpan,
    PaperSummaryProducer,
    ResolvedPaperArtifact,
)
from quantmind.knowledge.thesis import Thesis

__all__ = [
    # Base
    "BaseKnowledge",
    "Citation",
    "ExtractionRef",
    "SourceRef",
    # Shapes
    "FlattenKnowledge",
    "GraphKnowledge",
    "TreeKnowledge",
    "TreeNode",
    # Concrete
    "Earnings",
    "Factor",
    "News",
    "ArtifactLocator",
    "LegacyPaper",
    "PaperArtifact",
    "PaperArtifactKind",
    "PaperAssetInput",
    "PaperAssetRef",
    "PaperBoundingBox",
    "PaperChunk",
    "PaperChunkInput",
    "PaperChunkingConfig",
    "PaperChunkSet",
    "PaperCitation",
    "PaperCitationDraft",
    "PaperCitationValidationError",
    "PaperFlowResult",
    "PaperGlobalSummary",
    "PaperPageInput",
    "PaperParsedBlock",
    "PaperParsedManifest",
    "PaperParsedPage",
    "PaperSourceFacts",
    "PaperSourceRevision",
    "PaperSourceSpan",
    "PaperSummaryProducer",
    "ResolvedPaperArtifact",
    "Thesis",
]
