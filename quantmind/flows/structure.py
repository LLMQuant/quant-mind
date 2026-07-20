"""Build page-preserving paper structure artifacts."""

from quantmind.configs import PaperStructureCfg
from quantmind.flows._paper_structure import (
    PaperStructureError,
    _AgentsPaperStructureProvider,
    _PaperStructureProvider,
    _structure_instructions_hash,
)
from quantmind.knowledge import (
    PaperSourceRevision,
    PaperStructureProducer,
    PaperStructureTree,
)
from quantmind.preprocess import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
    TextBlock,
    extract_outline_signals,
)

__all__ = ["PaperStructureBuilder", "PaperStructureError"]


class PaperStructureBuilder:
    """Reuse one source-native paper structuring policy across documents.

    The builder owns only reusable model and provider policy. Each ``build``
    call receives an exact source revision and returns an immutable artifact;
    the builder never retains a current source or tree.
    """

    __slots__ = ("_cfg", "_provider")

    def __init__(
        self,
        cfg: PaperStructureCfg | None = None,
        *,
        _structure_provider: _PaperStructureProvider | None = None,
    ) -> None:
        self._cfg = (
            cfg.model_copy(deep=True)
            if cfg is not None
            else PaperStructureCfg()
        )
        self._provider = _structure_provider or _AgentsPaperStructureProvider()

    async def build(self, source: PaperSourceRevision) -> PaperStructureTree:
        """Build one validated structure tree from an exact source revision.

        Args:
            source: Immutable page-preserving paper source revision.

        Returns:
            A code-identified tree whose nodes cite physical source pages.

        Raises:
            PaperStructureError: If the model call exceeds its timeout.
            StructureTreeValidationError: If the proposed page tree is invalid.
        """
        signals = extract_outline_signals(_parsed_document(source))
        draft = await self._provider.structure(
            signals,
            source,
            cfg=self._cfg,
        )
        producer = PaperStructureProducer(
            model=self._cfg.model,
            prompt_version=self._cfg.prompt_version,
            instructions_hash=_structure_instructions_hash(self._cfg),
            page_text_chars=self._cfg.page_text_chars,
            max_output_tokens=self._cfg.max_output_tokens,
            max_depth=self._cfg.max_depth,
            max_nodes=self._cfg.max_nodes,
        )
        return PaperStructureTree.from_draft(
            source,
            producer=producer,
            draft=draft,
        )


def _parsed_document(source: PaperSourceRevision) -> ParsedDocument:
    """Project a canonical source manifest into outline extraction input."""
    return ParsedDocument(
        source_hash=source.parsed.source_hash,
        parser_name=source.parsed.parser_name,
        parser_version=source.parsed.parser_version,
        cleanup_version=source.parsed.cleanup_version,
        pages=tuple(
            ParsedPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text=page.text,
                blocks=tuple(
                    TextBlock(
                        text=block.text,
                        page_number=page.page_number,
                        bbox=BoundingBox(
                            x0=block.bbox.x0,
                            y0=block.bbox.y0,
                            x1=block.bbox.x1,
                            y1=block.bbox.y1,
                        ),
                        font_name=block.font_name,
                        font_size=block.font_size,
                        confidence=block.confidence,
                    )
                    for block in page.blocks
                ),
            )
            for page in source.parsed.pages
        ),
    )
