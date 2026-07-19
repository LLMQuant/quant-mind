"""Build page-preserving paper structure artifacts."""

from quantmind.configs import PaperFlowCfg
from quantmind.flows._paper_structure import (
    PaperStructureError,
    _AgentsPaperStructureProvider,
    _PaperStructureProvider,
    _structure_instructions_hash,
)
from quantmind.knowledge import (
    PaperChunkSet,
    PaperStructureProducer,
    PaperStructureTree,
)
from quantmind.preprocess import ParsedDocument, extract_outline_signals

__all__ = ["PaperStructureError", "build_paper_structure_tree"]


async def build_paper_structure_tree(
    document: ParsedDocument,
    chunk_set: PaperChunkSet,
    *,
    cfg: PaperFlowCfg | None = None,
    _structure_provider: _PaperStructureProvider | None = None,
) -> PaperStructureTree:
    """Build one independently versioned structure tree without persisting it.

    Args:
        document: Page-aware deterministic parser result for the paper.
        chunk_set: Canonical ordered chunks derived from the same source.
        cfg: Model, prompt, runtime, and tree-size bounds.

    Returns:
        A code-identified and validated paper structure-tree artifact.

    Raises:
        ValueError: If a chunk cites a physical page absent from the document.
        PaperStructureError: If the single model call exceeds its timeout.
    """
    cfg = cfg or PaperFlowCfg()
    physical_pages = {page.page_number for page in document.pages}
    if any(
        span.page_number not in physical_pages
        for chunk in chunk_set.chunks
        for span in chunk.source_spans
    ):
        raise ValueError(
            "paper chunk set cites a page absent from the document"
        )
    signals = extract_outline_signals(document)
    provider = _structure_provider or _AgentsPaperStructureProvider()
    draft = await provider.structure(signals, chunk_set, cfg=cfg)
    producer = PaperStructureProducer(
        model=cfg.model,
        prompt_version=cfg.structure_prompt_version,
        input_chunk_set_id=chunk_set.id,
        instructions_hash=_structure_instructions_hash(cfg),
        max_output_tokens=cfg.max_structure_output_tokens,
        max_depth=cfg.structure_max_depth,
        max_nodes=cfg.structure_max_nodes,
    )
    return PaperStructureTree.from_draft(
        chunk_set,
        producer=producer,
        draft=draft,
    )
