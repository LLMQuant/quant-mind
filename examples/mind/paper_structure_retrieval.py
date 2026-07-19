"""Build, persist, and retrieve from one local paper structure tree."""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperFlowCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import build_paper_structure_tree, paper_flow
from quantmind.library import LocalKnowledgeLibrary
from quantmind.mind import retrieve
from quantmind.preprocess import parse_pdf


async def main(pdf_path: Path) -> None:
    """Run the common vectorless retrieval path for one local PDF."""
    cfg = PaperFlowCfg(model="gpt-4o-mini")
    document = await parse_pdf(pdf_path.read_bytes())
    paper = await paper_flow(LocalFilePath(path=pdf_path), cfg=cfg)
    structure = await build_paper_structure_tree(
        document,
        paper.chunk_set,
        cfg=cfg,
    )
    library = await LocalKnowledgeLibrary.open(
        ":memory:",
        embedding_model="text-embedding-3-small",
    )
    try:
        await library.put_paper(paper)
        await library.put_paper_structure_tree(structure)
        evidence = await retrieve(
            structure,
            "What are the main method and limitations?",
            library=library,
            cfg=RetrievalCfg(model="gpt-4o-mini", grain="agentic"),
        )
        for item in evidence:
            pages = sorted(
                {citation.page for citation in item.citations if citation.page}
            )
            print(f"{item.title} — pages {pages}")
            print(item.content[:500])
    finally:
        await library.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/mind/paper_structure_retrieval.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
