"""Build, persist, and retrieve from one local paper structure tree."""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperFlowCfg, PaperStructureCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperStructureBuilder, paper_flow
from quantmind.library import LocalKnowledgeLibrary
from quantmind.mind import StructureRetriever


async def main(pdf_path: Path) -> None:
    """Run the common vectorless retrieval path for one local PDF."""
    paper = await paper_flow(
        LocalFilePath(path=pdf_path),
        cfg=PaperFlowCfg(model="gpt-4o-mini"),
    )
    # Structure construction consumes only the exact source revision; paper
    # chunk and summary artifacts do not affect tree identity or content.
    builder = PaperStructureBuilder(PaperStructureCfg(model="gpt-4o-mini"))
    structure = await builder.build(paper.source_revision)
    library = await LocalKnowledgeLibrary.open(
        ":memory:",
        embedding_model="text-embedding-3-small",
    )
    try:
        await library.put_paper_structure_tree(
            paper.source_revision,
            structure,
        )
        retriever = StructureRetriever(
            library=library,
            cfg=RetrievalCfg(model="gpt-4o-mini", grain="agentic"),
        )
        evidence = await retriever.retrieve(
            structure,
            "What are the main method and limitations?",
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
