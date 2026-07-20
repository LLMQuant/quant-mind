"""Build a self-contained structure tree and retrieve evidence in memory.

The core path needs no library. ``PaperFlow`` binds a build config once;
``build(input)`` fetches and parses the PDF and returns a self-contained
``PaperStructureTree`` whose leaf nodes carry their own page-cited text.
``AgenticRetriever`` binds a retrieval config; ``retrieve(tree, question)``
reasons over that tree value and returns evidence with the content already in
it — no library round-trip.

The clearly-labeled OPTIONAL section then shows dump/load symmetry: a library
stores the tree standalone (no source object needed) and reopens it as an
identical self-contained value, over which ``retrieve`` behaves exactly the
same. It is not needed to retrieve.

Running this end to end needs network access (a model provider). The example is
written so it imports and type-checks offline.
"""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperStructureCfg, RetrievalCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow
from quantmind.knowledge import PaperStructureTree
from quantmind.mind import AgenticRetriever

_QUESTION = "What are the main method and limitations?"


async def main(pdf_path: Path) -> None:
    """Build a structure tree and retrieve from it in memory (no library)."""
    # Bind the build config once; build(input) applies it per input. A batch
    # would call batch_run(flow.build, inputs) under this one setting.
    flow = PaperFlow(PaperStructureCfg(model="gpt-4o-mini"))
    tree = await flow.build(LocalFilePath(path=pdf_path))

    # Bind the retrieval config once; retrieve(tree, question) takes only the
    # operands. Evidence carries content directly — no library.
    retriever = AgenticRetriever(RetrievalCfg(model="gpt-4o-mini"))
    evidence = await retriever.retrieve(tree, _QUESTION)
    for item in evidence:
        print(item.title, "—", item.content[:500])

    await _optional_persist_reopen(tree)


async def _optional_persist_reopen(tree: PaperStructureTree) -> None:
    """OPTIONAL: dump the tree to a library and reopen an identical value.

    This section is not required to retrieve — the tree built above is already
    self-contained. It exists only to demonstrate that ``library.put(tree)`` /
    ``library.open_structure(tree.id)`` round-trip to the same value (the tree
    is stored standalone; no source object is needed), over which ``retrieve``
    behaves identically. Delete it if you only need in-memory retrieval.
    """
    from quantmind.library import LocalKnowledgeLibrary

    library = await LocalKnowledgeLibrary.open(
        ":memory:",
        embedding_model="text-embedding-3-small",
    )
    try:
        await library.put(tree)  # standalone: no source revision required
        reopened = await library.open_structure(tree.id)  # identical value

        retriever = AgenticRetriever(RetrievalCfg(model="gpt-4o-mini"))
        evidence = await retriever.retrieve(reopened, _QUESTION)
        for item in evidence:
            print(item.title, "—", item.content[:500])
    finally:
        await library.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/mind/paper_structure_retrieval.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
