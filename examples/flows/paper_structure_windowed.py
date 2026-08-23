"""Draft a structure tree from full page text in bounded windows.

``PaperFlow(PaperStructureCfg()).build`` reads every page complete: pages are
packed into character-bounded windows (``window_chars`` per model call, with
``window_overlap_pages`` shared pages for continuity), and a document larger
than one window is drafted across chained calls that each extend the prior
draft. Dense pages therefore keep their lower-page section starts and body
prose visible to the drafting model.

``page_text_chars`` stays available as an explicit per-page clip for cost
control on sparse inputs (short pages, a clean table of contents); the
``None`` default sends full pages.

Running this end to end needs network access (a model provider). The example
is written so it imports and type-checks offline.
"""

import asyncio
import sys
from pathlib import Path

from quantmind.configs import PaperStructureCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import PaperFlow


async def main(pdf_path: Path) -> None:
    """Build one windowed full-text structure tree for a local PDF."""
    # Defaults draft from full page text: ~80k chars per window, one page of
    # overlap between consecutive windows, and no per-page clipping.
    flow = PaperFlow(PaperStructureCfg(model="gpt-5.6-luna"))
    tree = await flow.build(LocalFilePath(path=pdf_path))

    producer = tree.producer
    print("orchestration:", producer.orchestration)
    print("window_chars:", producer.window_chars)
    print("window_overlap_pages:", producer.window_overlap_pages)
    print("page_text_chars:", producer.page_text_chars)
    for node in tree.nodes.values():
        pages = [c.page for c in node.citations if c.page is not None]
        print(f"{node.title} — pages {min(pages)}-{max(pages)}")

    # Sparse inputs can trade fidelity for cost with an explicit clip; the
    # producer records the policy, so both trees version independently.
    clipped_flow = PaperFlow(
        PaperStructureCfg(model="gpt-5.6-luna", page_text_chars=1_200)
    )
    clipped_tree = await clipped_flow.build(LocalFilePath(path=pdf_path))
    print("clipped tree id differs:", clipped_tree.id != tree.id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/flows/paper_structure_windowed.py paper.pdf"
        )
    asyncio.run(main(Path(sys.argv[1])))
