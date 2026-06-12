"""Command-line surface for the QuantMind corpus.

Used for seeding the initial corpus, manual queries, and as a shell-callable
backend for any tool that prefers a subprocess over MCP. Examples::

    python -m qm_mcp.cli ingest-arxiv 1105.3115
    python -m qm_mcp.cli ingest-pdf ~/papers/foo.pdf
    python -m qm_mcp.cli seed papers.txt          # one source per line
    python -m qm_mcp.cli query "What is gamma in Avellaneda-Stoikov?"
    python -m qm_mcp.cli list
    python -m qm_mcp.cli delete <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from qm_mcp import ingest as I
from qm_mcp.query import query as run_query
from qm_mcp.store import CorpusStore


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


async def _dispatch_source(src: str, *, force: bool):
    """Route one seed line to the right ingest fn by simple heuristics."""
    s = src.strip()
    if not s or s.startswith("#"):
        return None
    # Strip inline "# comment" trailers (seed files annotate ids). URLs may
    # legitimately contain '#', so only strip for non-URL lines.
    if not s.lower().startswith("http"):
        s = s.split("#", 1)[0].strip()
    if not s:
        return None
    low = s.lower()
    if (
        low.startswith("arxiv:")
        or low.startswith("http")
        and "arxiv.org" in low
    ):
        return await I.ingest_arxiv(
            s.split("arxiv:", 1)[-1].strip(), force=force
        )
    if low.startswith("http://") or low.startswith("https://"):
        return await I.ingest_url(s, force=force)
    if Path(s).expanduser().is_file():
        return await I.ingest_pdf(s, force=force)
    # bare token -> treat as arxiv id
    return await I.ingest_arxiv(s, force=force)


async def _amain(args: argparse.Namespace) -> int:
    if args.cmd == "ingest-arxiv":
        _print(await I.ingest_arxiv(args.value, force=args.force))
    elif args.cmd == "ingest-url":
        _print(await I.ingest_url(args.value, force=args.force))
    elif args.cmd == "ingest-pdf":
        _print(await I.ingest_pdf(args.value, force=args.force))
    elif args.cmd == "ingest-text":
        _print(await I.ingest_text(args.value, force=args.force))
    elif args.cmd == "seed":
        lines = Path(args.value).read_text(encoding="utf-8").splitlines()
        results = []
        for line in lines:
            try:
                res = await _dispatch_source(line, force=args.force)
            except Exception as exc:  # one bad source must not sink the batch
                res = {
                    "source": line.strip(),
                    "status": "error",
                    "error": str(exc),
                }
            if res is not None:
                results.append(res)
                print(
                    f"  [{res.get('status'):>8}] {res.get('title') or res.get('source') or res.get('error')}",
                    file=sys.stderr,
                )
        _print({"seeded": results, "total": len(results)})
    elif args.cmd == "query":
        _print(await run_query(args.value, k=args.k))
    elif args.cmd == "list":
        store = CorpusStore()
        _print({"count": len(store), "items": store.list_records(light=True)})
    elif args.cmd == "delete":
        _print({"id": args.value, "deleted": CorpusStore().delete(args.value)})
    else:  # pragma: no cover
        return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="qm_mcp.cli", description="QuantMind corpus CLI"
    )
    p.add_argument(
        "--force", action="store_true", help="re-ingest even if present"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in (
        "ingest-arxiv",
        "ingest-url",
        "ingest-pdf",
        "ingest-text",
        "seed",
        "delete",
    ):
        sp = sub.add_parser(name)
        sp.add_argument("value")
    qp = sub.add_parser("query")
    qp.add_argument("value")
    qp.add_argument("-k", type=int, default=5)
    sub.add_parser("list")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
