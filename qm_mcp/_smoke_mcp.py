"""Standalone MCP stdio smoke test: spawn the server, list tools, list corpus.

Run under the QuantMind venv:
    python -m qm_mcp._smoke_mcp
"""

from __future__ import annotations

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=os.sys.executable,
        args=["-m", "qm_mcp.server"],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await session.call_tool("qm_list_corpus", {})
            print("LIST_CORPUS:", res.content[0].text[:400])


if __name__ == "__main__":
    asyncio.run(main())
