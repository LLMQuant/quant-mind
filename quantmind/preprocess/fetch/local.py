"""Local-file fetch helper.

Reads a file from disk, returns a `Fetched` whose ``content_type`` is
inferred from the extension. Async only at the API boundary so flows can
batch local reads without blocking; the actual ``read_bytes`` runs in a
worker thread via :func:`asyncio.to_thread`.
"""

import asyncio
from pathlib import Path

from quantmind.preprocess.fetch._types import Fetched

_SUFFIX_TO_CONTENT_TYPE: dict[str, str] = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".xml": "application/xml",
}


def _infer_content_type(path: Path) -> str:
    return _SUFFIX_TO_CONTENT_TYPE.get(
        path.suffix.lower(), "application/octet-stream"
    )


async def read_local_file(path: str | Path) -> Fetched:
    """Read a file from disk into a ``Fetched`` value.

    Args:
        path: Path to the file. Tilde and relative paths are accepted; both
            are resolved to absolute form.

    Returns:
        ``Fetched`` whose ``bytes`` is the file payload, ``content_type`` is
        inferred from the suffix, and ``source_url`` is the resolved
        absolute path as a ``file://`` URL.

    Raises:
        FileNotFoundError: When the file does not exist.
        IsADirectoryError: When the path is a directory.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"local file not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"path is a directory, not a file: {p}")

    payload = await asyncio.to_thread(p.read_bytes)
    return Fetched(
        bytes=payload,
        content_type=_infer_content_type(p),
        source_url=p.as_uri(),
        headers={},
    )
