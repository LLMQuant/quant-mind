"""PDF -> markdown via PyMuPDF.

PR4 ships a single deterministic engine (``pymupdf``). High-quality
markdown engines (``marker-pdf``, ``llama-parse``) arrive as opt-in
``engine`` arguments in follow-up issues.

The actual fitz call is CPU-bound, so it runs through
:func:`asyncio.to_thread` to keep the event loop responsive when several
papers are being processed concurrently.
"""

import asyncio

import pymupdf


class PdfParseError(ValueError):
    """Raised when PyMuPDF refuses to open the byte stream."""


def _extract_text_sync(pdf_bytes: bytes) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfParseError(f"pymupdf could not open pdf bytes: {exc}") from exc

    try:
        page_texts: list[str] = []
        for page in doc:
            # pymupdf's stubs vary across versions: older releases miss
            # Page.get_text entirely; newer ones type the optional ``option``
            # arg as a Literal union that fans out the return into
            # ``str | list | dict``. Both cases need narrowing here.
            extracted = page.get_text()  # pyright: ignore[reportAttributeAccessIssue]
            text = extracted if isinstance(extracted, str) else ""
            if text.strip():
                page_texts.append(text)
    finally:
        doc.close()

    return "\n\n".join(page_texts)


async def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to a plain-text/markdown string.

    PyMuPDF returns plain text rather than rich markdown — there is no
    structural tree (headings, tables, math) reconstruction. Downstream
    consumers that need higher-fidelity markdown should wait for the
    marker-pdf engine option (follow-up issue) or pass the raw text to an
    LLM.

    Args:
        pdf_bytes: Raw PDF bytes (e.g. from
            :func:`quantmind.preprocess.fetch.fetch_url` or
            :func:`fetch_arxiv`).

    Returns:
        Concatenated per-page text, separated by blank lines. Empty pages
        are dropped.

    Raises:
        PdfParseError: If PyMuPDF cannot open the byte stream.
    """
    if not pdf_bytes:
        raise PdfParseError("pdf_bytes is empty")
    return await asyncio.to_thread(_extract_text_sync, pdf_bytes)
