"""CorpusStore — filesystem persistence for ingested knowledge + vectors.

Layout under ``QM_CORPUS_DIR`` (default ``~/.quantmind/corpus``)::

    items/<id>.json     one record per ingested item (metadata + full Paper)
    vectors/<id>.npy    aligned embedding vector (float32)

Item ``id`` is a stable hash of the source, so re-ingesting the same arXiv
id / URL / file is idempotent (dedup). The store has no global index file to
corrupt: listing globs ``items/``, and search loads the vectors on demand.
This is deliberately simple and good for hundreds–low-thousands of items.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from qm_mcp.config import corpus_dir


def make_id(source_type: str, source: str) -> str:
    """Stable dedup id for a source (sha1 of ``type:source``, 16 hex chars)."""
    digest = hashlib.sha1(f"{source_type}:{source}".encode("utf-8")).hexdigest()
    return digest[:16]


class CorpusStore:
    """Filesystem-backed corpus of extracted knowledge + embeddings."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or corpus_dir()
        self.items_dir = self.root / "items"
        self.vectors_dir = self.root / "vectors"
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    # ── paths ──────────────────────────────────────────────────────────
    def _item_path(self, item_id: str) -> Path:
        return self.items_dir / f"{item_id}.json"

    def _vec_path(self, item_id: str) -> Path:
        return self.vectors_dir / f"{item_id}.npy"

    # ── writes ─────────────────────────────────────────────────────────
    def exists(self, item_id: str) -> bool:
        return self._item_path(item_id).is_file()

    def add(self, record: dict[str, Any], vector: np.ndarray) -> None:
        """Persist one record + its embedding vector atomically-ish."""
        item_id = record["id"]
        tmp = self._item_path(item_id).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        tmp.replace(self._item_path(item_id))
        np.save(self._vec_path(item_id), vector.astype(np.float32))

    def delete(self, item_id: str) -> bool:
        removed = False
        for p in (self._item_path(item_id), self._vec_path(item_id)):
            if p.is_file():
                p.unlink()
                removed = True
        return removed

    # ── reads ──────────────────────────────────────────────────────────
    def get(self, item_id: str) -> dict[str, Any] | None:
        p = self._item_path(item_id)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list_records(self, *, light: bool = True) -> list[dict[str, Any]]:
        """All records. ``light=True`` drops the heavy ``paper`` tree."""
        out: list[dict[str, Any]] = []
        for p in sorted(self.items_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if light:
                rec = {
                    k: v
                    for k, v in rec.items()
                    if k not in ("paper", "full_context")
                }
            out.append(rec)
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self.items_dir.glob("*.json"))

    # ── search ─────────────────────────────────────────────────────────
    def search(
        self, query_vec: np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """Cosine top-k. Returns [(id, score)] sorted desc."""
        ids: list[str] = []
        mats: list[np.ndarray] = []
        for vp in self.vectors_dir.glob("*.npy"):
            ids.append(vp.stem)
            mats.append(np.load(vp))
        if not ids:
            return []
        matrix = np.vstack(mats).astype(np.float32)
        q = query_vec.astype(np.float32)
        denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)) + 1e-9
        scores = (matrix @ q) / denom
        order = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in order]
