"""Local file-based storage implementation for QuantMind."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from quantmind.config import LocalStorageConfig
from quantmind.models import KnowledgeItem
from quantmind.utils.logger import get_logger

from .base import BaseStorage

logger = get_logger(__name__)


class LocalStorage(BaseStorage):
    """Local file-based storage implementation.

    Organizes data into four directories:
    - raw_files/: Original files (PDFs, markdown, etc.)
    - knowledges/: KnowledgeItem objects as JSON
    - embeddings/: Embedding vectors as JSON arrays
    - extra/: Additional metadata and hashes

    Uses efficient indexing system for fast lookups:
    - raw_files_index.json: file_id -> {"path": "xxx", "extension": "yyy"}
    - knowledges_index.json: knowledge_id -> {"path": "xxx"}
    - embeddings_index.json: knowledge_id -> {"path": "xxx"}
    """

    def __init__(self, config: LocalStorageConfig):
        """Initialize local storage.

        Args:
            config: LocalStorageConfig instance
        """
        self.config = config
        self.config.model_post_init(None)  # Ensure directories exist

        # Initialize indexes
        self._raw_files_index: Dict[str, Dict[str, str]] = {}
        self._knowledges_index: Dict[str, Dict[str, str]] = {}
        self._embeddings_index: Dict[str, Dict[str, str]] = {}

        self._load_indexes()
        logger.info(f"LocalStorage initialized at {self.config.storage_dir}")

    def _get_index_path(self, index_type: str) -> Path:
        """Get path to index file."""
        return self.config.extra_dir / f"{index_type}_index.json"

    def _load_indexes(self) -> None:
        """Load all indexes from disk."""
        self._load_index("raw_files")
        self._load_index("knowledges")
        self._load_index("embeddings")

    def _load_index(self, index_type: str) -> None:
        """Load specific index from disk."""
        index_path = self._get_index_path(index_type)
        try:
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)

                if index_type == "raw_files":
                    self._raw_files_index = index_data
                elif index_type == "knowledges":
                    self._knowledges_index = index_data
                elif index_type == "embeddings":
                    self._embeddings_index = index_data
            else:
                # Build index from existing files if index doesn't exist
                self._rebuild_index(index_type)

        except Exception as e:
            logger.warning(
                f"Failed to load {index_type} index: {e}, rebuilding..."
            )
            self._rebuild_index(index_type)

    def _save_index(self, index_type: str) -> None:
        """Save specific index to disk."""
        index_path = self._get_index_path(index_type)
        try:
            if index_type == "raw_files":
                index_data = self._raw_files_index
            elif index_type == "knowledges":
                index_data = self._knowledges_index
            elif index_type == "embeddings":
                index_data = self._embeddings_index
            else:
                return

            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save {index_type} index: {e}")

    def _rebuild_index(self, index_type: str) -> None:
        """Rebuild index by scanning directory."""
        logger.info(f"Rebuilding {index_type} index...")

        if index_type == "raw_files":
            self._raw_files_index.clear()
            if self.config.raw_files_dir.exists():
                for file_path in self.config.raw_files_dir.iterdir():
                    if file_path.is_file():
                        # Extract file_id from filename (everything before last dot)
                        file_id = file_path.stem
                        self._raw_files_index[file_id] = {
                            "path": str(
                                file_path.relative_to(self.config.storage_dir)
                            ),
                            "extension": file_path.suffix,
                        }

        elif index_type == "knowledges":
            self._knowledges_index.clear()
            if self.config.knowledges_dir.exists():
                for file_path in self.config.knowledges_dir.glob("*.json"):
                    knowledge_id = file_path.stem
                    self._knowledges_index[knowledge_id] = {
                        "path": str(
                            file_path.relative_to(self.config.storage_dir)
                        )
                    }

        elif index_type == "embeddings":
            self._embeddings_index.clear()
            if self.config.embeddings_dir.exists():
                for file_path in self.config.embeddings_dir.glob("*.json"):
                    knowledge_id = file_path.stem
                    self._embeddings_index[knowledge_id] = {
                        "path": str(
                            file_path.relative_to(self.config.storage_dir)
                        )
                    }

        self._save_index(index_type)
        logger.info(
            f"Rebuilt {index_type} index with {len(getattr(self, f'_{index_type}_index'))} entries"
        )

    def rebuild_all_indexes(self) -> None:
        """Rebuild all indexes from scratch."""
        logger.info("Rebuilding all indexes...")
        self._rebuild_index("raw_files")
        self._rebuild_index("knowledges")
        self._rebuild_index("embeddings")
        logger.info("All indexes rebuilt successfully")

    # Raw Files Management
    def store_raw_file(
        self,
        file_id: str,
        file_path: Optional[Path] = None,
        content: Optional[bytes] = None,
        file_extension: str = "",
    ) -> str:
        """Store a raw file by copying or writing content directly."""
        try:
            # Validate input parameters
            if file_path is not None and content is not None:
                raise ValueError("Cannot specify both file_path and content")
            if file_path is None and content is None:
                raise ValueError("Must specify either file_path or content")

            # Determine target path and extension
            if file_path is not None:
                # Copy from existing file
                source_path = Path(file_path)
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Source file not found: {file_path}"
                    )

                target_path = (
                    self.config.raw_files_dir / f"{file_id}{source_path.suffix}"
                )
                extension = source_path.suffix

                # Copy file
                shutil.copy2(source_path, target_path)
                logger.debug(
                    f"Stored raw file {file_id} by copying from {file_path}"
                )

            else:
                # Write content directly
                if not file_extension:
                    file_extension = ".bin"  # Default extension
                if not file_extension.startswith("."):
                    file_extension = f".{file_extension}"

                target_path = (
                    self.config.raw_files_dir / f"{file_id}{file_extension}"
                )
                extension = file_extension

                # Write content to file
                with open(target_path, "wb") as f:
                    f.write(content)
                logger.debug(
                    f"Stored raw file {file_id} by writing content directly"
                )

            # Update index
            self._raw_files_index[file_id] = {
                "path": str(target_path.relative_to(self.config.storage_dir)),
                "extension": extension,
            }
            self._save_index("raw_files")

            return str(target_path)

        except Exception as e:
            logger.error(f"Failed to store raw file {file_id}: {e}")
            raise

    def get_raw_file(self, file_id: str) -> Optional[Path]:
        """Get path to a raw file using efficient index lookup."""
        try:
            # Fast index lookup
            if file_id in self._raw_files_index:
                relative_path = self._raw_files_index[file_id]["path"]
                file_path = self.config.storage_dir / relative_path

                if file_path.exists():
                    return file_path
                else:
                    # File was deleted externally, remove from index
                    logger.warning(
                        f"Raw file {file_id} in index but missing on disk, removing from index"
                    )
                    del self._raw_files_index[file_id]
                    self._save_index("raw_files")
                    return None

            # Fallback to directory scan if not in index
            for file_path in self.config.raw_files_dir.glob(f"{file_id}.*"):
                if file_path.is_file():
                    # Add to index for future lookups
                    self._raw_files_index[file_id] = {
                        "path": str(
                            file_path.relative_to(self.config.storage_dir)
                        ),
                        "extension": file_path.suffix,
                    }
                    self._save_index("raw_files")
                    return file_path

            return None

        except Exception as e:
            logger.error(f"Failed to get raw file {file_id}: {e}")
            return None

    def delete_raw_file(self, file_id: str) -> bool:
        """Delete a raw file and update index."""
        try:
            file_path = self.get_raw_file(file_id)
            if file_path and file_path.exists():
                file_path.unlink()

                # Remove from index
                if file_id in self._raw_files_index:
                    del self._raw_files_index[file_id]
                    self._save_index("raw_files")

                logger.debug(f"Deleted raw file {file_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete raw file {file_id}: {e}")
            return False

    # Knowledge Items Management
    def store_knowledge(self, knowledge: KnowledgeItem) -> str:
        """Store a knowledge item as JSON and update index."""
        try:
            knowledge_id = knowledge.get_primary_id()
            file_path = self.config.knowledges_dir / f"{knowledge_id}.json"

            # Save to JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    knowledge.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            # Update index
            self._knowledges_index[knowledge_id] = {
                "path": str(file_path.relative_to(self.config.storage_dir))
            }
            self._save_index("knowledges")

            logger.debug(f"Stored knowledge {knowledge_id} at {file_path}")
            return knowledge_id

        except Exception as e:
            logger.error(
                f"Failed to store knowledge {knowledge.get_primary_id()}: {e}"
            )
            raise

    def get_knowledge_path(self, knowledge_id: str) -> Optional[Path]:
        """Get the path to a knowledge item by ID using efficient index lookup."""
        if knowledge_id in self._knowledges_index:
            relative_path = self._knowledges_index[knowledge_id]["path"]
            return self.config.storage_dir / relative_path
        return None

    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID using efficient index lookup."""
        try:
            # Fast index lookup
            if knowledge_id in self._knowledges_index:
                relative_path = self._knowledges_index[knowledge_id]["path"]
                file_path = self.config.storage_dir / relative_path

                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return KnowledgeItem(**data)
                else:
                    # File was deleted externally, remove from index
                    logger.warning(
                        f"Knowledge {knowledge_id} in index but missing on disk, removing from index"
                    )
                    del self._knowledges_index[knowledge_id]
                    self._save_index("knowledges")
                    return None

            # Fallback to direct file check
            file_path = self.config.knowledges_dir / f"{knowledge_id}.json"
            if file_path.exists():
                # Add to index for future lookups
                self._knowledges_index[knowledge_id] = {
                    "path": str(file_path.relative_to(self.config.storage_dir))
                }
                self._save_index("knowledges")

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return KnowledgeItem(**data)

            return None

        except Exception as e:
            logger.error(f"Failed to get knowledge {knowledge_id}: {e}")
            return None

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete a knowledge item and update index."""
        try:
            # Check index first for fast lookup
            if knowledge_id in self._knowledges_index:
                relative_path = self._knowledges_index[knowledge_id]["path"]
                file_path = self.config.storage_dir / relative_path
            else:
                file_path = self.config.knowledges_dir / f"{knowledge_id}.json"

            if file_path.exists():
                file_path.unlink()

                # Remove from index
                if knowledge_id in self._knowledges_index:
                    del self._knowledges_index[knowledge_id]
                    self._save_index("knowledges")

                logger.debug(f"Deleted knowledge {knowledge_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete knowledge {knowledge_id}: {e}")
            return False

    # Embeddings Management
    def store_embedding(
        self, knowledge_id: str, embedding: List[float], model: str
    ) -> str:
        """Store an embedding vector and update index."""
        try:
            file_path = self.config.embeddings_dir / f"{knowledge_id}.json"

            embedding_data = {
                "knowledge_id": knowledge_id,
                "embedding": embedding,
                "model": model,
                "created_at": str(Path().stat().st_mtime),  # Simple timestamp
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(embedding_data, f, indent=2)

            # Update index
            self._embeddings_index[knowledge_id] = {
                "path": str(file_path.relative_to(self.config.storage_dir))
            }
            self._save_index("embeddings")

            logger.debug(
                f"Stored embedding for {knowledge_id} (model: {model})"
            )
            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to store embedding for {knowledge_id}: {e}")
            raise

    def get_embedding(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding data for a knowledge item using efficient index lookup."""
        try:
            # Fast index lookup
            if knowledge_id in self._embeddings_index:
                relative_path = self._embeddings_index[knowledge_id]["path"]
                file_path = self.config.storage_dir / relative_path

                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    # File was deleted externally, remove from index
                    logger.warning(
                        f"Embedding {knowledge_id} in index but missing on disk, removing from index"
                    )
                    del self._embeddings_index[knowledge_id]
                    self._save_index("embeddings")
                    return None

            # Fallback to direct file check
            file_path = self.config.embeddings_dir / f"{knowledge_id}.json"
            if file_path.exists():
                # Add to index for future lookups
                self._embeddings_index[knowledge_id] = {
                    "path": str(file_path.relative_to(self.config.storage_dir))
                }
                self._save_index("embeddings")

                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding for {knowledge_id}: {e}")
            return None

    def delete_embedding(self, knowledge_id: str) -> bool:
        """Delete an embedding and update index."""
        try:
            # Check index first for fast lookup
            if knowledge_id in self._embeddings_index:
                relative_path = self._embeddings_index[knowledge_id]["path"]
                file_path = self.config.storage_dir / relative_path
            else:
                file_path = self.config.embeddings_dir / f"{knowledge_id}.json"

            if file_path.exists():
                file_path.unlink()

                # Remove from index
                if knowledge_id in self._embeddings_index:
                    del self._embeddings_index[knowledge_id]
                    self._save_index("embeddings")

                logger.debug(f"Deleted embedding for {knowledge_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete embedding for {knowledge_id}: {e}")
            return False

    # Extra Data Management
    def store_extra(self, key: str, data: Any) -> str:
        """Store extra data (hashes, metadata, etc.)."""
        try:
            file_path = self.config.extra_dir / f"{key}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.debug(f"Stored extra data for key: {key}")
            return key

        except Exception as e:
            logger.error(f"Failed to store extra data for {key}: {e}")
            raise

    def get_extra(self, key: str) -> Optional[Any]:
        """Get extra data by key."""
        try:
            file_path = self.config.extra_dir / f"{key}.json"
            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to get extra data for {key}: {e}")
            return None

    def delete_extra(self, key: str) -> bool:
        """Delete extra data."""
        try:
            file_path = self.config.extra_dir / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted extra data for key: {key}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete extra data for {key}: {e}")
            return False

    # Utility Methods
    def get_all_knowledges(self) -> Iterator[KnowledgeItem]:
        """Get all knowledge items using efficient index."""
        for knowledge_id in self._knowledges_index.keys():
            knowledge = self.get_knowledge(knowledge_id)
            if knowledge:
                yield knowledge

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information including index statistics."""
        return {
            "type": self.__class__.__name__,
            "config": self.config.model_dump(),
            "storage_dir": str(self.config.storage_dir),
            "knowledge_count": len(self._knowledges_index),
            "raw_files_count": len(self._raw_files_index),
            "embeddings_count": len(self._embeddings_index),
            "indexes": {
                "raw_files": {
                    "entries": len(self._raw_files_index),
                    "index_file": str(self._get_index_path("raw_files")),
                },
                "knowledges": {
                    "entries": len(self._knowledges_index),
                    "index_file": str(self._get_index_path("knowledges")),
                },
                "embeddings": {
                    "entries": len(self._embeddings_index),
                    "index_file": str(self._get_index_path("embeddings")),
                },
            },
        }
