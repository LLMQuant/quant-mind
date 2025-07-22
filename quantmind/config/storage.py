"""Storage configuration models for QuantMind."""

from pathlib import Path
from typing import Dict, Any, Union

from pydantic import BaseModel, Field


class BaseStorageConfig(BaseModel):
    """Base configuration for all storage types."""

    storage_dir: str = Field(
        default="./data", description="Base storage directory"
    )


class LocalStorageConfig(BaseStorageConfig):
    """Configuration for local file-based storage."""

    base_dir: Path = Field(
        default=Path("./data"), description="Base storage directory"
    )

    def model_post_init(self, __context):
        """Ensure storage directory exists."""
        self.base_dir = Path(self.base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.base_dir / "raw_files").mkdir(exist_ok=True)
        (self.base_dir / "knowledges").mkdir(exist_ok=True)
        (self.base_dir / "embeddings").mkdir(exist_ok=True)
        (self.base_dir / "extra").mkdir(exist_ok=True)

    @property
    def raw_files_dir(self) -> Path:
        """Directory for raw files (PDFs, etc.)."""
        return self.base_dir / "raw_files"

    @property
    def knowledges_dir(self) -> Path:
        """Directory for knowledge JSONs."""
        return self.base_dir / "knowledges"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for embedding arrays."""
        return self.base_dir / "embeddings"

    @property
    def extra_dir(self) -> Path:
        """Directory for extra data."""
        return self.base_dir / "extra"
