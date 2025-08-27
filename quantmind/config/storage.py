"""Storage configuration models for QuantMind."""

from pathlib import Path

from pydantic import BaseModel, Field


class BaseStorageConfig(BaseModel):
    """Base configuration for all storage types."""

    storage_dir: Path = Field(
        default=Path("./data"), description="Base storage directory"
    )

    download_timeout: int = Field(
        default=30, description="Timeout in seconds for downloading files"
    )


class LocalStorageConfig(BaseStorageConfig):
    """Configuration for local file-based storage."""

    def model_post_init(self, __context):
        """Ensure storage directory exists."""
        self.storage_dir = Path(self.storage_dir).expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.storage_dir / "raw_files").mkdir(exist_ok=True)
        (self.storage_dir / "knowledges").mkdir(exist_ok=True)
        (self.storage_dir / "embeddings").mkdir(exist_ok=True)
        (self.storage_dir / "extra").mkdir(exist_ok=True)

    @property
    def raw_files_dir(self) -> Path:
        """Directory for raw files (PDFs, etc.)."""
        return self.storage_dir / "raw_files"

    @property
    def knowledges_dir(self) -> Path:
        """Directory for knowledge JSONs."""
        return self.storage_dir / "knowledges"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for embedding arrays."""
        return self.storage_dir / "embeddings"

    @property
    def extra_dir(self) -> Path:
        """Directory for extra data."""
        return self.storage_dir / "extra"
