"""JSON-based storage implementation for QuantMind knowledge base."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime

from quantmind.models.paper import Paper
from quantmind.storage.base import BaseStorage
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class JSONStorage(BaseStorage):
    """JSON file-based storage for papers.

    Stores papers as individual JSON files in a directory structure
    organized by categories and dates. Includes indexing for fast searches.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize JSON storage.

        Args:
            config: Configuration dictionary with:
                - storage_dir: Base directory for storage (default: ./data)
                - auto_backup: Enable automatic backups (default: True)
                - max_backup_count: Maximum number of backups to keep (default: 5)
                - index_file: Path to search index file (default: storage_dir/index.json)
        """
        super().__init__(config)

        self.storage_dir = Path(self.config.get("storage_dir", "./data"))
        self.auto_backup = self.config.get("auto_backup", True)
        self.max_backup_count = self.config.get("max_backup_count", 5)
        self.index_file = Path(
            self.config.get("index_file", self.storage_dir / "index.json")
        )

        # Create directory structure
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir = self.storage_dir / "papers"
        self.papers_dir.mkdir(exist_ok=True)

        # Initialize index
        self.index = self._load_index()

        logger.info(f"JSONStorage initialized at {self.storage_dir}")

    def store_paper(self, paper: Paper) -> str:
        """Store a paper as a JSON file.

        Args:
            paper: Paper object to store

        Returns:
            Unique identifier for the stored paper
        """
        try:
            paper_id = paper.get_primary_id()

            # Create file path based on date and category
            file_path = self._get_paper_file_path(paper)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save paper to JSON file
            paper.save_to_file(file_path)

            # Update index
            self._update_index(paper, str(file_path))

            logger.debug(f"Stored paper {paper_id} at {file_path}")
            return paper_id

        except Exception as e:
            logger.error(f"Failed to store paper {paper.get_primary_id()}: {e}")
            raise

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by its ID.

        Args:
            paper_id: Unique identifier of the paper

        Returns:
            Paper object if found, None otherwise
        """
        try:
            # Look up file path in index
            file_path = (
                self.index.get("papers", {}).get(paper_id, {}).get("file_path")
            )
            if not file_path or not os.path.exists(file_path):
                return None

            # Load paper from file
            paper = Paper.load_from_file(file_path)
            return paper

        except Exception as e:
            logger.error(f"Failed to retrieve paper {paper_id}: {e}")
            return None

    def update_paper(self, paper: Paper) -> bool:
        """Update an existing paper.

        Args:
            paper: Paper object with updated data

        Returns:
            True if update was successful
        """
        try:
            # Store will overwrite existing file
            self.store_paper(paper)
            return True
        except Exception as e:
            logger.error(
                f"Failed to update paper {paper.get_primary_id()}: {e}"
            )
            return False

    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from storage.

        Args:
            paper_id: Unique identifier of the paper to delete

        Returns:
            True if deletion was successful
        """
        try:
            # Get file path from index
            paper_info = self.index.get("papers", {}).get(paper_id)
            if not paper_info:
                return False

            file_path = paper_info.get("file_path")
            if file_path and os.path.exists(file_path):
                # Backup before deletion if enabled
                if self.auto_backup:
                    self._backup_file(file_path)

                # Delete file
                os.unlink(file_path)

            # Remove from index
            del self.index["papers"][paper_id]
            self._save_index()

            logger.debug(f"Deleted paper {paper_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            return False

    def search_papers(
        self,
        query: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Paper]:
        """Search for papers matching criteria.

        Args:
            query: Text query to search in title/abstract
            categories: List of categories to filter by
            tags: List of tags to filter by
            limit: Maximum number of results

        Returns:
            List of matching Paper objects
        """
        matching_papers = []
        papers_checked = 0

        for paper_id, paper_info in self.index.get("papers", {}).items():
            if limit and len(matching_papers) >= limit:
                break

            papers_checked += 1

            # Check category filter
            if categories:
                paper_categories = paper_info.get("categories", [])
                if not any(cat in paper_categories for cat in categories):
                    continue

            # Check tag filter
            if tags:
                paper_tags = paper_info.get("tags", [])
                if not any(tag in paper_tags for tag in tags):
                    continue

            # Check text query
            if query:
                title = paper_info.get("title", "").lower()
                abstract = paper_info.get("abstract", "").lower()
                query_lower = query.lower()
                if query_lower not in title and query_lower not in abstract:
                    continue

            # Load full paper
            paper = self.get_paper(paper_id)
            if paper:
                matching_papers.append(paper)

        logger.debug(
            f"Search found {len(matching_papers)} papers (checked {papers_checked})"
        )
        return matching_papers

    def get_all_papers(self) -> Iterator[Paper]:
        """Get all papers in storage.

        Returns:
            Iterator over all Paper objects
        """
        for paper_id in self.index.get("papers", {}):
            paper = self.get_paper(paper_id)
            if paper:
                yield paper

    def count_papers(
        self,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Count papers matching criteria.

        Args:
            categories: Optional list of categories to filter by
            tags: Optional list of tags to filter by

        Returns:
            Number of matching papers
        """
        if not categories and not tags:
            return len(self.index.get("papers", {}))

        count = 0
        for paper_info in self.index.get("papers", {}).values():
            # Check category filter
            if categories:
                paper_categories = paper_info.get("categories", [])
                if not any(cat in paper_categories for cat in categories):
                    continue

            # Check tag filter
            if tags:
                paper_tags = paper_info.get("tags", [])
                if not any(tag in paper_tags for tag in tags):
                    continue

            count += 1

        return count

    def get_categories(self) -> List[str]:
        """Get all unique categories in storage.

        Returns:
            List of category strings
        """
        categories = set()
        for paper_info in self.index.get("papers", {}).values():
            categories.update(paper_info.get("categories", []))
        return sorted(list(categories))

    def get_tags(self) -> List[str]:
        """Get all unique tags in storage.

        Returns:
            List of tag strings
        """
        tags = set()
        for paper_info in self.index.get("papers", {}).values():
            tags.update(paper_info.get("tags", []))
        return sorted(list(tags))

    def _get_paper_file_path(self, paper: Paper) -> Path:
        """Get file path for storing a paper.

        Args:
            paper: Paper object

        Returns:
            Path object for the paper file
        """
        # Use published date if available, otherwise current date
        if paper.published_date:
            date_str = paper.published_date.strftime("%Y-%m")
        else:
            date_str = datetime.now().strftime("%Y-%m")

        # Use primary category if available
        category = "uncategorized"
        if paper.categories:
            category = paper.categories[0].replace(" ", "_").replace("/", "_")

        # Create filename from paper ID
        paper_id = paper.get_primary_id()
        filename = f"{paper_id}.json"

        return self.papers_dir / date_str / category / filename

    def _load_index(self) -> Dict[str, Any]:
        """Load the search index.

        Returns:
            Index dictionary
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

        # Return empty index
        return {
            "created_at": datetime.utcnow().isoformat(),
            "papers": {},
            "statistics": {},
        }

    def _save_index(self):
        """Save the search index."""
        try:
            self.index["updated_at"] = datetime.utcnow().isoformat()
            self.index["statistics"] = {
                "total_papers": len(self.index.get("papers", {})),
                "categories": len(self.get_categories()),
                "tags": len(self.get_tags()),
            }

            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _update_index(self, paper: Paper, file_path: str):
        """Update the search index with paper information.

        Args:
            paper: Paper object
            file_path: Path to paper file
        """
        paper_id = paper.get_primary_id()

        # Store searchable information in index
        self.index.setdefault("papers", {})[paper_id] = {
            "file_path": file_path,
            "title": paper.title,
            "abstract": paper.abstract[
                :500
            ],  # Store first 500 chars for search
            "categories": paper.categories,
            "tags": paper.tags,
            "published_date": paper.published_date.isoformat()
            if paper.published_date
            else None,
            "stored_at": datetime.utcnow().isoformat(),
        }

        self._save_index()

    def _backup_file(self, file_path: str):
        """Create a backup of a file before deletion.

        Args:
            file_path: Path to file to backup
        """
        try:
            backup_dir = self.storage_dir / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Create backup with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{Path(file_path).stem}_{timestamp}.json"
            backup_path = backup_dir / backup_name

            # Copy file
            import shutil

            shutil.copy2(file_path, backup_path)

            # Clean old backups
            self._cleanup_backups()

        except Exception as e:
            logger.warning(f"Failed to backup file {file_path}: {e}")

    def _cleanup_backups(self):
        """Remove old backup files."""
        try:
            backup_dir = self.storage_dir / "backups"
            if not backup_dir.exists():
                return

            # Get all backup files sorted by modification time
            backups = sorted(
                backup_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Remove excess backups
            for backup in backups[self.max_backup_count :]:
                backup.unlink()

        except Exception as e:
            logger.warning(f"Failed to cleanup backups: {e}")

    def rebuild_index(self):
        """Rebuild the search index from stored files."""
        logger.info("Rebuilding search index...")

        self.index = {
            "created_at": datetime.utcnow().isoformat(),
            "papers": {},
            "statistics": {},
        }

        # Scan all JSON files
        for json_file in self.papers_dir.rglob("*.json"):
            try:
                paper = Paper.load_from_file(json_file)
                self._update_index(paper, str(json_file))
            except Exception as e:
                logger.warning(f"Failed to index {json_file}: {e}")

        logger.info(f"Index rebuilt with {len(self.index['papers'])} papers")
