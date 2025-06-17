"""Task definitions for the QuantMind workflow system."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from quantmind.models.paper import Paper
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Task(ABC):
    """Abstract base class for workflow tasks.

    Represents a single unit of work in the QuantMind pipeline,
    such as crawling, parsing, tagging, or storage operations.
    """

    def __init__(
        self,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize task.

        Args:
            task_id: Unique task identifier
            priority: Task priority level
            config: Task-specific configuration
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.config = config or {}
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result: Any = None
        self.progress = 0.0  # 0.0 to 1.0

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the task.

        Args:
            context: Execution context with shared data

        Returns:
            Task execution result

        Raises:
            Exception: If task execution fails
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate task configuration and requirements.

        Returns:
            True if task is valid and can be executed
        """
        pass

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()
        logger.info(f"Task {self.task_id} started")

    def complete(self, result: Any = None) -> None:
        """Mark task as completed.

        Args:
            result: Task execution result
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress = 1.0
        logger.info(f"Task {self.task_id} completed")

    def fail(self, error_message: str) -> None:
        """Mark task as failed.

        Args:
            error_message: Error description
        """
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        logger.error(f"Task {self.task_id} failed: {error_message}")

    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        logger.info(f"Task {self.task_id} cancelled")

    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds.

        Returns:
            Duration in seconds, None if not completed
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation.

        Returns:
            Dictionary with task information
        """
        return {
            "task_id": self.task_id,
            "type": self.__class__.__name__,
            "priority": self.priority.name,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat()
            if self.started_at
            else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration": self.get_duration(),
            "error_message": self.error_message,
            "config": self.config,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.task_id}, {self.status.value})"


class CrawlTask(Task):
    """Task for crawling papers from a source."""

    def __init__(
        self, source_name: str, query: str, max_results: int = 10, **kwargs
    ):
        """Initialize crawl task.

        Args:
            source_name: Name of the source to crawl
            query: Search query
            max_results: Maximum number of papers to fetch
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.source_name = source_name
        self.query = query
        self.max_results = max_results
        self.config.update(
            {
                "source_name": source_name,
                "query": query,
                "max_results": max_results,
            }
        )

    def execute(self, context: Dict[str, Any]) -> List[Paper]:
        """Execute paper crawling.

        Args:
            context: Context containing source instances

        Returns:
            List of crawled papers

        Raises:
            Exception: If source not available or crawling fails
        """
        sources = context.get("sources", {})
        if self.source_name not in sources:
            raise Exception(f"Source '{self.source_name}' not available")

        source = sources[self.source_name]
        papers = source.search(self.query, self.max_results)

        return papers

    def validate(self) -> bool:
        """Validate crawl task configuration."""
        return (
            bool(self.source_name) and bool(self.query) and self.max_results > 0
        )


class ParseTask(Task):
    """Task for parsing paper content."""

    def __init__(self, parser_name: str, papers: List[Paper], **kwargs):
        """Initialize parse task.

        Args:
            parser_name: Name of the parser to use
            papers: Papers to parse
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.parser_name = parser_name
        self.papers = papers
        self.config.update(
            {"parser_name": parser_name, "paper_count": len(papers)}
        )

    def execute(self, context: Dict[str, Any]) -> List[Paper]:
        """Execute paper parsing.

        Args:
            context: Context containing parser instances

        Returns:
            List of parsed papers

        Raises:
            Exception: If parser not available or parsing fails
        """
        parsers = context.get("parsers", {})
        if self.parser_name not in parsers:
            raise Exception(f"Parser '{self.parser_name}' not available")

        parser = parsers[self.parser_name]
        parsed_papers = []

        for i, paper in enumerate(self.papers):
            try:
                parsed_paper = parser.parse_paper(paper)
                parsed_papers.append(parsed_paper)
                self.progress = (i + 1) / len(self.papers)
            except Exception as e:
                logger.warning(
                    f"Failed to parse paper {paper.get_primary_id()}: {e}"
                )
                parsed_papers.append(paper)  # Keep original if parsing fails

        return parsed_papers

    def validate(self) -> bool:
        """Validate parse task configuration."""
        return bool(self.parser_name) and len(self.papers) > 0


class TagTask(Task):
    """Task for tagging papers."""

    def __init__(self, tagger_name: str, papers: List[Paper], **kwargs):
        """Initialize tag task.

        Args:
            tagger_name: Name of the tagger to use
            papers: Papers to tag
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.tagger_name = tagger_name
        self.papers = papers
        self.config.update(
            {"tagger_name": tagger_name, "paper_count": len(papers)}
        )

    def execute(self, context: Dict[str, Any]) -> List[Paper]:
        """Execute paper tagging.

        Args:
            context: Context containing tagger instances

        Returns:
            List of tagged papers

        Raises:
            Exception: If tagger not available or tagging fails
        """
        taggers = context.get("taggers", {})
        if self.tagger_name not in taggers:
            raise Exception(f"Tagger '{self.tagger_name}' not available")

        tagger = taggers[self.tagger_name]
        tagged_papers = []

        for i, paper in enumerate(self.papers):
            try:
                tagged_paper = tagger.tag_paper(paper)
                tagged_papers.append(tagged_paper)
                self.progress = (i + 1) / len(self.papers)
            except Exception as e:
                logger.warning(
                    f"Failed to tag paper {paper.get_primary_id()}: {e}"
                )
                tagged_papers.append(paper)  # Keep original if tagging fails

        return tagged_papers

    def validate(self) -> bool:
        """Validate tag task configuration."""
        return bool(self.tagger_name) and len(self.papers) > 0


class StoreTask(Task):
    """Task for storing papers in the knowledge base."""

    def __init__(self, storage_name: str, papers: List[Paper], **kwargs):
        """Initialize store task.

        Args:
            storage_name: Name of the storage backend to use
            papers: Papers to store
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.storage_name = storage_name
        self.papers = papers
        self.config.update(
            {"storage_name": storage_name, "paper_count": len(papers)}
        )

    def execute(self, context: Dict[str, Any]) -> List[str]:
        """Execute paper storage.

        Args:
            context: Context containing storage instances

        Returns:
            List of stored paper IDs

        Raises:
            Exception: If storage not available or storing fails
        """
        storages = context.get("storages", {})
        if self.storage_name not in storages:
            raise Exception(f"Storage '{self.storage_name}' not available")

        storage = storages[self.storage_name]
        stored_ids = []

        for i, paper in enumerate(self.papers):
            try:
                paper_id = storage.store_paper(paper)
                stored_ids.append(paper_id)
                self.progress = (i + 1) / len(self.papers)
            except Exception as e:
                logger.warning(
                    f"Failed to store paper {paper.get_primary_id()}: {e}"
                )

        return stored_ids

    def validate(self) -> bool:
        """Validate store task configuration."""
        return bool(self.storage_name) and len(self.papers) > 0
