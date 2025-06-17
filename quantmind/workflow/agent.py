"""Workflow agent for orchestrating QuantMind knowledge extraction pipeline."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from quantmind.workflow.pipeline import Pipeline
from quantmind.workflow.tasks import (
    Task,
    CrawlTask,
    ParseTask,
    TagTask,
    StoreTask,
    TaskPriority,
    TaskStatus,
)
from quantmind.models.paper import Paper
from quantmind.sources.base import BaseSource
from quantmind.tagger.base import BaseTagger
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowAgent:
    """Main orchestration agent for QuantMind knowledge extraction.

    Coordinates the entire Stage 1 pipeline: Source APIs → Parser → Tagger → Storage
    with workflow management, quality control, and error handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize workflow agent.

        Args:
            config: Configuration dictionary with:
                - max_workers: Maximum concurrent workers (default: 4)
                - retry_attempts: Number of retry attempts (default: 3)
                - timeout: Task timeout in seconds (default: 300)
                - enable_deduplication: Enable paper deduplication (default: True)
                - quality_threshold: Quality threshold for filtering (default: 0.5)
        """
        self.config = config or {}
        self.max_workers = self.config.get("max_workers", 4)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.timeout = self.config.get("timeout", 300)
        self.enable_deduplication = self.config.get(
            "enable_deduplication", True
        )
        self.quality_threshold = self.config.get("quality_threshold", 0.5)

        # Component registries
        self.sources: Dict[str, BaseSource] = {}
        self.parsers: Dict[
            str, Any
        ] = {}  # Will be defined when parsers are created
        self.taggers: Dict[str, BaseTagger] = {}
        self.storages: Dict[
            str, Any
        ] = {}  # Will be defined when storage is created

        # Pipeline tracking
        self.pipelines: Dict[str, Pipeline] = {}
        self.execution_history: List[Dict[str, Any]] = []

        # Quality control
        self.seen_papers: set = set()  # For deduplication
        self.quality_filters: List[callable] = []

        logger.info("WorkflowAgent initialized")

    def register_source(self, name: str, source: BaseSource) -> "WorkflowAgent":
        """Register a content source.

        Args:
            name: Source name
            source: Source instance

        Returns:
            Self for method chaining
        """
        self.sources[name] = source
        logger.info(f"Registered source: {name}")
        return self

    def register_parser(self, name: str, parser: Any) -> "WorkflowAgent":
        """Register a content parser.

        Args:
            name: Parser name
            parser: Parser instance

        Returns:
            Self for method chaining
        """
        self.parsers[name] = parser
        logger.info(f"Registered parser: {name}")
        return self

    def register_tagger(self, name: str, tagger: BaseTagger) -> "WorkflowAgent":
        """Register a content tagger.

        Args:
            name: Tagger name
            tagger: Tagger instance

        Returns:
            Self for method chaining
        """
        self.taggers[name] = tagger
        logger.info(f"Registered tagger: {name}")
        return self

    def register_storage(self, name: str, storage: Any) -> "WorkflowAgent":
        """Register a storage backend.

        Args:
            name: Storage name
            storage: Storage instance

        Returns:
            Self for method chaining
        """
        self.storages[name] = storage
        logger.info(f"Registered storage: {name}")
        return self

    def create_extraction_pipeline(
        self,
        name: str,
        source_name: str,
        query: str,
        max_papers: int = 50,
        parser_name: Optional[str] = None,
        tagger_name: Optional[str] = None,
        storage_name: Optional[str] = None,
    ) -> Pipeline:
        """Create a complete knowledge extraction pipeline.

        Args:
            name: Pipeline name
            source_name: Name of registered source
            query: Search query
            max_papers: Maximum papers to process
            parser_name: Optional parser name
            tagger_name: Optional tagger name
            storage_name: Optional storage name

        Returns:
            Configured pipeline

        Raises:
            ValueError: If required components are not registered
        """
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not registered")

        pipeline = Pipeline(
            name=name,
            max_workers=self.max_workers,
            retry_attempts=self.retry_attempts,
            timeout=self.timeout,
        )

        # Add component context
        pipeline.add_context("sources", self.sources)
        pipeline.add_context("parsers", self.parsers)
        pipeline.add_context("taggers", self.taggers)
        pipeline.add_context("storages", self.storages)
        pipeline.add_context("agent", self)

        # Create crawl task
        crawl_task = CrawlTask(
            source_name=source_name,
            query=query,
            max_results=max_papers,
            priority=TaskPriority.HIGH,
        )
        pipeline.add_task(crawl_task)

        # Create parse task if parser specified
        if parser_name:
            if parser_name not in self.parsers:
                raise ValueError(f"Parser '{parser_name}' not registered")

            parse_task = ParseTask(
                parser_name=parser_name,
                papers=[],  # Will be populated by crawl task
                priority=TaskPriority.MEDIUM,
            )
            pipeline.add_task(parse_task, dependencies=[crawl_task.task_id])

        # Create tag task if tagger specified
        if tagger_name:
            if tagger_name not in self.taggers:
                raise ValueError(f"Tagger '{tagger_name}' not registered")

            tag_task = TagTask(
                tagger_name=tagger_name,
                papers=[],  # Will be populated by previous task
                priority=TaskPriority.MEDIUM,
            )

            # Determine dependencies
            deps = [parse_task.task_id] if parser_name else [crawl_task.task_id]
            pipeline.add_task(tag_task, dependencies=deps)

        # Create store task if storage specified
        if storage_name:
            if storage_name not in self.storages:
                raise ValueError(f"Storage '{storage_name}' not registered")

            store_task = StoreTask(
                storage_name=storage_name,
                papers=[],  # Will be populated by previous task
                priority=TaskPriority.LOW,
            )

            # Determine dependencies
            if tagger_name:
                deps = [tag_task.task_id]
            elif parser_name:
                deps = [parse_task.task_id]
            else:
                deps = [crawl_task.task_id]

            pipeline.add_task(store_task, dependencies=deps)

        # Register event handlers
        pipeline.on_task_complete = self._on_task_complete
        pipeline.on_task_fail = self._on_task_fail
        pipeline.on_pipeline_complete = self._on_pipeline_complete

        self.pipelines[name] = pipeline
        logger.info(f"Created extraction pipeline: {name}")

        return pipeline

    def execute_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Execute a registered pipeline.

        Args:
            pipeline_name: Name of pipeline to execute

        Returns:
            Execution results

        Raises:
            ValueError: If pipeline not found
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        pipeline = self.pipelines[pipeline_name]

        logger.info(f"Executing pipeline: {pipeline_name}")
        start_time = datetime.utcnow()

        try:
            results = pipeline.execute()

            # Log execution summary
            execution_summary = {
                "pipeline_name": pipeline_name,
                "status": "completed",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "results": results,
                "stats": pipeline.get_pipeline_stats(),
            }

            self.execution_history.append(execution_summary)
            logger.info(f"Pipeline {pipeline_name} completed successfully")

            return results

        except Exception as e:
            # Log execution failure
            execution_summary = {
                "pipeline_name": pipeline_name,
                "status": "failed",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
                "stats": pipeline.get_pipeline_stats(),
            }

            self.execution_history.append(execution_summary)
            logger.error(f"Pipeline {pipeline_name} failed: {e}")

            raise

    def run_quick_extraction(
        self,
        source_name: str,
        query: str,
        max_papers: int = 10,
        tagger_name: Optional[str] = None,
    ) -> List[Paper]:
        """Run a quick extraction without full pipeline setup.

        Args:
            source_name: Name of registered source
            query: Search query
            max_papers: Maximum papers to fetch
            tagger_name: Optional tagger name

        Returns:
            List of processed papers
        """
        pipeline_name = (
            f"quick_extraction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        pipeline = self.create_extraction_pipeline(
            name=pipeline_name,
            source_name=source_name,
            query=query,
            max_papers=max_papers,
            tagger_name=tagger_name,
        )

        results = self.execute_pipeline(pipeline_name)

        # Extract papers from results
        papers = []
        for task_result in results.values():
            if isinstance(task_result, list) and len(task_result) > 0:
                if isinstance(task_result[0], Paper):
                    papers = task_result
                    break

        # Apply quality control
        papers = self.apply_quality_control(papers)

        return papers

    def apply_quality_control(self, papers: List[Paper]) -> List[Paper]:
        """Apply quality control filters to papers.

        Args:
            papers: List of papers to filter

        Returns:
            Filtered list of papers
        """
        filtered_papers = []

        for paper in papers:
            # Deduplication
            if self.enable_deduplication:
                paper_key = self._get_paper_key(paper)
                if paper_key in self.seen_papers:
                    logger.debug(
                        f"Skipping duplicate paper: {paper.get_primary_id()}"
                    )
                    continue
                self.seen_papers.add(paper_key)

            # Apply quality filters
            if self._passes_quality_filters(paper):
                filtered_papers.append(paper)
            else:
                logger.debug(
                    f"Paper failed quality filters: {paper.get_primary_id()}"
                )

        logger.info(
            f"Quality control: {len(papers)} -> {len(filtered_papers)} papers"
        )
        return filtered_papers

    def add_quality_filter(self, filter_func: callable) -> "WorkflowAgent":
        """Add a quality filter function.

        Args:
            filter_func: Function that takes a Paper and returns bool

        Returns:
            Self for method chaining
        """
        self.quality_filters.append(filter_func)
        return self

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history.

        Returns:
            List of execution summaries
        """
        return self.execution_history.copy()

    def get_pipeline_status(
        self, pipeline_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a specific pipeline.

        Args:
            pipeline_name: Name of pipeline

        Returns:
            Pipeline status information
        """
        if pipeline_name not in self.pipelines:
            return None

        pipeline = self.pipelines[pipeline_name]
        return pipeline.get_pipeline_stats()

    def _get_paper_key(self, paper: Paper) -> str:
        """Get unique key for paper deduplication.

        Args:
            paper: Paper object

        Returns:
            Unique key string
        """
        # Use arXiv ID if available, otherwise use title hash
        if paper.arxiv_id:
            return f"arxiv:{paper.arxiv_id}"
        elif paper.doi:
            return f"doi:{paper.doi}"
        else:
            return f"title:{hash(paper.title.lower())}"

    def _passes_quality_filters(self, paper: Paper) -> bool:
        """Check if paper passes all quality filters.

        Args:
            paper: Paper to check

        Returns:
            True if paper passes all filters
        """
        # Basic quality checks
        if not paper.title or len(paper.title.strip()) < 10:
            return False

        if not paper.abstract or len(paper.abstract.strip()) < 50:
            return False

        # Apply custom filters
        for filter_func in self.quality_filters:
            try:
                if not filter_func(paper):
                    return False
            except Exception as e:
                logger.warning(f"Quality filter error: {e}")
                return False

        return True

    def _on_task_complete(self, task: Task, result: Any):
        """Handle task completion event.

        Args:
            task: Completed task
            result: Task result
        """
        logger.info(
            f"Task completed: {task.task_id} ({task.__class__.__name__})"
        )

        # Update subsequent tasks with results
        if isinstance(task, CrawlTask) and isinstance(result, list):
            # Update parse/tag tasks with crawled papers
            pipeline = next(
                (p for p in self.pipelines.values() if task in p.tasks), None
            )
            if pipeline:
                for other_task in pipeline.tasks:
                    if isinstance(other_task, (ParseTask, TagTask, StoreTask)):
                        if (
                            hasattr(other_task, "papers")
                            and not other_task.papers
                        ):
                            other_task.papers = result

    def _on_task_fail(self, task: Task, error: str):
        """Handle task failure event.

        Args:
            task: Failed task
            error: Error message
        """
        logger.error(
            f"Task failed: {task.task_id} ({task.__class__.__name__}): {error}"
        )

    def _on_pipeline_complete(self, pipeline: Pipeline):
        """Handle pipeline completion event.

        Args:
            pipeline: Completed pipeline
        """
        logger.info(f"Pipeline completed: {pipeline.name}")
        stats = pipeline.get_pipeline_stats()
        logger.info(f"Pipeline stats: {json.dumps(stats, indent=2)}")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"WorkflowAgent(sources={len(self.sources)}, "
            f"parsers={len(self.parsers)}, "
            f"taggers={len(self.taggers)}, "
            f"storages={len(self.storages)})"
        )
