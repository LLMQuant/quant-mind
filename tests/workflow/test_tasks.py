"""Tests for workflow tasks."""

import pytest
from unittest.mock import Mock, MagicMock
from quantmind.models.paper import Paper
from quantmind.workflow.tasks import (
    Task,
    CrawlTask,
    ParseTask,
    TagTask,
    StoreTask,
    TaskStatus,
    TaskPriority,
)


class MockTask(Task):
    """Mock task for testing."""

    def __init__(self, should_fail=False, **kwargs):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self.executed = False

    def execute(self, context):
        self.executed = True
        if self.should_fail:
            raise Exception("Mock task failure")
        return "mock_result"

    def validate(self):
        return not self.should_fail


class TestTask:
    """Test cases for the base Task class."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = MockTask(priority=TaskPriority.HIGH)

        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert isinstance(task.task_id, str)
        assert task.progress == 0.0

    def test_task_execution_success(self):
        """Test successful task execution."""
        task = MockTask()
        context = {}

        task.start()
        result = task.execute(context)
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "mock_result"
        assert task.executed
        assert task.progress == 1.0
        assert task.get_duration() is not None

    def test_task_execution_failure(self):
        """Test task execution failure."""
        task = MockTask(should_fail=True)

        task.start()
        with pytest.raises(Exception):
            task.execute({})

        task.fail("Test failure")

        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Test failure"

    def test_task_cancellation(self):
        """Test task cancellation."""
        task = MockTask()

        task.cancel()

        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None

    def test_task_to_dict(self):
        """Test task serialization."""
        task = MockTask(priority=TaskPriority.HIGH)
        task.start()
        task.complete("result")

        data = task.to_dict()

        assert data["type"] == "MockTask"
        assert data["priority"] == "HIGH"
        assert data["status"] == "completed"
        assert data["duration"] is not None


class TestCrawlTask:
    """Test cases for CrawlTask."""

    def test_crawl_task_initialization(self):
        """Test crawl task initialization."""
        task = CrawlTask(
            source_name="arxiv", query="machine learning", max_results=50
        )

        assert task.source_name == "arxiv"
        assert task.query == "machine learning"
        assert task.max_results == 50

    def test_crawl_task_validation(self):
        """Test crawl task validation."""
        valid_task = CrawlTask("arxiv", "ml", 10)
        invalid_task = CrawlTask("", "ml", 10)
        invalid_task2 = CrawlTask("arxiv", "", 10)
        invalid_task3 = CrawlTask("arxiv", "ml", 0)

        assert valid_task.validate()
        assert not invalid_task.validate()
        assert not invalid_task2.validate()
        assert not invalid_task3.validate()

    def test_crawl_task_execution(self):
        """Test crawl task execution."""
        # Mock source
        mock_source = Mock()
        mock_papers = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]
        mock_source.search.return_value = mock_papers

        task = CrawlTask("test_source", "test query", 10)
        context = {"sources": {"test_source": mock_source}}

        result = task.execute(context)

        assert len(result) == 2
        assert all(isinstance(p, Paper) for p in result)
        mock_source.search.assert_called_once_with("test query", 10)

    def test_crawl_task_missing_source(self):
        """Test crawl task with missing source."""
        task = CrawlTask("missing_source", "query", 10)
        context = {"sources": {}}

        with pytest.raises(
            Exception, match="Source 'missing_source' not available"
        ):
            task.execute(context)


class TestParseTask:
    """Test cases for ParseTask."""

    def test_parse_task_initialization(self):
        """Test parse task initialization."""
        papers = [Paper(title="Test", abstract="Test")]
        task = ParseTask("pdf_parser", papers)

        assert task.parser_name == "pdf_parser"
        assert len(task.papers) == 1

    def test_parse_task_validation(self):
        """Test parse task validation."""
        papers = [Paper(title="Test", abstract="Test")]

        valid_task = ParseTask("pdf_parser", papers)
        invalid_task = ParseTask("", papers)
        invalid_task2 = ParseTask("pdf_parser", [])

        assert valid_task.validate()
        assert not invalid_task.validate()
        assert not invalid_task2.validate()

    def test_parse_task_execution(self):
        """Test parse task execution."""
        # Mock parser
        mock_parser = Mock()
        mock_parser.parse_paper.side_effect = (
            lambda p: p
        )  # Return paper unchanged

        papers = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]

        task = ParseTask("test_parser", papers)
        context = {"parsers": {"test_parser": mock_parser}}

        result = task.execute(context)

        assert len(result) == 2
        assert mock_parser.parse_paper.call_count == 2


class TestTagTask:
    """Test cases for TagTask."""

    def test_tag_task_initialization(self):
        """Test tag task initialization."""
        papers = [Paper(title="Test", abstract="Test")]
        task = TagTask("rule_tagger", papers)

        assert task.tagger_name == "rule_tagger"
        assert len(task.papers) == 1

    def test_tag_task_execution(self):
        """Test tag task execution."""
        # Mock tagger
        mock_tagger = Mock()
        mock_tagger.tag_paper.side_effect = (
            lambda p: p
        )  # Return paper unchanged

        papers = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]

        task = TagTask("test_tagger", papers)
        context = {"taggers": {"test_tagger": mock_tagger}}

        result = task.execute(context)

        assert len(result) == 2
        assert mock_tagger.tag_paper.call_count == 2


class TestStoreTask:
    """Test cases for StoreTask."""

    def test_store_task_initialization(self):
        """Test store task initialization."""
        papers = [Paper(title="Test", abstract="Test")]
        task = StoreTask("json_storage", papers)

        assert task.storage_name == "json_storage"
        assert len(task.papers) == 1

    def test_store_task_execution(self):
        """Test store task execution."""
        # Mock storage
        mock_storage = Mock()
        mock_storage.store_paper.side_effect = lambda p: p.get_primary_id()

        papers = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]

        task = StoreTask("test_storage", papers)
        context = {"storages": {"test_storage": mock_storage}}

        result = task.execute(context)

        assert len(result) == 2
        assert all(isinstance(paper_id, str) for paper_id in result)
        assert mock_storage.store_paper.call_count == 2
