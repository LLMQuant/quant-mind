"""Pipeline management for QuantMind workflow orchestration."""

from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from datetime import datetime

from quantmind.workflow.tasks import Task, TaskStatus, TaskPriority
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    """Pipeline for orchestrating QuantMind workflow tasks.

    Manages the execution of tasks in the knowledge extraction pipeline,
    including dependency management, parallel execution, and error handling.
    """

    def __init__(
        self,
        name: str,
        max_workers: int = 4,
        retry_attempts: int = 3,
        timeout: Optional[int] = None,
    ):
        """Initialize pipeline.

        Args:
            name: Pipeline name
            max_workers: Maximum number of concurrent workers
            retry_attempts: Number of retry attempts for failed tasks
            timeout: Task timeout in seconds
        """
        self.name = name
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.timeout = timeout

        self.tasks: List[Task] = []
        self.dependencies: Dict[
            str, List[str]
        ] = {}  # task_id -> [dependency_task_ids]
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: Dict[str, Future] = {}
        self.lock = threading.Lock()

        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = "initialized"

        # Event callbacks
        self.on_task_start: Optional[Callable[[Task], None]] = None
        self.on_task_complete: Optional[Callable[[Task, Any], None]] = None
        self.on_task_fail: Optional[Callable[[Task, str], None]] = None
        self.on_pipeline_complete: Optional[Callable[["Pipeline"], None]] = None

    def add_task(
        self, task: Task, dependencies: Optional[List[str]] = None
    ) -> "Pipeline":
        """Add a task to the pipeline.

        Args:
            task: Task to add
            dependencies: List of task IDs this task depends on

        Returns:
            Self for method chaining
        """
        with self.lock:
            self.tasks.append(task)
            if dependencies:
                self.dependencies[task.task_id] = dependencies

        logger.info(f"Added task {task.task_id} to pipeline {self.name}")
        return self

    def add_context(self, key: str, value: Any) -> "Pipeline":
        """Add data to the pipeline context.

        Args:
            key: Context key
            value: Context value

        Returns:
            Self for method chaining
        """
        with self.lock:
            self.context[key] = value
        return self

    def execute(self) -> Dict[str, Any]:
        """Execute the pipeline.

        Returns:
            Dictionary containing execution results

        Raises:
            Exception: If pipeline execution fails
        """
        logger.info(f"Starting pipeline {self.name}")
        self.started_at = datetime.utcnow()
        self.status = "running"

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self.executor = executor
                self._execute_tasks()

            self.status = "completed"
            self.completed_at = datetime.utcnow()

            if self.on_pipeline_complete:
                self.on_pipeline_complete(self)

            logger.info(f"Pipeline {self.name} completed successfully")
            return self.results

        except Exception as e:
            self.status = "failed"
            self.completed_at = datetime.utcnow()
            logger.error(f"Pipeline {self.name} failed: {e}")
            raise
        finally:
            self.executor = None
            self.futures.clear()

    def _execute_tasks(self):
        """Execute tasks with dependency management."""
        completed_tasks: set = set()
        executing_tasks: set = set()

        while len(completed_tasks) < len(self.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in self.tasks:
                if (
                    task.task_id not in completed_tasks
                    and task.task_id not in executing_tasks
                    and self._are_dependencies_met(
                        task.task_id, completed_tasks
                    )
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Check if we're deadlocked
                remaining_tasks = [
                    t for t in self.tasks if t.task_id not in completed_tasks
                ]
                if remaining_tasks:
                    # Wait for some tasks to complete
                    if self.futures:
                        # Wait for at least one future to complete
                        completed_futures = []
                        for task_id, future in list(self.futures.items()):
                            if future.done():
                                completed_futures.append(task_id)

                        if completed_futures:
                            for task_id in completed_futures:
                                self._handle_completed_task(
                                    task_id, completed_tasks, executing_tasks
                                )
                        else:
                            # All futures are still running, wait a bit
                            import time

                            time.sleep(0.1)
                    else:
                        raise Exception(
                            "Pipeline deadlock detected: no ready tasks and no running tasks"
                        )
                continue

            # Submit ready tasks
            for task in ready_tasks:
                if len(executing_tasks) < self.max_workers:
                    self._submit_task(task)
                    executing_tasks.add(task.task_id)

            # Check for completed tasks
            completed_futures = []
            for task_id, future in list(self.futures.items()):
                if future.done():
                    completed_futures.append(task_id)

            for task_id in completed_futures:
                self._handle_completed_task(
                    task_id, completed_tasks, executing_tasks
                )

    def _are_dependencies_met(self, task_id: str, completed_tasks: set) -> bool:
        """Check if all dependencies for a task are met.

        Args:
            task_id: Task ID to check
            completed_tasks: Set of completed task IDs

        Returns:
            True if all dependencies are met
        """
        dependencies = self.dependencies.get(task_id, [])
        return all(dep_id in completed_tasks for dep_id in dependencies)

    def _submit_task(self, task: Task):
        """Submit a task for execution.

        Args:
            task: Task to submit
        """
        if not task.validate():
            error_msg = f"Task {task.task_id} validation failed"
            task.fail(error_msg)
            return

        future = self.executor.submit(self._execute_task, task)
        self.futures[task.task_id] = future

        logger.debug(f"Submitted task {task.task_id} for execution")

    def _execute_task(self, task: Task) -> Any:
        """Execute a single task.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        task.start()
        if self.on_task_start:
            self.on_task_start(task)

        try:
            # Execute task with context
            result = task.execute(self.context)
            task.complete(result)

            if self.on_task_complete:
                self.on_task_complete(task, result)

            return result

        except Exception as e:
            error_msg = str(e)
            task.fail(error_msg)

            if self.on_task_fail:
                self.on_task_fail(task, error_msg)

            raise

    def _handle_completed_task(
        self, task_id: str, completed_tasks: set, executing_tasks: set
    ):
        """Handle a completed task.

        Args:
            task_id: ID of the completed task
            completed_tasks: Set of completed task IDs
            executing_tasks: Set of currently executing task IDs
        """
        future = self.futures.pop(task_id)
        executing_tasks.discard(task_id)

        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if not task:
            return

        try:
            result = future.result(timeout=self.timeout)
            self.results[task_id] = result
            completed_tasks.add(task_id)

            logger.debug(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            # Don't add to completed_tasks - task failed
            # This might cause pipeline to fail if other tasks depend on it

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID.

        Args:
            task_id: Task ID to find

        Returns:
            Task object if found, None otherwise
        """
        return next(
            (task for task in self.tasks if task.task_id == task_id), None
        )

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get tasks by their status.

        Args:
            status: Task status to filter by

        Returns:
            List of tasks with the specified status
        """
        return [task for task in self.tasks if task.status == status]

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        task_counts = {}
        for status in TaskStatus:
            task_counts[status.value] = len(self.get_tasks_by_status(status))

        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()

        return {
            "name": self.name,
            "status": self.status,
            "total_tasks": len(self.tasks),
            "task_counts": task_counts,
            "started_at": self.started_at.isoformat()
            if self.started_at
            else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration": duration,
            "results_count": len(self.results),
        }

    def cancel(self):
        """Cancel pipeline execution."""
        self.status = "cancelled"

        # Cancel all pending tasks
        for task in self.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.cancel()

        # Cancel futures
        for future in self.futures.values():
            future.cancel()

        logger.info(f"Pipeline {self.name} cancelled")

    def __str__(self) -> str:
        """String representation."""
        return f"Pipeline({self.name}, {len(self.tasks)} tasks, {self.status})"
