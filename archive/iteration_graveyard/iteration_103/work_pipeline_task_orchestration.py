"""
流水线任务编排模块

TaskOrchestrator — 任务编排器
Task — 任务
TaskDependency — 任务依赖
"""

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class TaskStatus(Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    task_id: str
    name: str
    task_func: Callable
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskOrchestrator:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._completed_tasks: Set[str] = set()
        self._running_tasks: Set[str] = set()

    def add_task(self, name: str, task_func: Callable, dependencies: Optional[Set[str]] = None) -> str:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            name=name,
            task_func=task_func,
            dependencies=dependencies or set()
        )
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_ready_tasks(self) -> List[Task]:
        with self._lock:
            ready = []
            for task in self._tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                if task.dependencies.issubset(self._completed_tasks):
                    task.status = TaskStatus.READY
                    ready.append(task)
            return ready

    def execute_task(self, task_id: str) -> Any:
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            task = self._tasks[task_id]

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        try:
            result = task.task_func()
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            with self._lock:
                self._completed_tasks.add(task_id)
            return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            raise

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
            task.status = TaskStatus.CANCELLED
            return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        task = self._tasks.get(task_id)
        return task.status if task else None
