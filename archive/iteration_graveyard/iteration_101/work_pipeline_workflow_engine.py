"""
流水线工作流引擎模块

WorkflowEngine — 工作流引擎
DAG — 有向无环图
WorkflowNode — 工作流节点
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class NodeStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class WorkflowNode:
    node_id: str
    name: str
    task: Callable
    dependencies: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str = ""


class DAG:
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        self.edges[from_id].add(to_id)
        self.reverse_edges[to_id].add(from_id)

    def get_executable_nodes(self, completed: Set[str]) -> List[WorkflowNode]:
        executable = []
        for node_id, node in self.nodes.items():
            if node_id in completed:
                continue
            if node.status != NodeStatus.PENDING:
                continue
            deps = set(node.dependencies)
            if deps.issubset(completed):
                executable.append(node)
        return executable

    def topological_sort(self) -> List[str]:
        in_degree = defaultdict(int)
        for node in self.nodes.values():
            if node.node_id not in in_degree:
                in_degree[node.node_id] = 0
            for dep in node.dependencies:
                in_degree[node.node_id] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            for neighbor in self.edges[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result


class WorkflowEngine:
    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
        self._lock = threading.Lock()
        self._running = False

    def execute(self, dag: DAG) -> Dict[str, Any]:
        completed = set()
        results = {}
        errors = {}

        while True:
            with self._lock:
                executable = dag.get_executable_nodes(completed)
                if not executable:
                    break

                for node in executable[:self.max_parallel]:
                    try:
                        node.status = NodeStatus.RUNNING
                        result = node.task()
                        node.status = NodeStatus.COMPLETED
                        node.result = result
                        results[node.node_id] = result
                        completed.add(node.node_id)
                    except Exception as e:
                        node.status = NodeStatus.FAILED
                        node.error = str(e)
                        errors[node.node_id] = str(e)
                        completed.add(node.node_id)

        return {"results": results, "errors": errors, "completed": list(completed)}

    def validate(self, dag: DAG) -> bool:
        try:
            dag.topological_sort()
            return True
        except Exception:
            return False
