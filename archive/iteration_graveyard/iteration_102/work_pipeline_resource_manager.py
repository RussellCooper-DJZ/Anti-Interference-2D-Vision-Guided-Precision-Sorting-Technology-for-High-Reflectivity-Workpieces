"""
流水线资源管理模块

ResourcePool — 资源池
Resource — 资源
ResourceAllocator — 资源分配器
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class ResourceType(Enum):
    CPU = "CPU"
    MEMORY = "MEMORY"
    GPU = "GPU"
    DISK = "DISK"
    NETWORK = "NETWORK"


class ResourceStatus(Enum):
    AVAILABLE = "AVAILABLE"
    ALLOCATED = "ALLOCATED"
    RESERVED = "RESERVED"


@dataclass
class Resource:
    resource_id: str
    name: str
    resource_type: ResourceType
    capacity: float
    used: float = 0.0
    status: ResourceStatus = ResourceStatus.AVAILABLE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Allocation:
    allocation_id: str
    resource_id: str
    task_id: str
    amount: float
    allocated_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None


class ResourcePool:
    def __init__(self, name: str):
        self.name = name
        self._resources: Dict[str, Resource] = {}
        self._allocations: Dict[str, Allocation] = {}
        self._lock = threading.RLock()

    def add_resource(self, resource: Resource) -> None:
        with self._lock:
            self._resources[resource.resource_id] = resource

    def allocate(self, resource_id: str, task_id: str, amount: float) -> Optional[Allocation]:
        with self._lock:
            if resource_id not in self._resources:
                return None

            resource = self._resources[resource_id]
            if resource.used + amount > resource.capacity:
                return None

            import uuid
            allocation = Allocation(
                allocation_id=str(uuid.uuid4()),
                resource_id=resource_id,
                task_id=task_id,
                amount=amount
            )
            resource.used += amount
            if resource.used >= resource.capacity:
                resource.status = ResourceStatus.ALLOCATED

            self._allocations[allocation.allocation_id] = allocation
            return allocation

    def release(self, allocation_id: str) -> bool:
        with self._lock:
            if allocation_id not in self._allocations:
                return False

            allocation = self._allocations[allocation_id]
            resource = self._resources.get(allocation.resource_id)
            if resource:
                resource.used -= allocation.amount
                if resource.used < resource.capacity:
                    resource.status = ResourceStatus.AVAILABLE

            allocation.released_at = datetime.now()
            return True

    def get_available(self, resource_type: Optional[ResourceType] = None) -> List[Resource]:
        with self._lock:
            resources = list(self._resources.values())
            if resource_type:
                resources = [r for r in resources if r.resource_type == resource_type]
            return [r for r in resources if r.status == ResourceStatus.AVAILABLE]


class ResourceAllocator:
    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._lock = threading.RLock()

    def create_pool(self, name: str) -> ResourcePool:
        with self._lock:
            pool = ResourcePool(name)
            self._pools[name] = pool
            return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        return self._pools.get(name)
