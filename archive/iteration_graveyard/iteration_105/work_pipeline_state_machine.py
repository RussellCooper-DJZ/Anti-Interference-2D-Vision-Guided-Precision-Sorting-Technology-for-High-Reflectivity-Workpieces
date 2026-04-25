"""
流水线状态机模块

StateMachine — 状态机
State — 状态
Transition — 转换
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class StateMachineState(Enum):
    INITIAL = "INITIAL"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Transition:
    from_state: StateMachineState
    to_state: StateMachineState
    event: str
    guard: Optional[Callable[[], bool]] = None
    action: Optional[Callable] = None


class StateMachine:
    def __init__(self, name: str):
        self.name = name
        self._states: Set[StateMachineState] = {StateMachineState.INITIAL}
        self._current_state: StateMachineState = StateMachineState.INITIAL
        self._transitions: List[Transition] = []
        self._history: List[tuple] = []
        self._lock = threading.RLock()

    def add_state(self, state: StateMachineState) -> None:
        self._states.add(state)

    def add_transition(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def trigger(self, event: str, *args, **kwargs) -> bool:
        with self._lock:
            for transition in self._transitions:
                if transition.from_state != self._current_state:
                    continue
                if transition.event != event:
                    continue
                if transition.guard and not transition.guard():
                    continue

                old_state = self._current_state
                if transition.action:
                    transition.action(*args, **kwargs)
                self._current_state = transition.to_state
                self._history.append((old_state, transition.to_state, event, datetime.now()))
                return True
            return False

    @property
    def current_state(self) -> StateMachineState:
        return self._current_state

    def is_state(self, state: StateMachineState) -> bool:
        return self._current_state == state

    def get_history(self) -> List[tuple]:
        return list(self._history)
