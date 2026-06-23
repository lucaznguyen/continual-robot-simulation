from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RobotTask:
    """A small tabletop robot task.

    The simulator is intentionally simple: a 2D end-effector can move around a
    table and, when close enough, push/pull an object or drawer handle.
    """

    name: str
    kind: str
    goal: tuple[float, float]
    object_start: tuple[float, float]
    color: str


DEFAULT_TASKS: tuple[RobotTask, ...] = (
    RobotTask(
        name="reach_red",
        kind="reach",
        goal=(-0.55, 0.55),
        object_start=(-0.15, 0.15),
        color="#d84f45",
    ),
    RobotTask(
        name="push_blue",
        kind="push",
        goal=(0.58, 0.42),
        object_start=(-0.38, -0.18),
        color="#3f7fdb",
    ),
    RobotTask(
        name="slide_green",
        kind="push",
        goal=(-0.48, -0.58),
        object_start=(0.35, 0.16),
        color="#41a85f",
    ),
    RobotTask(
        name="open_drawer",
        kind="drawer_open",
        goal=(0.62, -0.12),
        object_start=(-0.25, -0.12),
        color="#9b59b6",
    ),
    RobotTask(
        name="close_drawer",
        kind="drawer_close",
        goal=(-0.55, -0.38),
        object_start=(0.45, -0.38),
        color="#f39c12",
    ),
)


def task_catalog(names: Iterable[str] | None = None) -> list[RobotTask]:
    """Return tasks by name, preserving the requested order."""

    tasks = list(DEFAULT_TASKS)
    if names is None:
        return tasks

    by_name = {task.name: task for task in tasks}
    missing = [name for name in names if name not in by_name]
    if missing:
        known = ", ".join(by_name)
        raise ValueError(f"Unknown task(s): {missing}. Known tasks: {known}")
    return [by_name[name] for name in names]


def one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec
