from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_robot_sim.tasks import RobotTask, one_hot


@dataclass
class EnvState:
    ee: np.ndarray
    obj: np.ndarray
    step_count: int = 0


class TabletopRobotEnv:
    """Small 2D robot simulator for continual-learning experiments.

    This is not meant to replace MuJoCo/robosuite. It is a fast didactic
    simulator that makes continual-learning behavior visible in seconds.
    """

    def __init__(
        self,
        task: RobotTask,
        task_index: int,
        num_tasks: int,
        seed: int = 0,
        max_steps: int = 45,
    ) -> None:
        self.task = task
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = EnvState(ee=np.zeros(2, dtype=np.float32), obj=np.zeros(2, dtype=np.float32))
        self.goal = np.array(task.goal, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return 2 + 2 + 2 + 1 + self.num_tasks

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        ee = self.rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
        obj_center = np.array(self.task.object_start, dtype=np.float32)
        obj = obj_center + self.rng.normal(0.0, 0.035, size=2).astype(np.float32)
        if self.task.kind.startswith("drawer"):
            obj[1] = obj_center[1]
        self.state = EnvState(ee=ee, obj=np.clip(obj, -0.75, 0.75), step_count=0)
        return self.observe()

    def observe(self) -> np.ndarray:
        progress = np.array([self.state.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate(
            [
                self.state.ee.astype(np.float32),
                self.state.obj.astype(np.float32),
                self.goal.astype(np.float32),
                progress,
                one_hot(self.task_index, self.num_tasks),
            ]
        )

    def expert_action(self) -> np.ndarray:
        if self.task.kind == "reach":
            target = self.goal
        elif self._near_object():
            target = self._manipulation_target()
        else:
            target = self.state.obj
        delta = target - self.state.ee
        norm = float(np.linalg.norm(delta))
        if norm > 0.04:
            delta = delta / norm
        return self._clip_action(delta)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = self._clip_action(action)
        self.state.ee = np.clip(self.state.ee + 0.09 * action, -0.85, 0.85)

        if self.task.kind != "reach" and self._near_object(radius=0.24):
            if self.task.kind.startswith("drawer"):
                drawer_motion = np.array([action[0], 0.0], dtype=np.float32)
                useful_contact = float(action[0] * (self.goal[0] - self.state.obj[0])) > 0.02
                if useful_contact:
                    self.state.obj = np.clip(self.state.obj + 0.095 * drawer_motion, -0.85, 0.85)
                self.state.obj[1] = np.array(self.task.object_start, dtype=np.float32)[1]
            else:
                goal_direction = self.goal - self.state.obj
                norm = float(np.linalg.norm(goal_direction))
                if norm > 1e-6:
                    goal_direction = goal_direction / norm
                useful_contact = float(np.dot(action, goal_direction)) > 0.15
                if useful_contact:
                    self.state.obj = np.clip(self.state.obj + 0.09 * action, -0.85, 0.85)

        self.state.step_count += 1
        success = self.success()
        target = self.goal if self.task.kind == "reach" else self.state.obj
        reward = 1.0 if success else -float(np.linalg.norm(target - self.goal))
        done = success or self.state.step_count >= self.max_steps
        return self.observe(), reward, done, {"success": float(success)}

    def success(self) -> bool:
        if self.task.kind == "reach":
            return float(np.linalg.norm(self.state.ee - self.goal)) < 0.08
        if self.task.kind.startswith("drawer"):
            return abs(float(self.state.obj[0] - self.goal[0])) < 0.08
        return float(np.linalg.norm(self.state.obj - self.goal)) < 0.13

    def snapshot(self) -> dict[str, np.ndarray | str | bool]:
        return {
            "task": self.task.name,
            "kind": self.task.kind,
            "ee": self.state.ee.copy(),
            "obj": self.state.obj.copy(),
            "goal": self.goal.copy(),
            "success": self.success(),
        }

    def _near_object(self, radius: float = 0.22) -> bool:
        return float(np.linalg.norm(self.state.ee - self.state.obj)) < radius

    def _manipulation_target(self) -> np.ndarray:
        if self.task.kind == "drawer_open":
            return np.array([self.goal[0] + 0.08, self.state.obj[1]], dtype=np.float32)
        if self.task.kind == "drawer_close":
            return np.array([self.goal[0] - 0.08, self.state.obj[1]], dtype=np.float32)
        direction = self.goal - self.state.obj
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm
        return (self.goal + 0.25 * direction).astype(np.float32)

    @staticmethod
    def _clip_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action = action / norm
        return np.clip(action, -1.0, 1.0).astype(np.float32)
