from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from continual_robot_sim.envs import TabletopRobotEnv
from continual_robot_sim.tasks import RobotTask


@dataclass
class TaskDataset:
    name: str
    observations: torch.Tensor
    actions: torch.Tensor

    def as_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(self.observations, self.actions)


def collect_expert_dataset(
    task: RobotTask,
    task_index: int,
    num_tasks: int,
    episodes: int,
    seed: int,
    max_steps: int,
    behavior_noise: float = 0.12,
) -> TaskDataset:
    """Collect behavior-cloning samples from a hand-coded expert."""

    env = TabletopRobotEnv(
        task=task,
        task_index=task_index,
        num_tasks=num_tasks,
        seed=seed,
        max_steps=max_steps,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(episodes):
        obs = env.reset()
        for _step in range(max_steps):
            action = env.expert_action()
            observations.append(obs)
            actions.append(action)
            executed_action = action + env.rng.normal(0.0, behavior_noise, size=2).astype(np.float32)
            obs, _reward, done, _info = env.step(executed_action)
            if done:
                break

    return TaskDataset(
        name=task.name,
        observations=torch.tensor(np.asarray(observations), dtype=torch.float32),
        actions=torch.tensor(np.asarray(actions), dtype=torch.float32),
    )
