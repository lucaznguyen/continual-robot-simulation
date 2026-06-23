from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from continual_robot_sim.envs import TabletopRobotEnv
from continual_robot_sim.tasks import RobotTask


@dataclass
class Rollout:
    task_name: str
    success: bool
    snapshots: list[dict[str, np.ndarray | str | bool]]


def evaluate_policy(
    model: torch.nn.Module,
    tasks: list[RobotTask],
    eval_task_indices: list[int],
    episodes: int,
    seed: int,
    max_steps: int,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    scores: dict[str, float] = {}

    for task_index in eval_task_indices:
        task = tasks[task_index]
        successes = []
        for episode in range(episodes):
            env = TabletopRobotEnv(
                task=task,
                task_index=task_index,
                num_tasks=len(tasks),
                seed=seed + 1000 * task_index + episode,
                max_steps=max_steps,
            )
            obs = env.reset()
            done = False
            while not done:
                action = _predict_action(model, obs, device=device)
                obs, _reward, done, info = env.step(action)
            successes.append(info["success"])
        scores[task.name] = float(np.mean(successes))
    return scores


def collect_rollout(
    model: torch.nn.Module,
    task: RobotTask,
    task_index: int,
    num_tasks: int,
    seed: int,
    max_steps: int,
    device: str = "cpu",
) -> Rollout:
    model.eval()
    env = TabletopRobotEnv(
        task=task,
        task_index=task_index,
        num_tasks=num_tasks,
        seed=seed,
        max_steps=max_steps,
    )
    obs = env.reset()
    snapshots = [env.snapshot()]
    done = False
    while not done:
        action = _predict_action(model, obs, device=device)
        obs, _reward, done, _info = env.step(action)
        snapshots.append(env.snapshot())
    return Rollout(task_name=task.name, success=env.success(), snapshots=snapshots)


def _predict_action(model: torch.nn.Module, obs: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model(obs_tensor).squeeze(0).detach().cpu().numpy()
    return action.astype(np.float32)
