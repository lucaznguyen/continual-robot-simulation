from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continual_robot_sim.data import TaskDataset


MethodName = Literal["finetune", "replay", "ewc"]


@dataclass
class TrainerConfig:
    method: MethodName
    epochs_per_task: int = 12
    batch_size: int = 128
    lr: float = 3e-3
    replay_samples_per_task: int = 4096
    replay_epoch_multiplier: float = 1.6
    replay_balance_tasks: bool = True
    ewc_lambda: float = 80.0
    device: str = "cpu"


@dataclass
class ReplayMemory:
    samples_per_task: int
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)

    def add(self, dataset: TaskDataset, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        total = dataset.observations.shape[0]
        n = min(self.samples_per_task, total)
        indices = torch.randperm(total, generator=generator)[:n]
        self.observations.append(dataset.observations[indices].detach().cpu())
        self.actions.append(dataset.actions[indices].detach().cpu())

    def dataset_with(self, current: TaskDataset, balance_tasks: bool = True) -> TensorDataset:
        obs = [current.observations] + self.observations
        actions = [current.actions] + self.actions
        if balance_tasks:
            target = max(tensor.shape[0] for tensor in obs)
            obs = [_repeat_to_size(tensor, target) for tensor in obs]
            actions = [_repeat_to_size(tensor, target) for tensor in actions]
        return TensorDataset(torch.cat(obs, dim=0), torch.cat(actions, dim=0))


def _repeat_to_size(tensor: torch.Tensor, target: int) -> torch.Tensor:
    if tensor.shape[0] >= target:
        return tensor
    repeats = ceil(target / tensor.shape[0])
    return tensor.repeat((repeats, 1))[:target]


@dataclass
class EWCState:
    lambda_: float
    snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    fishers: list[dict[str, torch.Tensor]] = field(default_factory=list)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.snapshots:
            return torch.zeros((), device=next(model.parameters()).device)

        loss = torch.zeros((), device=next(model.parameters()).device)
        named_params = dict(model.named_parameters())
        for snapshot, fisher in zip(self.snapshots, self.fishers):
            for name, param in named_params.items():
                loss = loss + (fisher[name] * (param - snapshot[name]).pow(2)).sum()
        return 0.5 * self.lambda_ * loss

    def consolidate(
        self,
        model: nn.Module,
        dataset: TaskDataset,
        batch_size: int,
        device: str,
    ) -> None:
        model.eval()
        fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
        loader = DataLoader(dataset.as_tensor_dataset(), batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        batches = 0

        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(obs), actions)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            batches += 1

        scale = max(batches, 1)
        for name in fisher:
            fisher[name] = fisher[name] / scale

        snapshot = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.snapshots.append(snapshot)
        self.fishers.append(fisher)


class ContinualTrainer:
    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()
        self.replay = ReplayMemory(samples_per_task=config.replay_samples_per_task)
        self.ewc = EWCState(lambda_=config.ewc_lambda)

    def train_task(self, dataset: TaskDataset, task_number: int, seed: int) -> dict[str, float]:
        if self.config.method == "replay" and self.replay.observations:
            torch_dataset = self.replay.dataset_with(
                dataset,
                balance_tasks=self.config.replay_balance_tasks,
            )
        else:
            torch_dataset = dataset.as_tensor_dataset()

        generator = torch.Generator().manual_seed(seed + task_number)
        loader = DataLoader(
            torch_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.model.train()
        last_bc_loss = 0.0
        last_total_loss = 0.0
        epochs = self.config.epochs_per_task
        if self.config.method == "replay":
            epochs = max(epochs, ceil(epochs * self.config.replay_epoch_multiplier))
        for _epoch in range(epochs):
            for obs, actions in loader:
                obs = obs.to(self.config.device)
                actions = actions.to(self.config.device)
                pred = self.model(obs)
                bc_loss = self.criterion(pred, actions)
                total_loss = bc_loss
                if self.config.method == "ewc":
                    total_loss = total_loss + self.ewc.penalty(self.model)

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()
                last_bc_loss = float(bc_loss.detach().cpu())
                last_total_loss = float(total_loss.detach().cpu())

        if self.config.method == "replay":
            self.replay.add(dataset, seed=seed + task_number)
        if self.config.method == "ewc":
            self.ewc.consolidate(
                self.model,
                dataset=dataset,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )

        return {"bc_loss": last_bc_loss, "total_loss": last_total_loss}
