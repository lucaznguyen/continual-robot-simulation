from __future__ import annotations

import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
