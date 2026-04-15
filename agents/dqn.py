"""Assignment DQN — 3-action order decision network for Stage 2."""

import torch
import torch.nn as nn


class AssignmentDQN(nn.Module):
    """
    8-dim observation → 3 Q-values (Accept | Decline-idle | GoCharge).
    Architecture: Linear(8,128) → ReLU → Linear(128,128) → ReLU → Linear(128,3)
    """

    OBS_DIM    = 8
    ACTION_DIM = 3

    def __init__(self, obs_dim: int = 8, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,  128), nn.ReLU(),
            nn.Linear(128,      128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
