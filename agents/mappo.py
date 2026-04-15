"""MAPPO network architectures for Stage 3 (shared actor + centralised critic)."""

import torch
import torch.nn as nn


class AssignmentActor(nn.Module):
    """
    Decentralised actor shared across all robots.
    Input : 9-dim local observation (8 base features + is_eligible flag)
    Output: 3-dim action logits (Accept | Decline-idle | GoCharge)
    """

    OBS_DIM    = 9
    ACTION_DIM = 3

    def __init__(self, obs_dim: int = 9, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128,     128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CentralisedCritic(nn.Module):
    """
    Centralised critic — sees all agents' observations concatenated.
    Used only during training; not deployed at inference.
    Input : 27-dim global state (3 agents × 9 features)
    Output: scalar value estimate
    """

    def __init__(self, global_dim: int = 27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, 256), nn.ReLU(),
            nn.Linear(256,        256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def transfer_dqn_to_actor(dqn: nn.Module, actor: AssignmentActor):
    """
    Warm-start AssignmentActor from a trained Stage-2 AssignmentDQN.
    The first layer expands from Linear(8,128) to Linear(9,128);
    the extra column for is_eligible is zero-initialised.
    """
    sd_dqn = dqn.state_dict()
    sd_act = actor.state_dict()

    new_w = torch.zeros(128, 9)
    new_w[:, :8] = sd_dqn["net.0.weight"]
    sd_act["net.0.weight"] = new_w
    sd_act["net.0.bias"]   = sd_dqn["net.0.bias"].clone()

    for key in ["net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]:
        if key in sd_dqn:
            sd_act[key] = sd_dqn[key].clone()

    actor.load_state_dict(sd_act)
