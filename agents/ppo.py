"""Navigation network architectures for Stage 1 (DQN levels 1-2, PPO levels 3-5)."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """13-dim state → 6 Q-values. Used for navigation levels 1-2."""

    def __init__(self, state_dim: int = 13, action_dim: int = 6):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.head   = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.shared(x))


class PPO(nn.Module):
    """
    Actor-Critic PPO for navigation levels 3-5.
    Returns (action_logits [6], value [1]).
    """

    def __init__(self, state_dim: int = 13, action_dim: int = 6):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.actor  = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        return self.actor(f), self.critic(f)


def transfer_dqn_to_ppo(dqn: DQN, ppo: PPO, scale: float = 0.1):
    """Warm-start PPO shared and actor weights from a trained DQN."""
    ppo.shared[0].weight.data = dqn.shared[0].weight.data.clone() * scale
    ppo.shared[0].bias.data   = dqn.shared[0].bias.data.clone()   * scale
    ppo.actor[0].weight.data  = dqn.head[0].weight.data.clone()   * scale
    ppo.actor[0].bias.data    = dqn.head[0].bias.data.clone()     * scale


def select_action(model: nn.Module, state, temperature: float = 0.3) -> int:
    """Temperature-scaled softmax sampling. Works for both DQN and PPO outputs."""
    device = next(model.parameters()).device
    with torch.no_grad():
        out    = model(torch.tensor(state, device=device).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
        probs  = torch.softmax(logits / temperature, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()
