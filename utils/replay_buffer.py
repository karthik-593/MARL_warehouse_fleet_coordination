"""Shared experience replay buffer used by both Stage 1 (DQN) and Stage 2 (Assignment DQN)."""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.
    Stores (state, action, reward, next_state, done) tuples and returns
    GPU-ready tensors on sample.
    """

    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buf.append((state, int(action), float(reward), next_state, float(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch       = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(np.array(s),   dtype=torch.float32, device=device),
            torch.tensor(a,             dtype=torch.long,    device=device),
            torch.tensor(r,             dtype=torch.float32, device=device),
            torch.tensor(np.array(s_),  dtype=torch.float32, device=device),
            torch.tensor(d,             dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buf)
