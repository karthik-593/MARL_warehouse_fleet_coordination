"""
Stage 3 — Multi-Agent Warehouse Environment

Dispatch model
--------------
  1. One order arrives each round (shared pool of N_ORDERS per episode).
  2. The K_NEAREST = 2 closest robots (by BFS) are eligible to bid.
  3. All 3 robots act simultaneously:
       Eligible   : Accept(0) / Decline-idle(1) / GoCharge(2)
       Non-eligible: Accept is treated as Idle (action space is identical,
                     the environment simply overrides the outcome).
  4. Conflict resolution — multiple eligible robots Accept:
       → winner   = highest battery_margin score  (battery − trip_cost)
       → winner   executes the order, gets outcome reward
       → loser(s) fall back to Idle  (−20 to −17.5)

Observation (9-dim per robot)
------------------------------
  [0] battery           / 100
  [1] trip_cost         (estimated % needed) / 100
  [2] dist_pickup       / MAX_D
  [3] dist_dropoff      / MAX_D
  [4] dist_charger      / MAX_D
  [5] idle_time         / 50
  [6] orders_remaining  / N_ORDERS
  [7] battery_margin    clip((battery − trip_cost) / 100, −1, 1)
  [8] is_eligible       1.0 if in K nearest robots, 0.0 otherwise  ← NEW

Global state (critic) : concat of all 3 local obs = 27-dim
"""

import numpy as np
import random
import torch

from envs.assign_env import (
    WarehouseStage2, PICKUP_POINTS, bfs_dist, TRIP_COST_RATE
)

N_AGENTS  = 3
N_ORDERS  = 10    # orders per episode
K_NEAREST = 2     # robots eligible to bid per order
OBS_DIM   = 9     # 8 base features + is_eligible
GLOBAL_DIM = N_AGENTS * OBS_DIM   # 27


class MultiAgentWarehouse:
    """
    Cooperative multi-agent warehouse: 3 robots, shared order pool.

    Dispatch per order
    ------------------
    1. Pick K=2 nearest eligible robots.
    2. Each robot acts (all simultaneously).
    3. Resolve bids → winner executes order; non-eligible Accept → Idle.

    Usage
    -----
    obs, gs = env.reset()
    while True:
        actions = [0, 1, 2]  # one per robot
        obs, gs, rewards, done = env.step(actions, nav_model)
        if done: break
    """

    def __init__(self):
        self.agents = [WarehouseStage2() for _ in range(N_AGENTS)]
        self.orders  = []
        self.order_i = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.orders  = [random.choice(PICKUP_POINTS) for _ in range(N_ORDERS)]
        self.order_i = 0
        return self._obs_and_gs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions: list, nav_model: torch.nn.Module) -> tuple:
        """
        Execute one order-round for all robots.

        Parameters
        ----------
        actions   : list[int], length N_AGENTS  (0=Accept, 1=Idle, 2=Charge)
        nav_model : frozen Stage-1 PPO

        Returns
        -------
        next_obs     : np.ndarray  [N_AGENTS, OBS_DIM]
        global_state : np.ndarray  [GLOBAL_DIM]
        rewards      : list[float], length N_AGENTS
        done         : bool
        """
        pickup   = self.orders[self.order_i]
        eligible = self._k_nearest(pickup)           # set of agent indices
        rewards  = [0.0] * N_AGENTS

        # ── Identify who is bidding ────────────────────────────────────
        accepting = [i for i in eligible if actions[i] == 0]

        # ── Resolve bids ───────────────────────────────────────────────
        winner = None
        if len(accepting) == 1:
            winner = accepting[0]
        elif len(accepting) > 1:
            # Highest  battery − trip_cost  wins
            scores = {
                i: self.agents[i].battery
                   - bfs_dist(self.agents[i].pos, pickup) * TRIP_COST_RATE
                for i in accepting
            }
            winner = max(scores, key=scores.get)

        # ── Execute each robot's action ────────────────────────────────
        for i, agent in enumerate(self.agents):
            action = actions[i]

            if i == winner:                         # won the bid → execute order
                rewards[i] = agent.execute_order(nav_model, pickup)
                agent.idle_time = 0

            elif action == 2:                       # chose GoCharge (any robot)
                rewards[i] = agent.execute_go_charge(nav_model)

            else:                                   # Idle, lost bid, or non-eligible Accept
                rewards[i] = agent.execute_decline_idle()

        self.order_i += 1
        done = self.order_i >= N_ORDERS

        next_obs, gs = self._obs_and_gs()
        return next_obs, gs, rewards, done

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _k_nearest(self, pickup: tuple) -> set:
        """Return indices of the K_NEAREST robots by BFS distance to pickup."""
        dists  = [bfs_dist(a.pos, pickup) for a in self.agents]
        sorted_i = sorted(range(N_AGENTS), key=lambda i: dists[i])
        return set(sorted_i[:K_NEAREST])

    def _get_agent_obs(self, agent_idx: int, pickup: tuple,
                       is_eligible: bool) -> np.ndarray:
        """Build 9-dim obs for one robot given the current order."""
        agent      = self.agents[agent_idx]
        remaining  = max(0, N_ORDERS - self.order_i)
        base       = agent.get_obs(pickup, remaining).copy()  # 8-dim
        # Fix orders_remaining normalisation (get_obs divides by 5; we use N_ORDERS)
        base[6]    = remaining / N_ORDERS
        return np.append(base, float(is_eligible))            # 9-dim

    def _obs_and_gs(self) -> tuple:
        """Return ([N_AGENTS, OBS_DIM], [GLOBAL_DIM]) for current order."""
        pickup   = self.orders[min(self.order_i, N_ORDERS - 1)]
        eligible = self._k_nearest(pickup)
        obs = np.stack([
            self._get_agent_obs(i, pickup, i in eligible)
            for i in range(N_AGENTS)
        ])                                # [N_AGENTS, OBS_DIM]
        return obs, obs.flatten()         # [N_AGENTS, 9], [27]
