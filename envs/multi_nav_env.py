"""
Stage 1.6 — Nav fine-tuning with static robot obstacles.

One robot navigates; two others are placed at fixed random positions for the
entire episode, exactly matching Stage 3 deployment where robots are static
during sequential order execution (only the winner robot moves at a time).

Uses the same 13-dim obs format as assign_env._get_nav_state so the PPO
model architecture needs no changes.
"""

import numpy as np
import random

from envs.assign_env import (
    SHELVES, CHARGERS, PICKUP_POINTS, DROPOFF, FREE_CELLS,
    nearest_charger_info, GRID_SIZE, NAV_DRAIN, CHARGE_RATE,
)

MOVE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
MAX_STEPS    = 400
N_ROBOTS     = 3   # 1 navigator + 2 static obstacles


class StaticObstacleNavEnv:
    """
    Single-robot nav env with 2 static robot obstacles.

    At each episode reset, 3 distinct free cells are chosen: one for the
    navigator, two for the frozen obstacles.  The obstacles never move,
    so the nav policy must learn to route around them rather than waiting.

    reset() -> np.ndarray  (13-dim obs for the navigator)
    step(action) -> next_obs, reward, done
    """

    def __init__(self):
        self._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in SHELVES:
            self._grid[r, c] = -1
        for r, c in CHARGERS:
            self._grid[r, c] = 2
        self._grid[DROPOFF[0], DROPOFF[1]] = 3

        self._charger_set = frozenset(CHARGERS)

        self.pos         = None
        self.goal        = None
        self.battery     = 100.0
        self.done        = False
        self.success     = False
        self.steps       = 0
        self._static_pos = frozenset()
        self._visits     = {}
        self._prev_d     = 0.0

    def reset(self) -> np.ndarray:
        candidates = list(FREE_CELLS)
        random.shuffle(candidates)
        nav_pos, s1, s2  = candidates[0], candidates[1], candidates[2]
        self.pos         = nav_pos
        self._static_pos = frozenset([s1, s2])

        self.goal = random.choice(PICKUP_POINTS + [DROPOFF])

        self.battery = (float(random.randint(25, 70))
                        if random.random() < 0.4 else 100.0)

        self.done    = False
        self.success = False
        self.steps   = 0
        self._visits = {self.pos: 1}
        self._prev_d = self._mdist(self.pos, self.goal)

        return self._get_obs()

    def step(self, action: int) -> tuple:
        dr, dc = MOVE_ACTIONS[action]
        r, c   = self.pos
        nr     = int(np.clip(r + dr, 0, GRID_SIZE - 1))
        nc     = int(np.clip(c + dc, 0, GRID_SIZE - 1))

        reward = -1.0

        if self._grid[nr, nc] == -1:            # shelf wall — bounce
            reward -= 10.0
        elif (nr, nc) in self._static_pos:       # static robot — penalise waiting
            reward -= 2.0                        # nudges policy to find alternate path
        else:
            self.pos = (nr, nc)
            if action == 5 and self.pos in self._charger_set:
                self.battery = min(100.0, self.battery + CHARGE_RATE)
            elif action != 4:                    # not Stay
                self.battery = max(0.0, self.battery - NAV_DRAIN)

        # Goal-progress shaping
        curr_d = self._mdist(self.pos, self.goal)
        reward += (self._prev_d - curr_d) * 2.0
        self._prev_d = curr_d

        # Battery deficit penalty
        deficit = max(0.0, curr_d * 1.3 - self.battery)
        reward -= deficit * 0.05

        # Anti-loop
        cnt = self._visits.get(self.pos, 0) + 1
        self._visits[self.pos] = cnt
        if cnt > 2:
            reward -= 1.0

        # Terminal conditions
        if self.pos == self.goal:
            reward      += 200.0
            self.done    = True
            self.success = True
        elif self.battery <= 0:
            reward    -= 100.0
            self.done  = True

        self.steps += 1
        if self.steps >= MAX_STEPS:
            self.done = True

        return self._get_obs(), reward, self.done

    def _get_obs(self) -> np.ndarray:
        px, py    = self.pos
        gx, gy    = self.goal
        ch_pos, _ = nearest_charger_info(self.pos)
        cx, cy    = ch_pos

        sensor = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc  = px + dr, py + dc
            blocked = 1.0 if (
                nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE
                or self._grid[nr, nc] == -1
                or (nr, nc) in self._static_pos
            ) else 0.0
            d = 0
            cr2, cc2 = nr, nc
            while (0 <= cr2 < GRID_SIZE and 0 <= cc2 < GRID_SIZE
                   and self._grid[cr2, cc2] != -1
                   and (cr2, cc2) not in self._static_pos):
                d   += 1
                cr2 += dr
                cc2 += dc
            sensor.extend([blocked, d / GRID_SIZE])

        return np.array([
            self.battery / 100.0,
            (px - gx) / GRID_SIZE, (py - gy) / GRID_SIZE,
            (px - cx) / GRID_SIZE, (py - cy) / GRID_SIZE,
        ] + sensor, dtype=np.float32)

    def _mdist(self, a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
