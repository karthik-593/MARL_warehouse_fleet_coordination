"""Stage 1 navigation environment — 5-level curriculum (DQN L1-L2, PPO L3-L5)."""

import numpy as np
import random


class DynamicObstacle:
    """Randomly-walking obstacle used in Level 4."""

    def __init__(self, pos: tuple, size: int):
        self.pos  = pos
        self.size = size

    def move(self, blocked: set):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if (0 <= nr < self.size and 0 <= nc < self.size
                    and (nr, nc) not in blocked):
                self.pos = (nr, nc)
                return


class WarehouseEnv:
    """
    Single-robot warehouse navigation with battery management.

    State  : 13 features — battery, goal delta (2), charger delta (2),
             blocked + clearance per direction (8)
    Actions: Up, Down, Left, Right, Stay, Charge (at charger cell)
    Reward : -1.0 step, -10.0 obstacle, +goal-progress shaping (L3+),
             -deficit penalty (L3+), -1.0 anti-loop, +200 goal, -100 dead
    """

    ACTIONS      = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
    ACTION_NAMES = ["Up", "Down", "Left", "Right", "Stay", "Charge"]

    LEVEL_CONFIG = {
        1: dict(size=5,  n_static=0,  n_dynamic=0, low_bat_starts=False, warehouse=False),
        2: dict(size=10, n_static=0,  n_dynamic=0, low_bat_starts=False, warehouse=False),
        3: dict(size=10, n_static=10, n_dynamic=0, low_bat_starts=True,  warehouse=False),
        4: dict(size=10, n_static=5,  n_dynamic=3, low_bat_starts=True,  warehouse=False),
        5: dict(size=10, n_static=0,  n_dynamic=0, low_bat_starts=True,  warehouse=True),
    }

    STATE_DIM  = 13
    ACTION_DIM = 6

    def __init__(self, level: int = 1):
        cfg                 = self.LEVEL_CONFIG[level]
        self.size           = cfg["size"]
        self.level          = level
        self.n_static       = cfg["n_static"]
        self.n_dynamic      = cfg["n_dynamic"]
        self.low_bat_starts = cfg["low_bat_starts"]
        self.warehouse      = cfg["warehouse"]
        self.max_steps      = {1: 60, 2: 150, 3: 300, 4: 300, 5: 300}[level]
        self.use_shaping    = level >= 3
        self.dyn_obs: list  = []

        # Set at reset
        self.grid    = None
        self.pos     = None
        self.goal    = None
        self.charger = None
        self.battery = 100.0
        self.steps   = 0
        self.done    = False

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        if self.warehouse:
            return self._reset_warehouse()
        s          = self.size
        self.grid  = np.zeros((s, s), dtype=np.float32)
        self.charger = (0, 0)
        self.grid[0, 0] = 2

        # Place static obstacles
        placed = 0
        for _ in range(100_000):
            if placed >= self.n_static:
                break
            x, y = np.random.randint(0, s, 2)
            if (x, y) != (0, 0) and self.grid[x, y] == 0:
                self.grid[x, y] = -1
                placed += 1

        # Place goal
        while True:
            gx, gy = np.random.randint(0, s, 2)
            if self.grid[gx, gy] == 0:
                self.goal = (gx, gy)
                self.grid[gx, gy] = 3
                break

        # Place robot
        while True:
            px, py = np.random.randint(0, s, 2)
            if self.grid[px, py] == 0:
                self.pos = (px, py)
                break

        # Place dynamic obstacles
        occ = {(r, c) for r in range(s) for c in range(s)
               if self.grid[r, c] != 0} | {self.pos}
        self.dyn_obs = []
        for _ in range(self.n_dynamic):
            for _ in range(100_000):
                x, y = np.random.randint(0, s, 2)
                if (x, y) not in occ:
                    self.dyn_obs.append(DynamicObstacle((x, y), s))
                    occ.add((x, y))
                    break

        # Battery init: low starts for L3/L4 to force charging behaviour
        if self.low_bat_starts and np.random.random() < 0.50:
            self.battery = float(np.random.randint(20, 61))
        else:
            self.battery = 100.0

        self.steps        = 0
        self.done         = False
        self._prev_goal_d = self._mdist(self.pos, self.goal)
        self._visits      = {self.pos: 1}
        return self._get_state()

    def _reset_warehouse(self) -> np.ndarray:
        """Level 5: fixed warehouse shelf layout from Stage 2 assign_env."""
        from envs.assign_env import (SHELVES, CHARGERS, PICKUP_POINTS,
                                     DROPOFF, FREE_CELLS)
        s = self.size  # 10
        self.grid = np.zeros((s, s), dtype=np.float32)

        # Mark shelves as obstacles
        for r, c in SHELVES:
            self.grid[r, c] = -1

        # Pick a random charger; mark it on grid
        self.charger = random.choice(CHARGERS)
        self.grid[self.charger[0], self.charger[1]] = 2

        # Mark the dropoff cell
        self.grid[DROPOFF[0], DROPOFF[1]] = 3

        # Goal = random pickup point or the dropoff
        all_goals = PICKUP_POINTS + [DROPOFF]
        self.goal  = random.choice(all_goals)
        # Ensure goal cell shows as goal (overwrite 3 on dropoff if chosen)
        self.grid[self.goal[0], self.goal[1]] = 3

        # Robot starts on a free cell (not shelf, not charger, not goal)
        candidates = [c for c in FREE_CELLS
                      if c != self.goal and c != self.charger]
        self.pos = random.choice(candidates)

        self.dyn_obs = []

        if self.low_bat_starts and np.random.random() < 0.50:
            self.battery = float(np.random.randint(20, 61))
        else:
            self.battery = 100.0

        self.steps        = 0
        self.done         = False
        self._prev_goal_d = self._mdist(self.pos, self.goal)
        self._visits      = {self.pos: 1}
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action: int):
        dx, dy  = self.ACTIONS[action]
        x, y    = self.pos
        nx      = int(np.clip(x + dx, 0, self.size - 1))
        ny      = int(np.clip(y + dy, 0, self.size - 1))
        dyn     = self._dyn_set()

        reward  = -1.0
        hit_obs = self.grid[nx, ny] == -1 or (nx, ny) in dyn

        if hit_obs:
            reward -= 10.0
        else:
            self.pos = (nx, ny)
            if action == 5 and self.pos == self.charger:
                self.battery = min(100.0, self.battery + 20.0)
            elif action != 4:
                self.battery = max(0.0, self.battery - 1.0)

        if self.use_shaping:
            curr_gd = self._mdist(self.pos, self.goal)
            reward += (self._prev_goal_d - curr_gd) * 2.0
            self._prev_goal_d = curr_gd

        dist_to_goal = self._mdist(self.pos, self.goal)
        deficit      = max(0.0, dist_to_goal * 1.3 - self.battery)
        reward      -= deficit * 0.05

        if not hit_obs:
            cnt = self._visits.get(self.pos, 0) + 1
            self._visits[self.pos] = cnt
            if cnt > 2:
                reward -= 1.0

        blocked = ({(r, c) for r in range(self.size) for c in range(self.size)
                    if self.grid[r, c] != 0} | {self.pos} | dyn)
        for d in self.dyn_obs:
            d.move(blocked)

        success = False
        if self.pos == self.goal:
            reward += 200.0
            self.done = True
            success   = True
        elif self.battery <= 0:
            reward   -= 100.0
            self.done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, success

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        px, py = self.pos
        gx, gy = self.goal
        cx, cy = self.charger
        return np.array([
            self.battery / 100.0,
            (px - gx) / self.size, (py - gy) / self.size,
            (px - cx) / self.size, (py - cy) / self.size,
        ] + self._obstacle_sensors(), dtype=np.float32)

    def _mdist(self, a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _dyn_set(self) -> set:
        return {d.pos for d in self.dyn_obs}

    def _obstacle_sensors(self) -> list:
        px, py = self.pos
        dyn    = self._dyn_set()
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny  = px + dx, py + dy
            blocked = 1.0 if (
                nx < 0 or nx >= self.size or ny < 0 or ny >= self.size
                or self.grid[nx, ny] == -1 or (nx, ny) in dyn
            ) else 0.0
            d = 0
            cx2, cy2 = nx, ny
            while (0 <= cx2 < self.size and 0 <= cy2 < self.size
                   and self.grid[cx2, cy2] != -1 and (cx2, cy2) not in dyn):
                d += 1
                cx2 += dx
                cy2 += dy
            result.extend([blocked, d / self.size])
        return result
