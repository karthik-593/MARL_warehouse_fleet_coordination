"""Stage 2 assignment environment — fixed 10×10 warehouse, single robot, 3-action DQN."""

import numpy as np
import random
import torch
from collections import deque

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
GRID_SIZE        = 10
NAV_DRAIN        = 1      # % battery consumed per nav step (matches Stage 1)
TRIP_COST_RATE   = 5      # % per step — used ONLY for obs feature estimation
CHARGE_RATE      = 20     # % gained per Charge action at a charger
MAX_NAV_STEPS    = 300    # max steps for any single navigation leg

SHELVES: frozenset = frozenset([
    (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),   # Block A (top-left)
    (1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8),   # Block B (top-right)
    (6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3),   # Block C (bottom-left)
    (6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8),   # Block D (bottom-right)
])

PICKUP_POINTS: list = [
    (0, 2), (3, 2),   # Block A
    (0, 7), (3, 7),   # Block B
    (5, 2), (8, 2),   # Block C
    (5, 7), (8, 7),   # Block D
]

DROPOFF:  tuple = (9, 9)
CHARGERS: list  = [
    (0, 0),   # top-left corner
    (0, 9),   # top-right corner
    (9, 0),   # bottom-left corner
    (9, 5),   # bottom-centre  — near delivery zone, covers blocks C/D
    (4, 4),   # grid centre    — minimises max travel distance
]

_OCCUPIED  = SHELVES | set(CHARGERS) | {DROPOFF}
FREE_CELLS = [(r, c) for r in range(GRID_SIZE)
              for c in range(GRID_SIZE) if (r, c) not in _OCCUPIED]

# Centre-grid wait cells — equidistant from all pickup zones
WAIT_POSITIONS = [(r, c) for r in range(3, 6) for c in range(3, 6)
                  if (r, c) not in _OCCUPIED]

# ---------------------------------------------------------------------------
# BFS utilities
# ---------------------------------------------------------------------------
def bfs_dist(start: tuple, goal: tuple, blocked: frozenset = SHELVES) -> float:
    """BFS shortest path on the 10×10 grid.  Returns float('inf') if unreachable."""
    if start == goal:
        return 0
    visited = {start}
    q       = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                    and (nr, nc) not in blocked and (nr, nc) not in visited):
                if (nr, nc) == goal:
                    return d + 1
                visited.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return float("inf")


def nearest_charger_info(pos: tuple) -> tuple:
    """Return (charger_pos, bfs_distance) for the closest charger."""
    best_pos, best_d = CHARGERS[0], float("inf")
    for ch in CHARGERS:
        d = bfs_dist(pos, ch)
        if d < best_d:
            best_pos, best_d = ch, d
    return best_pos, best_d


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class WarehouseStage2:
    """
    Fixed 10×10 warehouse for Stage 2 assignment training.
    Observation: 8-dim normalised vector (battery, trip_cost, distances, idle_time,
                 orders_remaining, battery_margin).
    Actions    : 0=Accept, 1=Decline-idle, 2=GoCharge.
    """

    MOVE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
    OBS_DIM      = 8
    ACTION_DIM   = 3

    def __init__(self):
        # Static grid used by nav-state sensor
        self._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in SHELVES:
            self._grid[r, c] = -1
        for r, c in CHARGERS:
            self._grid[r, c] = 2
        self._grid[DROPOFF[0], DROPOFF[1]] = 3

        self.pos       = None
        self.battery   = 100.0
        self.idle_time = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.pos       = random.choice(FREE_CELLS)
        self.battery   = random.uniform(50.0, 100.0)
        self.idle_time = 0

    # ------------------------------------------------------------------
    # Assignment observation
    # ------------------------------------------------------------------
    def get_obs(self, pickup: tuple, orders_remaining: int = 5) -> np.ndarray:
        """Return normalised 8-dim observation for the Assignment DQN."""
        d_to_pickup   = bfs_dist(self.pos, pickup)
        d_pickup_drop = bfs_dist(pickup, DROPOFF)
        trip_cost     = (d_to_pickup + d_pickup_drop) * TRIP_COST_RATE
        _, d_charger  = nearest_charger_info(self.pos)
        MAX_D         = 18.0
        battery_margin = (self.battery - trip_cost) / 100.0
        return np.array([
            self.battery                / 100.0,
            min(trip_cost, 100.0)       / 100.0,
            min(d_to_pickup,   MAX_D)   / MAX_D,
            min(d_pickup_drop, MAX_D)   / MAX_D,
            min(d_charger,     MAX_D)   / MAX_D,
            min(self.idle_time, 50)     / 50.0,
            orders_remaining            / 5.0,
            np.clip(battery_margin, -1.0, 1.0),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def execute_order(self, nav_model: torch.nn.Module, pickup: tuple) -> float:
        """Navigate pos → pickup → dropoff. Returns +100 / -80 / -10."""
        _, reached = self._navigate(nav_model, pickup)
        if not reached:
            return -80.0 if self.battery <= 0 else -10.0
        _, reached = self._navigate(nav_model, DROPOFF)
        if not reached:
            return -80.0 if self.battery <= 0 else -10.0
        return 100.0

    def execute_decline_idle(self) -> float:
        """Idle 1-5 steps. Reward = -20 + 0.5 × idle_steps."""
        idle_steps      = random.randint(1, 5)
        self.idle_time += idle_steps
        return -20.0 + 0.5 * idle_steps

    def execute_go_charge(self, nav_model: torch.nn.Module) -> float:
        """Navigate to nearest charger, charge to full, reposition. Reward = -25."""
        ch_pos, _ = nearest_charger_info(self.pos)
        _, reached = self._navigate(nav_model, ch_pos)
        if reached:
            self.battery = 100.0
            wait_pos = min(WAIT_POSITIONS, key=lambda p: bfs_dist(self.pos, p))
            self._navigate(nav_model, wait_pos)
        self.idle_time = 0
        return -25.0

    # ------------------------------------------------------------------
    # Navigation (outcome-only — reward does not enter assignment replay)
    # ------------------------------------------------------------------
    def _navigate(self, nav_model: torch.nn.Module, target: tuple):
        """
        Drive frozen nav policy toward target.
        Returns (None, reached: bool).
        Internal nav rewards are discarded — execute_order returns
        a clean outcome reward to the assignment DQN instead.
        """
        device = next(nav_model.parameters()).device
        for _ in range(MAX_NAV_STEPS):
            state = self._get_nav_state(target)
            with torch.no_grad():
                logits, _ = nav_model(
                    torch.tensor(state, device=device).unsqueeze(0))
                logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
                probs  = torch.softmax(logits / 0.3, dim=-1)
                action = torch.distributions.Categorical(probs).sample().item()
            _, breakdown = self._nav_step(action)
            if breakdown:
                return None, False
            if self.pos == target:
                return None, True
        return None, False   # timeout

    def _nav_step(self, action: int):
        """Execute one raw navigation step.  Returns (reward, breakdown_flag)."""
        dr, dc = self.MOVE_ACTIONS[action]
        r, c   = self.pos
        nr     = int(np.clip(r + dr, 0, GRID_SIZE - 1))
        nc     = int(np.clip(c + dc, 0, GRID_SIZE - 1))
        reward = 0.0
        if self._grid[nr, nc] == -1:        # shelf collision
            reward -= 20.0
        else:
            self.pos = (nr, nc)
            if action == 5 and self.pos in set(CHARGERS):
                self.battery = min(100.0, self.battery + CHARGE_RATE)
            elif action != 4:               # not Stay
                self.battery = max(0.0, self.battery - NAV_DRAIN)
        if self.battery <= 0:
            reward -= 150.0
        return reward, self.battery <= 0

    def _get_nav_state(self, target: tuple) -> np.ndarray:
        """13-dim state matching Stage 1 PPO's expected input format."""
        px, py      = self.pos
        gx, gy      = target
        ch_pos, _   = nearest_charger_info(self.pos)
        cx, cy      = ch_pos
        obs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc  = px + dr, py + dc
            blocked = 1.0 if (
                nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE
                or self._grid[nr, nc] == -1
            ) else 0.0
            d = 0
            cr2, cc2 = nr, nc
            while (0 <= cr2 < GRID_SIZE and 0 <= cc2 < GRID_SIZE
                   and self._grid[cr2, cc2] != -1):
                d += 1
                cr2 += dr
                cc2 += dc
            obs.extend([blocked, d / GRID_SIZE])
        return np.array([
            self.battery / 100.0,
            (px - gx) / GRID_SIZE, (py - gy) / GRID_SIZE,
            (px - cx) / GRID_SIZE, (py - cy) / GRID_SIZE,
        ] + obs, dtype=np.float32)
