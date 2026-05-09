# Warehouse Multi-Agent Reinforcement Learning

A 3-stage curriculum that trains autonomous warehouse robots to navigate, accept orders, and coordinate deliveries while managing battery — progressing from single-agent navigation to multi-agent MAPPO.

---

## Stages

| Stage | Algorithm | Agents | Task |
|-------|-----------|--------|------|
| 1 — Navigation | DQN (L1–L2) → PPO (L3–L6) | 1 | Reach goal, avoid obstacles, manage battery |
| 2 — Assignment | Double DQN | 1 | Decide Accept / Decline-idle / GoCharge per order |
| 3 — Coordination | MAPPO (CTDE) | 3 | Coordinate order assignment across a shared fleet |

---

## Results

**Stage 2 — Assignment DQN** (200 greedy episodes, 5 orders/episode)

| Metric | Value |
|--------|-------|
| Mean episode reward | 374.65 |
| Deliveries / episode | 3.98 / 5  (79.6%) |
| Accept rate | 80.1% |
| GoCharge rate | 16.5% |
| Idle rate | 0.8% |
| Breakdowns / episode | 0.12 |

**Stage 3 — MAPPO** (200 greedy episodes, 10 orders/episode, 3 robots)

| Metric | Value |
|--------|-------|
| Mean team reward | 306.14 |
| Deliveries / episode | 8.06 / 10  (80.6%) |
| Accept rate | 60.7% |
| GoCharge rate | 34.5% |
| Idle rate | 4.8% |
| Breakdowns / episode | 0.47 |

The 3-robot team achieves **80.6% delivery rate** on 10 orders per episode, matching the single-robot Stage-2 efficiency (79.6%) while handling twice the workload.

---

## Environment

Fixed 10×10 warehouse grid:

```
Shelf blocks  : 4 blocks of 2×3 cells (one per quadrant)
Chargers (5)  : (0,0)  (0,9)  (9,0)  (9,5)  (4,4)
Dropoff       : (9,9)
Pickup points : (0,2) (3,2) (0,7) (3,7) (5,2) (8,2) (5,7) (8,7)
```

Battery drains 1% per movement step. Charging at a charger cell restores 20% per action.

---

## State & Action Spaces

**Stage 1 — Navigation (13 features)**

| Index | Feature |
|-------|---------|
| 0 | `battery / 100` |
| 1–2 | `(pos − goal) / grid_size` (row, col) |
| 3–4 | `(pos − charger) / grid_size` (row, col) |
| 5–12 | `[blocked, clearance]` × 4 directions |

Actions: Up, Down, Left, Right, Stay, Charge (6 discrete)

**Stage 2 — Assignment (8 features)**

| Index | Feature |
|-------|---------|
| 0 | `battery / 100` |
| 1 | `trip_cost_estimate / 100` |
| 2 | `BFS(pos, pickup) / 18` |
| 3 | `BFS(pickup, dropoff) / 18` |
| 4 | `BFS(pos, nearest_charger) / 18` |
| 5 | `idle_time / 50` |
| 6 | `orders_remaining / 5` |
| 7 | `clip((battery − trip_cost) / 100, −1, 1)` |

Actions: Accept (0), Decline-idle (1), GoCharge (2)

**Stage 3 — MAPPO** adds one feature: `is_eligible` (1 if robot is in K=2 nearest, else 0).
Global state for the centralised critic: 27-dim (3 agents × 9).

---

## Architecture

| Module | Input | Hidden | Output |
|--------|-------|--------|--------|
| Nav DQN (L1–L2) | 13 | 256 → 256 | 6 Q-values |
| Nav PPO (L3–L6) | 13 | 256 → 256 | 6 logits + value |
| Assignment DQN | 8 | 128 → 128 | 3 Q-values |
| MAPPO Actor | 9 | 128 → 128 | 3 logits |
| MAPPO Critic | 27 | 256 → 256 | scalar value |

Each stage warm-starts from the previous stage's weights.

---

## Training

### Run all stages

```bash
# Stage 1 — navigation curriculum L1–L4
python -m training.train_nav

# Stage 1 — L5 warehouse fine-tune (run after train_nav)
python -m training.train_nav_l5

# Stage 1 — L6 warehouse + robot obstacles (run after train_nav_l5)
python -m training.train_nav_l6

# Stage 2 — assignment DQN
python -m training.train_assign

# Stage 3 — MAPPO
python -m training.train_mappo
```

### Hyperparameters

**Stage 1 — DQN (L1, L2)**

| Param | Value |
|-------|-------|
| Episodes per level | 8 000 |
| Replay buffer | 100 000 |
| Batch | 256 |
| γ | 0.99 |
| LR | 5e-4 |
| ε decay | 0.9995/step, min 0.02 |
| Target update τ | 0.005 |

**Stage 1 — PPO (L3, L4)**

| Param | Value |
|-------|-------|
| Episodes | L3: 20 000  ·  L4: 20 000 |
| Rollout | 32 |
| Clip ε | 0.2 |
| Entropy coef | 0.03 |
| LR | 1e-4, cosine annealed to 1e-5 |

**Stage 1 — PPO L5 fine-tune**

| Param | Value |
|-------|-------|
| Episodes | 30 000 |
| Training mix | 100% warehouse layout |
| LR | 3e-5 (constant) |
| Entropy coef | 0.02 |

**Stage 1 — PPO L6 fine-tune**

| Param | Value |
|-------|-------|
| Episodes | 30 000 |
| Training mix | Warehouse + 2 static robot obstacles |
| LR | 3e-5, cosine annealed to 3e-6 |
| Entropy coef | 0.02 |

**Stage 2 — Assignment DQN**

| Param | Value |
|-------|-------|
| Episodes | 16 000 |
| Batch | 128 |
| γ | 0.99 |
| LR | 3e-4 |
| ε decay | 0.9993/episode, min 0.01 |
| Warmup | 300 transitions |

**Stage 3 — MAPPO**

| Param | Value |
|-------|-------|
| Episodes | 50 000 |
| Rollout | 16 |
| Clip ε | 0.2 |
| PPO epochs | 4 |
| LR actor | 5e-5 |
| LR critic | 5e-5 |
| Entropy coef | 0.05 → 0.01 (annealed) |
| Checkpoint interval | 1 000 episodes |

Stage 3 resumes automatically from `checkpoints/mappo_actor_ckpt.pt` if it exists.

---

## Animations

Interactive HTML players for 6 scenarios (pause, scrub, speed control):

```bash
# Open notebooks/animations.ipynb and run all cells
# Output: checkpoints/animations/s01.html ... s06.html
```

| Scenario | Stage | Content |
|----------|-------|---------|
| S1 | Navigation | 5×5 plain grid |
| S2 | Navigation | 10×10 plain grid |
| S3 | Navigation | 10×10 + stationary obstacles |
| S4 | Navigation | 10×10 + moving obstacles |
| S5 | Navigation | L6 warehouse + 2 frozen robot obstacles |
| S6 | MAPPO | 3-robot coordination (best of 20 seeds) |

---

## Project Structure

```
RL_multi_agent/
├── agents/
│   ├── ppo.py              # DQN + PPO for navigation
│   ├── dqn.py              # AssignmentDQN (Stage 2)
│   └── mappo.py            # AssignmentActor + CentralisedCritic (Stage 3)
├── envs/
│   ├── nav_env.py          # Navigation environment, 6-level curriculum
│   ├── multi_nav_env.py    # Multi-robot nav env for L6 obstacle fine-tuning
│   ├── assign_env.py       # Assignment environment (Stage 2)
│   └── marl_env.py         # Multi-agent environment (Stage 3)
├── training/
│   ├── train_nav.py        # Stage 1: L1–L4 curriculum
│   ├── train_nav_l5.py     # Stage 1: L5 warehouse fine-tune
│   ├── train_nav_l6.py     # Stage 1: L6 warehouse + robot obstacles
│   ├── train_assign.py     # Stage 2: Assignment DQN
│   └── train_mappo.py      # Stage 3: MAPPO
├── utils/
│   ├── replay_buffer.py    # Experience replay for DQN stages
│   ├── plotting.py         # Training curve plots
│   └── visualize.py        # Animation engine (6 scenarios, HTML)
├── notebooks/
│   ├── stage1_navigation.ipynb
│   ├── stage2_assignment.ipynb
│   ├── stage3_mappo.ipynb
│   └── animations.ipynb
├── checkpoints/            # Saved weights (git-ignored)
└── requirements.txt
```

---

## Requirements

```
torch >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

GPU (CUDA) recommended. CPU works but Stage 3 is slow.

```bash
pip install -r requirements.txt

# GPU build
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## K=2 Dispatch (Stage 3)

Each order is offered to the two nearest robots by BFS distance.

- Both eligible robots independently choose Accept / Decline-idle / GoCharge.
- If both bid Accept, the robot with higher `battery − trip_cost` wins; the other reverts to Idle.
- Non-eligible robots decide freely.

The 4.8% idle rate in evaluation reflects bid-conflict losers, not policy failure.
