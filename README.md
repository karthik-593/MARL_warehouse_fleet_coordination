# Warehouse Multi-Agent Reinforcement Learning

A 3-stage curriculum that trains autonomous warehouse robots to navigate, accept orders, and coordinate deliveries while managing battery — progressing from single-agent navigation to multi-agent MAPPO.

---

## Stages

| Stage | Algorithm | Agents | Task |
|-------|-----------|--------|------|
| 1 — Navigation | DQN (L1–L2) → PPO (L3–L5) | 1 | Reach goal, avoid obstacles, manage battery |
| 2 — Assignment | Double DQN | 1 | Decide Accept / Decline-idle / GoCharge per order |
| 3 — Coordination | MAPPO (CTDE) | 3 | Coordinate order assignment across a shared fleet |

---

## Results

**Stage 2 — Assignment DQN** (200 greedy episodes, 5 orders/episode)

| Metric | Value |
|--------|-------|
| Mean episode reward | 300.48 |
| Deliveries / episode | 3.40 / 5  (68.0%) |
| Accept rate | 72.0% |
| GoCharge rate | 18.1% |
| Idle rate | 0.0% |
| Breakdowns / episode | 0.21 |

**Stage 3 — MAPPO** (200 greedy episodes, 10 orders/episode, 3 robots)

| Metric | Value |
|--------|-------|
| Mean team reward | 517.01 |
| Deliveries / episode | 9.46 / 10  (94.6%) |
| Accept rate | 80.65% |
| GoCharge rate | 17.10% |
| Idle rate | 2.25% |
| Breakdowns / episode | 0.25 |

3 robots deliver **2.78×** more than a single Stage 2 robot (92.7% of the theoretical maximum).

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
| Nav PPO (L3–L5) | 13 | 256 → 256 | 6 logits + value |
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
| ε decay | 0.9995/step, min 0.10 |
| Target update τ | 0.005 |

**Stage 1 — PPO (L3, L4)**

| Param | Value |
|-------|-------|
| Episodes | L3: 20 000  ·  L4: 16 000 |
| Rollout | 32 |
| Clip ε | 0.2 |
| Entropy coef | 0.03 |
| LR | 1e-4, cosine annealed to 1e-5 |

**Stage 1 — PPO L5 fine-tune**

| Param | Value |
|-------|-------|
| Episodes | 15 000 |
| Training mix | 70% warehouse + 30% random obstacles |
| LR | 3e-5 (constant) |

**Stage 2 — Assignment DQN**

| Param | Value |
|-------|-------|
| Episodes | 8 000 |
| Batch | 128 |
| γ | 0.99 |
| LR | 3e-4 |
| ε decay | 0.9993/episode, min 0.05 |
| Warmup | 300 transitions |

**Stage 3 — MAPPO**

| Param | Value |
|-------|-------|
| Episodes | 10 000 |
| Rollout | 16 |
| Clip ε | 0.2 |
| PPO epochs | 4 |
| LR actor | 5e-5 |
| LR critic | 1e-4 |
| Entropy coef | 0.02 |
| Checkpoint interval | 1 000 episodes |

Stage 3 resumes automatically from `checkpoints/mappo_actor_ckpt.pt` if it exists.

---

## Animations

Interactive HTML players for 25 scenarios (pause, scrub, speed control):

```bash
# Open notebooks/animations.ipynb and run all cells
# Output: checkpoints/animations/s01.html ... s25.html
```

| Scenarios | Stage | Content |
|-----------|-------|---------|
| S1–S9 | Navigation | Untrained vs trained, static/dynamic obstacles, battery |
| S10–S15 | Assignment | Accept, GoCharge, low-battery edge cases |
| S16–S25 | MAPPO | K=2 dispatch, bid conflicts, fleet coordination, baseline comparison |

---

## Project Structure

```
RL_multi_agent/
├── agents/
│   ├── ppo.py              # DQN + PPO for navigation
│   ├── dqn.py              # AssignmentDQN (Stage 2)
│   └── mappo.py            # AssignmentActor + CentralisedCritic (Stage 3)
├── envs/
│   ├── nav_env.py          # Navigation environment, 5-level curriculum
│   ├── assign_env.py       # Assignment environment (Stage 2)
│   └── marl_env.py         # Multi-agent environment (Stage 3)
├── training/
│   ├── train_nav.py        # Stage 1: L1–L4 curriculum
│   ├── train_nav_l5.py     # Stage 1: L5 warehouse fine-tune
│   ├── train_assign.py     # Stage 2: Assignment DQN
│   └── train_mappo.py      # Stage 3: MAPPO
├── utils/
│   ├── replay_buffer.py    # Experience replay for DQN stages
│   ├── plotting.py         # Training curve plots
│   └── visualize.py        # Animation engine (25 scenarios, HTML/GIF)
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

The 2.25% idle rate in evaluation reflects bid-conflict losers, not policy failure.
