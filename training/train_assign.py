"""
Stage 2 — Assignment DQN Training (Accept | Decline-idle | GoCharge)

Run:  python -m training.train_assign
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.assign_env import WarehouseStage2, PICKUP_POINTS
from agents.ppo import PPO
from agents.dqn import AssignmentDQN
from utils.replay_buffer import ReplayBuffer
from utils.plotting import plot_assign_history, decision_heatmap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED     = 42
# Absolute path so checkpoints always land in the project root,
# regardless of whether the script is run from a notebook or the CLI.
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CKPT_DIR, exist_ok=True)



def load_nav_policy(path: str = None) -> PPO:
    if path is None:
        path = os.path.join(CKPT_DIR, "ppo_final.pt")
    nav = PPO(state_dim=13, action_dim=6).to(DEVICE)
    if os.path.exists(path):
        nav.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✓  Loaded nav policy from {path}")
    else:
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run Stage 1 first:  python -m training.train_nav"
        )
    nav.eval()
    for p in nav.parameters():
        p.requires_grad_(False)
    return nav


def _update_dqn(online: AssignmentDQN, target: AssignmentDQN,
                replay: ReplayBuffer, optimizer: torch.optim.Optimizer,
                gamma: float = 0.99, batch: int = 128) -> float:
    s, a, r, s_, done = replay.sample(batch, DEVICE)
    with torch.no_grad():
        a_next   = online(s_).argmax(dim=1)
        q_next   = target(s_).gather(1, a_next.unsqueeze(1)).squeeze(1)
        q_target = r + gamma * q_next * (1 - done)
    q_pred = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss   = nn.functional.smooth_l1_loss(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def _soft_update(online: AssignmentDQN, target: AssignmentDQN, tau: float = 0.005):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.copy_(tau * po.data + (1 - tau) * pt.data)


def train_stage2(
    nav_model:     PPO,
    episodes:      int   = 8_000,
    n_orders:      int   = 5,
    gamma:         float = 0.99,
    lr:            float = 3e-4,
    batch:         int   = 128,
    tau:           float = 0.005,
    eps_start:     float = 1.0,
    eps_decay:     float = 0.9993,
    eps_min:       float = 0.05,
    warmup:        int   = 300,
    verbose_every: int   = 500,
) -> tuple:
    """Train the Assignment DQN. Returns (online_dqn, history_dict)."""
    env        = WarehouseStage2()
    online_dqn = AssignmentDQN().to(DEVICE)
    target_dqn = AssignmentDQN().to(DEVICE)
    target_dqn.load_state_dict(online_dqn.state_dict())
    target_dqn.eval()
    optimizer  = optim.Adam(online_dqn.parameters(), lr=lr)
    replay     = ReplayBuffer(capacity=50_000)
    epsilon    = eps_start

    history: dict = {
        "rewards":        [],
        "losses":         [],
        "accept_rates":   [],
        "delivery_rates": [],
        "charge_rates":   [],
    }

    for ep in range(1, episodes + 1):
        env.reset()
        ep_reward  = 0.0
        loss_sum   = 0.0
        loss_cnt   = 0
        accepts    = 0
        deliveries = 0
        charges    = 0

        for order_i in range(n_orders):
            orders_remaining = n_orders - order_i
            pickup = random.choice(PICKUP_POINTS)
            feat   = env.get_obs(pickup, orders_remaining)

            # ε-greedy over 3 actions
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q      = online_dqn(
                        torch.tensor(feat, device=DEVICE).unsqueeze(0))
                    action = q.argmax().item()

            if action == 0:       # Accept
                accepts   += 1
                reward     = env.execute_order(nav_model, pickup)
                env.idle_time = 0
                if reward > 50:
                    deliveries += 1
            elif action == 1:     # Decline-idle
                reward = env.execute_decline_idle()
            else:                 # GoCharge
                charges += 1
                reward  = env.execute_go_charge(nav_model)

            ep_reward += reward
            done       = env.battery <= 0

            if order_i < n_orders - 1 and not done:
                next_remaining = n_orders - order_i - 1
                next_pickup    = random.choice(PICKUP_POINTS)
                next_feat      = env.get_obs(next_pickup, next_remaining)
            else:
                next_feat = feat.copy()
                done      = True

            replay.push(feat, action, reward, next_feat, done)

            if len(replay) >= warmup and len(replay) >= batch:
                loss = _update_dqn(online_dqn, target_dqn, replay,
                                   optimizer, gamma, batch)
                _soft_update(online_dqn, target_dqn, tau)
                loss_sum += loss
                loss_cnt += 1

            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)
        history["rewards"].append(ep_reward)
        history["losses"].append(loss_sum / max(loss_cnt, 1))
        history["accept_rates"].append(accepts / n_orders)
        history["delivery_rates"].append(deliveries / max(accepts, 1))
        history["charge_rates"].append(charges / n_orders)

        if ep % verbose_every == 0:
            w = verbose_every
            print(f"  Ep {ep:5d} | "
                  f"Reward {np.mean(history['rewards'][-w:]):+7.1f} | "
                  f"Accept {np.mean(history['accept_rates'][-w:]):.2f} | "
                  f"Charge {np.mean(history['charge_rates'][-w:]):.2f} | "
                  f"Deliv {np.mean(history['delivery_rates'][-w:]):.2f} | "
                  f"Loss {np.mean(history['losses'][-w:]):.4f} | "
                  f"ε={epsilon:.3f}")

    return online_dqn, history


def evaluate_stage2(nav_model: PPO, assign_dqn: AssignmentDQN,
                    n_eval: int = 200, n_orders: int = 5) -> dict:
    """Greedy evaluation.  Returns mean metrics dict."""
    env = WarehouseStage2()
    assign_dqn.eval()
    records = []

    for _ in range(n_eval):
        env.reset()
        ep = {"reward": 0.0, "deliveries": 0, "breakdowns": 0,
              "accepts": 0, "idles": 0, "charges": 0}

        for order_i in range(n_orders):
            orders_remaining = n_orders - order_i
            pickup = random.choice(PICKUP_POINTS)
            feat   = env.get_obs(pickup, orders_remaining)
            with torch.no_grad():
                action = assign_dqn(
                    torch.tensor(feat, device=DEVICE).unsqueeze(0)
                ).argmax().item()

            if action == 0:
                ep["accepts"] += 1
                reward = env.execute_order(nav_model, pickup)
                env.idle_time = 0
                if reward > 50:
                    ep["deliveries"] += 1
                if env.battery <= 0:
                    ep["breakdowns"] += 1
            elif action == 1:
                ep["idles"]  += 1
                reward = env.execute_decline_idle()
            else:
                ep["charges"] += 1
                reward = env.execute_go_charge(nav_model)

            ep["reward"] += reward
            if env.battery <= 0:
                break

        records.append(ep)

    assign_dqn.train()
    metrics = {k: np.mean([r[k] for r in records])
               for k in ["reward", "deliveries", "breakdowns",
                         "accepts", "idles", "charges"]}
    metrics["accept_rate"] = metrics.pop("accepts")  / n_orders
    metrics["idle_rate"]   = metrics.pop("idles")    / n_orders
    metrics["charge_rate"] = metrics.pop("charges")  / n_orders

    print("─" * 55)
    print(f"  Stage 2 — Evaluation  (n={n_eval} greedy episodes)")
    print("─" * 55)
    labels = {
        "reward":       "Mean episode reward",
        "deliveries":   "Deliveries / episode",
        "breakdowns":   "Breakdowns / episode",
        "accept_rate":  "Accept rate   (action=0)",
        "idle_rate":    "Idle rate     (action=1)",
        "charge_rate":  "GoCharge rate (action=2, learned)",
    }
    for k, label in labels.items():
        print(f"  {label:<36}: {metrics[k]:.4f}")
    print("─" * 55)
    return metrics


def main():
    print(f"Device : {DEVICE}")
    print("=" * 60)
    print("  Stage 2 — Assignment DQN  (3-action, learned charging)")
    print("=" * 60)

    nav_model = load_nav_policy()

    assign_dqn, history = train_stage2(
        nav_model     = nav_model,
        episodes      = 8_000,
        n_orders      = 5,
        gamma         = 0.99,
        lr            = 3e-4,
        batch         = 128,
        tau           = 0.005,
        eps_start     = 1.0,
        eps_decay     = 0.9993,
        eps_min       = 0.05,
        warmup        = 300,
        verbose_every = 500,
    )
    print("\nTraining complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────
    eval_metrics = evaluate_stage2(nav_model, assign_dqn)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_assign_history(history, out_dir=CKPT_DIR)
    decision_heatmap(assign_dqn, DEVICE, out_dir=CKPT_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, "assign_dqn.pt")
    torch.save(assign_dqn.state_dict(), ckpt_path)
    print(f"✓  Saved assignment DQN  → {ckpt_path}")

    for key, arr in history.items():
        np.save(os.path.join(CKPT_DIR, f"assign_{key}.npy"), np.array(arr))
    print("✓  Saved history arrays  → checkpoints/assign_*.npy")

    with open(os.path.join(CKPT_DIR, "assign_eval.json"), "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items()}, f, indent=2)
    print("✓  Saved eval metrics    → checkpoints/assign_eval.json")


if __name__ == "__main__":
    main()
