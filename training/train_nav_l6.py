"""
Stage 1.6 — Nav Fine-Tuning with Static Robot Obstacles

Loads ppo_l5_warehouse.pt, fine-tunes on single-robot navigation with 2
static robot obstacles.  This matches Stage 3 deployment exactly: when a
robot executes an order, the other two robots are frozen at their last
positions (sequential execution in marl_env.step).

Saves ppo_l6_static.pt AND overwrites ppo_final.pt so Stage 2 + Stage 3
automatically use the improved nav model.

Run:  python -m training.train_nav_l6
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.ppo import PPO
from envs.multi_nav_env import StaticObstacleNavEnv
from training.train_nav import CKPT_DIR, DEVICE
from utils.plotting import plot_nav_history


def train_static_nav(
    model:         PPO,
    episodes:      int   = 30_000,
    n_rollout:     int   = 32,
    clip_eps:      float = 0.2,
    ppo_epochs:    int   = 4,
    entropy_coef:  float = 0.02,
    gamma:         float = 0.99,
    lr:            float = 3e-5,
    verbose_every: int   = 1000,
) -> tuple:
    """Fine-tune PPO on single-robot nav with 2 static obstacles.
    Returns (model, ep_rewards, ep_success_rates).
    """
    env       = StaticObstacleNavEnv()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, episodes // n_rollout), eta_min=lr * 0.1)

    ep_rewards, ep_success = [], []

    for ep_start in range(0, episodes, n_rollout):
        buf_s, buf_a, buf_r, buf_v, buf_lp = [], [], [], [], []
        rollout_rewards = []
        rollout_success = []

        for _ in range(n_rollout):
            obs  = env.reset()
            ep_r = 0.0
            bufs = {"s": [], "a": [], "r": [], "v": [], "lp": []}

            while True:
                st = torch.tensor(obs, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    logits, value = model(st)
                logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
                dist   = Categorical(F.softmax(logits, dim=-1))
                action = dist.sample()
                lp     = dist.log_prob(action)

                bufs["s"].append(obs)
                bufs["a"].append(action.item())
                bufs["v"].append(value.item())
                bufs["lp"].append(lp.item())

                obs, reward, done = env.step(action.item())
                bufs["r"].append(reward)
                ep_r += reward
                if done:
                    break

            # Discounted returns
            G       = 0.0
            returns = []
            for r in reversed(bufs["r"]):
                G = r + gamma * G
                returns.insert(0, G)

            buf_s.extend(bufs["s"])
            buf_a.extend(bufs["a"])
            buf_r.extend(returns)
            buf_v.extend(bufs["v"])
            buf_lp.extend(bufs["lp"])

            rollout_rewards.append(ep_r)
            rollout_success.append(float(env.success))

        ep_rewards.extend(rollout_rewards)
        ep_success.extend(rollout_success)

        if not buf_s:
            scheduler.step()
            continue

        s_t   = torch.tensor(np.array(buf_s),  device=DEVICE, dtype=torch.float32)
        a_t   = torch.tensor(buf_a,            device=DEVICE, dtype=torch.long)
        r_t   = torch.tensor(buf_r,            device=DEVICE, dtype=torch.float32)
        v_t   = torch.tensor(buf_v,            device=DEVICE, dtype=torch.float32)
        lp_t  = torch.tensor(buf_lp,           device=DEVICE, dtype=torch.float32)
        adv_t = r_t - v_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(ppo_epochs):
            idx = torch.randperm(len(s_t), device=DEVICE)
            for mb in idx.split(256):
                logits, vals = model(s_t[mb])
                logits   = torch.clamp(logits, -20.0, 20.0)
                dist     = Categorical(F.softmax(logits, dim=-1))
                new_lp   = dist.log_prob(a_t[mb])
                entropy  = dist.entropy().mean()
                ratio    = torch.exp(new_lp - lp_t[mb])
                adv      = adv_t[mb]
                a_loss   = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
                ).mean()
                c_loss   = 0.5 * F.mse_loss(vals.squeeze(-1), r_t[mb])
                loss     = a_loss + c_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        scheduler.step()

        ep_done = ep_start + n_rollout
        if ep_done % verbose_every < n_rollout:
            w = verbose_every
            print(f"  L6  Ep {ep_done:5d}/{episodes}"
                  f"  Reward {np.mean(ep_rewards[-w:]):+7.1f}"
                  f"  Success {np.mean(ep_success[-w:]):.2f}")

    return model, ep_rewards, ep_success


def main():
    l5_path = os.path.join(CKPT_DIR, "ppo_l5_warehouse.pt")
    if not os.path.exists(l5_path):
        raise FileNotFoundError(
            f"ppo_l5_warehouse.pt not found at {l5_path}.\n"
            "Run the full nav curriculum first: python -m training.train_nav"
        )

    print(f"Device : {DEVICE}")
    print("=" * 60)
    print("  Stage 1.6 — Nav Fine-Tune  (2 static robot obstacles)")
    print("=" * 60)
    print(f"  Loading L5 weights from {l5_path}")

    model = PPO(state_dim=13, action_dim=6).to(DEVICE)
    model.load_state_dict(torch.load(l5_path, map_location=DEVICE,
                                     weights_only=True))

    print("\n[L6] PPO  static-obstacle nav  (30 000 episodes)")
    model, rewards, success = train_static_nav(
        model,
        episodes      = 30_000,
        lr            = 3e-5,
        entropy_coef  = 0.02,
        verbose_every = 1000,
    )

    out_l6     = os.path.join(CKPT_DIR, "ppo_l6_static.pt")
    final_path = os.path.join(CKPT_DIR, "ppo_final.pt")

    torch.save(model.state_dict(), out_l6)
    print(f"\n  Saved L6 static nav -> {out_l6}")

    torch.save(model.state_dict(), final_path)
    print(f"  Updated ppo_final.pt -> {final_path}")

    np.save(os.path.join(CKPT_DIR, "nav_l6_rewards.npy"), np.array(rewards))
    plot_nav_history(rewards, success, level=6, out_dir=CKPT_DIR)
    print("  Saved reward history -> checkpoints/nav_l6_rewards.npy")


if __name__ == "__main__":
    main()
