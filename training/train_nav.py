"""
Stage 1 — Navigation Training
4-level curriculum:
  L1  DQN   5×5 plain grid
  L2  DQN   10×10 plain grid     (warm-starts from L1 weights)
  L3  PPO   10×10 + 10 static obstacles + low-battery starts
  L4  PPO   10×10 + 5 static + 3 dynamic obstacles

Run from the project root:
  python -m training.train_nav
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from envs.nav_env import WarehouseEnv
from agents.ppo import DQN, PPO, transfer_dqn_to_ppo, select_action
from utils.replay_buffer import ReplayBuffer
from utils.plotting import plot_nav_history, plot_nav_full_curriculum



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




def train_dqn(level: int, episodes: int = 8000, pretrained=None,
              verbose_every: int = 1000):
    """Double-DQN with soft target updates for nav levels 1–2."""
    GAMMA     = 0.99
    BATCH     = 256
    TAU       = 0.005
    WARMUP    = 3000
    LR        = 5e-4
    EPS_DECAY = 0.9995
    EPS_MIN   = 0.02

    env    = WarehouseEnv(level=level)
    online = DQN(state_dim=13, action_dim=6).to(DEVICE)
    target = DQN(state_dim=13, action_dim=6).to(DEVICE)

    if pretrained is not None:
        online.load_state_dict(pretrained.state_dict())
        epsilon = 0.5
        print(f"  Warm-starting from Level {level-1} weights  (ε=0.5)")
    else:
        epsilon = 1.0

    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer  = optim.Adam(online.parameters(), lr=LR)
    replay     = ReplayBuffer(capacity=100_000)
    ep_rewards, ep_success = [], []

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_r  = 0.0

        while not env.done:
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                with torch.no_grad():
                    action = online(
                        torch.tensor(state, device=DEVICE).unsqueeze(0)
                    ).argmax().item()

            next_s, reward, done, success = env.step(action)
            replay.push(state, action, reward, next_s, done)
            ep_r  += reward
            state  = next_s

            if len(replay) >= WARMUP and len(replay) >= BATCH:
                s, a, r, s_, d = replay.sample(BATCH, DEVICE)
                with torch.no_grad():
                    a_next   = online(s_).argmax(dim=1)
                    q_next   = target(s_).gather(1, a_next.unsqueeze(1)).squeeze(1)
                    q_target = r + GAMMA * q_next * (1 - d)
                q_pred = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss   = F.smooth_l1_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                for po, pt in zip(online.parameters(), target.parameters()):
                    pt.data.copy_(TAU * po.data + (1 - TAU) * pt.data)

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        ep_rewards.append(ep_r)
        ep_success.append(float(success))

        if ep % verbose_every == 0:
            w = verbose_every
            print(f"  L{level} DQN  Ep {ep:5d}/{episodes}"
                  f"  Reward {np.mean(ep_rewards[-w:]):+7.1f}"
                  f"  Success {np.mean(ep_success[-w:]):.2f}"
                  f"  ε={epsilon:.3f}")

    return online, ep_rewards, ep_success


def train_ppo(level: int, model: PPO, episodes: int = 12000,
              clip_eps: float = 0.2, ppo_epochs: int = 4,
              entropy_coef: float = 0.03, gamma: float = 0.99,
              n_rollout: int = 32, lr: float = 1e-4,
              verbose_every: int = 1000):
    """PPO with rollout buffer + cosine LR schedule for nav levels 3–4."""
    env       = WarehouseEnv(level=level)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, episodes // n_rollout), eta_min=lr * 0.1)

    ep_rewards, ep_success = [], []

    for ep_start in range(0, episodes, n_rollout):
        buf_s, buf_a, buf_r, buf_v, buf_lp = [], [], [], [], []
        rollout_success = []

        for _ in range(n_rollout):
            state   = env.reset()
            ep_r    = 0.0
            success = False
            ep_s, ep_a, ep_r_list, ep_v, ep_lp = [], [], [], [], []

            while not env.done:
                st = torch.tensor(state, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    logits, value = model(st)
                logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
                dist   = torch.distributions.Categorical(
                    torch.softmax(logits, dim=-1))
                action = dist.sample()
                log_p  = dist.log_prob(action)

                next_s, reward, done, success = env.step(action.item())
                ep_r += reward
                ep_s.append(state);     ep_a.append(action.item())
                ep_r_list.append(reward); ep_v.append(value.item())
                ep_lp.append(log_p.item())
                state = next_s

            returns = []
            G = 0.0
            for r in reversed(ep_r_list):
                G = r + gamma * G
                returns.insert(0, G)

            buf_s.extend(ep_s);   buf_a.extend(ep_a)
            buf_r.extend(returns); buf_v.extend(ep_v); buf_lp.extend(ep_lp)
            ep_rewards.append(ep_r); rollout_success.append(float(success))
        ep_success.extend(rollout_success)

        s_t   = torch.tensor(np.array(buf_s), device=DEVICE, dtype=torch.float32)
        a_t   = torch.tensor(buf_a,           device=DEVICE, dtype=torch.long)
        r_t   = torch.tensor(buf_r,           device=DEVICE, dtype=torch.float32)
        v_t   = torch.tensor(buf_v,           device=DEVICE, dtype=torch.float32)
        lp_t  = torch.tensor(buf_lp,          device=DEVICE, dtype=torch.float32)
        adv_t = r_t - v_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(ppo_epochs):
            idx = torch.randperm(len(s_t), device=DEVICE)
            for mb in idx.split(256):
                logits, vals = model(s_t[mb])
                logits = torch.clamp(logits, -20.0, 20.0)
                dist   = torch.distributions.Categorical(
                    torch.softmax(logits, dim=-1))
                new_lp  = dist.log_prob(a_t[mb])
                entropy = dist.entropy().mean()
                ratio   = torch.exp(new_lp - lp_t[mb])
                adv     = adv_t[mb]
                actor_loss  = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
                ).mean()
                critic_loss = 0.5 * F.mse_loss(vals.squeeze(-1), r_t[mb])
                loss        = actor_loss + critic_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        scheduler.step()

        ep_done = ep_start + n_rollout
        if ep_done % verbose_every < n_rollout:
            w = verbose_every
            print(f"  L{level} PPO  Ep {ep_done:5d}/{episodes}"
                  f"  Reward {np.mean(ep_rewards[-w:]):+7.1f}"
                  f"  Success {np.mean(ep_success[-w:]):.2f}")

    return model, ep_rewards, ep_success




def main():
    print(f"Device : {DEVICE}")
    print("=" * 60)
    print("  Stage 1 — Navigation Curriculum  (L1 → L4)")
    print("=" * 60)

    all_rewards = {}

    # ── Level 1: DQN 5×5 ─────────────────────────────────────────────────
    print("\n[L1] DQN  5×5 plain grid  (8 000 episodes)")
    dqn_l1, r1, s1 = train_dqn(level=1, episodes=8_000)
    all_rewards[1] = r1
    torch.save(dqn_l1.state_dict(), os.path.join(CKPT_DIR, "dqn_l1.pt"))
    plot_nav_history(r1, s1, level=1, out_dir=CKPT_DIR)

    # ── Level 2: DQN 10×10 (warm-start from L1) ──────────────────────────
    print("\n[L2] DQN  10×10 plain grid  (8 000 episodes)")
    dqn_l2, r2, s2 = train_dqn(level=2, episodes=8_000, pretrained=dqn_l1)
    all_rewards[2] = r2
    torch.save(dqn_l2.state_dict(), os.path.join(CKPT_DIR, "dqn_l2.pt"))
    plot_nav_history(r2, s2, level=2, out_dir=CKPT_DIR)

    # ── Level 3: PPO 10×10 + static obstacles ────────────────────────────
    print("\n[L3] PPO  10×10 + static obstacles  (20 000 episodes)")
    ppo_l3 = PPO(state_dim=13, action_dim=6).to(DEVICE)
    transfer_dqn_to_ppo(dqn_l2, ppo_l3)
    ppo_l3, r3, s3 = train_ppo(level=3, model=ppo_l3, episodes=20_000,
                                lr=1e-4, entropy_coef=0.03)
    all_rewards[3] = r3
    torch.save(ppo_l3.state_dict(), os.path.join(CKPT_DIR, "ppo_l3.pt"))
    plot_nav_history(r3, s3, level=3, out_dir=CKPT_DIR)

    # ── Level 4: PPO 10×10 + static + dynamic obstacles ──────────────────
    print("\n[L4] PPO  10×10 + dynamic obstacles  (16 000 episodes)")
    ppo_l4 = PPO(state_dim=13, action_dim=6).to(DEVICE)
    ppo_l4.load_state_dict(ppo_l3.state_dict())
    ppo_l4, r4, s4 = train_ppo(level=4, model=ppo_l4, episodes=16_000)
    all_rewards[4] = r4
    torch.save(ppo_l4.state_dict(), os.path.join(CKPT_DIR, "ppo_final.pt"))
    print(f"\n✓  Saved final nav policy → {CKPT_DIR}/ppo_final.pt")
    plot_nav_history(r4, s4, level=4, out_dir=CKPT_DIR)

    # ── Full curriculum overview ──────────────────────────────────────────
    plot_nav_full_curriculum(all_rewards, out_dir=CKPT_DIR)

    # Save training histories
    for lvl, arr in zip([1, 2, 3, 4], [r1, r2, r3, r4]):
        np.save(os.path.join(CKPT_DIR, f"nav_l{lvl}_rewards.npy"), np.array(arr))
    print("✓  Saved reward histories → checkpoints/nav_l*.npy")


if __name__ == "__main__":
    main()
