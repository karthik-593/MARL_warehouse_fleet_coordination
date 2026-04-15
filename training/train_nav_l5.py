"""
Stage 1 — Level 5 warehouse navigation fine-tuning.
Fine-tunes ppo_final.pt on the real warehouse shelf layout.
70% L5 (warehouse) + 30% L3 (random obstacles) per rollout.

Run:  python -m training.train_nav_l5
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.nav_env import WarehouseEnv
from agents.ppo import PPO



SEED       = 42
CKPT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "checkpoints")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES      = 15_000
N_ROLLOUT     = 32
CLIP_EPS      = 0.2
PPO_EPOCHS    = 4
GAMMA         = 0.99
LR            = 3e-5
ENTROPY_COEF  = 0.02
VERBOSE_EVERY = 1_000
MIX_WAREHOUSE = 0.70

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



def train_l5_mixed(model: PPO, episodes: int = EPISODES,
                   n_rollout: int = N_ROLLOUT) -> tuple:
    """
    PPO fine-tuning with mixed L5 (warehouse) + L3 (random obstacles) rollouts.
    No LR scheduler — constant fine-tuning rate.
    """
    env_wh = WarehouseEnv(level=5)   # warehouse layout
    env_rn = WarehouseEnv(level=3)   # random obstacles (L3)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    ep_rewards, ep_success = [], []
    n_wh = round(n_rollout * MIX_WAREHOUSE)
    n_rn = n_rollout - n_wh

    for ep_start in range(0, episodes, n_rollout):
        buf_s, buf_a, buf_r, buf_v, buf_lp = [], [], [], [], []
        rollout_success = []

        for env, count in [(env_wh, n_wh), (env_rn, n_rn)]:
            for _ in range(count):
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
                    ep_s.append(state);       ep_a.append(action.item())
                    ep_r_list.append(reward); ep_v.append(value.item())
                    ep_lp.append(log_p.item())
                    state = next_s

                returns = []
                G = 0.0
                for r in reversed(ep_r_list):
                    G = r + GAMMA * G
                    returns.insert(0, G)

                buf_s.extend(ep_s);    buf_a.extend(ep_a)
                buf_r.extend(returns); buf_v.extend(ep_v)
                buf_lp.extend(ep_lp)
                ep_rewards.append(ep_r)
                rollout_success.append(float(success))

        ep_success.extend(rollout_success)

        s_t   = torch.tensor(np.array(buf_s), device=DEVICE, dtype=torch.float32)
        a_t   = torch.tensor(buf_a,           device=DEVICE, dtype=torch.long)
        r_t   = torch.tensor(buf_r,           device=DEVICE, dtype=torch.float32)
        v_t   = torch.tensor(buf_v,           device=DEVICE, dtype=torch.float32)
        lp_t  = torch.tensor(buf_lp,          device=DEVICE, dtype=torch.float32)
        adv_t = r_t - v_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
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
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv,
                ).mean()
                critic_loss = 0.5 * F.mse_loss(vals.squeeze(-1), r_t[mb])
                loss        = actor_loss + critic_loss - ENTROPY_COEF * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        ep_done = ep_start + n_rollout
        if ep_done % VERBOSE_EVERY < n_rollout:
            w = VERBOSE_EVERY
            print(f"  L5 mix  Ep {ep_done:5d}/{episodes}"
                  f"  Reward {np.mean(ep_rewards[-w:]):+7.1f}"
                  f"  Success {np.mean(ep_success[-w:]):.2f}")

    return model, ep_rewards, ep_success



def evaluate_nav(model: PPO, level: int, n_eval: int = 200) -> dict:
    """Quick evaluation — returns mean steps and success rate."""
    env = WarehouseEnv(level=level)
    steps_list, success_list = [], []
    model.eval()
    for _ in range(n_eval):
        state   = env.reset()
        success = False
        steps   = 0
        while not env.done:
            st = torch.tensor(state, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(st)
            logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
            probs  = torch.softmax(logits / 0.3, dim=-1)
            action = torch.distributions.Categorical(probs).sample().item()
            state, _, _, success = env.step(action)
            steps += 1
        steps_list.append(steps)
        success_list.append(float(success))
    model.train()
    return {
        "success_rate": float(np.mean(success_list)),
        "mean_steps":   float(np.mean(steps_list)),
    }



def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Device      : {DEVICE}")
    print(f"Checkpoints : {CKPT_DIR}")
    print("=" * 60)
    print("  Stage 1 — Level 5  Warehouse Nav Fine-tuning")
    print(f"  Mix: {int(MIX_WAREHOUSE*100)}% warehouse + "
          f"{int((1-MIX_WAREHOUSE)*100)}% random  |  LR={LR}  |  {EPISODES} eps")
    print("=" * 60)

    # Load existing nav policy
    base_ckpt = os.path.join(CKPT_DIR, "ppo_final.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_ckpt}\n"
            "Run training/train_nav.py first (L1–L4).")
    model = PPO(state_dim=13, action_dim=6).to(DEVICE)
    model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))
    print(f"  Loaded base nav policy  ← {base_ckpt}")

    # Baseline evaluation before fine-tuning
    print("\n  Baseline evaluation (before L5):")
    for lvl in [3, 5]:
        try:
            m = evaluate_nav(model, level=lvl, n_eval=200)
            print(f"    L{lvl}  success={m['success_rate']:.2f}  "
                  f"mean_steps={m['mean_steps']:.1f}")
        except Exception as e:
            print(f"    L{lvl}  eval failed: {e}")

    # Fine-tune
    print(f"\n  Fine-tuning for {EPISODES} episodes...")
    model, rewards, successes = train_l5_mixed(model, episodes=EPISODES)

    # Post fine-tune evaluation
    print("\n  Post-training evaluation (after L5):")
    for lvl in [3, 5]:
        try:
            m = evaluate_nav(model, level=lvl, n_eval=200)
            print(f"    L{lvl}  success={m['success_rate']:.2f}  "
                  f"mean_steps={m['mean_steps']:.1f}")
        except Exception as e:
            print(f"    L{lvl}  eval failed: {e}")

    # Save
    out_l5    = os.path.join(CKPT_DIR, "ppo_l5_warehouse.pt")
    out_final = os.path.join(CKPT_DIR, "ppo_final.pt")
    torch.save(model.state_dict(), out_l5)
    torch.save(model.state_dict(), out_final)
    np.save(os.path.join(CKPT_DIR, "nav_l5_rewards.npy"), np.array(rewards))
    print(f"\n✓  Saved  {out_l5}")
    print(f"✓  Overwritten  {out_final}")
    print("    Stage 2/3 will now use the warehouse-aware nav policy.")


if __name__ == "__main__":
    main()
