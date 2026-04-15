"""
Stage 3 — MAPPO Training (3 robots, K=2 dispatch, CTDE)

Run:  python -m training.train_mappo
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.marl_env import MultiAgentWarehouse, N_AGENTS, N_ORDERS, OBS_DIM, GLOBAL_DIM
from agents.ppo import PPO
from agents.dqn import AssignmentDQN
from agents.mappo import AssignmentActor, CentralisedCritic, transfer_dqn_to_actor
from utils.plotting import plot_mappo_history



SEED     = 42
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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\nRun Stage 1 first: python -m training.train_nav")
    nav.load_state_dict(torch.load(path, map_location=DEVICE))
    nav.eval()
    for p in nav.parameters():
        p.requires_grad_(False)
    print(f"  Loaded nav policy    <- {path}")
    return nav


def load_assign_dqn(path: str = None) -> AssignmentDQN:
    if path is None:
        path = os.path.join(CKPT_DIR, "assign_dqn.pt")
    dqn = AssignmentDQN().to(DEVICE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\nRun Stage 2 first: python -m training.train_assign")
    dqn.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"  Loaded assign DQN    <- {path}")
    return dqn




def train_mappo(
    nav_model:     PPO,
    assign_dqn:    AssignmentDQN,
    episodes:      int   = 10_000,
    n_rollout:     int   = 16,
    clip_eps:      float = 0.2,
    ppo_epochs:    int   = 4,
    gamma:         float = 0.99,
    lr_actor:      float = 5e-5,
    lr_critic:     float = 1e-4,
    entropy_coef:  float = 0.02,
    verbose_every: int   = 500,
    resume_actor:  str   = None,
    resume_critic: str   = None,
) -> tuple:
    """Train MAPPO. Returns (actor, critic, history_dict)."""
    env    = MultiAgentWarehouse()
    actor  = AssignmentActor(obs_dim=OBS_DIM, action_dim=3).to(DEVICE)
    critic = CentralisedCritic(global_dim=GLOBAL_DIM).to(DEVICE)

    if resume_actor and resume_critic:
        actor.load_state_dict(torch.load(resume_actor,  map_location=DEVICE))
        critic.load_state_dict(torch.load(resume_critic, map_location=DEVICE))
        print(f"  Resumed from checkpoint: {resume_actor}")
    else:
        transfer_dqn_to_actor(assign_dqn, actor)
        print("  Warm-started actor from Stage-2 DQN weights")

    opt_a = optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=lr_critic)

    history = {
        "team_rewards":   [],
        "actor_losses":   [],
        "critic_losses":  [],
        "accept_rates":   [],
        "charge_rates":   [],
        "delivery_rates": [],
    }

    for ep_start in range(0, episodes, n_rollout):

        # rollout buffer
        buf_obs  = []   # [N_AGENTS, OBS_DIM]  per step
        buf_gs   = []   # [GLOBAL_DIM]          per step
        buf_acts = []   # [N_AGENTS]            per step
        buf_lp   = []   # [N_AGENTS]            per step  (log probs)
        buf_ret  = []   # scalar                per step  (MC return)
        buf_val  = []   # scalar                per step  (critic estimate)

        roll_team_r  = []
        roll_accepts = []
        roll_charges = []
        roll_delivs  = []

        for _ in range(n_rollout):
            obs, gs    = env.reset()
            ep_trans   = []      # (obs, gs, acts_np, lp_np, mean_r, val)
            ep_team_r  = 0.0
            ep_acc = ep_chg = ep_del = 0

            while True:
                obs_t = torch.tensor(obs, device=DEVICE,
                                     dtype=torch.float32)       # [N_A, 9]
                gs_t  = torch.tensor(gs,  device=DEVICE,
                                     dtype=torch.float32).unsqueeze(0)  # [1, 27]

                with torch.no_grad():
                    logits = actor(obs_t)           # [N_A, 3]
                    value  = critic(gs_t).squeeze() # scalar

                logits = torch.clamp(logits, -20.0, 20.0)
                dist   = torch.distributions.Categorical(
                    torch.softmax(logits, dim=-1))
                acts   = dist.sample()              # [N_A]
                lp     = dist.log_prob(acts)        # [N_A]

                next_obs, next_gs, rewards, done = env.step(
                    acts.tolist(), nav_model)
                mean_r = float(np.mean(rewards)) / N_AGENTS   # normalise scale

                ep_trans.append((
                    obs, gs,
                    acts.cpu().numpy(), lp.cpu().numpy(),
                    mean_r, value.item()
                ))

                ep_team_r += float(np.sum(rewards))
                for a, r in zip(acts.tolist(), rewards):
                    if a == 0:
                        ep_acc += 1
                        if r > 50:
                            ep_del += 1
                    elif a == 2:
                        ep_chg += 1

                obs, gs = next_obs, next_gs
                if done:
                    break

            # Monte-Carlo returns for this episode
            G       = 0.0
            ep_rets = []
            for *_, mr, _ in reversed(ep_trans):
                G = mr + gamma * G
                ep_rets.insert(0, G)

            for i, (o, g, a, lp_, mr, v) in enumerate(ep_trans):
                buf_obs.append(o)
                buf_gs.append(g)
                buf_acts.append(a)
                buf_lp.append(lp_)
                buf_ret.append(ep_rets[i])
                buf_val.append(v)

            total_dec = N_AGENTS * N_ORDERS
            roll_team_r.append(ep_team_r)
            roll_accepts.append(ep_acc / total_dec)
            roll_charges.append(ep_chg / total_dec)
            roll_delivs.append(ep_del / max(ep_acc, 1))

        # prepare tensors
        T      = len(buf_obs)
        obs_t  = torch.tensor(np.array(buf_obs),  device=DEVICE,
                               dtype=torch.float32)   # [T, N_A, OBS_DIM]
        gs_t   = torch.tensor(np.array(buf_gs),   device=DEVICE,
                               dtype=torch.float32)   # [T, GLOBAL_DIM]
        act_t  = torch.tensor(np.array(buf_acts), device=DEVICE,
                               dtype=torch.long)       # [T, N_A]
        lp_t   = torch.tensor(np.array(buf_lp),   device=DEVICE,
                               dtype=torch.float32)   # [T, N_A]
        ret_t  = torch.tensor(np.array(buf_ret),  device=DEVICE,
                               dtype=torch.float32)   # [T]
        val_t  = torch.tensor(np.array(buf_val),  device=DEVICE,
                               dtype=torch.float32)   # [T]

        adv_t  = ret_t - val_t
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ppo update
        a_loss_sum = c_loss_sum = 0.0

        for _ in range(ppo_epochs):
            idx = torch.randperm(T, device=DEVICE)
            for mb in idx.split(64):
                # Actor — flatten agents into batch dim
                obs_mb = obs_t[mb].reshape(-1, OBS_DIM)   # [mb*N_A, OBS_DIM]
                act_mb = act_t[mb].reshape(-1)             # [mb*N_A]
                lp_mb  = lp_t[mb].reshape(-1)             # [mb*N_A]
                adv_mb = adv_t[mb].unsqueeze(1).expand(
                    -1, N_AGENTS).reshape(-1)              # [mb*N_A]

                logits  = actor(obs_mb)
                logits  = torch.clamp(logits, -20.0, 20.0)
                dist    = torch.distributions.Categorical(
                    torch.softmax(logits, dim=-1))
                new_lp  = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()
                ratio   = torch.exp(new_lp - lp_mb)

                a_loss  = -torch.min(
                    ratio * adv_mb,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb,
                ).mean() - entropy_coef * entropy

                opt_a.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()
                a_loss_sum += a_loss.item()

                # Critic
                val_mb = critic(gs_t[mb]).squeeze(-1)
                c_loss = 0.5 * F.mse_loss(val_mb, ret_t[mb])

                opt_c.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_c.step()
                c_loss_sum += c_loss.item()

        # history
        history["team_rewards"].extend(roll_team_r)
        history["accept_rates"].extend(roll_accepts)
        history["charge_rates"].extend(roll_charges)
        history["delivery_rates"].extend(roll_delivs)
        history["actor_losses"].append(a_loss_sum)
        history["critic_losses"].append(c_loss_sum)

        ep_done = ep_start + n_rollout
        if ep_done % verbose_every < n_rollout:
            w = verbose_every
            print(
                f"  Ep {ep_done:5d}/{episodes}"
                f"  TeamR {np.mean(history['team_rewards'][-w:]):+8.1f}"
                f"  Accept {np.mean(history['accept_rates'][-w:]):.2f}"
                f"  Charge {np.mean(history['charge_rates'][-w:]):.2f}"
                f"  Deliv {np.mean(history['delivery_rates'][-w:]):.2f}"
            )

        # checkpoint every 1000 episodes
        if ep_done % 1000 < n_rollout:
            torch.save(actor.state_dict(),
                       os.path.join(CKPT_DIR, "mappo_actor_ckpt.pt"))
            torch.save(critic.state_dict(),
                       os.path.join(CKPT_DIR, "mappo_critic_ckpt.pt"))
            for key, arr in history.items():
                np.save(os.path.join(CKPT_DIR, f"mappo_{key}.npy"),
                        np.array(arr))
            print(f"  [ckpt] saved at ep {ep_done}")

    return actor, critic, history




def evaluate_mappo(nav_model: PPO, actor: AssignmentActor,
                   n_eval: int = 200) -> dict:
    """Greedy evaluation of the trained MAPPO actor."""
    env = MultiAgentWarehouse()
    actor.eval()
    records = []

    for _ in range(n_eval):
        obs, _ = env.reset()
        ep = {"team_reward": 0.0, "deliveries": 0, "breakdowns": 0,
              "accepts": 0, "idles": 0, "charges": 0}

        while True:
            obs_t = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                actions = actor(obs_t).argmax(dim=-1).tolist()

            next_obs, _, rewards, done = env.step(actions, nav_model)

            for a, r in zip(actions, rewards):
                ep["team_reward"] += r
                if a == 0:
                    ep["accepts"] += 1
                    if r > 50:
                        ep["deliveries"] += 1
                    if r < -50:
                        ep["breakdowns"] += 1
                elif a == 1:
                    ep["idles"] += 1
                else:
                    ep["charges"] += 1

            obs = next_obs
            if done:
                break

        records.append(ep)

    actor.train()
    total_dec = N_AGENTS * N_ORDERS
    metrics = {
        "team_reward":  np.mean([r["team_reward"]  for r in records]),
        "deliveries":   np.mean([r["deliveries"]   for r in records]),
        "breakdowns":   np.mean([r["breakdowns"]   for r in records]),
        "accept_rate":  np.mean([r["accepts"]      for r in records]) / total_dec,
        "idle_rate":    np.mean([r["idles"]        for r in records]) / total_dec,
        "charge_rate":  np.mean([r["charges"]      for r in records]) / total_dec,
    }

    print("─" * 58)
    print(f"  Stage 3 — MAPPO Evaluation  (n={n_eval} greedy episodes)")
    print("─" * 58)
    labels = {
        "team_reward":  "Mean team reward / episode",
        "deliveries":   "Deliveries / episode  (all agents)",
        "breakdowns":   "Breakdowns / episode",
        "accept_rate":  "Accept rate",
        "idle_rate":    "Idle rate",
        "charge_rate":  "GoCharge rate (learned)",
    }
    for k, lbl in labels.items():
        print(f"  {lbl:<38}: {metrics[k]:.4f}")
    print("─" * 58)
    return metrics




def main():
    print(f"Device : {DEVICE}")
    print("=" * 60)
    print(f"  Stage 3 — MAPPO  ({N_AGENTS} agents, K={2} nearest eligible)")
    print("=" * 60)

    nav_model  = load_nav_policy()
    assign_dqn = load_assign_dqn()

    # Resume from mid-training checkpoint if available
    ckpt_actor  = os.path.join(CKPT_DIR, "mappo_actor_ckpt.pt")
    ckpt_critic = os.path.join(CKPT_DIR, "mappo_critic_ckpt.pt")
    resume_actor  = ckpt_actor  if os.path.exists(ckpt_actor)  else None
    resume_critic = ckpt_critic if os.path.exists(ckpt_critic) else None

    actor, critic, history = train_mappo(
        nav_model     = nav_model,
        assign_dqn    = assign_dqn,
        episodes      = 10_000,
        resume_actor  = resume_actor,
        resume_critic = resume_critic,
    )
    print("\nTraining complete.")

    eval_metrics = evaluate_mappo(nav_model, actor)
    plot_mappo_history(history, out_dir=CKPT_DIR)

    # save────
    torch.save(actor.state_dict(),
               os.path.join(CKPT_DIR, "mappo_actor.pt"))
    torch.save(critic.state_dict(),
               os.path.join(CKPT_DIR, "mappo_critic.pt"))
    print(f"  Saved actor  -> {CKPT_DIR}/mappo_actor.pt")
    print(f"  Saved critic -> {CKPT_DIR}/mappo_critic.pt")

    for key, arr in history.items():
        np.save(os.path.join(CKPT_DIR, f"mappo_{key}.npy"), np.array(arr))
    print("  Saved history arrays -> checkpoints/mappo_*.npy")

    with open(os.path.join(CKPT_DIR, "mappo_eval.json"), "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items()}, f, indent=2)
    print("  Saved eval metrics   -> checkpoints/mappo_eval.json")


if __name__ == "__main__":
    main()
