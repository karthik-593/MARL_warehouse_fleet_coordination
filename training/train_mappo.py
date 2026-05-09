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

_CKPT_FILE = "mappo_ckpt.pt"




def load_nav_policy(path: str = None) -> PPO:
    if path is None:
        path = os.path.join(CKPT_DIR, "ppo_final.pt")
    nav = PPO(state_dim=13, action_dim=6).to("cpu")   # batch-size-1 sequential calls — CPU wins
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\nRun Stage 1 first: python -m training.train_nav")
    nav.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
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
    dqn.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    print(f"  Loaded assign DQN    <- {path}")
    return dqn




def train_mappo(
    nav_model:     PPO,
    assign_dqn:    AssignmentDQN,
    episodes:      int   = 10_000,
    n_rollout:     int   = 16,
    clip_eps:      float = 0.2,
    val_clip:      float = 50.0,       # PPO value-clip half-range (reward-scale units)
    ppo_epochs:    int   = 4,
    gamma:         float = 0.99,
    gae_lam:       float = 0.95,
    lr_actor:      float = 5e-5,
    lr_critic:     float = 5e-5,
    entropy_start: float = 0.05,
    entropy_end:   float = 0.01,
    kl_target:     float = 0.015,
    verbose_every: int   = 500,
    resume_ckpt:   str   = None,
) -> tuple:
    """Train MAPPO. Returns (actor, critic, history_dict)."""

    # ── Fix 8: Startup sanity print ──────────────────────────────────────
    print(f"  OBS_DIM={OBS_DIM}  GLOBAL_DIM={GLOBAL_DIM}  N_AGENTS={N_AGENTS}")
    if GLOBAL_DIM == N_AGENTS * OBS_DIM:
        print("  [WARN] global state = concat of local obs — critic has no extra info")

    env    = MultiAgentWarehouse()
    actor  = AssignmentActor(obs_dim=OBS_DIM, action_dim=3).to(DEVICE)
    critic = CentralisedCritic(global_dim=GLOBAL_DIM).to(DEVICE)
    opt_a  = optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_c  = optim.Adam(critic.parameters(), lr=lr_critic)

    start_ep = 0
    history = {
        "team_rewards":    [],
        "actor_losses":    [],
        "critic_losses":   [],
        "accept_rates":    [],
        "charge_rates":    [],
        "delivery_rates":  [],
        "breakdown_rates": [],
    }

    # ── Fix 2: Restore full checkpoint (weights + optimisers + ep_start) ─
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        opt_a.load_state_dict(ckpt["opt_a"])
        opt_c.load_state_dict(ckpt["opt_c"])
        start_ep = ckpt.get("ep_done", 0)
        for key in history:
            arr_path = os.path.join(CKPT_DIR, f"mappo_{key}.npy")
            if os.path.exists(arr_path):
                history[key] = np.load(arr_path).tolist()
        print(f"  Resumed from checkpoint: ep {start_ep}  ({resume_ckpt})")
    else:
        transfer_dqn_to_actor(assign_dqn, actor)
        print("  Fresh start  (ep 0) — warm-started actor from Stage-2 DQN weights")

    # ── Fix 4: tracker-based verbose / checkpoint triggers ───────────────
    last_log_ep  = start_ep - verbose_every   # ensures first print fires promptly
    last_ckpt_ep = start_ep

    for ep_start in range(start_ep, episodes, n_rollout):
        ep_done = ep_start + n_rollout

        # ── Fix 3: Entropy schedule ──────────────────────────────────────
        frac = min(ep_start / max(episodes // 2, 1), 1.0)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * frac

        # rollout buffer
        buf_obs  = []   # [N_AGENTS, OBS_DIM]  per step
        buf_gs   = []   # [GLOBAL_DIM]          per step
        buf_acts = []   # [N_AGENTS]            per step
        buf_lp   = []   # [N_AGENTS]            per step  (log probs)
        buf_ret  = []   # [N_AGENTS]            per step  (per-agent GAE return)
        buf_val  = []   # scalar                per step  (V_old, raw scale)

        roll_team_r  = []
        roll_accepts = []
        roll_charges = []
        roll_delivs  = []
        roll_brkdwns = []

        for _ in range(n_rollout):
            obs, gs   = env.reset()
            ep_trans  = []      # (obs, gs, acts_np, lp_np, rewards_np[N_A], val)
            ep_team_r = 0.0
            ep_acc = ep_chg = ep_del = ep_brk = 0

            while True:
                obs_t = torch.tensor(obs, device=DEVICE,
                                     dtype=torch.float32)       # [N_A, OBS_DIM]
                gs_t  = torch.tensor(gs,  device=DEVICE,
                                     dtype=torch.float32).unsqueeze(0)  # [1, GLOBAL_DIM]

                with torch.no_grad():
                    logits = actor(obs_t)           # [N_A, 3]
                    value  = critic(gs_t).squeeze() # scalar, raw reward scale

                logits = torch.clamp(logits, -20.0, 20.0)
                dist   = torch.distributions.Categorical(
                    torch.softmax(logits, dim=-1))
                acts   = dist.sample()              # [N_A]
                lp     = dist.log_prob(acts)        # [N_A]

                next_obs, next_gs, rewards, done = env.step(
                    acts.tolist(), nav_model)
                rewards_np = np.array(rewards, dtype=np.float32)  # [N_A]

                ep_trans.append((
                    obs, gs,
                    acts.cpu().numpy(), lp.cpu().numpy(),
                    rewards_np, value.item()
                ))

                ep_team_r += float(rewards_np.sum())
                # ── Fix 5: track breakdowns separately ───────────────────
                for a, r in zip(acts.tolist(), rewards):
                    if a == 0:
                        ep_acc += 1
                        if r > 50:
                            ep_del += 1
                        elif r < -50:
                            ep_brk += 1
                    elif a == 2:
                        ep_chg += 1

                obs, gs = next_obs, next_gs
                if done:
                    break

            # Per-agent GAE: individual reward streams, shared critic baseline
            gae_vec = np.zeros(N_AGENTS, dtype=np.float32)
            ep_rets = []
            T_ep    = len(ep_trans)
            ep_vals = [v for *_, v in ep_trans]
            for t in reversed(range(T_ep)):
                r_vec = ep_trans[t][4]                          # [N_AGENTS]
                v_nxt = ep_vals[t + 1] if t + 1 < T_ep else 0.0
                delta = r_vec + gamma * v_nxt - ep_vals[t]     # [N_AGENTS]
                gae_vec = delta + gamma * gae_lam * gae_vec    # [N_AGENTS]
                ep_rets.insert(0, gae_vec.copy() + ep_vals[t]) # [N_AGENTS]

            for i, (o, g, a, lp_, mr, v) in enumerate(ep_trans):
                buf_obs.append(o)
                buf_gs.append(g)
                buf_acts.append(a)
                buf_lp.append(lp_)
                buf_ret.append(ep_rets[i])
                buf_val.append(v)

            total_dec = N_AGENTS * N_ORDERS
            roll_team_r.append(ep_team_r)
            roll_accepts.append(ep_acc  / total_dec)
            roll_charges.append(ep_chg  / total_dec)
            roll_delivs.append(ep_del   / max(ep_acc, 1))
            roll_brkdwns.append(ep_brk  / max(ep_acc, 1))

        # Prepare tensors
        T      = len(buf_obs)
        T_full = (T // 64) * 64 or T   # drop remainder; fall back to T if T < 64

        obs_t  = torch.tensor(np.array(buf_obs),  device=DEVICE,
                               dtype=torch.float32)   # [T, N_A, OBS_DIM]
        gs_t   = torch.tensor(np.array(buf_gs),   device=DEVICE,
                               dtype=torch.float32)   # [T, GLOBAL_DIM]
        act_t  = torch.tensor(np.array(buf_acts), device=DEVICE,
                               dtype=torch.long)       # [T, N_A]
        lp_t   = torch.tensor(np.array(buf_lp),   device=DEVICE,
                               dtype=torch.float32)   # [T, N_A]
        ret_t  = torch.tensor(np.array(buf_ret),  device=DEVICE,
                               dtype=torch.float32)   # [T, N_AGENTS]  per-agent GAE returns
        val_t  = torch.tensor(np.array(buf_val),  device=DEVICE,
                               dtype=torch.float32)   # [T]  V_old, raw scale

        adv_t  = ret_t - val_t.unsqueeze(1)           # [T, N_AGENTS]
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO update
        a_loss_sum = c_loss_sum = 0.0
        kl_stop = False

        for epoch in range(ppo_epochs):
            # ── Fix 7: KL early-stopping ──────────────────────────────────
            if kl_stop:
                break
            idx      = torch.randperm(T, device=DEVICE)
            epoch_kl = 0.0
            n_mb     = 0

            for mb in idx[:T_full].split(64):
                # Actor — flatten agents into batch dim
                obs_mb = obs_t[mb].reshape(-1, OBS_DIM)   # [mb*N_A, OBS_DIM]
                act_mb = act_t[mb].reshape(-1)             # [mb*N_A]
                lp_mb  = lp_t[mb].reshape(-1)             # [mb*N_A]
                adv_mb = adv_t[mb].reshape(-1)             # [mb*N_A] per-agent adv

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

                epoch_kl += (lp_mb - new_lp.detach()).mean().item()
                n_mb     += 1

                # Critic — PPO-style value clipping; target = mean per-agent return
                val_mb     = critic(gs_t[mb]).squeeze(-1)          # [mb]
                ret_mb_avg = ret_t[mb].mean(dim=1)                 # [mb]
                val_clp    = val_t[mb] + torch.clamp(
                    val_mb - val_t[mb], -val_clip, val_clip)
                c_loss  = 0.5 * torch.max(
                    F.mse_loss(val_mb,  ret_mb_avg),
                    F.mse_loss(val_clp, ret_mb_avg),
                )

                opt_c.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_c.step()
                c_loss_sum += c_loss.item()

            if n_mb > 0 and epoch_kl / n_mb > kl_target:
                print(f"  [KL early-stop] ep {ep_done}  epoch {epoch+1}/{ppo_epochs}"
                      f"  mean KL={epoch_kl/n_mb:.4f}")
                kl_stop = True

        # History
        history["team_rewards"].extend(roll_team_r)
        history["accept_rates"].extend(roll_accepts)
        history["charge_rates"].extend(roll_charges)
        history["delivery_rates"].extend(roll_delivs)
        history["breakdown_rates"].extend(roll_brkdwns)
        history["actor_losses"].append(a_loss_sum)
        history["critic_losses"].append(c_loss_sum)

        # ── Fix 4: tracker-based print ────────────────────────────────────
        if ep_done - last_log_ep >= verbose_every:
            last_log_ep = ep_done
            w = verbose_every
            print(
                f"  Ep {ep_done:5d}/{episodes}"
                f"  TeamR {np.mean(history['team_rewards'][-w:]):+8.1f}"
                f"  Accept {np.mean(history['accept_rates'][-w:]):.2f}"
                f"  Charge {np.mean(history['charge_rates'][-w:]):.2f}"
                f"  Deliv {np.mean(history['delivery_rates'][-w:]):.2f}"
                f"  Brk {np.mean(history['breakdown_rates'][-w:]):.2f}"
                f"  Ent {entropy_coef:.3f}"
            )

        # ── Fix 4: tracker-based checkpoint ──────────────────────────────
        if ep_done - last_ckpt_ep >= 1000:
            last_ckpt_ep = ep_done
            ckpt_path = os.path.join(CKPT_DIR, _CKPT_FILE)
            # Fix 2: Save full state including optimiser buffers and episode
            torch.save({
                "actor":   actor.state_dict(),
                "critic":  critic.state_dict(),
                "opt_a":   opt_a.state_dict(),
                "opt_c":   opt_c.state_dict(),
                "ep_done": ep_done,
            }, ckpt_path)
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

    # Fix 2: single combined checkpoint file
    ckpt_path   = os.path.join(CKPT_DIR, _CKPT_FILE)
    resume_ckpt = ckpt_path if os.path.exists(ckpt_path) else None

    actor, critic, history = train_mappo(
        nav_model   = nav_model,
        assign_dqn  = assign_dqn,
        episodes    = 50_000,
        resume_ckpt = resume_ckpt,
    )
    print("\nTraining complete.")

    eval_metrics = evaluate_mappo(nav_model, actor)
    plot_mappo_history(history, out_dir=CKPT_DIR)

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
