"""
Standalone Level 4 nav retraining.
Loads ppo_l3.pt, trains L4 (20k episodes), saves ppo_final.pt.

Run from project root:
  python -m training.train_nav_l4
"""

import os
import numpy as np
import torch

from agents.ppo import PPO
from training.train_nav import train_ppo, CKPT_DIR, DEVICE
from utils.plotting import plot_nav_history


def main():
    l3_path = os.path.join(CKPT_DIR, "ppo_l3.pt")
    if not os.path.exists(l3_path):
        raise FileNotFoundError(
            f"ppo_l3.pt not found at {l3_path}. Run the full curriculum first."
        )

    print(f"Device : {DEVICE}")
    print(f"Loading L3 weights from {l3_path}")

    ppo_l4 = PPO(state_dim=13, action_dim=6).to(DEVICE)
    ppo_l4.load_state_dict(torch.load(l3_path, weights_only=True))

    print("\n[L4] PPO  10×10 + dynamic obstacles  (20 000 episodes)")
    ppo_l4, r4, s4 = train_ppo(level=4, model=ppo_l4, episodes=20_000)

    out = os.path.join(CKPT_DIR, "ppo_final.pt")
    torch.save(ppo_l4.state_dict(), out)
    np.save(os.path.join(CKPT_DIR, "nav_l4_rewards.npy"), np.array(r4))
    plot_nav_history(r4, s4, level=4, out_dir=CKPT_DIR)
    print(f"\n✓  Saved → {out}")


if __name__ == "__main__":
    main()
