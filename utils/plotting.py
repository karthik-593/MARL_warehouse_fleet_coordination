"""
Shared plotting utilities for Stage 1 (navigation) and Stage 2 (assignment).
All functions save the figure to disk and also call plt.show() for notebook use.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smooth(x, window: int = 100) -> np.ndarray:
    """Rolling-mean smoothing for training curves."""
    x = np.asarray(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------------
# Stage 1 — Navigation training curves
# ---------------------------------------------------------------------------
def plot_nav_history(rewards: list, successes: list, level: int,
                     out_dir: str = "checkpoints"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Stage 1 — Level {level} Training", fontweight="bold")

    axes[0].plot(smooth(rewards), color="#1565C0", lw=1.5)
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].axhline(0, color="black", lw=0.5, ls="--")
    axes[0].grid(alpha=0.3)

    axes[1].plot(smooth(successes, window=50), color="#2E7D32", lw=1.5)
    axes[1].set_title("Success Rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f"nav_l{level}_curves.png")
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")
    plt.show()


def plot_nav_full_curriculum(all_rewards: dict, out_dir: str = "checkpoints"):
    """Plot all 4 level curves in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Stage 1 — Full Curriculum Learning Curves", fontsize=13,
                 fontweight="bold")
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    for ax, (lvl, rewards), color in zip(axes.flat, all_rewards.items(), colors):
        ax.plot(smooth(rewards), color=color, lw=1.5)
        ax.set_title(f"Level {lvl}")
        ax.set_xlabel("Episode")
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "nav_full_curriculum.png")
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Stage 2 — Assignment training curves
# ---------------------------------------------------------------------------
def plot_assign_history(history: dict, out_dir: str = "checkpoints"):
    """
    6-panel training dashboard.
    history keys: rewards, losses, accept_rates, delivery_rates, charge_rates
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(
        "Stage 2 — Assignment DQN  (3-action: Accept / Idle / GoCharge)",
        fontsize=13, fontweight="bold",
    )

    axes[0, 0].plot(smooth(history["rewards"]), color="#1565C0", lw=1.5)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].axhline(0, color="black", lw=0.5, ls="--")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(smooth(history["losses"]), color="#C62828", lw=1.5)
    axes[0, 1].set_title("TD Loss (Smooth L1)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(smooth(history["delivery_rates"]), color="#6A1B9A", lw=1.5)
    axes[0, 2].set_title("Delivery Rate | Accepted Orders")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(smooth(history["accept_rates"]), color="#2E7D32", lw=1.5)
    axes[1, 0].set_title("Accept Rate  (action=0)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(smooth(history["charge_rates"]), color="#E65100", lw=1.5)
    axes[1, 1].set_title("GoCharge Rate  (action=2) — learned threshold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)

    # Stacked action distribution
    ar = np.array(history["accept_rates"])
    cr = np.array(history["charge_rates"])
    ir = 1.0 - ar - cr
    x  = np.arange(len(ar))
    axes[1, 2].stackplot(
        x, ar, cr, ir,
        labels=["Accept", "GoCharge", "Idle"],
        colors=["#2E7D32", "#E65100", "#9E9E9E"],
        alpha=0.7,
    )
    axes[1, 2].set_title("Action Distribution over Training")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend(loc="upper right", fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "assign_training_curves.png")
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Stage 2 — Decision heatmap
# ---------------------------------------------------------------------------
def decision_heatmap(model: torch.nn.Module, device: torch.device,
                     out_dir: str = "checkpoints"):
    """
    Three side-by-side heatmaps: P(Accept), P(Idle), P(GoCharge) over the
    battery × trip_cost grid.  Reveals the charging threshold the agent learned.
    Other features held at median values: orders_remaining=3/5, dist_charger=0.2,
    dist_pickup=0.3, dist_dropoff=0.3, idle=0.1.
    """
    model.eval()
    N          = 25
    batteries  = np.linspace(0, 1, N)
    trip_costs = np.linspace(0, 1, N)
    maps       = np.zeros((3, N, N))  # [accept, idle, charge]

    with torch.no_grad():
        for i, bat in enumerate(batteries):
            for j, tc in enumerate(trip_costs):
                margin = float(np.clip(bat - tc, -1.0, 1.0))
                feat   = torch.tensor(
                    [bat, tc, 0.3, 0.3, 0.2, 0.1, 0.6, margin],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                q    = model(feat).squeeze()
                prob = torch.softmax(q, dim=0).cpu().numpy()
                maps[:, i, j] = prob

    model.train()
    titles = ["P(Accept)", "P(Idle)", "P(GoCharge) — learned charge region"]
    cmaps  = ["RdYlGn",   "Greys",   "YlOrRd"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Assignment DQN Decision Surface  "
        "(battery × trip_cost;  orders_remaining=3,  dist_charger=0.2)",
        fontsize=11, fontweight="bold",
    )
    for ax, hm, title, cmap in zip(axes, maps, titles, cmaps):
        im = ax.imshow(hm, origin="lower", aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, extent=[0, 100, 0, 100])
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Trip Cost (% battery needed)", fontsize=10)
        ax.set_ylabel("Battery Level (%)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.plot([0, 100], [0, 100], "b--", lw=1.2, label="trip=battery")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(out_dir, "assign_decision_heatmap.png")
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Stage 3 — MAPPO training curves
# ---------------------------------------------------------------------------
def plot_mappo_history(history: dict, out_dir: str = "checkpoints"):
    """
    4-panel dashboard for MAPPO training.
    history keys: team_rewards, actor_losses, critic_losses,
                  accept_rates, charge_rates, delivery_rates
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Stage 3 — MAPPO  (3 agents, CTDE: shared actor + centralised critic)",
        fontsize=13, fontweight="bold",
    )

    axes[0, 0].plot(smooth(history["team_rewards"]), color="#1565C0", lw=1.5)
    axes[0, 0].set_title("Team Reward / Episode  (sum of all agents)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].axhline(0, color="black", lw=0.5, ls="--")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(smooth(history["delivery_rates"]), color="#6A1B9A", lw=1.5)
    axes[0, 1].set_title("Delivery Rate | Accepted Orders")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(smooth(history["accept_rates"]), color="#2E7D32", lw=1.5,
                    label="Accept")
    axes[1, 0].plot(smooth(history["charge_rates"]), color="#E65100", lw=1.5,
                    label="GoCharge")
    axes[1, 0].set_title("Accept / GoCharge Rates per Agent")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # Stacked action distribution
    ar = np.array(history["accept_rates"])
    cr = np.array(history["charge_rates"])
    ir = np.clip(1.0 - ar - cr, 0.0, 1.0)
    x  = np.arange(len(ar))
    axes[1, 1].stackplot(
        x, ar, cr, ir,
        labels=["Accept", "GoCharge", "Idle"],
        colors=["#2E7D32", "#E65100", "#9E9E9E"],
        alpha=0.7,
    )
    axes[1, 1].set_title("Action Distribution over Training")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "mappo_training_curves.png")
    plt.savefig(out, dpi=120)
    print(f"Saved -> {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Stage 2 — Warehouse grid layout
# ---------------------------------------------------------------------------
def render_assign_grid(robot_pos=None, path=None, title="Stage 2 Warehouse Layout",
                       out_path=None):
    """Render the fixed 10×10 warehouse with shelves, chargers, pickup points."""
    from envs.assign_env import (
        GRID_SIZE, SHELVES, CHARGERS, DROPOFF, PICKUP_POINTS,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            pos = (r, c)
            y   = GRID_SIZE - 1 - r
            if pos in SHELVES:
                fc = "#795548"
            elif pos in set(CHARGERS):
                fc = "#4CAF50"
            elif pos == DROPOFF:
                fc = "#FF9800"
            elif pos in set(PICKUP_POINTS):
                fc = "#90CAF9"
            else:
                fc = "#ECEFF1"
            ax.add_patch(plt.Rectangle((c, y), 1, 1, fc=fc, ec="#B0BEC5", lw=0.5))

    if path and len(path) > 1:
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            ax.annotate("", xy=(c2 + 0.5, GRID_SIZE - 1 - r2 + 0.5),
                        xytext=(c1 + 0.5, GRID_SIZE - 1 - r1 + 0.5),
                        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))

    if robot_pos:
        r, c = robot_pos
        ax.add_patch(plt.Circle((c + 0.5, GRID_SIZE - 1 - r + 0.5), 0.35,
                                fc="red", ec="darkred", zorder=5))

    for r, c in CHARGERS:
        ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, "C",
                ha="center", va="center", fontweight="bold",
                color="white", fontsize=9)
    ax.text(DROPOFF[1] + 0.5, GRID_SIZE - 1 - DROPOFF[0] + 0.5, "D",
            ha="center", va="center", fontweight="bold", color="white", fontsize=9)
    for i, (r, c) in enumerate(PICKUP_POINTS):
        ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, f"P{i+1}",
                ha="center", va="center", fontsize=7, color="#1565C0")

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels(range(GRID_SIZE + 1), fontsize=7)
    ax.set_yticklabels(range(GRID_SIZE, -1, -1), fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(fc="#795548", label="Shelf (obstacle)"),
        mpatches.Patch(fc="#4CAF50", label="Charger (C)"),
        mpatches.Patch(fc="#FF9800", label="Dropoff (D)"),
        mpatches.Patch(fc="#90CAF9", label="Pickup point (P)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"Saved → {out_path}")
    plt.show()
