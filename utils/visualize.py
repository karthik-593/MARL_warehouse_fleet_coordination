"""
Warehouse MARL — Simulation Animations  (6 scenarios)

Scenario  Stage   Description
   1      Nav     5×5 plain grid             — trained PPO (L6 model)
   2      Nav     10×10 plain grid           — trained PPO
   3      Nav     10×10 stationary obstacles — trained PPO
   4      Nav     10×10 moving obstacles     — trained PPO
   5      Nav     L6 warehouse + 2 frozen robots — trained PPO (real deployment setup)
   6      MAPPO   Stage 3 — 3-robot coordination

All navigation uses real policy inference — no BFS.

Usage (notebook):
    from utils.visualize import build_scenario_catalog, make_html_animation
    cat = build_scenario_catalog(ckpt_dir, device)
    frames = cat[1]["record"]()
    make_html_animation(frames, "s01.html", fps=8)
"""

import os, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import torch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
ROBOT_COLORS = ["#E53935", "#1E88E5", "#43A047"]
ROBOT_LABELS = ["R0",      "R1",      "R2"]
C_SHELF    = "#5D4037";  C_CHARGER = "#00897B";  C_DROPOFF = "#FB8C00"
C_PICKUP   = "#90CAF9";  C_FREE    = "#ECEFF1";  C_GOAL    = "#FDD835"
C_OBS      = "#546E7A";  C_DYNOBS  = "#E91E63";  C_ELIGIBLE= "#FFF176"
C_BG       = "#263238";  C_PANEL   = "#1C2B33"
C_ACCEPT   = "#388E3C";  C_IDLE    = "#616161";  C_CHARGE  = "#0277BD"
C_NAV_TRAIL= "#FFB300";  C_FROZEN  = "#90A4AE"

ACTION_NAMES_ASSIGN = ["ACCEPT", "IDLE", "GOCHARGE"]
ACTION_COLORS       = {"ACCEPT": C_ACCEPT, "IDLE": C_IDLE, "GOCHARGE": C_CHARGE}


# ===========================================================================
# Real-policy nav path builder
# ===========================================================================
def _trace_nav_path(start_pos, start_battery, nav_model, target, device,
                    other_positions=frozenset(), max_steps=300, skip=1):
    """
    Run the frozen nav PPO policy step-by-step from start_pos to target.
    Returns list of (pos, battery) tuples — actual policy behaviour, no BFS.
    other_positions : cells occupied by other robots (treated as blocked).
    skip            : keep every skip-th frame (1 = every step).
    """
    from envs.assign_env import WarehouseStage2
    agent         = WarehouseStage2()
    agent.pos     = start_pos
    agent.battery = float(start_battery)
    full_path     = [(start_pos, float(start_battery))]

    for _ in range(max_steps):
        if agent.pos == target or agent.battery <= 0:
            break
        state = agent._get_nav_state(target, other_positions)
        st    = torch.tensor(state, device=device).unsqueeze(0)
        with torch.no_grad():
            out    = nav_model(st)
            logits = out[0] if isinstance(out, tuple) else out
        logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
        probs  = torch.softmax(logits / 0.3, dim=-1)
        action = torch.distributions.Categorical(probs).sample().item()
        agent._nav_step(action, other_positions)
        full_path.append((agent.pos, agent.battery))

    if skip > 1:
        thinned = full_path[::skip]
        if not thinned or thinned[-1] != full_path[-1]:
            thinned.append(full_path[-1])
        return thinned
    return full_path


# ===========================================================================
# Grid drawing helpers
# ===========================================================================
def _rc_to_xy(r, c, size): return c, size - 1 - r

def _draw_cell(ax, r, c, size, fc, ec="#B0BEC5", lw=0.4):
    x, y = _rc_to_xy(r, c, size)
    ax.add_patch(plt.Rectangle((x, y), 1, 1, fc=fc, ec=ec, lw=lw, zorder=1))

def _draw_robot(ax, r, c, size, color, label, battery, winner=False, eligible=True):
    x, y   = _rc_to_xy(r, c, size)
    cx, cy = x + 0.5, y + 0.5
    if eligible:
        ax.add_patch(plt.Circle((cx, cy), 0.46, fc=C_ELIGIBLE, ec="none",
                                 zorder=2, alpha=0.38))
    edge = "gold" if winner else "white"
    lw2  = 2.8   if winner else 1.2
    ax.add_patch(plt.Circle((cx, cy), 0.34, fc=color, ec=edge, lw=lw2, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=4)
    bx, by = x + 0.1, y + 0.05
    bw, bh = 0.8, 0.13
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc="#37474F", ec="none", zorder=4))
    bc = "#4CAF50" if battery > 50 else "#FFC107" if battery > 25 else "#F44336"
    ax.add_patch(plt.Rectangle((bx, by), bw * battery / 100, bh,
                                fc=bc, ec="none", zorder=5))

def _draw_frozen_robot(ax, r, c, size, label):
    """Draw a static/frozen robot obstacle — grey, no battery bar."""
    x, y   = _rc_to_xy(r, c, size)
    cx, cy = x + 0.5, y + 0.5
    ax.add_patch(plt.Circle((cx, cy), 0.34, fc=C_FROZEN, ec="white",
                             lw=1.0, zorder=3, alpha=0.85))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=7, fontweight="bold", color="#263238", zorder=4)

def _draw_trail(ax, trail, size, color, alpha_max=0.7):
    n = len(trail)
    for i, (r, c) in enumerate(trail):
        x, y = _rc_to_xy(r, c, size)
        alpha = 0.08 + alpha_max * (i / max(n - 1, 1))
        ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.14, fc=color,
                                 ec="none", alpha=alpha, zorder=2))

def _highlight_cell(ax, r, c, size, color, label=""):
    x, y = _rc_to_xy(r, c, size)
    ax.add_patch(plt.Rectangle((x, y), 1, 1, fc="none", ec=color,
                                lw=3.0, zorder=6))
    if label:
        ax.text(x + 0.5, y + 0.82, label, ha="center", va="center",
                fontsize=7, color=color, fontweight="bold", zorder=7)

def _setup_nav_bg(ax, grid, size, dyn_obs_pos=None):
    dyn = set(dyn_obs_pos or [])
    ax.set_facecolor(C_BG)
    for r in range(size):
        for c in range(size):
            v = grid[r, c]
            if   (r, c) in dyn: fc = C_DYNOBS
            elif v == -1:       fc = C_OBS
            elif v == 2:        fc = C_CHARGER
            elif v == 3:        fc = C_GOAL
            else:               fc = C_FREE
            _draw_cell(ax, r, c, size, fc)
    ax.set_xlim(0, size); ax.set_ylim(0, size)
    ax.set_aspect("equal"); ax.axis("off")

def _setup_warehouse_bg(ax):
    from envs.assign_env import SHELVES, CHARGERS, DROPOFF, PICKUP_POINTS
    size = 10
    ax.set_facecolor(C_BG)
    for r in range(size):
        for c in range(size):
            pos = (r, c)
            if   pos in SHELVES:            fc = C_SHELF
            elif pos in set(CHARGERS):      fc = C_CHARGER
            elif pos == DROPOFF:            fc = C_DROPOFF
            elif pos in set(PICKUP_POINTS): fc = C_PICKUP
            else:                           fc = C_FREE
            _draw_cell(ax, r, c, size, fc)
    for cr, cc in CHARGERS:
        x, y = _rc_to_xy(cr, cc, size)
        ax.text(x+.5, y+.5, "C", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)
    dr, dc = DROPOFF
    x, y = _rc_to_xy(dr, dc, size)
    ax.text(x+.5, y+.5, "D", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal"); ax.axis("off")


# ===========================================================================
# Info panel helpers
# ===========================================================================
def _clear_info(ax):
    ax.cla(); ax.set_facecolor(C_PANEL); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def _info_text(ax, x, y, txt, fs=8, color="white", bold=False):
    ax.text(x, y, txt, ha="center", va="top", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal",
            transform=ax.transAxes, clip_on=False)

def _phase_badge(ax, phase, y=0.86):
    styles = {
        "nav_pickup":  ("#E65100", "➤  NAVIGATING → PICKUP"),
        "nav_dropoff": ("#6A1B9A", "➤  NAVIGATING → DROPOFF"),
        "nav_charger": ("#00695C", "➤  NAVIGATING → CHARGER"),
        "decision":    ("#1565C0", "▶  DECISION"),
        "outcome":     ("#2E7D32", "✓  OUTCOME"),
    }
    bg, txt = styles.get(phase, ("#37474F", phase.upper()))
    ax.add_patch(plt.Rectangle((0.04, y - 0.030), 0.92, 0.062,
                                fc=bg, ec="none", zorder=2,
                                transform=ax.transAxes))
    ax.text(0.5, y + 0.002, txt, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            transform=ax.transAxes, zorder=3)

def _battery_strip(ax, battery, y, label="Battery"):
    bc = "#4CAF50" if battery > 50 else "#FFC107" if battery > 25 else "#F44336"
    ax.add_patch(plt.Rectangle((0.04, y), 0.92, 0.044,
                                fc="#37474F", ec="none", zorder=2,
                                transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0.04, y), 0.92 * battery / 100, 0.044,
                                fc=bc, ec="none", zorder=3,
                                transform=ax.transAxes))
    ax.text(0.5, y + 0.022, f"{label}  {int(battery)}%",
            ha="center", va="center", fontsize=7.5, color="white",
            fontweight="bold", zorder=4, transform=ax.transAxes)

def _action_badge(ax, action, reward=None, y=0.50):
    fc = ACTION_COLORS.get(action, "#607D8B")
    ax.add_patch(plt.Rectangle((0.04, y - 0.027), 0.92, 0.060,
                                fc=fc, ec="none", zorder=2,
                                transform=ax.transAxes))
    label = action if reward is None else f"{action}   {reward:+.0f}"
    ax.text(0.5, y + 0.004, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            transform=ax.transAxes, zorder=3)


# ===========================================================================
# Frame renderers
# ===========================================================================
def _render_nav_frame(ax_grid, ax_info, f):
    ax_grid.cla(); _clear_info(ax_info)
    size = f["size"]
    _setup_nav_bg(ax_grid, f["grid"], size, f.get("dyn_obs", []))
    _draw_trail(ax_grid, f.get("trail", []), size, "#78909C")

    gr, gc = f["goal"]
    gx, gy = _rc_to_xy(gr, gc, size)
    ax_grid.add_patch(plt.Circle((gx+.5, gy+.5), 0.45,
                                  fc="none", ec=C_GOAL, lw=2.5, zorder=5))

    # Draw frozen robots (L6 only)
    for fi, (fr, fc_) in enumerate(f.get("frozen_robots", [])):
        _draw_frozen_robot(ax_grid, fr, fc_, size, f"F{fi+1}")

    _draw_robot(ax_grid, *f["pos"], size, "#E53935", "R0", f["battery"])

    lvl_desc = {1: "5×5 plain grid",
                2: "10×10 plain grid",
                3: "10×10 + stationary obstacles",
                4: "10×10 + moving obstacles",
                5: "L6 warehouse + 2 frozen robots"}
    desc = lvl_desc.get(f["level"], f"Level {f['level']}")
    ax_grid.set_title(f"Stage 1 — Navigation  [{desc}]",
                      fontsize=10, color="white", pad=4)
    ax_grid.set_facecolor(C_BG)

    _info_text(ax_info, 0.5, 0.99, "STAGE 1 — NAVIGATION", fs=8.5, bold=True)
    _info_text(ax_info, 0.5, 0.92, desc, fs=7, color="#B0BEC5")
    _info_text(ax_info, 0.5, 0.83, f"Step  {f['step']}", fs=12, bold=True)
    _battery_strip(ax_info, f["battery"], 0.69)
    status = f.get("status", "navigating...")
    sc = ("#F44336" if any(w in status for w in ("DEAD", "TIMEOUT", "FAIL"))
          else "#66BB6A" if any(w in status for w in ("REACHED", "GOAL", "SUCCESS"))
          else "#FFB300")
    _info_text(ax_info, 0.5, 0.59, status, fs=8, color=sc, bold=True)

    if f.get("frozen_robots"):
        _info_text(ax_info, 0.5, 0.44, "Frozen robots:", fs=7, color="#90A4AE")
        _info_text(ax_info, 0.5, 0.38, "F1 & F2 = static obstacles", fs=6.5, color=C_FROZEN)
        _info_text(ax_info, 0.5, 0.30, "R0 navigates around them", fs=6.5, color="#78909C")


def _render_mappo_frame(ax_grid, ax_info, f):
    ax_grid.cla(); _clear_info(ax_info)
    from envs.assign_env import DROPOFF
    _setup_warehouse_bg(ax_grid)
    phase  = f.get("phase", "decision")
    winner = f.get("winner")

    if phase in ("decision", "nav_pickup"):
        _highlight_cell(ax_grid, *f["pickup"], 10, "#FDD835", "ORDER")
    elif phase == "nav_dropoff":
        _highlight_cell(ax_grid, *f["pickup"], 10, "#90CAF9", "")
        _highlight_cell(ax_grid, *DROPOFF, 10, C_DROPOFF, "DROPOFF")
    elif phase == "nav_charger":
        tgt = f.get("target")
        if tgt:
            _highlight_cell(ax_grid, *tgt, 10, C_CHARGE, "CHARGE")
    elif phase == "outcome":
        _highlight_cell(ax_grid, *f["pickup"], 10, "#FDD835", "")

    if phase.startswith("nav_"):
        _draw_trail(ax_grid, f.get("trail", []), 10, C_NAV_TRAIL)

    for robot in f["robots"]:
        r, c = robot["pos"]
        _draw_robot(ax_grid, r, c, 10, robot["color"], robot["label"],
                    robot["battery"],
                    winner=robot.get("winner", False) and phase == "outcome",
                    eligible=robot.get("eligible", True))
        x, y = _rc_to_xy(r, c, 10)
        ax_grid.text(x+.5, y+1.10, robot["action"],
                     ha="center", va="bottom", fontsize=6.5,
                     color=ACTION_COLORS.get(robot["action"], "white"),
                     fontweight="bold", zorder=7, clip_on=False)

    ax_grid.set_title("Stage 3 — MAPPO   (3 robots, K=2 dispatch)",
                      fontsize=10, color="white", pad=4)
    ax_grid.set_facecolor(C_BG)

    _info_text(ax_info, 0.5, 0.99, "STAGE 3 — MAPPO", fs=8.5, bold=True)
    _info_text(ax_info, 0.5, 0.92, "3 robots · K=2 dispatch · CTDE", fs=6.5, color="#B0BEC5")
    _info_text(ax_info, 0.5, 0.84, f"Order  {f['order_i']} / {f['n_orders']}",
               fs=11, bold=True)
    _phase_badge(ax_info, phase, y=0.73)

    card_y = [0.59, 0.40, 0.21]
    for i, robot in enumerate(f["robots"]):
        y0 = card_y[i]
        ax_info.add_patch(plt.Rectangle((0.04, y0+0.07), 0.92, 0.075,
                                         fc=robot["color"], ec="none", zorder=2,
                                         transform=ax_info.transAxes))
        elig = "  ★" if robot.get("eligible") else "  ·"
        win  = "  WIN" if robot.get("winner") and phase == "outcome" else ""
        nav  = "  ➤" if robot.get("navigating") else ""
        ax_info.text(0.5, y0+0.107, f"{robot['label']}{elig}{win}{nav}",
                     ha="center", va="center", fontsize=7, fontweight="bold",
                     color="white", transform=ax_info.transAxes, zorder=3)
        bc = ("#4CAF50" if robot["battery"] > 50
              else "#FFC107" if robot["battery"] > 25 else "#F44336")
        ax_info.add_patch(plt.Rectangle((0.04, y0+0.04), 0.92, 0.030,
                                         fc="#37474F", ec="none", zorder=2,
                                         transform=ax_info.transAxes))
        ax_info.add_patch(plt.Rectangle((0.04, y0+0.04),
                                         0.92 * robot["battery"] / 100, 0.030,
                                         fc=bc, ec="none", zorder=3,
                                         transform=ax_info.transAxes))
        ax_info.text(0.5, y0+0.055, f"{int(robot['battery'])}%",
                     ha="center", va="center", fontsize=6.5, color="white",
                     transform=ax_info.transAxes, zorder=4)
        ac  = ACTION_COLORS.get(robot["action"], "#607D8B")
        rwd = (f"  {robot['reward']:+.0f}"
               if phase == "outcome" and "reward" in robot else "")
        ax_info.text(0.5, y0+0.022, f"{robot['action']}{rwd}",
                     ha="center", va="center", fontsize=7, color=ac,
                     fontweight="bold", transform=ax_info.transAxes)


RENDERERS = {"nav": _render_nav_frame, "mappo": _render_mappo_frame}

LEGENDS = {
    "nav": [mpatches.Patch(fc=C_GOAL,      label="Goal"),
            mpatches.Patch(fc=C_CHARGER,   label="Charger"),
            mpatches.Patch(fc=C_OBS,       label="Obstacle"),
            mpatches.Patch(fc=C_DYNOBS,    label="Dyn.Obs"),
            mpatches.Patch(fc=C_FROZEN,    label="Frozen robot")],
    "mappo": [mpatches.Patch(fc=ROBOT_COLORS[0], label="R0"),
              mpatches.Patch(fc=ROBOT_COLORS[1], label="R1"),
              mpatches.Patch(fc=ROBOT_COLORS[2], label="R2"),
              mpatches.Patch(fc=C_ELIGIBLE,       label="★ Eligible"),
              mpatches.Patch(fc=C_NAV_TRAIL,      label="Path taken")],
}


# ===========================================================================
# Figure factory
# ===========================================================================
def _build_fig_and_ani(frames, fps, figsize=(9.5, 6.5)):
    ftype    = frames[0]["type"]
    renderer = RENDERERS[ftype]
    handles  = LEGENDS[ftype]

    fig = plt.figure(figsize=figsize, facecolor=C_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1],
                            wspace=0.04, left=0.02, right=0.98,
                            top=0.91, bottom=0.09)
    ax_grid = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    def update(i):
        renderer(ax_grid, ax_info, frames[i])
        ax_grid.legend(handles=handles, loc="lower center",
                       bbox_to_anchor=(0.5, -0.10), ncol=len(handles),
                       fontsize=7, framealpha=0.25,
                       labelcolor="white", facecolor="#37474F",
                       handlelength=1.2)
        return ax_grid, ax_info

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False)
    return fig, ani


# ===========================================================================
# Output formats
# ===========================================================================
def make_animation(frames, out_path, fps=8, figsize=(9.5, 6.5)):
    """Save as .gif."""
    if not frames:
        print(f"  No frames — skipping {out_path}"); return
    fig, ani = _build_fig_and_ani(frames, fps, figsize)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps, dpi=110)
    plt.close(fig)
    print(f"  Saved GIF  -> {out_path}")

def make_html_animation(frames, out_path, fps=8, figsize=(9.5, 6.5)):
    """Save as interactive .html player (play/pause/scrub/speed)."""
    if not frames:
        print(f"  No frames — skipping {out_path}"); return
    fig, ani = _build_fig_and_ani(frames, fps, figsize)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    html_str = ani.to_jshtml(fps=fps, default_mode="loop")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
    plt.close(fig)
    print(f"  Saved HTML -> {out_path}  ({len(frames)} frames @ {fps} FPS)")


# ===========================================================================
# Episode recorders
# ===========================================================================
def record_nav_episode(model, level, device, seed=0) -> list:
    """
    Record a single successful navigation episode using the real PPO policy.
    Tries multiple seeds and returns the first successful run.
    """
    from envs.nav_env import WarehouseEnv

    for attempt_seed in range(seed, seed + 50):
        random.seed(attempt_seed); np.random.seed(attempt_seed)
        torch.manual_seed(attempt_seed)
        env   = WarehouseEnv(level=level)
        state = env.reset()

        frames  = []
        trail   = []
        step_i  = 0
        success = False

        while not env.done:
            trail.append(env.pos)
            frames.append({
                "type": "nav", "level": level, "size": env.size,
                "grid": env.grid.copy(), "pos": env.pos,
                "goal": env.goal, "battery": env.battery,
                "dyn_obs": [d.pos for d in env.dyn_obs],
                "trail": list(trail[-25:]),
                "step": step_i,
            })
            st = torch.tensor(state, device=device).unsqueeze(0)
            with torch.no_grad():
                out    = model(st)
                logits = out[0] if isinstance(out, tuple) else out
            logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
            probs  = torch.softmax(logits / 0.3, dim=-1)
            action = torch.distributions.Categorical(probs).sample().item()
            state, _, done, success = env.step(action)
            step_i += 1
            if done:
                status = ("SUCCESS — GOAL REACHED!" if success
                          else "BATTERY DEAD" if env.battery <= 0
                          else "TIMEOUT")
                frames.append({**frames[-1], "pos": env.pos,
                               "battery": env.battery,
                               "dyn_obs": [d.pos for d in env.dyn_obs],
                               "step": step_i, "status": status})
                break

        if success:
            print(f"  L{level} nav: success at seed={attempt_seed}, {step_i} steps")
            return frames

    print(f"  L{level} nav: no success in 50 seeds — returning last attempt")
    return frames


def record_nav_l6_episode(model, device, seed=0) -> list:
    """
    L6 warehouse navigation with 2 frozen robot obstacles — actual deployment setup.
    Uses StaticObstacleNavEnv (the exact env used in L6 training).
    The 2 frozen robots are shown as grey markers on the warehouse grid.
    Tries multiple seeds for a successful run.
    """
    from envs.multi_nav_env import StaticObstacleNavEnv

    for attempt_seed in range(seed, seed + 50):
        random.seed(attempt_seed); np.random.seed(attempt_seed)
        torch.manual_seed(attempt_seed)
        env = StaticObstacleNavEnv()
        obs = env.reset()

        # Build a static warehouse-style grid for rendering
        from envs.assign_env import SHELVES, CHARGERS, DROPOFF, GRID_SIZE
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in SHELVES:
            grid[r, c] = -1
        for r, c in CHARGERS:
            grid[r, c] = 2
        grid[DROPOFF[0], DROPOFF[1]] = 3

        frozen = list(env._static_pos)  # 2 frozen robot positions
        frames = []
        trail  = []
        step_i = 0

        while True:
            trail.append(env.pos)
            frames.append({
                "type": "nav", "level": 5,   # level 5 key = L6 warehouse desc
                "size": GRID_SIZE,
                "grid": grid.copy(),
                "pos": env.pos,
                "goal": env.goal,
                "battery": env.battery,
                "dyn_obs": [],
                "frozen_robots": list(frozen),
                "trail": list(trail[-25:]),
                "step": step_i,
            })
            st = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(st)
            logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
            probs  = torch.softmax(logits / 0.3, dim=-1)
            action = torch.distributions.Categorical(probs).sample().item()
            obs, _, done = env.step(action)
            step_i += 1
            if done:
                status = ("SUCCESS — GOAL REACHED!" if env.success
                          else "BATTERY DEAD" if env.battery <= 0
                          else "TIMEOUT")
                frames.append({**frames[-1], "pos": env.pos,
                               "battery": env.battery,
                               "step": step_i, "status": status})
                if env.success:
                    print(f"  L6 nav: success at seed={attempt_seed}, {step_i} steps, "
                          f"frozen={frozen}")
                    return frames
                break

    print(f"  L6 nav: no success in 50 seeds — returning last attempt")
    return frames


def record_mappo_episode(nav_model, actor, device, seed=0,
                         show_nav=True, nav_skip=2) -> list:
    """
    Record one full MAPPO episode (10 orders, 3 robots).
    Tries multiple seeds and returns the best (most deliveries).
    show_nav : insert step-by-step nav frames for the winning robot.
    """
    from envs.marl_env import MultiAgentWarehouse, N_AGENTS, N_ORDERS
    from envs.assign_env import bfs_dist, TRIP_COST_RATE, DROPOFF

    best_frames    = []
    best_deliveries = -1

    for attempt_seed in range(seed, seed + 20):
        random.seed(attempt_seed); np.random.seed(attempt_seed)
        torch.manual_seed(attempt_seed)
        env     = MultiAgentWarehouse()
        obs, gs = env.reset()
        frames  = []
        deliveries = 0

        for order_i in range(N_ORDERS):
            pickup   = env.orders[order_i]
            eligible = env._k_nearest(pickup)
            obs_t    = torch.tensor(obs, device=device, dtype=torch.float32)

            with torch.no_grad():
                actions = actor(obs_t).argmax(dim=-1).tolist()

            accepting = [i for i in eligible if actions[i] == 0]
            winner    = None
            if len(accepting) == 1:
                winner = accepting[0]
            elif len(accepting) > 1:
                scores = {i: env.agents[i].battery
                             - bfs_dist(env.agents[i].pos, pickup) * TRIP_COST_RATE
                          for i in accepting}
                winner = max(scores, key=scores.get)

            def robot_state(overrides=None, rewards=None, navigating_idx=None):
                overrides = overrides or {}
                return [{"pos":       overrides.get(i, {}).get("pos",     env.agents[i].pos),
                         "battery":   overrides.get(i, {}).get("battery", env.agents[i].battery),
                         "color":     ROBOT_COLORS[i], "label": ROBOT_LABELS[i],
                         "action":    ACTION_NAMES_ASSIGN[actions[i]],
                         "eligible":  i in eligible,
                         "winner":    i == winner,
                         "navigating": i == navigating_idx,
                         **({"reward": rewards[i]} if rewards else {})}
                        for i in range(N_AGENTS)]

            base = {"type": "mappo", "pickup": pickup,
                    "winner": winner,
                    "order_i": order_i + 1, "n_orders": N_ORDERS}

            frames.append({**base, "robots": robot_state(),
                           "phase": "decision"})

            if show_nav and winner is not None:
                w_agent   = env.agents[winner]
                w_pos, w_bat = w_agent.pos, w_agent.battery
                other_pos = frozenset(env.agents[j].pos
                                      for j in range(N_AGENTS) if j != winner)
                trail = []
                nav_pickup = _trace_nav_path(w_pos, w_bat, nav_model,
                                             pickup, device,
                                             other_positions=other_pos,
                                             skip=nav_skip)
                for si, (pos, bat) in enumerate(nav_pickup):
                    trail.append(pos)
                    ov = {winner: {"pos": pos, "battery": bat}}
                    frames.append({**base,
                                   "robots": robot_state(overrides=ov,
                                                          navigating_idx=winner),
                                   "phase": "nav_pickup",
                                   "trail": list(trail[-30:]),
                                   "nav_step": si + 1})

                last_pos, last_bat = nav_pickup[-1]
                trail2 = [last_pos]
                nav_drop = _trace_nav_path(last_pos, last_bat, nav_model,
                                           DROPOFF, device,
                                           other_positions=other_pos,
                                           skip=nav_skip)
                for si, (pos, bat) in enumerate(nav_drop):
                    trail2.append(pos)
                    ov = {winner: {"pos": pos, "battery": bat}}
                    frames.append({**base,
                                   "robots": robot_state(overrides=ov,
                                                          navigating_idx=winner),
                                   "phase": "nav_dropoff",
                                   "trail": list(trail2[-30:]),
                                   "nav_step": si + 1})

            obs, gs, rewards, done = env.step(actions, nav_model)
            for a, r in zip(actions, rewards):
                if a == 0 and r > 50:
                    deliveries += 1
            frames.append({**base, "robots": robot_state(rewards=rewards),
                           "phase": "outcome"})
            if done:
                break

        if deliveries > best_deliveries:
            best_deliveries = deliveries
            best_frames     = frames
            best_seed       = attempt_seed

    print(f"  MAPPO: best seed={best_seed}, deliveries={best_deliveries}/10")
    return best_frames


# ===========================================================================
# Model loaders
# ===========================================================================
def _load_nav_ppo(ckpt_dir, device, name="ppo_final.pt"):
    from agents.ppo import PPO
    path = os.path.join(ckpt_dir, name)
    m = PPO(state_dim=13, action_dim=6).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def _load_mappo_actor(ckpt_dir, device):
    from agents.mappo import AssignmentActor
    from envs.marl_env import OBS_DIM
    actor = AssignmentActor(obs_dim=OBS_DIM).to(device)
    actor.load_state_dict(torch.load(os.path.join(ckpt_dir, "mappo_actor.pt"),
                                     map_location=device, weights_only=True))
    actor.eval()
    return actor


# ===========================================================================
# 6-scenario catalog
# ===========================================================================
def build_scenario_catalog(ckpt_dir: str, device) -> dict:
    """Return dict mapping scenario number → {title, desc, out, record}."""
    cat = {}

    # Shared nav model — L6 PPO works on all levels (it's the most capable)
    def nav():
        return _load_nav_ppo(ckpt_dir, device)

    cat[1] = dict(
        title="Navigation — 5×5 plain grid",
        desc="Trained PPO navigates a 5×5 grid. Clean direct path to goal.",
        out="s01_nav_5x5.html",
        record=lambda: record_nav_episode(nav(), 1, device, seed=0),
    )

    cat[2] = dict(
        title="Navigation — 10×10 plain grid",
        desc="Trained PPO on the full 10×10 warehouse grid, no obstacles.",
        out="s02_nav_10x10.html",
        record=lambda: record_nav_episode(nav(), 2, device, seed=0),
    )

    cat[3] = dict(
        title="Navigation — 10×10 with stationary obstacles",
        desc="PPO routes around 10 static obstacles to reach the goal.",
        out="s03_nav_static_obs.html",
        record=lambda: record_nav_episode(nav(), 3, device, seed=1),
    )

    cat[4] = dict(
        title="Navigation — 10×10 with moving obstacles",
        desc="PPO adapts in real time as dynamic obstacles change position each step.",
        out="s04_nav_moving_obs.html",
        record=lambda: record_nav_episode(nav(), 4, device, seed=3),
    )

    cat[5] = dict(
        title="Navigation — L6 warehouse + 2 frozen robot obstacles",
        desc="Real warehouse layout (shelves, chargers, dropoff). "
             "Two other robots are frozen at fixed positions. "
             "The nav policy routes around them — exactly as in Stage 3 deployment.",
        out="s05_nav_l6_warehouse.html",
        record=lambda: record_nav_l6_episode(nav(), device, seed=0),
    )

    cat[6] = dict(
        title="Stage 3 — MAPPO: 3-robot coordination (best of 20 seeds)",
        desc="Trained MAPPO actor. K=2 dispatch: 2 nearest robots bid per order. "
             "Winner navigates pickup → dropoff using the frozen nav policy. "
             "Battery managed autonomously with GoCharge decisions.",
        out="s06_mappo_3robots.html",
        record=lambda: record_mappo_episode(
            nav(), _load_mappo_actor(ckpt_dir, device), device, seed=0),
    )

    return cat


# ===========================================================================
# CLI runner
# ===========================================================================
def run_all(ckpt_dir: str, out_dir: str, device, fmt: str = "html"):
    cat    = build_scenario_catalog(ckpt_dir, device)
    fps    = {1: 10, 2: 8, 3: 8, 4: 8, 5: 8, 6: 4}
    save_fn = make_html_animation if fmt == "html" else make_animation
    ext     = ".html" if fmt == "html" else ".gif"

    for n in sorted(cat.keys()):
        s = cat[n]
        print(f"\n[{n}] {s['title']}")
        try:
            frames   = s["record"]()
            out_name = os.path.splitext(s["out"])[0] + ext
            save_fn(frames, os.path.join(out_dir, out_name), fps=fps[n])
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["html", "gif"], default="html")
    args = parser.parse_args()
    root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    out_dir  = os.path.join(ckpt_dir, "animations")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\nFormat : {args.fmt}\nOutput : {out_dir}\n")
    run_all(ckpt_dir, out_dir, device, fmt=args.fmt)


if __name__ == "__main__":
    main()
