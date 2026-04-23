"""
Simulation animations for all 3 stages — 9 scenarios.

Two output formats:
    make_animation()      → .gif  (portable)
    make_html_animation() → .html (interactive player: pause / scrub / speed)

Usage (CLI):
    python -m utils.visualize

Usage (notebook):
    from utils.visualize import build_scenario_catalog, make_html_animation
    frames = build_scenario_catalog(ckpt_dir, device)[1]["record"]()
    make_html_animation(frames, "s01.html")
"""

import os, random, argparse
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
ROBOT_COLORS  = ["#E53935", "#1E88E5", "#43A047"]
ROBOT_LABELS  = ["R0",      "R1",      "R2"]
C_SHELF    = "#5D4037";  C_CHARGER = "#00897B";  C_DROPOFF = "#FB8C00"
C_PICKUP   = "#90CAF9";  C_FREE    = "#ECEFF1";  C_GOAL    = "#FDD835"
C_OBS      = "#546E7A";  C_DYNOBS  = "#E91E63";  C_ELIGIBLE= "#FFF176"
C_BG       = "#263238";  C_PANEL   = "#1C2B33"
C_ACCEPT   = "#388E3C";  C_IDLE    = "#616161";  C_CHARGE  = "#0277BD"
C_NAV_TRAIL= "#FFB300"

ACTION_NAMES_ASSIGN = ["ACCEPT", "IDLE", "GOCHARGE"]
ACTION_COLORS       = {"ACCEPT": C_ACCEPT, "IDLE": C_IDLE, "GOCHARGE": C_CHARGE}


# ===========================================================================
# Navigation path builder  (BFS optimal path + battery simulation)
# ===========================================================================
def _bfs_path(start, target, blocked, grid_size=None):
    """
    BFS shortest path on a grid.
    blocked   : set of (r,c) cells that cannot be entered.
    grid_size : int side length; defaults to the warehouse 10×10.
    Returns list of positions from start to target (inclusive).
    """
    from collections import deque
    if grid_size is None:
        from envs.assign_env import GRID_SIZE
        grid_size = GRID_SIZE
    if start == target:
        return [start]
    visited = {start}
    queue   = deque([(start, [start])])
    while queue:
        pos, path = queue.popleft()
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            npos   = (nr, nc)
            if (0 <= nr < grid_size and 0 <= nc < grid_size
                    and npos not in blocked and npos not in visited):
                new_path = path + [npos]
                if npos == target:
                    return new_path
                visited.add(npos)
                queue.append((npos, new_path))
    return [start]   # unreachable — return start only


def _trace_nav_path(start_pos, start_battery, nav_model, target, device,
                    other_positions=frozenset(), max_steps=300, skip=1):
    """
    Step-by-step nav model execution from start_pos to target.
    Returns list of (pos, battery) tuples — actual policy behaviour.
    other_positions: cells occupied by other robots (treated as blocked).
    skip: keep every skip-th frame (1 = all steps).
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
    # battery bar
    bx, by = x + 0.1, y + 0.05
    bw, bh = 0.8, 0.13
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc="#37474F", ec="none", zorder=4))
    bc = "#4CAF50" if battery > 50 else "#FFC107" if battery > 25 else "#F44336"
    ax.add_patch(plt.Rectangle((bx, by), bw * battery / 100, bh,
                                fc=bc, ec="none", zorder=5))

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
    ax.cla()
    ax.set_facecolor(C_PANEL)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def _info_text(ax, x, y, txt, fs=8, color="white", bold=False, ha="center"):
    ax.text(x, y, txt, ha=ha, va="top", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal",
            transform=ax.transAxes, clip_on=False)

def _phase_badge(ax, phase, y=0.86):
    styles = {
        "decision":     ("#1565C0", "▶  DECISION"),
        "outcome":      ("#2E7D32", "✓  OUTCOME"),
        "nav_pickup":   ("#E65100", "➤  NAVIGATING → PICKUP"),
        "nav_dropoff":  ("#6A1B9A", "➤  NAVIGATING → DROPOFF"),
        "nav_charger":  ("#00695C", "➤  NAVIGATING → CHARGER"),
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
    _draw_robot(ax_grid, *f["pos"], size, "#E53935", "R0", f["battery"])

    policy = "UNTRAINED (random)" if f.get("untrained") else f"L{f['level']} Trained"
    ax_grid.set_title(f"Stage 1 — Navigation  [{policy}]",
                      fontsize=10, color="white", pad=4)
    ax_grid.set_facecolor(C_BG)

    _info_text(ax_info, 0.5, 0.99, "STAGE 1", fs=9, bold=True)
    _info_text(ax_info, 0.5, 0.93, "Navigation", fs=7.5, color="#B0BEC5")
    _info_text(ax_info, 0.5, 0.84, f"Step  {f['step']}", fs=12, bold=True)
    _battery_strip(ax_info, f["battery"], 0.70)
    status = f.get("status", f.get("action", "navigating..."))
    sc = ("#F44336" if any(w in status for w in ("DEAD", "TIMEOUT", "FAIL"))
          else "#66BB6A" if any(w in status for w in ("REACHED", "GOAL"))
          else "#B0BEC5")
    _info_text(ax_info, 0.5, 0.60, status, fs=8, color=sc, bold=True)
    lvl_desc = {1: "5×5 plain", 2: "10×10 plain",
                3: "10×10 shelves", 4: "10×10 + moving obstacles"}
    _info_text(ax_info, 0.5, 0.46, f"Level {f['level']}:", fs=7.5, color="#90A4AE")
    _info_text(ax_info, 0.5, 0.39, lvl_desc.get(f["level"], ""), fs=7, color="#78909C")


def _render_assign_frame(ax_grid, ax_info, f):
    ax_grid.cla(); _clear_info(ax_info)
    from envs.assign_env import DROPOFF
    _setup_warehouse_bg(ax_grid)
    phase = f.get("phase", "decision")

    # Highlight active target
    if phase in ("decision", "nav_pickup"):
        _highlight_cell(ax_grid, *f["pickup"], 10,
                        C_ACCEPT if f["action"] != "GOCHARGE" else C_CHARGE,
                        "PICKUP" if phase == "decision" else "")
    elif phase == "nav_dropoff":
        _highlight_cell(ax_grid, *f["pickup"], 10, "#90CAF9", "")   # pickup done
        _highlight_cell(ax_grid, *DROPOFF, 10, C_DROPOFF, "DROPOFF")
    elif phase == "nav_charger":
        tgt = f.get("target")
        if tgt:
            _highlight_cell(ax_grid, *tgt, 10, C_CHARGE, "CHARGE")
    elif phase == "outcome":
        ac = ACTION_COLORS.get(f["action"], "#607D8B")
        _highlight_cell(ax_grid, *f["pickup"], 10, ac, "")

    # Trail during navigation
    if phase.startswith("nav_"):
        _draw_trail(ax_grid, f.get("trail", []), 10, C_NAV_TRAIL)

    _draw_robot(ax_grid, *f["pos"], 10, "#E53935", "R0", f["battery"])
    ax_grid.set_title("Stage 2 — Assignment DQN", fontsize=10, color="white", pad=4)
    ax_grid.set_facecolor(C_BG)

    # info panel
    _info_text(ax_info, 0.5, 0.99, "STAGE 2", fs=9, bold=True)
    _info_text(ax_info, 0.5, 0.93, "Assignment DQN", fs=7.5, color="#B0BEC5")
    _info_text(ax_info, 0.5, 0.84, f"Order  {f['order_i']} / {f['n_orders']}",
               fs=11, bold=True)
    _phase_badge(ax_info, phase, y=0.72)
    _battery_strip(ax_info, f["battery"], 0.59)

    reward = f.get("reward") if phase == "outcome" else None
    _action_badge(ax_info, f["action"], reward, y=0.46)

    if phase == "outcome" and "reward" in f:
        r   = f["reward"]
        txt = ("DELIVERED  +100" if r > 50
               else "BREAKDOWN   -80" if r < -50
               else "CHARGED    -25"  if f["action"] == "GOCHARGE"
               else f"IDLED  {r:.0f}")
        oc  = "#66BB6A" if r > 50 else "#EF5350" if r < -50 else "#78909C"
        _info_text(ax_info, 0.5, 0.33, txt, fs=8.5, color=oc, bold=True)

    if phase.startswith("nav_"):
        step = f.get("nav_step", "")
        _info_text(ax_info, 0.5, 0.33, f"step  {step}", fs=8, color="#FFB300", bold=True)


def _render_mappo_frame(ax_grid, ax_info, f):
    ax_grid.cla(); _clear_info(ax_info)
    from envs.assign_env import DROPOFF
    _setup_warehouse_bg(ax_grid)
    phase  = f.get("phase", "decision")
    winner = f.get("winner")

    # Highlight target cell
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

    # Winner trail during navigation
    if phase.startswith("nav_"):
        _draw_trail(ax_grid, f.get("trail", []), 10, C_NAV_TRAIL)

    # Draw all robots
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

    # info panel
    _info_text(ax_info, 0.5, 0.99, "STAGE 3", fs=9, bold=True)
    _info_text(ax_info, 0.5, 0.93, "MAPPO", fs=7.5, color="#B0BEC5")
    _info_text(ax_info, 0.5, 0.84, f"Order  {f['order_i']} / {f['n_orders']}",
               fs=11, bold=True)
    _phase_badge(ax_info, phase, y=0.73)

    card_y = [0.59, 0.40, 0.21]
    for i, robot in enumerate(f["robots"]):
        y0 = card_y[i]
        # Header
        ax_info.add_patch(plt.Rectangle((0.04, y0+0.07), 0.92, 0.075,
                                         fc=robot["color"], ec="none", zorder=2,
                                         transform=ax_info.transAxes))
        elig = "  ★" if robot.get("eligible") else "  ·"
        win  = "  WIN" if robot.get("winner") and phase == "outcome" else ""
        nav  = "  ➤" if robot.get("navigating") else ""
        ax_info.text(0.5, y0+0.107, f"{robot['label']}{elig}{win}{nav}",
                     ha="center", va="center", fontsize=7, fontweight="bold",
                     color="white", transform=ax_info.transAxes, zorder=3)
        # Battery
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
        # Action + reward
        ac  = ACTION_COLORS.get(robot["action"], "#607D8B")
        rwd = (f"  {robot['reward']:+.0f}"
               if phase == "outcome" and "reward" in robot else "")
        ax_info.text(0.5, y0+0.022, f"{robot['action']}{rwd}",
                     ha="center", va="center", fontsize=7, color=ac,
                     fontweight="bold", transform=ax_info.transAxes)


RENDERERS = {"nav":    _render_nav_frame,
             "assign": _render_assign_frame,
             "mappo":  _render_mappo_frame}

LEGENDS = {
    "nav": [mpatches.Patch(fc=C_GOAL,      label="Goal"),
            mpatches.Patch(fc=C_CHARGER,   label="Charger"),
            mpatches.Patch(fc=C_OBS,       label="Obstacle"),
            mpatches.Patch(fc=C_DYNOBS,    label="Dyn.Obs")],
    "assign": [mpatches.Patch(fc=C_PICKUP,    label="Pickup"),
               mpatches.Patch(fc=C_CHARGER,   label="Charger"),
               mpatches.Patch(fc=C_DROPOFF,   label="Dropoff"),
               mpatches.Patch(fc=C_SHELF,     label="Shelf"),
               mpatches.Patch(fc=C_NAV_TRAIL, label="Path taken")],
    "mappo":  [mpatches.Patch(fc=ROBOT_COLORS[0], label="R0"),
               mpatches.Patch(fc=ROBOT_COLORS[1], label="R1"),
               mpatches.Patch(fc=ROBOT_COLORS[2], label="R2"),
               mpatches.Patch(fc=C_ELIGIBLE,       label="★ Eligible"),
               mpatches.Patch(fc=C_NAV_TRAIL,      label="Path taken")],
}


# ===========================================================================
# Figure factory (shared by GIF and HTML builders)
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
def make_animation(frames: list, out_path: str, fps: int = 4,
                   figsize: tuple = (9.5, 6.5)):
    """Save as .gif (portable, no browser needed)."""
    if not frames:
        print(f"  No frames — skipping {out_path}"); return
    fig, ani = _build_fig_and_ani(frames, fps, figsize)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps, dpi=110)
    plt.close(fig)
    print(f"  Saved GIF  -> {out_path}")


def make_html_animation(frames: list, out_path: str, fps: int = 4,
                        figsize: tuple = (9.5, 6.5)):
    """
    Save as .html with an interactive player (play/pause/scrub/speed).
    Open in any browser or display inline in Jupyter with:
        from IPython.display import HTML
        display(HTML(open(path).read()))
    """
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
def record_nav_episode(model, level: int, device, seed=0,
                       untrained=False, force_battery=None) -> list:
    """Record a navigation episode. Trained: BFS path. Untrained: random walk."""
    from envs.nav_env import WarehouseEnv
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env   = WarehouseEnv(level=level)
    state = env.reset()
    if force_battery is not None:
        env.battery = float(force_battery)

    # untrained: random walk
    if untrained or model is None:
        frames, trail = [], []
        while not env.done:
            trail.append(env.pos)
            action = random.randint(0, 5)
            frames.append({"type": "nav", "level": level, "size": env.size,
                            "grid": env.grid.copy(), "pos": env.pos,
                            "goal": env.goal, "battery": env.battery,
                            "dyn_obs": [d.pos for d in env.dyn_obs],
                            "trail": list(trail[-25:]),
                            "action": env.ACTION_NAMES[action],
                            "step": env.steps, "untrained": True})
            state, _, done, success = env.step(action)
            if done:
                frames.append({**frames[-1], "pos": env.pos, "battery": env.battery,
                               "status": ("REACHED GOAL!" if success
                                          else "BATTERY DEAD" if env.battery <= 0
                                          else "TIMEOUT")})
                break
        return frames

    # trained: step-by-step model inference (shows actual policy behavior)
    frames  = []
    trail   = []
    success = False
    step_i  = 0
    while not env.done:
        trail.append(env.pos)
        frames.append({
            "type": "nav", "level": level, "size": env.size,
            "grid": env.grid.copy(), "pos": env.pos,
            "goal": env.goal, "battery": env.battery,
            "dyn_obs": [d.pos for d in env.dyn_obs],
            "trail": list(trail[-25:]),
            "action": "Move",
            "step": step_i, "untrained": False,
        })
        st = torch.tensor(state, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(st)
            logits = out[0] if isinstance(out, tuple) else out  # PPO→tuple, DQN→tensor
        logits = torch.clamp(logits.squeeze(0), -20.0, 20.0)
        probs  = torch.softmax(logits / 0.3, dim=-1)
        action = torch.distributions.Categorical(probs).sample().item()
        state, _, done, success = env.step(action)
        step_i += 1
        if done:
            frames.append({**frames[-1], "pos": env.pos,
                           "battery": env.battery,
                           "dyn_obs": [d.pos for d in env.dyn_obs],
                           "status": ("REACHED GOAL!" if success
                                      else "BATTERY DEAD" if env.battery <= 0
                                      else "TIMEOUT")})
            break
    return frames


def record_assign_episode(nav_model, assign_model, device, seed=0,
                          n_orders=5, force_battery=None,
                          force_position=None,
                          force_actions: dict = None,
                          show_nav: bool = True,
                          nav_skip: int = 2) -> list:
    """
    show_nav  : if True, insert step-by-step navigation frames between decisions
    nav_skip  : keep every nav_skip-th nav step (reduces frame count for GIF)
    """
    from envs.assign_env import WarehouseStage2, PICKUP_POINTS, DROPOFF, CHARGERS
    random.seed(seed)
    env = WarehouseStage2(); env.reset()
    if force_battery  is not None: env.battery = float(force_battery)
    if force_position is not None: env.pos     = force_position
    force_actions = force_actions or {}
    frames = []

    for order_i in range(n_orders):
        pickup    = random.choice(PICKUP_POINTS)
        remaining = n_orders - order_i
        feat      = env.get_obs(pickup, remaining)

        if order_i in force_actions:
            action = force_actions[order_i]
        else:
            with torch.no_grad():
                q      = assign_model(
                    torch.tensor(feat, device=device).unsqueeze(0))
                action = q.argmax().item()

        action_name = ACTION_NAMES_ASSIGN[action]
        base = {"type": "assign", "pickup": pickup,
                "action": action_name,
                "order_i": order_i + 1, "n_orders": n_orders}

        # ── Decision frame ────────────────────────────────────────────
        frames.append({**base, "pos": env.pos, "battery": env.battery,
                       "phase": "decision",
                       "status": f"Order {order_i+1}/{n_orders}  →  deciding..."})

        # ── Navigation frames ─────────────────────────────────────────
        if show_nav and action == 0:   # Accept → navigate pickup → dropoff
            trail = []
            nav_to_pickup = _trace_nav_path(env.pos, env.battery, nav_model,
                                            pickup, device, skip=nav_skip)
            for step_i, (pos, bat) in enumerate(nav_to_pickup):
                trail.append(pos)
                frames.append({**base, "pos": pos, "battery": bat,
                                "phase": "nav_pickup",
                                "trail": list(trail[-30:]),
                                "nav_step": step_i + 1,
                                "status": f"R0 → PICKUP  (step {step_i+1})"})
            # now trace pickup → dropoff (using last known pos/battery)
            last_pos, last_bat = nav_to_pickup[-1]
            trail2 = [last_pos]
            nav_to_drop = _trace_nav_path(last_pos, last_bat, nav_model,
                                          DROPOFF, device, skip=nav_skip)
            for step_i, (pos, bat) in enumerate(nav_to_drop):
                trail2.append(pos)
                frames.append({**base, "pos": pos, "battery": bat,
                                "phase": "nav_dropoff",
                                "trail": list(trail2[-30:]),
                                "nav_step": step_i + 1,
                                "status": f"R0 → DROPOFF  (step {step_i+1})"})

        elif show_nav and action == 2:  # GoCharge → navigate to nearest charger
            from envs.assign_env import nearest_charger_info
            ch_pos, _ = nearest_charger_info(env.pos)
            trail = []
            # skip=1 always — charger paths are short, every step must be visible
            nav_path = _trace_nav_path(env.pos, env.battery, nav_model,
                                       ch_pos, device, skip=1)
            for step_i, (pos, bat) in enumerate(nav_path):
                trail.append(pos)
                frames.append({**base, "pos": pos, "battery": bat,
                                "phase": "nav_charger",
                                "target": ch_pos,
                                "trail": list(trail[-30:]),
                                "nav_step": step_i + 1,
                                "status": f"R0 → CHARGER  (step {step_i+1})"})

        # ── Execute & outcome frame ───────────────────────────────────
        if action == 0:
            reward = env.execute_order(nav_model, pickup); env.idle_time = 0
        elif action == 1:
            reward = env.execute_decline_idle()
        else:
            reward = env.execute_go_charge(nav_model)

        frames.append({**base, "pos": env.pos, "battery": env.battery,
                       "reward": reward, "phase": "outcome"})
    return frames


def record_mappo_episode(nav_model, actor, device, seed=0,
                         random_policy=False,
                         show_nav: bool = True,
                         nav_skip: int = 2) -> list:
    """
    show_nav : if True, insert step-by-step nav frames for the winning robot
    """
    from envs.marl_env import MultiAgentWarehouse, N_AGENTS, N_ORDERS
    from envs.assign_env import bfs_dist, TRIP_COST_RATE, DROPOFF, nearest_charger_info
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env     = MultiAgentWarehouse()
    obs, gs = env.reset()
    frames  = []

    for order_i in range(N_ORDERS):
        pickup   = env.orders[order_i]
        eligible = env._k_nearest(pickup)
        obs_t    = torch.tensor(obs, device=device, dtype=torch.float32)

        if random_policy:
            actions = [random.randint(0, 2) for _ in range(N_AGENTS)]
        else:
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

        # ── Decision frame ────────────────────────────────────────────
        winner_str = f"R{winner} assigned" if winner is not None else "No accept"
        frames.append({**base,
                       "robots": robot_state(),
                       "phase": "decision",
                       "status": f"Order {order_i+1}/{N_ORDERS} — {winner_str}"})

        # ── Navigation frames for winner ──────────────────────────────
        if show_nav and winner is not None:
            w_agent   = env.agents[winner]
            w_pos     = w_agent.pos
            w_bat     = w_agent.battery
            other_pos = frozenset(env.agents[j].pos
                                  for j in range(N_AGENTS) if j != winner)
            trail     = []

            # Winner → pickup
            nav_to_pickup = _trace_nav_path(w_pos, w_bat, nav_model,
                                            pickup, device,
                                            other_positions=other_pos,
                                            skip=nav_skip)
            for step_i, (pos, bat) in enumerate(nav_to_pickup):
                trail.append(pos)
                ov = {winner: {"pos": pos, "battery": bat}}
                frames.append({**base,
                                "robots": robot_state(overrides=ov,
                                                       navigating_idx=winner),
                                "phase": "nav_pickup",
                                "trail": list(trail[-30:]),
                                "nav_step": step_i + 1,
                                "status": f"R{winner} → PICKUP  (step {step_i+1})"})

            last_pos, last_bat = nav_to_pickup[-1]
            trail2 = [last_pos]

            # Winner → dropoff
            nav_to_drop = _trace_nav_path(last_pos, last_bat, nav_model,
                                          DROPOFF, device,
                                          other_positions=other_pos,
                                          skip=nav_skip)
            for step_i, (pos, bat) in enumerate(nav_to_drop):
                trail2.append(pos)
                ov = {winner: {"pos": pos, "battery": bat}}
                frames.append({**base,
                                "robots": robot_state(overrides=ov,
                                                       navigating_idx=winner),
                                "phase": "nav_dropoff",
                                "trail": list(trail2[-30:]),
                                "nav_step": step_i + 1,
                                "status": f"R{winner} → DROPOFF  (step {step_i+1})"})

        # ── Execute & outcome frame ───────────────────────────────────
        obs, gs, rewards, done = env.step(actions, nav_model)
        frames.append({**base,
                       "robots": robot_state(rewards=rewards),
                       "phase": "outcome"})
        if done:
            break
    return frames


# ===========================================================================
# Model loaders
# ===========================================================================
def _load_nav_dqn(ckpt_dir, device, level):
    from agents.ppo import DQN
    path = os.path.join(ckpt_dir, f"dqn_l{level}.pt")
    m = DQN(state_dim=13, action_dim=6).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    return m

def _load_nav_ppo(ckpt_dir, device, name="ppo_final.pt"):
    from agents.ppo import PPO
    path = os.path.join(ckpt_dir, name)
    m = PPO(state_dim=13, action_dim=6).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def _load_assign(ckpt_dir, device):
    from agents.dqn import AssignmentDQN
    nav = _load_nav_ppo(ckpt_dir, device)
    dqn = AssignmentDQN().to(device)
    dqn.load_state_dict(torch.load(os.path.join(ckpt_dir, "assign_dqn.pt"),
                                   map_location=device)); dqn.eval()
    return nav, dqn

def _load_mappo(ckpt_dir, device):
    from agents.mappo import AssignmentActor
    from envs.marl_env import OBS_DIM
    nav   = _load_nav_ppo(ckpt_dir, device)
    actor = AssignmentActor(obs_dim=OBS_DIM).to(device)
    actor.load_state_dict(torch.load(os.path.join(ckpt_dir, "mappo_actor.pt"),
                                     map_location=device)); actor.eval()
    return nav, actor

def _with_assign(ckpt_dir, device, fn):
    nav, dqn = _load_assign(ckpt_dir, device)
    return fn(nav, dqn)

def _with_mappo(ckpt_dir, device, fn):
    nav, actor = _load_mappo(ckpt_dir, device)
    return fn(nav, actor)

# ===========================================================================
# 9-scenario catalog
# ===========================================================================
def build_scenario_catalog(ckpt_dir: str, device) -> dict:
    cat = {}

    # ── Stage 1 — Navigation ─────────────────────────────────────────────────
    cat[1] = dict(title="Nav failure — random policy (5×5)",
                  desc="Untrained random policy. Robot wanders and never reaches the goal.",
                  out="s01_nav_failure_5x5.html",
                  record=lambda: record_nav_episode(
                      None, 1, device, untrained=True, seed=7))

    cat[2] = dict(title="Nav success — trained DQN (5×5)",
                  desc="DQN trained on 5×5 grid. Near-direct path to the goal.",
                  out="s02_nav_success_5x5.html",
                  record=lambda: record_nav_episode(
                      _load_nav_dqn(ckpt_dir, device, 1), 1, device, seed=0))

    cat[3] = dict(title="Nav success — 10×10 grid, no obstacles",
                  desc="Trained DQN on 10×10 plain grid. Efficient path with battery intact.",
                  out="s03_nav_success_10x10.html",
                  record=lambda: record_nav_episode(
                      _load_nav_dqn(ckpt_dir, device, 2), 2, device, seed=0))

    cat[4] = dict(title="Nav success — 10×10 with stationary obstacles",
                  desc="PPO routes around static obstacles to goal.",
                  out="s04_nav_success_obstacles.html",
                  record=lambda: record_nav_episode(
                      _load_nav_ppo(ckpt_dir, device, "ppo_final.pt"), 3, device, seed=1))

    cat[5] = dict(title="Nav success — 10×10 with moving dynamic obstacles",
                  desc="Step-by-step execution. Dynamic obstacles move each frame; robot navigates around them.",
                  out="s05_nav_success_dynobs.html",
                  record=lambda: record_nav_episode(
                      _load_nav_ppo(ckpt_dir, device), 4, device, seed=3))

    # ── Stage 2 — Assignment DQN ─────────────────────────────────────────────
    cat[6] = dict(title="Assignment failure — breakdown on low battery",
                  desc="Forced Accept at 15% battery. Robot runs out of charge mid-trip. Outcome: −80.",
                  out="s06_assign_failure.html",
                  record=lambda: _with_assign(ckpt_dir, device, lambda n, d:
                      record_assign_episode(n, d, device, seed=1,
                                            force_battery=15.0, force_actions={0: 0})))

    cat[7] = dict(title="Assignment success — delivery at high battery",
                  desc="DQN accepts at 90% battery. Robot navigates pickup → dropoff. Outcome: +100.",
                  out="s07_assign_success.html",
                  record=lambda: _with_assign(ckpt_dir, device, lambda n, d:
                      record_assign_episode(n, d, device, seed=3,
                                            force_battery=90.0, force_actions={0: 0})))

    # ── Stage 3 — MAPPO ──────────────────────────────────────────────────────
    cat[8] = dict(title="MAPPO failure — random policy (no coordination)",
                  desc="Untrained random actor. Robots pick actions randomly — frequent breakdowns, missed orders.",
                  out="s08_mappo_failure.html",
                  record=lambda: _with_mappo(ckpt_dir, device, lambda n, a:
                      record_mappo_episode(n, a, device, seed=0, random_policy=True)))

    cat[9] = dict(title="MAPPO success — 3-robot coordination",
                  desc="Trained MAPPO. K=2 dispatch, winner navigates pickup → dropoff. High delivery rate.",
                  out="s09_mappo_success.html",
                  record=lambda: _with_mappo(ckpt_dir, device, lambda n, a:
                      record_mappo_episode(n, a, device, seed=92)))

    return cat


# ===========================================================================
# CLI runner
# ===========================================================================
def run_all(ckpt_dir: str, out_dir: str, device, fmt: str = "html"):
    cat     = build_scenario_catalog(ckpt_dir, device)
    fps_map = {**{i: 8 for i in range(1, 6)},
               **{i: 4 for i in range(6, 10)}}
    save_fn = make_html_animation if fmt == "html" else make_animation
    ext     = ".html" if fmt == "html" else ".gif"

    for n in sorted(cat.keys()):
        s = cat[n]
        print(f"  [{n}] {s['title']}")
        try:
            frames   = s["record"]()
            out_name = os.path.splitext(s["out"])[0] + ext
            save_fn(frames, os.path.join(out_dir, out_name),
                    fps=fps_map.get(n, 4))
        except Exception as e:
            import traceback
            print(f"       ERROR: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["html", "gif"], default="html",
                        help="Output format (default: html)")
    args = parser.parse_args()

    root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    out_dir  = os.path.join(ckpt_dir, "animations")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Format : {args.fmt}")
    print(f"Output : {out_dir}\n")
    run_all(ckpt_dir, out_dir, device, fmt=args.fmt)


if __name__ == "__main__":
    main()
