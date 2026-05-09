"""Quick diagnostic: test fixed _navigate on isolated 200 orders."""
import sys, random, torch, numpy as np, time
sys.stdout.reconfigure(encoding="utf-8")

from envs.assign_env import WarehouseStage2, PICKUP_POINTS, FREE_CELLS
from agents.ppo import PPO

DEVICE = torch.device("cpu")
nav = PPO(state_dim=13, action_dim=6).to(DEVICE)
nav.load_state_dict(torch.load("checkpoints/ppo_final.pt", map_location=DEVICE, weights_only=True))
nav.eval()
for p in nav.parameters():
    p.requires_grad_(False)

random.seed(0); np.random.seed(0)
N = 200
res = {"del": 0, "to": 0, "brk": 0}
t0 = time.time()

for i in range(N):
    agent = WarehouseStage2()
    agent.reset()
    others = random.sample([c for c in FREE_CELLS if c != agent.pos], 2)
    pickup = random.choice(PICKUP_POINTS)
    r = agent.execute_order(nav, pickup, frozenset(others))
    if r > 50:     res["del"] += 1
    elif r == -80: res["brk"] += 1
    else:          res["to"]  += 1
    if (i+1) % 50 == 0:
        print(f"  {i+1}/200  ({time.time()-t0:.1f}s)", flush=True)

elapsed = time.time() - t0
print(f"\nN={N}  time={elapsed:.2f}s  ({elapsed/N*1000:.1f}ms/ep)")
print(f"  Delivered  {res['del']} ({res['del']/N*100:.1f}%)")
print(f"  Timeout    {res['to']}  ({res['to']/N*100:.1f}%)")
print(f"  Breakdown  {res['brk']} ({res['brk']/N*100:.1f}%)")
