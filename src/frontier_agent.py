# src/frontier_agent.py
"""
Frontier-based baseline runner for SARGridEnv.

Generates a CSV with episodes:
map_id, run, seed, method, time_to_find, success, path_length, coverage_percent

Usage:
python -m src.frontier_agent --config configs/frontier.yaml --maps 30 --runs 10 --out results/logs/frontier_final.csv
"""

import argparse
import os
from pathlib import Path
import heapq
import math
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.env import SARGridEnv

# -------- A* planner (grid, 4-neigh) ----------
def astar(start, goal, occ_grid):
    """
    occ_grid: 2D array where 0 = free, 1 = occupied/blocked
    start/goal: (x,y)
    Returns list of (x,y) path or None.
    """
    h, w = occ_grid.shape

    def neighbors(n):
        x, y = n
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and occ_grid[nx, ny] == 0:
                yield (nx, ny)

    def hcost(a, b):
        # Manhattan heuristic
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    openq = []
    heapq.heappush(openq, (hcost(start, goal), 0, start))
    gscore = {start: 0}
    came = {}
    visited = set()

    while openq:
        f, g, current = heapq.heappop(openq)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return path[::-1]
        if current in visited:
            continue
        visited.add(current)
        for nb in neighbors(current):
            ng = g + 1
            if ng < gscore.get(nb, 1e9):
                gscore[nb] = ng
                came[nb] = current
                heapq.heappush(openq, (ng + hcost(nb, goal), ng, nb))
    return None

# -------- Frontier detection (WFD-like simple) ----------
def detect_frontiers(known_map):
    """
    known_map: values in {0.0 free, 1.0 occupied}
    Returns list of frontier cells (x,y) - free cells adjacent to unknown.
    """
    h, w = known_map.shape
    frontiers = []
    for i in range(h):
        for j in range(w):
            if known_map[i, j] == 0.0:
                # check 4-neighbor for unknown
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < h and 0 <= y < w and known_map[x, y] == 0.5:
                        frontiers.append((i, j))
                        break
    return frontiers

# -------- mapping from desired step to action given heading ----------
def step_to_action(agent_heading, from_pos, to_pos):
    """
    agent_heading: 0=up,1=right,2=down,3=left
    from_pos, to_pos: (x,y)
    returns action in {0=forward,1=turn_left,2=turn_right,3=stay}
    Strategy: if forward matches desired delta -> forward else choose left/right to rotate.
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    desired = None
    if (dx, dy) == (-1, 0):
        desired = 0
    elif (dx, dy) == (0, 1):
        desired = 1
    elif (dx, dy) == (1, 0):
        desired = 2
    elif (dx, dy) == (0, -1):
        desired = 3
    else:
        return 3  # stay if unexpected

    if desired == agent_heading:
        return 0  # forward
    # compute minimal rotation: (desired - heading) mod 4
    diff = (desired - agent_heading) % 4
    if diff == 1:
        return 2  # right
    elif diff == 3:
        return 1  # left
    else:
        # opposite (diff==2) - choose left (arbitrary)
        return 1

# -------- main runner ----------
def run_frontier(cfg_path, out_csv="results/logs/frontier_results.csv", maps_to_run=None, runs_per_map_override=None, visualize=False):
    cfg = load_config(cfg_path)
    env = SARGridEnv(cfg)
    env_cfg = cfg.get("_env", cfg)

    test_count = int(env_cfg.get("maps", {}).get("test_count", 30)) if maps_to_run is None else int(maps_to_run)
    runs_per_map = int(env_cfg.get("seeds", {}).get("runs_per_map", 10)) if runs_per_map_override is None else int(runs_per_map_override)

    results = []
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    figdir = Path("results/figures")
    if visualize:
        figdir.mkdir(parents=True, exist_ok=True)

    for map_id in range(test_count):
        for run in range(runs_per_map):
            seed = int(env.base_seed + map_id * 100 + run)
            obs, info = env.reset(seed=seed)

            # initialize path length based on Manhattan traveled
            prev_pos = env._agent_pos
            path_length = 0

            t = 0
            success = False

            # main episode loop
            while t < env.T_max:
                # current known map
                known = env._known  # values {0.0, 1.0, 0.5}
                frontiers = detect_frontiers(known)

                if len(frontiers) == 0:
                    # nothing to explore: random policy fallback
                    action = env.action_space.sample()
                else:
                    # choose nearest frontier (Manhattan) as a simple heuristic
                    agent_pos = env._agent_pos
                    frontiers.sort(key=lambda p: abs(p[0]-agent_pos[0]) + abs(p[1]-agent_pos[1]))
                    goal = frontiers[0]

                    # build safe occupancy for planning: known free=0, others blocked
                    occ_grid = np.where(env._known == 0.0, 0, 1)

                    # compute path to goal with A*
                    path = astar(agent_pos, goal, occ_grid)
                    if (path is None) or (len(path) < 2):
                        # cannot plan: stay or random small move
                        action = env.action_space.sample()
                    else:
                        # move toward the next cell in path
                        next_cell = path[1]
                        action = step_to_action(env._agent_heading, agent_pos, next_cell)

                # perform step
                obs, reward, terminated, truncated, info = env.step(int(action))

                # update traveled path length (Manhattan)
                new_pos = env._agent_pos
                step_dist = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                path_length += int(step_dist)
                prev_pos = new_pos

                t += 1

                # detection check (env provides 'detected' in info)
                if info.get("detected", False):
                    success = True
                    break

            # coverage efficiency: percentage of cells explored
           
            num_free = int(np.sum(env._true_map == 0))
            num_visited = len(env._visited)
            coverage = (num_visited / max(1, num_free)) * 100.0

            results.append({
                "map_id": int(map_id),
                "run": int(run),
                "seed": int(seed),
                "method": "frontier",
                "time_to_find": int(t) if success else int(env.T_max),
                "success": int(success),
                "path_length": int(path_length),
                "coverage_percent": float(coverage)
            })

            # optional visualization per-first-run of each map
            if visualize and run == 0:
                try:
                    import matplotlib.pyplot as plt
                    vis = np.zeros((env.size, env.size, 3), dtype=np.uint8) + 255
                    vis[env._true_map == 1] = [0,0,0]  # walls black
                    for (x,y) in env._visited:
                        vis[x,y] = [0,255,0]  # visited green
                    vx, vy = env._victim_pos
                    vis[vx,vy] = [255,0,0]  # victim red
                    plt.figure(figsize=(6,6))
                    plt.imshow(vis)
                    plt.title(f"Map {map_id} visited (green), victim (red)")
                    plt.axis('off')
                    plt.savefig(figdir / f"heatmap_map{map_id}.png")
                    plt.close()
                except Exception:
                    pass

    # write results CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved frontier results to {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frontier.yaml")
    parser.add_argument("--out", default="results/logs/frontier_results.csv")
    parser.add_argument("--maps", type=int, default=None, help="override number of test maps (quick runs)")
    parser.add_argument("--runs", type=int, default=None, help="override runs per map (quick runs)")
    parser.add_argument("--visualize", action="store_true", help="save one heatmap per map (run==0)")
    args = parser.parse_args()
    run_frontier(args.config, out_csv=args.out, maps_to_run=args.maps, runs_per_map_override=args.runs, visualize=args.visualize)
