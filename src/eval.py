# src/eval.py
import argparse
import os
import json
import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path
from src.config_loader import load_config
from src.env import SARGridEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import heapq
import matplotlib.pyplot as plt

# --------------- simple A* for planning on known grid -----------------
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
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # manhattan
    openq = []
    heapq.heappush(openq, (hcost(start, goal), 0, start, None))
    came = {}
    gscore = {start: 0}
    visited = set()
    parent = {}
    while openq:
        f, g, current, _ = heapq.heappop(openq)
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
                heapq.heappush(openq, (ng + hcost(nb, goal), ng, nb, current))
    return None

# --------------- frontier helper (WFD-like) -----------------
def detect_frontiers(known_map):
    """
    Fast and safe frontier detection.
    Always returns a list of (x, y) tuples.
    known_map values:
        0.0 = free
        1.0 = occupied
        0.5 = unknown
    """
    # ---- Safety: ensure numpy array ----
    known_map = np.asarray(known_map)
    if known_map.ndim != 2:
        return []

    free = (known_map == 0.0)
    unknown = (known_map == 0.5)

    # If no unknown space exists → no frontier
    if not np.any(unknown):
        return []

    # 4-neighborhood shifts
    up    = np.pad(unknown[1:],   ((0,1),(0,0)), constant_values=False)
    down  = np.pad(unknown[:-1],  ((1,0),(0,0)), constant_values=False)
    left  = np.pad(unknown[:,1:], ((0,0),(0,1)), constant_values=False)
    right = np.pad(unknown[:,:-1],((0,0),(1,0)), constant_values=False)

    # frontier = free cell next to an unknown cell
    has_unknown_neighbor = up | down | left | right

    frontier_mask = free & has_unknown_neighbor

    # no frontier found → return empty list
    if not np.any(frontier_mask):
        return []

    coords = np.argwhere(frontier_mask)
    return [tuple(p) for p in coords]



# --------------- evaluation runner -----------------
def evaluate(cfg_path, model_path=None, mode="model", out_csv="results/logs/eval_results.csv",
             visualize=False, maps_dir: str = None):
    """
    If maps_dir is provided (path to folder with map_*.npz files), evaluation will load and
    run one episode per file using env.load_from_file(file). Otherwise it uses the
    original seed-based reset loop (map_id, run).
    """
    cfg = load_config(cfg_path)
    env = SARGridEnv(cfg)
    env_cfg = cfg.get("_env", {})

    # if user provided maps_dir, we will iterate over npz files
    files_list = None
    if maps_dir is not None:
        p = Path(maps_dir)
        if not p.exists():
            raise FileNotFoundError(f"maps_dir not found: {maps_dir}")
        files_list = sorted(p.glob("map_*.npz"))
        if len(files_list) == 0:
            raise FileNotFoundError(f"No map_*.npz files found in {maps_dir}")

    # load model if requested (same as before)
    model = None
    model_name = None
    if model_path is not None and mode != "frontier":
        model_path = Path(model_path)
        if model_path.exists():
            try:
                model = PPO.load(str(model_path))
                model_name = "PPO"
            except Exception:
                model = DQN.load(str(model_path))
                model_name = "DQN"
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    results = []
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if files_list is not None:
        # --- evaluate on saved maps (one file = one episode) ---
        for f in files_list:
            # try to parse map_id/run from filename (optional, for consistent CSV)
            map_id = None
            run = None
            try:
                name = f.stem  # e.g., "map_000_run_00"
                parts = name.split("_")
                # basic parse: map_{id}_run_{r}
                if "map" in parts and "run" in parts:
                    map_idx = parts.index("map")
                    run_idx = parts.index("run")
                    map_id = int(parts[map_idx+1])
                    run = int(parts[run_idx+1])
            except Exception:
                map_id = None
                run = None

            # load the exact saved map/state into env (needs your env.load_from_file)
            env.load_from_file(str(f))

            # get initial observation (for models) if needed
            try:
                obs = env._get_observation()
            except Exception:
                obs = None

            t = 0
            path_length = 0
            prev_pos = env._agent_pos
            success = False

            while t < env.T_max:
                if mode == "frontier":
                    known = env._known
                    frontiers = detect_frontiers(known)
                    if len(frontiers) == 0:
                        action = env.action_space.sample()
                    else:
                        agent_pos = env._agent_pos
                        frontiers.sort(key=lambda p: abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]))
                        goal = frontiers[0]
                        occ_grid = np.where(env._known==0.0, 0, 1)
                        path = astar(agent_pos, goal, occ_grid)
                        if path is None or len(path) < 2:
                            action = 3
                        else:
                            nx, ny = path[1]
                            dx = nx - agent_pos[0]
                            dy = ny - agent_pos[1]
                            heading = env._agent_heading
                            forward = {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}[heading]
                            if (dx,dy) == forward:
                                action = 0
                            else:
                                desired = None
                                if (dx,dy) == (-1,0): desired = 0
                                elif (dx,dy) == (0,1): desired = 1
                                elif (dx,dy) == (1,0): desired = 2
                                elif (dx,dy) == (0,-1): desired = 3
                                if desired is None:
                                    action = 3
                                else:
                                    if (desired - heading) % 4 == 1:
                                        action = 2
                                    elif (desired - heading) % 4 == 3:
                                        action = 1
                                    else:
                                        action = 1
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(int(action))

                # update path length correctly (Manhattan)
                new_pos = env._agent_pos
                step_dist = abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1])
                path_length += int(step_dist)
                prev_pos = new_pos

                t += 1
                if info.get("detected", False):
                    success = True
                    break

            # coverage as observed cells (same definition as in report)
            known_mask = (env._known != 0.5)
            num_known = int(np.sum(known_mask))
            num_free = int(np.sum(env._true_map == 0))
            coverage = (num_known / max(1, num_free)) * 100.0

            results.append({
                "map_file": f.name,
                "map_id": int(map_id) if map_id is not None else None,
                "run": int(run) if run is not None else None,
                "seed": int(getattr(env, "base_seed", -1)),
                "method": mode if mode!="model" else (model_name or "model"),
                "time_to_find": int(t) if success else int(env.T_max),
                "success": int(success),
                "path_length": int(path_length),
                "coverage_percent": float(coverage)
            })

    else:
        # --- original seed-based evaluation (unchanged behaviour) ---
        runs_per_map = int(env_cfg.get("seeds", {}).get("runs_per_map", 10))
        test_count = int(env_cfg.get("maps", {}).get("test_count", 30))
        for map_id in range(test_count):
            for run in range(runs_per_map):
                seed = env.base_seed + map_id * 100 + run
                obs, info = env.reset(seed=seed)
                terminated = False
                truncated = False
                t = 0
                path_length = 0
                success = False
                prev_pos = env._agent_pos
                while (not terminated) and (not truncated):
                    if mode == "frontier":
                        known = env._known
                        frontiers = detect_frontiers(known)
                        if len(frontiers) == 0:
                            action = env.action_space.sample()
                        else:
                            agent_pos = env._agent_pos
                            frontiers.sort(key=lambda p: abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]))
                            goal = frontiers[0]
                            occ_grid = np.where(env._known==0.0, 0, 1)
                            path = astar(agent_pos, goal, occ_grid)
                            if path is None or len(path) < 2:
                                action = 3
                            else:
                                nx, ny = path[1]
                                dx = nx - agent_pos[0]
                                dy = ny - agent_pos[1]
                                heading = env._agent_heading
                                forward = {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}[heading]
                                if (dx,dy) == forward:
                                    action = 0
                                else:
                                    desired = None
                                    if (dx,dy) == (-1,0): desired = 0
                                    elif (dx,dy) == (0,1): desired = 1
                                    elif (dx,dy) == (1,0): desired = 2
                                    elif (dx,dy) == (0,-1): desired = 3
                                    if desired is None:
                                        action = 3
                                    else:
                                        if (desired - heading) % 4 == 1:
                                            action = 2
                                        elif (desired - heading) % 4 == 3:
                                            action = 1
                                        else:
                                            action = 1
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(int(action))

                    # update path length correctly (Manhattan)
                    new_pos = env._agent_pos
                    step_dist = abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1])
                    path_length += int(step_dist)
                    prev_pos = new_pos

                    t += 1
                    if info.get("detected", False):
                        success = True
                    if t >= env.T_max:
                        break

                coverage = (len(env._visited) / max(1, np.sum(env._true_map == 0))) * 100.0
                results.append({
                    "map_id": map_id,
                    "run": run,
                    "seed": seed,
                    "method": mode if mode!="model" else (model_name or "model"),
                    "time_to_find": int(t) if success else int(env.T_max),
                    "success": int(success),
                    "path_length": int(path_length),
                    "coverage_percent": float(coverage)
                })

    # write results to CSV via pandas
    df = pd.DataFrame(results)
    outdir = Path(out_csv).parent
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved evaluation results to {out_csv}")
    return df





# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml", help="Eval config")
    parser.add_argument("--model", default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--mode", default="model", choices=["model","frontier"], help="mode to evaluate")
    parser.add_argument("--out", default="results/logs/eval_results.csv", help="CSV output")
    parser.add_argument("--visualize", action="store_true", help="save heatmaps")
    parser.add_argument("--maps_dir", default="configs/test_maps", help="Optional: path to folder with saved map .npz files")
    args = parser.parse_args()
    df = evaluate(args.config, model_path=args.model, mode=args.mode, out_csv=args.out,
                  visualize=args.visualize, maps_dir=args.maps_dir)
    print(df.head())