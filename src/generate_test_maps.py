import os
import numpy as np
from pathlib import Path

from src.config_loader import load_config
from src.env import SARGridEnv

# Execute as: python -m src.generate_test_maps


def generate_and_save(cfg_path="configs/env.yaml", out_dir="configs/test_maps", test_count=30, runs_per_map=1):
    cfg = load_config(cfg_path if cfg_path.endswith(".yaml") else "configs/env.yaml")
    env = SARGridEnv(cfg)  # use same generator
    os.makedirs(out_dir, exist_ok=True)

    base_seed = env.base_seed
    for map_id in range(test_count):
        # use a deterministic seed per map
        map_seed = int(base_seed + map_id * 1000)
        # create a map once per map_id
        true_map = env._generate_map(seed=map_seed)
        # for runs_per_map we may want different agent/victim positions
        for run in range(runs_per_map):
            rng = np.random.RandomState(int(map_seed + run))
            # sample agent and victim on free cells within this true_map (use env helpers)
            # we'll directly choose free cells here
            free_cells = list(zip(*np.where(true_map == 0)))
            agent_pos = tuple(map(int, free_cells[rng.randint(len(free_cells))]))
            # choose victim in same connected component? optional
            victim_pos = tuple(map(int, free_cells[rng.randint(len(free_cells))]))
            # ensure not too close
            if abs(agent_pos[0]-victim_pos[0]) + abs(agent_pos[1]-victim_pos[1]) < 4:
                # try a few times
                for _ in range(100):
                    v = tuple(map(int, free_cells[rng.randint(len(free_cells))]))
                    if abs(agent_pos[0]-v[0]) + abs(agent_pos[1]-v[1]) >= 4:
                        victim_pos = v
                        break
            fname = Path(out_dir) / f"map_{map_id:03d}_run_{run:02d}.npz"
            np.savez_compressed(str(fname),
                                true_map=true_map,
                                agent_pos=np.array(agent_pos),
                                victim_pos=np.array(victim_pos),
                                map_seed=map_seed,
                                run_seed=int(map_seed + run))
            print("Saved", fname)

if __name__ == "__main__":
    # usage: adjust test_count and runs_per_map as needed
    generate_and_save(cfg_path="configs/env.yaml", out_dir="configs/test_maps", test_count=30, runs_per_map=10)
