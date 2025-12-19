# src/env.py
import numpy as np
import math
import random
from gymnasium import Env, spaces

# --------- Utility helpers ----------
def bresenham_line(x0, y0, x1, y1):
    """Bresenham integer line from (x0,y0) to (x1,y1) inclusive."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# --------- SARGridEnv ----------
class SARGridEnv(Env):
    """
    Gym-style 2D occupancy-grid environment for SAR.
    Observation: (C, patch, patch) float32 in [0,1] for C channels.
      channels = [known_occupancy, heading_channel, timestep_channel]
    Action space: Discrete(4): 0=forward,1=turn_left,2=turn_right,3=stay
    Uses gymnasium step() signature: (obs, reward, terminated, truncated, info)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        # config is the dict returned by load_config(cfg_path)
        cfg_env = config.get("_env", config)  # merged env config
        grid_cfg = cfg_env.get("grid", {})
        self.size = int(grid_cfg.get("size", 60))
        self.cell_size = float(grid_cfg.get("cell_size", 0.2))
        self.T_max = int(cfg_env.get("episode", {}).get("max_steps", 500))

        # sensor params
        sensor = cfg_env.get("sensor", {})
        self.num_rays = int(sensor.get("rays", 36))
        self.max_range = int(sensor.get("max_range", 12))
        self.range_noise = float(sensor.get("range_noise", 0.2))

        # motion
        motion = cfg_env.get("motion", {})
        self.p_slip = float(motion.get("slip_probability", 0.02))

        # victim
        victim_cfg = cfg_env.get("victim", {})
        self.detection_mode = victim_cfg.get("detection_mode", "los")
        self.detection_range = int(victim_cfg.get("detection_range", 1))

        # maps procedural parameters
        proc = cfg_env.get("maps", {}).get("procedural", {})
        self.obs_density_range = proc.get("obstacle_density_range", [0.05, 0.15])

        # seeds & runs per map
        seeds_cfg = cfg_env.get("seeds", {})
        self.base_seed = int(seeds_cfg.get("base_seed", 1000))
        self.runs_per_map = int(seeds_cfg.get("runs_per_map", 10))

        # RL observation patch size: prefer top-level 'agent' (algo-specific), else fallback to included env
        agent_cfg = config.get("agent") or config.get("_env", {}).get("agent") or {}
        self.patch = int(agent_cfg.get("obs_patch", 11))
        if self.patch % 2 == 0:
            self.patch += 1

               # observation & action spaces
        # observation: flattened local patch (patch*patch) + heading (1) + time (1)
        obs_dim = (self.patch * self.patch) + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # internal state placeholders
        self._true_map = None       # ground truth: 0 free, 1 occupied
        self._known = None          # agent-known map: 0 free, 1 occ, 0.5 unknown
        self._agent_pos = None      # (x,y)
        self._agent_heading = 0     # 0=up,1=right,2=down,3=left
        self._victim_pos = None
        self._t = 0
        self._visited = set()

    # ---------------- Map generation ----------------
    def _generate_map(self, seed=None):
        rng = np.random.RandomState(seed)
        density = float(rng.uniform(self.obs_density_range[0], self.obs_density_range[1]))
        grid = (rng.rand(self.size, self.size) < density).astype(np.uint8)
        # carve border-free area for start
        margin = 2
        grid[:margin, :] = 0
        grid[:, :margin] = 0
        grid[self.size-margin:, :] = 0
        grid[:, self.size-margin:] = 0
        # ensure connectivity: naive approach - remove isolated islands near agent later
        return grid

    def _sample_free_cell(self, rng):
        for _ in range(10000):
            x = rng.randint(0, self.size)
            y = rng.randint(0, self.size)
            if self._true_map[x, y] == 0:
                return (x, y)
        # fallback
        for i in range(self.size):
            for j in range(self.size):
                if self._true_map[i, j] == 0:
                    return (i, j)
        return (0, 0)

    # ---------------- LOS detection ----------------
    def _has_line_of_sight(self, a, b):
        line = bresenham_line(a[0], a[1], b[0], b[1])
        # exclude the first point (agent pos), check intermediate points for obstacles
        for (x, y) in line[1:]:
            if not (0 <= x < self.size and 0 <= y < self.size):
                return False
            if self._true_map[x, y] == 1:
                return False
        return True

    # ---------------- Sensor simulation ----------------
    def _simulate_scan(self, pos):
        """Return list of ranges (float) for each ray, with noise."""
        px, py = pos
        ranges = []
        # angles across 360 deg
        for i in range(self.num_rays):
            theta = 2.0 * math.pi * i / self.num_rays
            dx = math.cos(theta)
            dy = math.sin(theta)
            r = 0
            hit = False
            while r < self.max_range:
                r += 1
                x = int(round(px + dx * r))
                y = int(round(py + dy * r))
                if x < 0 or y < 0 or x >= self.size or y >= self.size:
                    break
                if self._true_map[x, y] == 1:
                    hit = True
                    break
            measured = float(r if hit else self.max_range)
            measured += np.random.normal(0.0, self.range_noise)
            measured = clamp(measured, 0.0, self.max_range)
            ranges.append(measured)
        return np.array(ranges, dtype=np.float32)

    # ---------------- Occupancy update (simple) ----------------
    def _update_known_from_scan(self, scan, pos):
        """Mark free cells along ray and mark hit as occupied if within range."""
        px, py = pos
        for i in range(self.num_rays):
            theta = 2.0 * math.pi * i / self.num_rays
            dx = math.cos(theta)
            dy = math.sin(theta)
            r = int(round(scan[i]))
            for step in range(1, r):
                x = int(round(px + dx * step))
                y = int(round(py + dy * step))
                if 0 <= x < self.size and 0 <= y < self.size:
                    self._known[x, y] = 0.0
            # if hit occurred and within grid mark occupied
            if r < self.max_range:
                x = int(round(px + dx * r))
                y = int(round(py + dy * r))
                if 0 <= x < self.size and 0 <= y < self.size:
                    self._known[x, y] = 1.0

    # ---------------- Observation construction ----------------
    def _get_local_patch(self):
        """
        Return an (patch, patch) array centered on agent position, with values
        from self._known (0.0 free, 1.0 occ, 0.5 unknown). Out-of-bounds are filled with 0.5.
        """
        half = self.patch // 2
        px, py = self._agent_pos
        patch = np.full((self.patch, self.patch), 0.5, dtype=np.float32)
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                x = px + i
                y = py + j
                if 0 <= x < self.size and 0 <= y < self.size:
                    patch[i + half, j + half] = float(self._known[x, y])
        return patch
    

    def _get_observation(self):
        """
        Build flat observation vector:
          [flattened local patch (patch*patch), heading_norm (1), time_norm (1)]
        """
        patch = self._get_local_patch()  # (patch,patch)
        patch_flat = patch.flatten()     # length patch*patch

        heading = np.array([self._agent_heading / 3.0], dtype=np.float32)
        time_norm = np.array([self._t / float(self.T_max)], dtype=np.float32)

        obs = np.concatenate([patch_flat, heading, time_norm]).astype(np.float32)
        return obs
    
    # ---------------- reset, step, render ----------------
    def reset(self, seed=None, options=None):
        rng = np.random.RandomState(seed if seed is not None else None)
        # create true map
        gen_seed = rng.randint(0, 2**30 - 1)
        self._true_map = self._generate_map(seed=gen_seed)
        # init known map to unknown (0.5)
        self._known = np.full_like(self._true_map, 0.5, dtype=np.float32)
        # sample agent and victim positions on free cells
        self._agent_pos = self._sample_free_cell(rng)
        self._agent_heading = rng.randint(0, 4)
        self._victim_pos = self._sample_free_cell(rng)
        # ensure victim is not too close to agent
        if (abs(self._victim_pos[0] - self._agent_pos[0]) + abs(self._victim_pos[1] - self._agent_pos[1])) < 6:
            # try relocate victim
            for _ in range(100):
                cand = self._sample_free_cell(rng)
                if (abs(cand[0] - self._agent_pos[0]) + abs(cand[1] - self._agent_pos[1])) >= 6:
                    self._victim_pos = cand
                    break
        self._t = 0
        self._visited = set()
        # initial scan & update known
        scan = self._simulate_scan(self._agent_pos)
        self._update_known_from_scan(scan, self._agent_pos)
        obs = self._get_observation()
        info = {"agent_pos": self._agent_pos, "victim_pos": self._victim_pos}
        return obs, info


    def load_from_file(self, filepath):
        """
        Load a pre-generated map + agent/victim placement from .npz file.
        Sets internal state so reset() is not used for the evaluation episode.
        """
        data = np.load(filepath)
        self._true_map = data["true_map"].astype(np.uint8)
        # set known map to unknown
        self._known = np.full_like(self._true_map, 0.5, dtype=np.float32)
        # set agent/victim
        self._agent_pos = tuple(map(int, data["agent_pos"]))
        self._agent_heading = 0 if not hasattr(self, "_agent_heading") else self._agent_heading
        self._victim_pos = tuple(map(int, data["victim_pos"]))
        self._t = 0
        self._visited = set()
        self._visited.add(self._agent_pos)
        # perform initial scan and update known (same done in reset)
        scan = self._simulate_scan(self._agent_pos)
        self._update_known_from_scan(scan, self._agent_pos)
        return

    def step(self, action):
        """
        action: 0 forward,1 left,2 right,3 stay
        """
        terminated = False
        truncated = False
        reward = 0.0
        # apply action with slip
        if action == 0:  # forward
            if np.random.rand() > self.p_slip:
                # move in heading direction
                dx, dy = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}[self._agent_heading]
                nx = self._agent_pos[0] + dx
                ny = self._agent_pos[1] + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self._true_map[nx, ny] == 0:
                    self._agent_pos = (nx, ny)
                    self._visited.add(self._agent_pos)
                else:
                    # collision with wall or occupied -> penalize
                    reward += -5.0
            else:
                # slipped: stays in place (no extra reward)
                pass
        elif action == 1:  # left
            self._agent_heading = (self._agent_heading - 1) % 4
        elif action == 2:  # right
            self._agent_heading = (self._agent_heading + 1) % 4
        elif action == 3:  # stay
            pass

        # step penalty
        reward += -0.05
        self._t += 1

        # scan and update known map
        scan = self._simulate_scan(self._agent_pos)
        self._update_known_from_scan(scan, self._agent_pos)

        # detection check
        detected = False
        if self.detection_mode == "los":
            if self._has_line_of_sight(self._agent_pos, self._victim_pos):
                detected = True
        elif self.detection_mode == "proximity":
            if (abs(self._agent_pos[0] - self._victim_pos[0]) + abs(self._agent_pos[1] - self._victim_pos[1])) <= self.detection_range:
                detected = True

        if detected:
            reward += 100.0
            terminated = True

        if self._t >= self.T_max:
            truncated = True

        obs = self._get_observation()
        info = {"agent_pos": self._agent_pos, "victim_pos": self._victim_pos, "detected": detected}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="human"):
        # Optional: implement visualization later
        pass

    def close(self):
        pass
