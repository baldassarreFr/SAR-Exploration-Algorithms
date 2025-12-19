# src/train_dqn.py
import argparse
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from src.config_loader import load_config
from src.env import SARGridEnv

def make_env(cfg, seed=None):
    def _init():
        env = SARGridEnv(cfg)
        if seed is not None:
            np.random.seed(seed)
        return env
    return _init

def train(cfg_path, quick=False, device=None):
    cfg = load_config(cfg_path)
    total_steps = int(cfg.get("training", {}).get("total_timesteps", 200000))
    if quick:
        total_steps = min(total_steps, 20000)

    tb_log = cfg.get("logging", {}).get("tensorboard_log", "results/logs/dqn_tb")
    os.makedirs(tb_log, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    env = DummyVecEnv([make_env(cfg)])
    policy = cfg.get("policy", "CnnPolicy")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dqn_cfg = cfg.get("dqn", {})
    lr = float(dqn_cfg.get("learning_rate", 1e-4))
    buffer_size = int(dqn_cfg.get("buffer_size", 100000))
    batch_size = int(dqn_cfg.get("batch_size", 32))
    target_update_interval = int(dqn_cfg.get("target_update_interval", 1000))
    gamma = float(dqn_cfg.get("gamma", 0.99))

    model = DQN(policy,
                env,
                verbose=1,
                tensorboard_log=tb_log,
                device=device,
                learning_rate=lr,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update_interval=target_update_interval,
                gamma=gamma)

    ckpt_cb = CheckpointCallback(save_freq=cfg.get("training", {}).get("checkpoint_interval_steps", 10000),
                                 save_path="checkpoints/",
                                 name_prefix="dqn_ckpt")

    model.learn(total_timesteps=total_steps, callback=ckpt_cb)
    final_path = "checkpoints/dqn_final"
    model.save(final_path)
    env.close()
    print("Saved final DQN model to:", final_path)
    return final_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_small.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.config, quick=args.quick, device=args.device)
