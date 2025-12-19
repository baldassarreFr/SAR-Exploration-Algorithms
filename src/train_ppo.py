from stable_baselines3 import PPO
from src.env import SARGridEnv
from src.config_loader import load_config
import argparse
import os

def train(cfg_path, total_steps=50_000):
    cfg = load_config(cfg_path)
    env = SARGridEnv(cfg)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64
    )

    model.learn(total_timesteps=total_steps)

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ppo_simple")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_small.yaml")
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()

    train(args.config, args.steps)
