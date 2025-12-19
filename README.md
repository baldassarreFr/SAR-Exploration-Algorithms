# SAR exploration experiments

## Setup
Conda env: conda env create -f env.yml
Activate: conda activate sar_experiments

## Config
All environment and experiment parameters live in `configs/`:
- `configs/env.yaml` - shared env properties (grid, sensor, maps, seeds)
- `configs/frontier.yaml` - frontier baseline params
- `configs/ppo_small.yaml` - PPO training params
- `configs/dqn_small.yaml` - DQN training params
- `configs/eval.yaml` - evaluation params

## Run
- Baseline: `python src/frontier_agent.py --config configs/frontier.yaml`
- Train PPO: `python src/train_ppo.py --config configs/ppo_small.yaml`
- Train DQN: `python src/train_dqn.py --config configs/dqn_small.yaml`
- Eval: `python src/eval.py --config configs/eval.yaml`



# Complete project structure:
project/
├─ configs/
│  ├─ env.yaml               # core environment parameters (shared)
│  ├─ frontier.yaml          # frontier-specific params
│  ├─ ppo_small.yaml         # PPO training config
│  ├─ dqn_small.yaml         # DQN training config
│  └─ eval.yaml              # evaluation settings
|
├─ src/
│  ├─ __init__.py
│  ├─ config_loader.py       # helper to load YAML configs
│  ├─ env.py                 # Gym-style env skeleton (reads configs/env.yaml)
│  ├─ frontier_agent.py      # frontier baseline runner
│  ├─ train_ppo.py           # PPO training runner (SB3)
│  ├─ train_dqn.py           # DQN training runner (SB3)
│  └─ eval.py                # evaluation harness (runs policies + baseline, logs)
|
├─ results/
│  ├─ logs/
│  └─ figures/
|
├─ Report.pdf
|
├─ README.md
|
├─ env.yml                   # conda environment file