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
