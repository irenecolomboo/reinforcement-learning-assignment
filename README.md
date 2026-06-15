# Deep Reinforcement Learning — Final Assignment
**Environment:** LunarLander-v3 &nbsp;|&nbsp; **Algorithms:** A2C, PPO &nbsp;|&nbsp; **Framework:** PyTorch

## Setup

```bash
pip install -r requirements.txt
```


## File Overview

| File | Description |
|---|---|
| `Final_assignment.ipynb` | Main notebook: training runs, all plots, and analysis |
| `networks.py` | Shared Actor-Critic network (shared MLP trunk, orthogonal init) |
| `utils.py` | `set_seed`, `compute_gae`, `CSVLogger`, checkpoint save/load |
| `train_ppo.py` | PPO training script (vectorised envs, obs normalisation, LR annealing) |
| `train_a2c.py` | A2C training script (shares rollout infrastructure with PPO) |
| `train_partial_obs.py` | Partial observability ablations (masks sensor groups via a Gym wrapper) |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Saved model checkpoints (`.pt` files, one per 100k steps per seed) |
| `results/` | CSV logs: `episode_returns.csv`, `ppo_metrics.csv` / `a2c_metrics.csv`, `config.json` |



## Reproducing the Results

All results are already pre-computed. To view the analysis, open `Final_assignment.ipynb`.

To retrain from scratch:

```bash
# PPO and A2C experiments (baseline, entropy sweep, network size, learning rate)
python train_ppo.py --seed 42

# Partial observability and sensor ablations
python train_partial_obs.py --configs lg_partial_obs lg_no_position lg_no_angle lg_no_legs lg_minimal_obs --seeds 42 123 456
```

All hyperparameters are defined as plain dicts in the notebook (cell 2) and saved to `results/<exp_name>/seed_<N>/config.json` at the start of each run. Seeds used: **42, 123, 456**.



## Video

`Final Assignment.mp4` — screen recording walkthrough of the code and results (≤ 10 min).
