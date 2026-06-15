# PPO training with partial observability ablations

import argparse
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import train_ppo

# Each entry: exp_name - visible dimension indices (out of the total of 8)
OBS_CONFIGS = {
    "lg_partial_obs": {
        "visible":     np.array([0, 1, 4, 6, 7]),       # mask vx, vy, ang_vel
        "description": "No velocities (5/8): masks vx(2), vy(3), angular_vel(5)",
    },
    "lg_no_position": {
        "visible":     np.array([2, 3, 4, 5, 6, 7]),    # mask x, y
        "description": "No position (6/8): masks x(0), y(1)",
    },
    "lg_no_angle": {
        "visible":     np.array([0, 1, 2, 3, 6, 7]),    # mask angle, ang_vel
        "description": "No angle info (6/8): masks angle(4), angular_vel(5)",
    },
    "lg_no_legs": {
        "visible":     np.array([0, 1, 2, 3, 4, 5]),    # mask leg contacts
        "description": "No leg contacts (6/8): masks left_leg(6), right_leg(7)",
    },
    "lg_minimal_obs": {
        "visible":     np.array([0, 1, 4]),              # mask vx, vy, ang_vel, legs
        "description": "Minimal (3/8): only x, y, angle visible",
    },
}

# PPO hyperparameters (like lg_baseline)
BASE_CFG = dict(
    env_id="LunarLander-v3",
    total_timesteps=3_000_000,
    n_envs=8, n_steps=512, batch_size=256, n_epochs=4,
    lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    max_grad_norm=0.5, anneal_lr=True,
    hidden_sizes=[256, 256], checkpoint_freq=100_000,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

#  Wrapper 
class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, visible_dims: np.ndarray):
        super().__init__(env)
        self._dims = visible_dims
        low  = env.observation_space.low[visible_dims]
        high = env.observation_space.high[visible_dims]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self._dims]


def _make_patched_vec_env(visible_dims: np.ndarray): # Return a make_vec_env replacement that injects PartialObsWrapper
    def _patched(env_id, n_envs, seed):
        def _make(i):
            def _init():
                e = gym.make(env_id)
                e = PartialObsWrapper(e, visible_dims)
                e.reset(seed=seed + i)
                e.action_space.seed(seed + i)
                return e
            return _init
        return gym.vector.SyncVectorEnv([_make(i) for i in range(n_envs)])
    return _patched


_orig_make_vec_env = train_ppo.make_vec_env


def already_done(exp_name: str, seed: int) -> bool:
    path = os.path.join(RESULTS_DIR, exp_name, f"seed_{seed}", "episode_returns.csv")
    return os.path.exists(path)


def train_config(exp_name: str, seeds: list[int]) -> None:
    cfg = OBS_CONFIGS[exp_name]
    print(f"\n{'='*60}")
    print(f"Config: {exp_name}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    train_ppo.make_vec_env = _make_patched_vec_env(cfg["visible"])
    try:
        for seed in seeds:
            if already_done(exp_name, seed):
                print(f"  seed {seed}: already complete, skipping.")
                continue
            print(f"  Training seed {seed}…")
            train_ppo.train({**BASE_CFG, "exp_name": exp_name}, seed)
    finally:
        train_ppo.make_vec_env = _orig_make_vec_env  # restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="+", choices=list(OBS_CONFIGS), default=list(OBS_CONFIGS),
        help="Which observation configs to train (default: all)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    for exp_name in args.configs:
        train_config(exp_name, args.seeds)

    print("\nAll done. Results saved to results/<exp_name>/seed_*/")


if __name__ == "__main__":
    main()
