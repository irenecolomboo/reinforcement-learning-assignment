"""
A2C Training Script for LunarLander-v3

References / code adapted from:
- CleanRL library— structure for the training loop (same infrastructure as train_ppo.py, which was inspired by CleanRL ppo.py).
  https://github.com/vwxyzjn/cleanrl
"""

import argparse
import json
import os
import signal
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networks import ActorCritic
from train_ppo import (RunningMeanStd, collect_rollout_vec, make_vec_env)
from utils import CSVLogger, compute_gae, save_checkpoint, set_seed

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train A2C on LunarLander-v3")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def a2c_update( 
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: dict,
) -> dict: # Single full-batch A2C gradient step (no clipping, no mini-batches)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #stabilize training

    _, log_probs, entropy, values = model.get_action_and_value(obs, actions) # evaluate action during current policy

    policy_loss = -(log_probs * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    entropy_loss = -entropy.mean()

    loss = (
        policy_loss
        + cfg["vf_coef"]  * value_loss
        + cfg["ent_coef"] * entropy_loss
    )

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     entropy.mean().item(),
    }

def train(cfg: dict, seed: int) -> ActorCritic: # Full A2C training loop (vectorized envs + obs norm + LR annealing)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["exp_name"]
    n_envs = cfg.get("n_envs", 8)
    anneal_lr = cfg.get("anneal_lr", True)
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, "results", exp_name, f"seed_{seed}")
    ckpt_dir = os.path.join(project_root, "checkpoints", exp_name, f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir,    exist_ok=True)

    envs = make_vec_env(cfg["env_id"], n_envs, seed)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    hidden_sizes = tuple(cfg["hidden_sizes"])
    model = ActorCritic(obs_dim, act_dim, hidden_sizes).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=cfg["lr"],
                              eps=1e-5, alpha=0.99)

    ep_logger = CSVLogger(
        os.path.join(results_dir, "episode_returns.csv"),
        fieldnames=["global_step", "episode", "ep_return", "ep_length"],
    )
    update_logger = CSVLogger(
        os.path.join(results_dir, "a2c_metrics.csv"),
        fieldnames=["global_step", "policy_loss", "value_loss", "entropy", "lr"],
    )

    cfg_copy = dict(cfg)
    cfg_copy["seed"]      = seed
    cfg_copy["n_envs"]    = n_envs
    cfg_copy["algorithm"] = "a2c"
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(cfg_copy, f, indent=2)

    interrupted = [False]
    def _handle_sigint(sig, frame):
        print("\n[!] Interrupted – saving checkpoint before exit.")
        interrupted[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    global_step = 0
    episode_count = 0
    obs_curr, _ = envs.reset(seed=seed)
    recent_returns: list = []
    start_time = time.time()

    steps_per_update = cfg["n_steps"] * n_envs
    total_updates = cfg["total_timesteps"] // steps_per_update
    lr_init = cfg["lr"]

    print(f"\n{'='*60}")
    print(f"Algorithm: A2C")
    print(f"Experiment: {exp_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device}   n_envs: {n_envs}")
    print(f"Env: {cfg['env_id']}  (obs={obs_dim}, act={act_dim})")
    print(f"Network: {hidden_sizes}")
    print(f"Timesteps: {cfg['total_timesteps']:,}")
    print(f"ent_coef: {cfg['ent_coef']}   lr: {lr_init}  anneal: {anneal_lr}")
    print(f"{'='*60}\n")

    for update in range(1, total_updates + 1):
        if interrupted[0]:
            break

        if anneal_lr: # linear lr decay
            frac   = 1.0 - (update - 1) / total_updates
            lr_now = lr_init * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = lr_init

        # rollout 
        (obs_t, act_t, logp_t, rew_t, done_t, val_t,
         next_value, obs_curr, ep_stats) = collect_rollout_vec(
            envs, model, cfg["n_steps"], n_envs, device, obs_curr, obs_rms
        )
        global_step += steps_per_update

        for (ep_ret, ep_len) in ep_stats:
            episode_count += 1
            ep_logger.log({
                "global_step": global_step,
                "episode":     episode_count,
                "ep_return":   round(ep_ret, 3),
                "ep_length":   ep_len,
            })
            recent_returns.append(ep_ret)

        all_adv, all_ret = [], []
        for env_i in range(n_envs):
            adv_i, ret_i = compute_gae(
                rew_t[:, env_i], val_t[:, env_i], done_t[:, env_i],
                float(next_value[env_i]),
                cfg["gamma"], cfg["gae_lambda"],
            )
            all_adv.append(adv_i)
            all_ret.append(ret_i)

        advantages = torch.stack(all_adv, dim=1).reshape(-1)
        returns    = torch.stack(all_ret, dim=1).reshape(-1)

        # update 
        metrics = a2c_update(
            model, optimizer,
            obs_t.to(device), act_t.to(device),
            advantages.to(device), returns.to(device),
            cfg,
        )

        update_logger.log({
            "global_step": global_step,
            "policy_loss": round(metrics["policy_loss"], 4),
            "value_loss": round(metrics["value_loss"], 4),
            "entropy": round(metrics["entropy"], 4),
            "lr": round(lr_now, 7),
        })

        log_every = max(1, total_updates // 20)
        if update % log_every == 0 or update == total_updates:
            elapsed  = time.time() - start_time
            mean_ret = np.mean(recent_returns[-20:]) if recent_returns else float("nan")
            sps      = global_step / elapsed
            print(
                f"Update {update:>5}/{total_updates}  "
                f"step={global_step:>8,}  "
                f"ret(last20)={mean_ret:>8.1f}  "
                f"ent={metrics['entropy']:.3f}  "
                f"lr={lr_now:.5f}  sps={sps:.0f}"
            )

        ckpt_freq = cfg.get("checkpoint_freq", 100_000)
        if global_step % ckpt_freq == 0 or update == total_updates or interrupted[0]:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "obs_rms": obs_rms.state_dict(),
                    "global_step": global_step,
                    "config": cfg_copy,
                    "seed": seed,
                },
                ckpt_path,
            )
            print(f"  ✓ Checkpoint → {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nDone. Total steps: {global_step:,}  "
          f"Episodes: {episode_count}  "
          f"Time: {total_time:.0f}s\n")
    envs.close()
    return model


if __name__ == "__main__":
    args = parse_args()
    cfg  = dict(
        exp_name="a2c_run", env_id="LunarLander-v3",
        total_timesteps=3_000_000, n_envs=8, n_steps=20,
        batch_size=160, n_epochs=1, lr=7e-4, gamma=0.99,
        gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, anneal_lr=True, hidden_sizes=[256, 256],
        checkpoint_freq=100_000,
    )
    train(cfg, args.seed)
