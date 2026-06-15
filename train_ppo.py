"""
PPO Training Script for LunarLander-v3

References / code adapted from:
- "Proximal Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347
- "The 37 Implementation Details of PPO" https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- from CleanRL structure for the training loop https://github.com/vwxyzjn/cleanrl
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
from utils import CSVLogger, compute_gae, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--render", action="store_true",
                   help="Render the environment visually during training (slow)")
    return p.parse_args()


def make_env(env_id: str, seed: int, render: bool = False) -> gym.Env: #Create a single environment instance with episode-statistics recording
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_vec_env(env_id: str, n_envs: int, seed: int) -> gym.vector.VectorEnv: #Create n_envs parallel environments via SyncVectorEnv.
    def _make(i):
        def _init():
            e = gym.make(env_id)
            e.reset(seed=seed + i)        # unique seed per env
            e.action_space.seed(seed + i)
            return e
        return _init
    return gym.vector.SyncVectorEnv([_make(i) for i in range(n_envs)])



class RunningMeanStd: #Tracks the running mean and variance of a stream of vectors, to normalize observations online. Used in collect_rollout_vec to normalise obs before feeding to the network, and to update statistics from raw obs.
    # Welford (1962) online algorithm – numerically stable.
    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon          # avoid division by zero at startup

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of observations (shape: [N, *shape])."""
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta       = batch_mean - self.mean
        new_mean    = self.mean + delta * (batch_count / total_count)
        m_a         = self.var   * self.count
        m_b         = batch_var  * batch_count
        new_var     = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count

        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def collect_rollout_vec(
    envs: gym.vector.VectorEnv,
    model: "ActorCritic",
    n_steps: int,
    n_envs: int,
    device: torch.device,
    obs_curr: np.ndarray,
    obs_rms: "RunningMeanStd",
) -> tuple: # Rollout collection (vectorized)
    # Collect n_steps x n_envs transitions using n_envs parallel environments.
    obs_dim = envs.single_observation_space.shape[0]

    obs_buf  = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    act_buf  = np.zeros((n_steps, n_envs), dtype=np.int64)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    rew_buf  = np.zeros((n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf  = np.zeros((n_steps, n_envs), dtype=np.float32)

    ep_stats: list = []

    # Manual per-env episode trackers
    ep_ret_running = np.zeros(n_envs, dtype=np.float64)
    ep_len_running = np.zeros(n_envs, dtype=np.int32)

    for t in range(n_steps):
        # Feed raw obs into RunningMeanStd so the statistics keep improving, then normalise
        obs_rms.update(obs_curr)
        obs_norm = obs_rms.normalise(obs_curr).astype(np.float32)

        obs_t = torch.FloatTensor(obs_norm).to(device)
        with torch.no_grad():    # no gradients needed during rollout collection
            actions, log_probs, _, values = model.get_action_and_value(obs_t)

        # Step all n_envs environments simultaneously.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions.cpu().numpy()
        )
        dones = terminations | truncations   # either reason ends the episode

        obs_buf[t]  = obs_norm
        act_buf[t]  = actions.cpu().numpy()
        logp_buf[t] = log_probs.cpu().numpy()
        rew_buf[t]  = rewards
        done_buf[t] = dones.astype(np.float32)
        val_buf[t]  = values.cpu().numpy()

        # Track episode returns manually
        ep_ret_running += rewards
        ep_len_running += 1
        for env_i in range(n_envs):
            if dones[env_i]:
                ep_stats.append((float(ep_ret_running[env_i]),
                                  int(ep_len_running[env_i])))
                ep_ret_running[env_i] = 0.0
                ep_len_running[env_i] = 0

        obs_curr = next_obs

    # Bootstrap: estimate V(s_{T+1}) for the state we stopped at
    obs_rms.update(obs_curr)
    obs_norm_last = obs_rms.normalise(obs_curr).astype(np.float32)
    obs_last = torch.FloatTensor(obs_norm_last).to(device)
    with torch.no_grad():
        _, next_values = model(obs_last)
    next_value = next_values.cpu().squeeze(-1).numpy() 

    T = n_steps * n_envs
    return (
        torch.FloatTensor(obs_buf.reshape(T, obs_dim)),
        torch.LongTensor(act_buf.reshape(T)),
        torch.FloatTensor(logp_buf.reshape(T)),
        torch.FloatTensor(rew_buf),      
        torch.FloatTensor(done_buf),
        torch.FloatTensor(val_buf),
        next_value,
        obs_curr,
        ep_stats,
    )


# Rollout collection ( for single env, kept for backward compat / evaluate)
def collect_rollout(
    env: gym.Env,
    model: "ActorCritic",
    n_steps: int,
    device: torch.device,
    obs_curr: np.ndarray,
) -> tuple:
    obs_dim = env.observation_space.shape[0]

    obs_buf  = np.zeros((n_steps, obs_dim), dtype=np.float32)
    act_buf  = np.zeros(n_steps, dtype=np.int64)
    logp_buf = np.zeros(n_steps, dtype=np.float32)
    rew_buf  = np.zeros(n_steps, dtype=np.float32)
    done_buf = np.zeros(n_steps, dtype=np.float32)
    val_buf  = np.zeros(n_steps, dtype=np.float32)

    ep_stats: list = []
    ep_ret_running = 0.0
    ep_len_running = 0

    for t in range(n_steps):
        obs_t = torch.FloatTensor(obs_curr).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs_t)

        action_int = action.item()
        next_obs, reward, terminated, truncated, _ = env.step(action_int)
        done = terminated or truncated

        obs_buf[t]  = obs_curr
        act_buf[t]  = action_int
        logp_buf[t] = log_prob.item()
        rew_buf[t]  = float(reward)
        done_buf[t] = float(done)
        val_buf[t]  = value.item()
        ep_ret_running += float(reward)
        ep_len_running += 1

        if done:
            ep_stats.append((ep_ret_running, ep_len_running))
            ep_ret_running = 0.0
            ep_len_running = 0
            obs_curr, _ = env.reset()
        else:
            obs_curr = next_obs

    obs_last = torch.FloatTensor(obs_curr).unsqueeze(0).to(device)
    with torch.no_grad():
        _, next_value = model(obs_last)
    next_value = next_value.item()

    return (
        torch.FloatTensor(obs_buf),
        torch.LongTensor(act_buf),
        torch.FloatTensor(logp_buf),
        torch.FloatTensor(rew_buf),
        torch.FloatTensor(done_buf),
        torch.FloatTensor(val_buf),
        next_value,
        obs_curr,
        ep_stats,
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: dict,
) -> dict: # Run n_epochs of mini-batch PPO updates on the collected rollout
 
    T = len(obs)
    indices = np.arange(T)

    # Normalise advantages: reduces variance of gradient estimates
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(cfg["n_epochs"]):
        np.random.shuffle(indices)

        for start in range(0, T, cfg["batch_size"]):
            mb_idx = indices[start : start + cfg["batch_size"]]

            mb_obs     = obs[mb_idx]
            mb_act     = actions[mb_idx]
            mb_old_lp  = old_log_probs[mb_idx]
            mb_adv     = advantages[mb_idx]
            mb_ret     = returns[mb_idx]

            # Evaluate the current policy
            _, log_prob, entropy, value = model.get_action_and_value(mb_obs, mb_act)

            # clipped policy loss 
            ratio = torch.exp(log_prob - mb_old_lp)

            # surr1: unclipped 
            surr1 = ratio * mb_adv

            # surr2: clipped
            surr2 = torch.clamp(ratio, 1.0 - cfg["clip_eps"],
                                        1.0 + cfg["clip_eps"]) * mb_adv

            # Take the minimum: always use the more conservative of the two
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(value, mb_ret)
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

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.mean().item())
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train(cfg: dict, seed: int, render: bool = False) -> "ActorCritic":
    # Full PPO training loop with vectorized environments, observation normalisation, and linear LR annealing

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name     = cfg["exp_name"]
    n_envs       = cfg.get("n_envs", 8)
    anneal_lr    = cfg.get("anneal_lr", True)
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir  = os.path.join(project_root, "results",     exp_name, f"seed_{seed}")
    ckpt_dir     = os.path.join(project_root, "checkpoints", exp_name, f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir,    exist_ok=True)

    envs    = make_vec_env(cfg["env_id"], n_envs, seed)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n 
    obs_rms = RunningMeanStd(shape=(obs_dim,)) # normalize

    hidden_sizes = tuple(cfg["hidden_sizes"])
    model     = ActorCritic(obs_dim, act_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)

    ep_logger = CSVLogger(
        os.path.join(results_dir, "episode_returns.csv"),
        fieldnames=["global_step", "episode", "ep_return", "ep_length"],
    )
    update_logger = CSVLogger(
        os.path.join(results_dir, "ppo_metrics.csv"),
        fieldnames=["global_step", "policy_loss", "value_loss",
                    "entropy", "approx_kl", "lr"],
    )

    cfg_copy = dict(cfg)
    cfg_copy["seed"]   = seed
    cfg_copy["n_envs"] = n_envs
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(cfg_copy, f, indent=2)

    interrupted = [False]
    def _handle_sigint(sig, frame):
        print("\n[!] Interrupted – saving checkpoint before exit.")
        interrupted[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)


    global_step   = 0
    episode_count = 0
    obs_curr, _   = envs.reset(seed=seed)
    recent_returns: list = []
    start_time    = time.time()

    steps_per_update = cfg["n_steps"] * n_envs
    total_updates    = cfg["total_timesteps"] // steps_per_update
    lr_init          = cfg["lr"]

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}   n_envs: {n_envs}")
    print(f"  Env: {cfg['env_id']}  (obs={obs_dim}, act={act_dim})")
    print(f"  Network: {hidden_sizes}")
    print(f"  Timesteps: {cfg['total_timesteps']:,}")
    print(f"  ent_coef: {cfg['ent_coef']}   lr: {lr_init}  anneal: {anneal_lr}")
    print(f"  obs_norm: True   n_envs: {n_envs}")
    print(f"{'='*60}\n")

    for update in range(1, total_updates + 1):
        if interrupted[0]:
            break
        #  LR linear decay to 0
        if anneal_lr:
            frac = 1.0 - (update - 1) / total_updates
            lr_now = lr_init * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = lr_init

        #  Collect rollout (vectorized)
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

        #  Compute advantages (GAE) per-env, then flatten
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

        # PPO update
        metrics = ppo_update(
            model, optimizer,
            obs_t.to(device), act_t.to(device), logp_t.to(device),
            advantages.to(device), returns.to(device),
            cfg,
        )

        update_logger.log({
            "global_step":  global_step,
            "policy_loss":  round(metrics["policy_loss"], 4),
            "value_loss":   round(metrics["value_loss"],  4),
            "entropy":      round(metrics["entropy"],     4),
            "approx_kl":    round(metrics["approx_kl"],  6),
            "lr":           round(lr_now, 7),
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
                f"kl={metrics['approx_kl']:.5f}  "
                f"lr={lr_now:.5f}  sps={sps:.0f}"
            )

        ckpt_freq = cfg.get("checkpoint_freq", 100_000)
        if global_step % ckpt_freq == 0 or update == total_updates or interrupted[0]:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
            save_checkpoint(
                {
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "obs_rms":         obs_rms.state_dict(),
                    "global_step":     global_step,
                    "config":          cfg_copy,
                    "seed":            seed,
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
        exp_name="ppo_run", env_id="LunarLander-v3",
        total_timesteps=3_000_000, n_envs=8, n_steps=512,
        batch_size=256, n_epochs=4, lr=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, anneal_lr=True, hidden_sizes=[256, 256],
        checkpoint_freq=100_000,
    )
    train(cfg, args.seed, render=args.render)
