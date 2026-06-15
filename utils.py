# Shared utilities for the PPO project
import csv
import os
import random
from typing import Tuple
import numpy as np
import torch

def set_seed(seed: int) -> None: # reproducibility helper
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filepath: str) -> None: #persist model + optimizer state to disk
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module,  
                    optimizer: torch.optim.Optimizer = None) -> Tuple[int, dict]: # restore from a saved checkpoint
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("global_step", 0), checkpoint.get("config", {})


class CSVLogger: # lightweight CSV-based metric logger
    def __init__(self, filepath: str, fieldnames: list) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.filepath = filepath
        self.fieldnames = fieldnames
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: dict) -> None:
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]: # Generalised Advantage Estimation
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    # Traverse backwards through the rollout to accumulate GAE
    for t in reversed(range(T)):
        # last timestep, use externally bootstrapped value;
        #  all others, use the value at the next step in the rollout
        next_val = next_value if t == T - 1 else values[t + 1].item()
        mask = 1.0 - dones[t].item()           # 0 at terminal, 1 otherwise

        # One-step TD error
        delta = rewards[t].item() + gamma * next_val * mask - values[t].item()
        # Accumulate GAE running backwards
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
