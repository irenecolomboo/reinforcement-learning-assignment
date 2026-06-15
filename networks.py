"""
Actor-Critic Neural Network for PPO

Architecture: shared MLP trunk , separate policy head + value head

References
- from CleanRL Agent class in ppo.py (shared trunk, orthogonal init, gain=0.01 for policy head): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- Stable-Baselines3 ActorCriticPolicy (shared MLP trunk convention): https://github.com/DLR-RM/stable-baselines3
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module): # Shared trunk Actor-Critic network for discrete-action PPO.

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple = (64, 64)):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # Policy head 
        self.policy_head = nn.Linear(in_dim, act_dim)

        # Value head 
        self.value_head = nn.Linear(in_dim, 1)

        # Apply orthogonal initialisation to all layers
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias) 
        # Policy head: small gain
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

        # Value head: default gain (1.0) is fine
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor): # forward pass returns (logits, value)
        features = self.shared(obs)     
        logits = self.policy_head(features) 


        value = self.value_head(features).squeeze(-1) 
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        ):
        
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()   # draw one action per env during rollout

        log_prob = dist.log_prob(action) #rollout: old policy; update: new
        entropy = dist.entropy()

        return action, log_prob, entropy, value
