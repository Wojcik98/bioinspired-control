import numpy as np
import torch
from typing import Tuple


class LearnDQN:
    """Takes care of the estimation of the learning function."""
    def __init__(self, discount_factor: float) -> None:
        """
        Sets the attributes.

        Args:
            discount_factor: the discount factor in the Bellman equation.
        """
        self.discount_factor = discount_factor
    
    def __call__(self, net: torch.nn.Module,
        states: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        states_: torch.tensor,
        dones: torch.tensor
        ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the learning function.

        Args:
            net: network
            states: batch of states
            actions: batch of actions
            rewards: batch of rewards
            states_: batch of the future states
            dones: batch of the ended episodes
        
        Returns:
            q: predicted q value
            q_hat: q target value
        """
        # DONE compute q target and q values
        actions = actions[0].numpy()
        qs = net.forward(states)[np.arange(len(actions)), actions]
        future_qs = torch.max(net.forward(states_), dim=1)

        q_targets = future_qs*self.discount_factor + rewards

        return qs, q_targets
