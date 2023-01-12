import torch
from torch import optim
from torch import nn
from pathlib import Path
from typing import Tuple


class NetworkCartPole(nn.Module):
    """Network used in the CartPole task"""

    def __init__(self, input_dim: Tuple, output_dim: int, lr: float, device: torch.device) -> None:
        """
        Creates the network, the optimized, the loss function and the attributes

        Args:
            input_dim: input dimension of the network
            output_dim: output dimension of the network
            lr: learning rate
            device: the device in which runs the code. IT WORKS ONLY IN CPU
        """
        super(NetworkCartPole, self).__init__()

        self.in_dim = input_dim[0]
        self.out_dim = output_dim

        # DONE creates the network
        self.net = torch.nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.out_dim),
        )

        # DONE init the loss function
        self.loss_fn = nn.MSELoss()

        # DONE init the optimizer
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, state: torch.tensor) -> torch.tensor:
        """
        The inference of the network.

        Returns:
            prediction of the network.
        """
        self.optim.zero_grad()
        return self.net(state)

    def learn(self, y: torch.tensor, y_hat: torch.tensor) -> None:
        """
        In this method, the network compute the loss and the backpropagation

        Args:
            y: prediction
            y_hat: target
        """
        loss = self.loss_fn(y, y_hat)
        loss.backward()
        self.optim.step()
    
    def save(self, path: Path) -> None:
        """Saves the network"""
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        """Loads the network"""
        self.load_state_dict(torch.load(path))
