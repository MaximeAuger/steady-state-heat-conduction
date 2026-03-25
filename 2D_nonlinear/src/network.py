"""
Architecture MLP pour PINN et VPINN (2D).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_input=2, n_output=1, n_hidden=64, n_layers=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
