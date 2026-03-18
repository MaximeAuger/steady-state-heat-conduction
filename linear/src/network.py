"""
Fully connected neural network (Multi-Layer Perceptron) for PINN and VPINN.

The architecture follows the standard design used in physics-informed learning:
a feed-forward network with tanh activation and Xavier initialisation, mapping
a single spatial coordinate x to a scalar approximation u_theta(x).

References
----------
    - Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
      neural networks: A deep learning framework for solving forward and inverse
      problems involving nonlinear partial differential equations. J. Comput.
      Phys., 378, 686-707.

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with tanh activations.

    Architecture
    ------------
        x (R^1) -> [Linear -> Tanh] x n_layers -> Linear -> u_theta (R^1)

    Parameters
    ----------
    n_input : int
        Input dimension (1 for a 1-D spatial coordinate).
    n_output : int
        Output dimension (1 for a scalar field).
    n_hidden : int
        Number of neurons per hidden layer.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        n_input: int = 1,
        n_output: int = 1,
        n_hidden: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, n_output))

        self.net = nn.Sequential(*layers)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Xavier / Glorot normal initialisation for all linear layers."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (N, 1)
            Spatial coordinates.

        Returns
        -------
        Tensor of shape (N, 1)
            Network prediction u_theta(x).
        """
        return self.net(x)
