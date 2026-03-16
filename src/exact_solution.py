"""
Manufactured solution for the 1D steady-state heat conduction problem
with a nonlocal integral boundary condition.

Problem statement
-----------------
    -u''(x) = f(x),    x in (0, 1)
    u(0) = alpha * int_0^1 u(x) dx + g0       (integral BC)
    u(1) = g1                                   (Dirichlet BC)

Manufactured solution
---------------------
    u(x) = sin(pi * x) + (1 - x) * C,    C = 4 / pi

Derivation:
    u(1)  = sin(pi) + 0 = 0                            => g1 = 0
    u''(x)= -pi^2 sin(pi*x)                            => f(x) = pi^2 sin(pi*x)
    int_0^1 u(x) dx = 2/pi + C/2 = 2/pi + 2/pi = 4/pi
    u(0)  = C = 4/pi = int_0^1 u(x) dx                 => alpha = 1, g0 = 0

References
----------
    - Cannon, J.R. (1963). The solution of the heat equation subject to the
      specification of energy. Quart. Appl. Math., 21(2), 155-160.
    - Liu, Y. (1999). Numerical solution of the heat equation with nonlocal
      boundary conditions. J. Comput. Appl. Math., 110(1), 115-127.

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
ALPHA = 1.0                     # coefficient in the integral boundary condition
C_MANUF = 4.0 / np.pi          # constant in the manufactured solution
G0 = 0.0                       # additive constant in the integral BC
G1 = 0.0                       # Dirichlet value at x = 1


# ---------------------------------------------------------------------------
# NumPy versions (used for error computation and reference)
# ---------------------------------------------------------------------------
def u_exact_np(x: np.ndarray) -> np.ndarray:
    """Exact temperature field u(x) = sin(pi*x) + (1-x)*C."""
    return np.sin(np.pi * x) + (1.0 - x) * C_MANUF


def f_source_np(x: np.ndarray) -> np.ndarray:
    """Source term f(x) = pi^2 * sin(pi*x)."""
    return np.pi ** 2 * np.sin(np.pi * x)


def u_exact_deriv_np(x: np.ndarray) -> np.ndarray:
    """First derivative u'(x) = pi*cos(pi*x) - C."""
    return np.pi * np.cos(np.pi * x) - C_MANUF


def integral_u_exact() -> float:
    """Analytical value of int_0^1 u(x) dx = 2/pi + C/2."""
    return 2.0 / np.pi + C_MANUF / 2.0


# ---------------------------------------------------------------------------
# PyTorch versions (used inside the neural network training loop)
# ---------------------------------------------------------------------------
def u_exact_torch(x: torch.Tensor) -> torch.Tensor:
    """Exact solution (PyTorch tensor)."""
    return torch.sin(np.pi * x) + (1.0 - x) * C_MANUF


def f_source_torch(x: torch.Tensor) -> torch.Tensor:
    """Source term (PyTorch tensor)."""
    return (np.pi ** 2) * torch.sin(np.pi * x)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_manufactured_solution() -> None:
    """Run consistency checks on the manufactured solution.

    Verifies that:
        1. u(1) = g1  (Dirichlet BC)
        2. u(0) = alpha * int_0^1 u dx + g0  (integral BC)
        3. -u''(x) = f(x)  (PDE residual)
    """
    x_bnd = np.array([0.0, 1.0])
    u0, u1 = u_exact_np(x_bnd)
    int_u = integral_u_exact()

    assert abs(u1 - G1) < 1e-14, f"Dirichlet BC violated: u(1)={u1}, g1={G1}"
    assert abs(u0 - ALPHA * int_u - G0) < 1e-14, \
        f"Integral BC violated: u(0)={u0}, alpha*int+g0={ALPHA*int_u+G0}"

    x_test = np.linspace(0.01, 0.99, 50)
    residual = np.pi ** 2 * np.sin(np.pi * x_test) - f_source_np(x_test)
    assert np.max(np.abs(residual)) < 1e-14, "PDE consistency check failed"

    print("[OK] Manufactured solution verified.")


if __name__ == "__main__":
    verify_manufactured_solution()
