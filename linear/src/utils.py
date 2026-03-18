"""
Utility functions: Gauss-Legendre quadrature, Legendre test functions for
VPINN, automatic differentiation helpers, error metrics, and a finite
difference reference solver.

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch
from numpy.polynomial import legendre as leg

DTYPE = torch.float64
DEVICE = "cpu"


# ===================================================================
# Gauss-Legendre quadrature
# ===================================================================
def gauss_legendre(n: int, a: float = 0.0, b: float = 1.0):
    """Gauss-Legendre nodes and weights on [a, b].

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a, b : float
        Integration bounds.

    Returns
    -------
    x : ndarray of shape (n,)
        Quadrature nodes.
    w : ndarray of shape (n,)
        Quadrature weights.
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def gauss_legendre_torch(n: int, a: float = 0.0, b: float = 1.0):
    """Gauss-Legendre nodes and weights as PyTorch tensors of shape (n, 1)."""
    x, w = gauss_legendre(n, a, b)
    return (
        torch.tensor(x, dtype=DTYPE).reshape(-1, 1),
        torch.tensor(w, dtype=DTYPE).reshape(-1, 1),
    )


# ===================================================================
# Legendre test functions for VPINN
# ===================================================================
class LegendreTestFunctions:
    """Global test functions based on shifted Legendre polynomials.

    Each test function is defined as

        v_k(x) = P_k(2x - 1) - 1,    k = 1, 2, ..., n_test

    where P_k is the k-th Legendre polynomial.  This construction ensures
    that v_k(1) = P_k(1) - 1 = 0 for all k (since P_k(1) = 1).

    The derivative is obtained via the chain rule:

        v_k'(x) = 2 * P_k'(2x - 1)

    Values at the left boundary:

        v_k(0) = P_k(-1) - 1 = (-1)^k - 1

    These functions form a hierarchical polynomial basis on [0, 1] that
    naturally satisfies the homogeneous Dirichlet condition at x = 1.

    Parameters
    ----------
    n_test : int
        Number of test functions.

    References
    ----------
        - Kharazmi, E., Zhang, Z. & Karniadakis, G.E. (2021). hp-VPINNs:
          variational physics-informed neural networks with domain
          decomposition. CMAME, 374, 113547.
    """

    def __init__(self, n_test: int):
        self.n_test = n_test

    def eval_v(self, x01: np.ndarray) -> np.ndarray:
        """Evaluate all test functions at points in [0, 1].

        Returns
        -------
        V : ndarray of shape (N, n_test)
        """
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        V = np.zeros((N, self.n_test))
        for k in range(self.n_test):
            coeffs = np.zeros(k + 2)
            coeffs[-1] = 1.0
            V[:, k] = leg.legval(xi, coeffs) - 1.0
        return V

    def eval_dv(self, x01: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of all test functions.

        Returns
        -------
        dV : ndarray of shape (N, n_test)
        """
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        dV = np.zeros((N, self.n_test))
        for k in range(self.n_test):
            coeffs = np.zeros(k + 2)
            coeffs[-1] = 1.0
            dcoeffs = leg.legder(coeffs)
            dV[:, k] = 2.0 * leg.legval(xi, dcoeffs)
        return dV

    def v_at_zero(self) -> np.ndarray:
        """Values v_k(0) = (-1)^k - 1 for k = 1, ..., n_test."""
        return np.array([(-1.0) ** (k + 1) - 1.0 for k in range(self.n_test)])


# ===================================================================
# Automatic differentiation helper
# ===================================================================
def diff(u: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute d^n u / dx^n using PyTorch automatic differentiation.

    Parameters
    ----------
    u : Tensor of shape (N, 1)
        Function values (network output).
    x : Tensor of shape (N, 1)
        Input coordinates (must have requires_grad=True).
    order : int
        Differentiation order (1 or 2).

    Returns
    -------
    Tensor of shape (N, 1)
        The n-th order derivative.
    """
    d = u
    for _ in range(order):
        d = torch.autograd.grad(
            d, x, grad_outputs=torch.ones_like(d),
            create_graph=True, retain_graph=True,
        )[0]
    return d


# ===================================================================
# Error metrics
# ===================================================================
def compute_errors(net, u_exact_fn, u_deriv_fn=None, n_eval: int = 1000):
    """Compute L2, L-infinity, and (optionally) H1 errors on a fine grid.

    Parameters
    ----------
    net : nn.Module
        Trained neural network.
    u_exact_fn : callable
        Exact solution (NumPy signature).
    u_deriv_fn : callable or None
        Exact first derivative.  If provided, H1 errors are computed.
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    dict
        Keys: 'x', 'u_exact', 'u_pred', 'error', 'L2', 'Linf',
              and optionally 'H1_semi', 'H1'.
    """
    x_np = np.linspace(0.0, 1.0, n_eval)
    u_ex = u_exact_fn(x_np)
    dx = 1.0 / (n_eval - 1)

    x_t = torch.tensor(x_np, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        u_pred = net(x_t).cpu().numpy().flatten()

    err = u_pred - u_ex
    l2 = np.sqrt(np.trapz(err ** 2, dx=dx))
    linf = np.max(np.abs(err))

    result = {
        "x": x_np, "u_exact": u_ex, "u_pred": u_pred,
        "error": err, "L2": l2, "Linf": linf,
    }

    if u_deriv_fn is not None:
        x_t2 = torch.tensor(x_np, dtype=DTYPE).reshape(-1, 1)
        x_t2.requires_grad_(True)
        u_p = net(x_t2)
        du_p = diff(u_p, x_t2, 1).detach().cpu().numpy().flatten()
        du_ex = u_deriv_fn(x_np)
        err_d = du_p - du_ex
        h1_semi = np.sqrt(np.trapz(err_d ** 2, dx=dx))
        result["H1_semi"] = h1_semi
        result["H1"] = np.sqrt(l2 ** 2 + h1_semi ** 2)

    return result


# ===================================================================
# Finite difference reference solver
# ===================================================================
def solve_fdm(f_fn, alpha: float, g0: float, g1: float, N: int = 1000):
    """Solve the BVP by second-order centred finite differences.

    The integral boundary condition is discretised using the trapezoidal
    rule and incorporated as the first row of the linear system.

    Parameters
    ----------
    f_fn : callable
        Source term f(x).
    alpha : float
        Coefficient in the integral BC.
    g0 : float
        Additive constant in the integral BC.
    g1 : float
        Dirichlet value at x = 1.
    N : int
        Number of sub-intervals (N+1 grid points).

    Returns
    -------
    x : ndarray of shape (N+1,)
    u : ndarray of shape (N+1,)
    """
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)

    # Unknowns: u_0, u_1, ..., u_{N-1}  (u_N = g1 is known)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Row 0 — integral boundary condition (trapezoidal rule)
    A[0, 0] = 1.0 - alpha * h / 2.0
    for j in range(1, N):
        A[0, j] = -alpha * h
    b[0] = alpha * h / 2.0 * g1 + g0

    # Rows 1 to N-1 — interior second-order centred differences
    for i in range(1, N):
        A[i, i - 1] = -1.0
        A[i, i] = 2.0
        if i + 1 < N:
            A[i, i + 1] = -1.0
        else:
            b[i] += g1
        b[i] += h ** 2 * f_fn(x[i])

    u_int = np.linalg.solve(A, b)
    u = np.zeros(N + 1)
    u[:N] = u_int
    u[N] = g1
    return x, u
