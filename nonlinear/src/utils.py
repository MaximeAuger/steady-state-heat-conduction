"""
Utility functions for the nonlinear PINN / VPINN benchmark.

Contents
--------
- Gauss-Legendre quadrature
- Legendre polynomial test functions for VPINN
- Automatic differentiation helper
- Error metrics
- Linear FDM solver  (for initialisation and beta=0 reference)
- Nonlinear FDM solver with Newton-Raphson iteration

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch
from numpy.polynomial import legendre as leg

from exact_solution import (
    ALPHA, G0, G1, DTYPE,
    u_exact, u_exact_deriv,
    k_conductivity, dk_conductivity,
)


# ===================================================================
# 1. Gauss-Legendre quadrature on [a, b]
# ===================================================================

def gauss_legendre(n, a=0.0, b=1.0):
    """Return nodes and weights of an *n*-point Gauss-Legendre rule on [a, b]."""
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


# ===================================================================
# 2. Legendre test functions for VPINN
# ===================================================================

class LegendreTestFunctions:
    """Shifted Legendre test functions v_k(x) = P_k(2x-1) - 1.

    These satisfy v_k(1) = 0, which enforces the homogeneous Dirichlet
    condition at the right boundary.

    Parameters
    ----------
    n_test : int
        Number of test functions (k = 1, ..., n_test).
    """

    def __init__(self, n_test: int):
        self.n_test = n_test

    def eval_v(self, x01):
        """Evaluate all test functions at points in [0, 1]."""
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        V = np.zeros((N, self.n_test))
        for k in range(1, self.n_test + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            V[:, k - 1] = leg.legval(xi, c) - 1.0
        return V

    def eval_dv(self, x01):
        """Evaluate derivatives: dv_k/dx = 2 * P_k'(2x - 1)."""
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        dV = np.zeros((N, self.n_test))
        for k in range(1, self.n_test + 1):
            c = np.zeros(k + 1)
            c[k] = 1.0
            dc = leg.legder(c)
            dV[:, k - 1] = 2.0 * leg.legval(xi, dc)
        return dV

    def v_at_zero(self):
        """Return v_k(0) = P_k(-1) - 1 = (-1)^k - 1."""
        return np.array([(-1.0) ** k - 1.0 for k in range(1, self.n_test + 1)])


# ===================================================================
# 3. Automatic differentiation helper
# ===================================================================

def diff(u, x, order=1):
    """Compute du/dx (or d^2u/dx^2) via automatic differentiation.

    Parameters
    ----------
    u : Tensor, shape (N, 1)
    x : Tensor, shape (N, 1), requires_grad=True
    order : int  (1 or 2)

    Returns
    -------
    Tensor, shape (N, 1)
    """
    du = u
    for _ in range(order):
        du = torch.autograd.grad(
            du, x,
            grad_outputs=torch.ones_like(du),
            create_graph=True,
            retain_graph=True,
        )[0]
    return du


# ===================================================================
# 4. Error metrics
# ===================================================================

def compute_errors(net, u_exact_fn, u_deriv_fn=None, n_eval=1000):
    """Compute L2, Linf, and (optionally) H1 errors.

    Returns
    -------
    dict with keys 'x', 'u_exact', 'u_pred', 'error', 'L2', 'Linf',
    and optionally 'H1_semi', 'H1'.
    """
    x_np = np.linspace(0, 1, n_eval)
    u_ex = u_exact_fn(x_np)

    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=DTYPE, requires_grad=True)
    with torch.no_grad():
        u_pr = net(x_t).numpy().ravel()

    err = u_pr - u_ex
    dx = 1.0 / (n_eval - 1)

    L2 = np.sqrt(np.trapz(err ** 2, dx=dx))
    Linf = np.max(np.abs(err))

    out = {
        "x": x_np, "u_exact": u_ex, "u_pred": u_pr, "error": err,
        "L2": L2, "Linf": Linf,
    }

    if u_deriv_fn is not None:
        du_ex = u_deriv_fn(x_np)
        x_t2 = torch.tensor(x_np.reshape(-1, 1), dtype=DTYPE, requires_grad=True)
        u_t2 = net(x_t2)
        du_t2 = diff(u_t2, x_t2, order=1)
        du_pr = du_t2.detach().numpy().ravel()
        derr = du_pr - du_ex
        H1_semi = np.sqrt(np.trapz(derr ** 2, dx=dx))
        out["H1_semi"] = H1_semi
        out["H1"] = np.sqrt(L2 ** 2 + H1_semi ** 2)

    return out


# ===================================================================
# 5. Linear FDM solver  -u'' = f  (for initialisation / beta=0)
# ===================================================================

def solve_fdm_linear(f_fn, alpha, g0, g1, N=1000):
    """Finite-difference solver for the linear problem -u'' = f."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Row 0: integral BC via trapezoidal rule
    A[0, 0] = 1.0 - alpha * h / 2.0
    for j in range(1, N):
        A[0, j] = -alpha * h
    b[0] = alpha * h / 2.0 * g1 + g0

    # Rows 1..N-1: centred finite differences  -u'' = f
    for i in range(1, N):
        A[i, i] = 2.0
        if i - 1 >= 0:
            A[i, i - 1] = -1.0
        if i + 1 < N:
            A[i, i + 1] = -1.0
        b[i] = h ** 2 * f_fn(x[i])
    b[N - 1] += g1  # boundary: u_N = g1

    u_int = np.linalg.solve(A, b)
    return x, np.append(u_int, g1)


# ===================================================================
# 6. Nonlinear FDM solver  -[k(u) u']' = f  (Newton-Raphson)
# ===================================================================

def solve_fdm_nonlinear(f_fn, k_fn, dk_fn, alpha, g0, g1,
                        N=1000, tol=1e-12, max_iter=100,
                        verbose=False, beta=1.0):
    """
    Newton-Raphson FDM solver for  -[k(u) u']' = f(x).

    Interior discretisation (i = 1..N-1):
        F_i = -[k_{i+1/2}(u_{i+1}-u_i) - k_{i-1/2}(u_i-u_{i-1})] / h^2 - f_i = 0

    where k_{i+1/2} = k( (u_i + u_{i+1})/2 ).

    Row 0: integral BC   u_0 = alpha * trap(u) + g0
    u_N = g1 is fixed (eliminated from unknowns).

    Parameters
    ----------
    f_fn : callable(x, beta=...) -> source term
    k_fn : callable(u, beta) -> conductivity
    dk_fn : callable(u, beta) -> dk/du
    alpha, g0, g1 : BC parameters
    N : int  (number of intervals)
    tol, max_iter : Newton parameters
    beta : nonlinearity strength

    Returns
    -------
    x, u : ndarrays of shape (N+1,)
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    f_vals = f_fn(x, beta=beta)

    # --- Initial guess: linear FDM (beta=0) ---
    from exact_solution import f_source
    _, u_lin = solve_fdm_linear(lambda xx: f_source(xx, beta=0.0), alpha, g0, g1, N)
    u = u_lin.copy()

    for it in range(max_iter):
        F = np.zeros(N)
        J = np.zeros((N, N))

        # ---- Row 0: integral BC ----
        # u_0 - alpha * trap(u) - g0 = 0
        trap = h / 2.0 * u[0] + h * np.sum(u[1:N]) + h / 2.0 * u[N]
        F[0] = u[0] - alpha * trap - g0
        J[0, 0] = 1.0 - alpha * h / 2.0
        for j in range(1, N):
            J[0, j] = -alpha * h

        # ---- Rows 1..N-1: PDE ----
        for i in range(1, N):
            u_ip = 0.5 * (u[i] + u[i + 1])   # midpoint right
            u_im = 0.5 * (u[i] + u[i - 1])   # midpoint left
            k_ip = k_fn(u_ip, beta)
            k_im = k_fn(u_im, beta)
            dk_ip = dk_fn(u_ip, beta)
            dk_im = dk_fn(u_im, beta)
            du_r = u[i + 1] - u[i]
            du_l = u[i] - u[i - 1]

            # Residual
            F[i] = -(k_ip * du_r - k_im * du_l) / h ** 2 - f_vals[i]

            # --- Jacobian entries ---

            # dF_i / du_{i-1}
            # d/du_{i-1} [-(k_ip*du_r - k_im*du_l)/h^2]
            #   k_ip, du_r don't depend on u_{i-1}
            #   dk_im/du_{i-1} = dk_im * 0.5
            #   d(du_l)/du_{i-1} = -1
            # = -( 0 - (dk_im*0.5*du_l + k_im*(-1)) ) / h^2
            # = (dk_im*0.5*du_l - k_im) / h^2
            if i - 1 < N:  # u_{i-1} is an unknown
                J[i, i - 1] = (0.5 * dk_im * du_l - k_im) / h ** 2

            # dF_i / du_i
            # right flux: d(k_ip*du_r)/du_i = dk_ip*0.5*du_r + k_ip*(-1)
            # left  flux: d(k_im*du_l)/du_i = dk_im*0.5*du_l + k_im*(+1)
            dFr = 0.5 * dk_ip * du_r - k_ip
            dFl = 0.5 * dk_im * du_l + k_im
            J[i, i] = -(dFr - dFl) / h ** 2

            # dF_i / du_{i+1}
            # d(k_ip*du_r)/du_{i+1} = dk_ip*0.5*du_r + k_ip*(+1)
            # k_im, du_l don't depend on u_{i+1}
            if i + 1 < N:  # u_{i+1} is an unknown
                J[i, i + 1] = -(0.5 * dk_ip * du_r + k_ip) / h ** 2

        # ---- Newton step ----
        delta = np.linalg.solve(J, -F)
        u[:N] += delta

        res_norm = np.linalg.norm(F, ord=np.inf)
        if verbose and (it < 5 or it % 10 == 0):
            print(f"  Newton iter {it:3d}: ||F||_inf = {res_norm:.2e}")

        if res_norm < tol:
            if verbose:
                print(f"  Newton converged in {it + 1} iterations.")
            break
    else:
        print(f"  WARNING: Newton did not converge in {max_iter} iterations "
              f"(||F|| = {res_norm:.2e})")

    return x, u
