"""
Manufactured exact solution for the nonlinear 1-D steady-state heat equation.

PDE:  -[k(u) u'(x)]' = f(x),   x in (0, 1)
      k(u) = 1 + beta * u^2     (temperature-dependent conductivity)

BCs:  u(0) = alpha * int_0^1 u(x) dx + g0   (nonlocal integral BC)
      u(1) = g1                               (Dirichlet)

Manufactured solution (same as linear case):
      u(x) = sin(pi x) + (1 - x) * C,   C = 4/pi

The source term f(x) is computed by substitution into the PDE so that
the manufactured u satisfies the nonlinear equation exactly.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Physical / manufactured-solution parameters
# ---------------------------------------------------------------------------
ALPHA = 1.0
C_MANUF = 4.0 / np.pi          # constant in manufactured solution
G0 = 0.0
G1 = 0.0
BETA_DEFAULT = 1.0              # default nonlinearity strength

DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Conductivity  k(u) = 1 + beta * u^2
# ---------------------------------------------------------------------------

def k_conductivity(u, beta=BETA_DEFAULT):
    """Nonlinear conductivity coefficient."""
    return 1.0 + beta * u ** 2


def dk_conductivity(u, beta=BETA_DEFAULT):
    """Derivative dk/du = 2 * beta * u."""
    return 2.0 * beta * u


# ---------------------------------------------------------------------------
# Exact solution and its derivatives (NumPy)
# ---------------------------------------------------------------------------

def u_exact(x, numpy=True):
    """Exact manufactured solution."""
    if numpy:
        return np.sin(np.pi * x) + (1.0 - x) * C_MANUF
    return torch.sin(np.pi * x) + (1.0 - x) * C_MANUF


def u_exact_deriv(x, numpy=True):
    """First derivative of exact solution."""
    if numpy:
        return np.pi * np.cos(np.pi * x) - C_MANUF
    return np.pi * torch.cos(np.pi * x) - C_MANUF


def u_exact_deriv2(x, numpy=True):
    """Second derivative of exact solution."""
    if numpy:
        return -(np.pi ** 2) * np.sin(np.pi * x)
    return -(np.pi ** 2) * torch.sin(np.pi * x)


def integral_u_exact():
    """Exact value of int_0^1 u(x) dx."""
    return 2.0 / np.pi + C_MANUF / 2.0  # = 4/pi


# ---------------------------------------------------------------------------
# Source term (nonlinear)
#   f(x) = -[k(u) u']' = -[k'(u)(u')^2 + k(u) u'']
# ---------------------------------------------------------------------------

def f_source(x, beta=BETA_DEFAULT, numpy=True):
    """
    Source term computed from the manufactured solution so that
    -[k(u) u']' = f(x) holds exactly.

    f(x) = -[ dk(u)*u'^2 + k(u)*u'' ]
         = -[ 2*beta*u*(u')^2 + (1 + beta*u^2)*u'' ]
    """
    u = u_exact(x, numpy=numpy)
    du = u_exact_deriv(x, numpy=numpy)
    d2u = u_exact_deriv2(x, numpy=numpy)
    k = k_conductivity(u, beta)
    dk = dk_conductivity(u, beta)
    return -(dk * du ** 2 + k * d2u)


def f_source_torch(x, beta=BETA_DEFAULT):
    """PyTorch version of f_source for use in loss functions."""
    return f_source(x, beta=beta, numpy=False)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(beta=BETA_DEFAULT, n=5000):
    """
    Verify the manufactured solution by checking:
      1. -[k(u) u']' = f  (via finite differences on the flux k(u)*u')
      2. Boundary conditions
      3. Integral constraint
    """
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]

    u = u_exact(x)
    du = u_exact_deriv(x)
    f = f_source(x, beta=beta)

    # Check 1: PDE via finite differences on flux = k(u)*u'
    flux = k_conductivity(u, beta) * du
    flux_deriv_fd = np.gradient(flux, h, edge_order=2)
    residual = -flux_deriv_fd - f
    pde_err = np.max(np.abs(residual[10:-10]))  # avoid boundary artefacts

    # Check 2: BCs
    bc_left = u[0] - (ALPHA * integral_u_exact() + G0)
    bc_right = u[-1] - G1

    # Check 3: integral constraint
    integral_num = np.trapz(u, x)
    integral_exact = integral_u_exact()
    int_err = abs(integral_num - integral_exact)

    print(f"=== Verification (beta = {beta}) ===")
    print(f"  PDE residual (FD)   : {pde_err:.2e}")
    print(f"  BC left  error      : {abs(bc_left):.2e}")
    print(f"  BC right error      : {abs(bc_right):.2e}")
    print(f"  Integral error      : {int_err:.2e}")
    print(f"  u(0) = {u[0]:.6f},  alpha*int(u)+g0 = {ALPHA*integral_exact+G0:.6f}")
    print(f"  u(1) = {u[-1]:.6f},  g1 = {G1:.6f}")

    ok = pde_err < 1e-3 and abs(bc_left) < 1e-12 and abs(bc_right) < 1e-12
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    for b in [0.0, 0.5, 1.0, 2.0, 5.0]:
        verify(beta=b)
        print()
