"""
Solution manufacturee pour le probleme de Poisson 2D stationnaire
avec condition au bord integrale non locale.

Probleme :
    -Delta u(x,y) = f(x,y),   (x,y) in (0,1)^2
    u(0,y) = alpha * int_0^1 u(x,y) dx + g0    (integrale, pour chaque y)
    u(1,y) = 0                                    (Dirichlet)
    u(x,0) = 0                                    (Dirichlet)
    u(x,1) = 0                                    (Dirichlet)

Solution manufacturee :
    u(x,y) = sin(pi*y) * [sin(pi*x) + (1-x) * C],   C = 4/pi

Verification :
    u(1,y) = 0, u(x,0) = 0, u(x,1) = 0
    int_0^1 u(x,y) dx = sin(pi*y) * 4/pi = u(0,y)
    -Delta u = pi^2 * sin(pi*y) * [2*sin(pi*x) + (1-x)*C]
"""

import numpy as np
import torch

# Parametres du probleme
ALPHA = 1.0
C_MANUF = 4.0 / np.pi
G0 = 0.0
DTYPE = torch.float64


# ============================================================
# Fonctions NumPy (pour evaluation / FDM / pre-calcul)
# ============================================================
def u_exact_np(x, y):
    """Solution exacte u(x,y). x, y : arrays ou scalaires."""
    return np.sin(np.pi * y) * (np.sin(np.pi * x) + (1.0 - x) * C_MANUF)


def f_source_np(x, y):
    """Terme source f(x,y) = -Delta u."""
    return np.pi ** 2 * np.sin(np.pi * y) * (
        2.0 * np.sin(np.pi * x) + (1.0 - x) * C_MANUF
    )


def u_exact_dx_np(x, y):
    """du/dx."""
    return np.sin(np.pi * y) * (np.pi * np.cos(np.pi * x) - C_MANUF)


def u_exact_dy_np(x, y):
    """du/dy."""
    return np.pi * np.cos(np.pi * y) * (
        np.sin(np.pi * x) + (1.0 - x) * C_MANUF
    )


# ============================================================
# Fonctions PyTorch (pour loss PINN/VPINN)
# ============================================================
def u_exact_torch(xy):
    """Solution exacte. xy : (N, 2) tensor."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return torch.sin(np.pi * y) * (torch.sin(np.pi * x) + (1.0 - x) * C_MANUF)


def f_source_torch(xy):
    """Terme source. xy : (N, 2) tensor."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return np.pi ** 2 * torch.sin(np.pi * y) * (
        2.0 * torch.sin(np.pi * x) + (1.0 - x) * C_MANUF
    )


# ============================================================
# Lambda exact (flux au bord gauche pour le VPINN)
# ============================================================
def lambda_exact():
    """
    Lambda = -u_x(0,y) decompose en base sin(j*pi*y).
    u_x(0,y) = sin(pi*y) * (pi - C_MANUF)
    => lam_1 = -(pi - C_MANUF) = C_MANUF - pi = 4/pi - pi ~ -1.8684
    => lam_j = 0 pour j > 1
    """
    return C_MANUF - np.pi  # ~ -1.8684


# ============================================================
# Verification
# ============================================================
def verify(n_pts=200, tol=1e-10):
    """Verifie la solution manufacturee sur une grille."""
    x = np.linspace(0, 1, n_pts)
    y = np.linspace(0, 1, n_pts)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    u = u_exact_np(xx, yy)

    # BC Dirichlet
    assert np.max(np.abs(u[-1, :])) < tol, "u(1,y) != 0"
    assert np.max(np.abs(u[:, 0])) < tol, "u(x,0) != 0"
    assert np.max(np.abs(u[:, -1])) < tol, "u(x,1) != 0"

    # BC integrale : u(0,y) = alpha * int_0^1 u(x,y) dx pour chaque y
    u_left = u[0, :]
    integral_x = np.trapz(u, x=x, axis=0)  # integrale sur x pour chaque y
    residual_bc = u_left - ALPHA * integral_x - G0
    assert np.max(np.abs(residual_bc)) < 1e-3, (
        f"Condition integrale non satisfaite: max|res| = {np.max(np.abs(residual_bc)):.2e}"
    )

    # Laplacien = -f
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    u_xx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx ** 2
    u_yy = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy ** 2
    laplacian = u_xx + u_yy
    f_vals = f_source_np(xx[1:-1, 1:-1], yy[1:-1, 1:-1])
    residual_pde = -laplacian - f_vals
    assert np.max(np.abs(residual_pde)) < 1e-2, (
        f"Residu PDE trop grand: max = {np.max(np.abs(residual_pde)):.2e}"
    )

    print("[OK] Solution manufacturee 2D verifiee.")
    return True


if __name__ == "__main__":
    verify()
