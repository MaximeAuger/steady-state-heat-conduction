"""
Solution manufacturee pour le probleme de conduction thermique 2D non-lineaire.

EDP :
    -div(k(u)*grad(u)) = f(x,y),   (x,y) in (0,1)^2
    k(u) = 1 + beta * u^2          (conductivite dependant de la temperature)

BCs :
    u(0,y) = alpha * int_0^1 u(x,y) dx + g0  (integrale, pour chaque y)
    u(1,y) = 0, u(x,0) = 0, u(x,1) = 0       (Dirichlet)

Solution manufacturee (MEME que le cas lineaire) :
    u(x,y) = sin(pi*y) * [sin(pi*x) + (1-x) * 4/pi]

Le terme source f(x,y,beta) est calcule par substitution :
    div(k(u)*grad(u)) = dk/du * |grad(u)|^2 + k(u) * Delta(u)
    f = -div(k(u)*grad(u))
"""

import numpy as np
import torch

# Parametres du probleme
ALPHA = 1.0
C_MANUF = 4.0 / np.pi
G0 = 0.0
BETA_DEFAULT = 1.0
DTYPE = torch.float64


# ============================================================
# Conductivite k(u) = 1 + beta * u^2
# ============================================================
def k_conductivity(u, beta=BETA_DEFAULT):
    return 1.0 + beta * u ** 2


def dk_conductivity(u, beta=BETA_DEFAULT):
    return 2.0 * beta * u


# ============================================================
# Solution exacte et derivees (NumPy)
# ============================================================
def u_exact_np(x, y):
    return np.sin(np.pi * y) * (np.sin(np.pi * x) + (1.0 - x) * C_MANUF)


def u_exact_dx_np(x, y):
    return np.sin(np.pi * y) * (np.pi * np.cos(np.pi * x) - C_MANUF)


def u_exact_dy_np(x, y):
    return np.pi * np.cos(np.pi * y) * (np.sin(np.pi * x) + (1.0 - x) * C_MANUF)


def u_exact_dxx_np(x, y):
    return -np.pi ** 2 * np.sin(np.pi * y) * np.sin(np.pi * x)


def u_exact_dyy_np(x, y):
    return -np.pi ** 2 * np.sin(np.pi * y) * (np.sin(np.pi * x) + (1.0 - x) * C_MANUF)


# ============================================================
# Terme source f(x,y,beta)
# ============================================================
def f_source_np(x, y, beta=BETA_DEFAULT):
    """
    f = -div(k(u)*grad(u))
      = -(dk/du * |grad(u)|^2 + k(u) * Delta(u))
      = -(2*beta*u * (u_x^2 + u_y^2) + (1+beta*u^2) * (u_xx + u_yy))
    """
    u = u_exact_np(x, y)
    ux = u_exact_dx_np(x, y)
    uy = u_exact_dy_np(x, y)
    uxx = u_exact_dxx_np(x, y)
    uyy = u_exact_dyy_np(x, y)
    grad_sq = ux ** 2 + uy ** 2
    laplacian = uxx + uyy
    return -(2.0 * beta * u * grad_sq + (1.0 + beta * u ** 2) * laplacian)


# ============================================================
# Fonctions PyTorch
# ============================================================
def u_exact_torch(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return torch.sin(np.pi * y) * (torch.sin(np.pi * x) + (1.0 - x) * C_MANUF)


def f_source_torch(xy, beta=BETA_DEFAULT):
    x, y = xy[:, 0:1], xy[:, 1:2]
    u = torch.sin(np.pi * y) * (torch.sin(np.pi * x) + (1.0 - x) * C_MANUF)
    ux = torch.sin(np.pi * y) * (np.pi * torch.cos(np.pi * x) - C_MANUF)
    uy = np.pi * torch.cos(np.pi * y) * (torch.sin(np.pi * x) + (1.0 - x) * C_MANUF)
    uxx = -np.pi ** 2 * torch.sin(np.pi * y) * torch.sin(np.pi * x)
    uyy = -np.pi ** 2 * torch.sin(np.pi * y) * (torch.sin(np.pi * x) + (1.0 - x) * C_MANUF)
    grad_sq = ux ** 2 + uy ** 2
    laplacian = uxx + uyy
    return -(2.0 * beta * u * grad_sq + (1.0 + beta * u ** 2) * laplacian)


# ============================================================
# Lambda exact (flux au bord gauche pour le VPINN)
# ============================================================
def lambda_exact(beta=BETA_DEFAULT):
    """
    Lambda = -k(u(0,y)) * u_x(0,y) decompose en base sin(j*pi*y).
    u(0,y) = (4/pi)*sin(pi*y)
    u_x(0,y) = sin(pi*y)*(pi - 4/pi)
    k(u(0,y)) = 1 + beta*(4/pi)^2*sin^2(pi*y)

    lam(y) = -k(u(0,y))*u_x(0,y) = -(1 + beta*(4/pi)^2*sin^2(pi*y)) * sin(pi*y)*(pi-4/pi)

    Decomposition en Fourier-sinus :
    lam(y) = -(pi-4/pi)*sin(pi*y) - beta*(4/pi)^2*(pi-4/pi)*sin^2(pi*y)*sin(pi*y)

    Utiliser sin^2(a)*sin(a) = sin(a)*(1-cos(2a))/2 = sin(a)/2 - sin(a)*cos(2a)/2
    sin(a)*cos(2a) = (sin(3a) - sin(a))/2
    Donc sin^2(a)*sin(a) = sin(a)/2 - (sin(3a)-sin(a))/4 = 3*sin(a)/4 - sin(3a)/4

    lam_1 = -(pi-4/pi) - beta*(4/pi)^2*(pi-4/pi)*3/4
    lam_3 = beta*(4/pi)^2*(pi-4/pi)/4
    """
    u0 = C_MANUF  # u(0,y)/sin(pi*y) = 4/pi
    du0 = np.pi - C_MANUF  # u_x(0,y)/sin(pi*y) = pi - 4/pi
    k0_base = 1.0 + beta * u0 ** 2  # k(u(0,y)) quand sin(pi*y)=1

    # Coefficient lam_1 exact par integration numerique
    from utils import gauss_legendre
    yq, wyq = gauss_legendre(100)
    u_0y = C_MANUF * np.sin(np.pi * yq)
    ux_0y = (np.pi - C_MANUF) * np.sin(np.pi * yq)
    k_0y = 1.0 + beta * u_0y ** 2
    lam_y = -k_0y * ux_0y  # lam(y) aux points de quadrature

    # Projection sur sin(j*pi*y)
    n_lam = 5
    lam_coeffs = np.zeros(n_lam)
    for j in range(n_lam):
        wj = np.sin((j + 1) * np.pi * yq)
        lam_coeffs[j] = 2.0 * np.sum(wyq * lam_y * wj)

    return lam_coeffs


# ============================================================
# Verification
# ============================================================
def verify(beta=BETA_DEFAULT, n_pts=200, tol=1e-10):
    """Verifie la solution manufacturee sur une grille."""
    x = np.linspace(0, 1, n_pts)
    y = np.linspace(0, 1, n_pts)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    u = u_exact_np(xx, yy)

    # BC Dirichlet
    assert np.max(np.abs(u[-1, :])) < tol, "u(1,y) != 0"
    assert np.max(np.abs(u[:, 0])) < tol, "u(x,0) != 0"
    assert np.max(np.abs(u[:, -1])) < tol, "u(x,1) != 0"

    # BC integrale
    u_left = u[0, :]
    integral_x = np.trapz(u, x=x, axis=0)
    residual_bc = u_left - ALPHA * integral_x - G0
    assert np.max(np.abs(residual_bc)) < 1e-3, (
        f"Condition integrale non satisfaite: max|res| = {np.max(np.abs(residual_bc)):.2e}"
    )

    # PDE : -div(k(u)*grad(u)) = f par differences finies sur le flux
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    k_u = k_conductivity(u, beta)
    ux = u_exact_dx_np(xx, yy)
    uy = u_exact_dy_np(xx, yy)
    f_vals = f_source_np(xx, yy, beta)

    # Verifier au centre du domaine (eviter les bords)
    # Differences finies centrees sur grille interieure [1:-1, 1:-1]
    flux_x = k_u * ux
    flux_y = k_u * uy
    ii = slice(1, -1)
    # d(flux_x)/dx par diff centrees sur l'axe 0
    dfx_dx = (flux_x[2:, ii] - flux_x[:-2, ii]) / (2 * dx)  # (n-2, n-2)
    # d(flux_y)/dy par diff centrees sur l'axe 1
    dfy_dy = (flux_y[ii, 2:] - flux_y[ii, :-2]) / (2 * dy)  # (n-2, n-2)
    residual_pde = -dfx_dx - dfy_dy - f_vals[1:-1, 1:-1]
    pde_err = np.max(np.abs(residual_pde[3:-3, 3:-3]))

    assert pde_err < 1e-1, f"Residu PDE trop grand: max = {pde_err:.2e}"

    print(f"[OK] Solution 2D NL verifiee (beta={beta}, PDE res={pde_err:.2e}).")
    return True


if __name__ == "__main__":
    for b in [0.0, 0.5, 1.0, 2.0, 5.0]:
        verify(beta=b)
