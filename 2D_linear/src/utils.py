"""
Utilitaires 2D : quadrature, fonctions de test (Legendre + sinus),
derivees AD, metriques d'erreur, solveur FDM 2D de reference.
"""

import numpy as np
import torch
from numpy.polynomial import legendre as leg
from scipy import sparse
from scipy.sparse.linalg import spsolve

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Quadrature de Gauss-Legendre (1D et 2D)
# ============================================================
def gauss_legendre(n, a=0.0, b=1.0):
    """Noeuds et poids GL sur [a, b] (NumPy)."""
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def gauss_legendre_torch(n, a=0.0, b=1.0):
    """Noeuds et poids GL sur [a, b] (PyTorch tensors, shape (n, 1))."""
    x, w = gauss_legendre(n, a, b)
    return (
        torch.tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1, 1),
        torch.tensor(w, dtype=DTYPE, device=DEVICE).reshape(-1, 1),
    )


def gauss_legendre_2d(nx, ny, ax=0.0, bx=1.0, ay=0.0, by=1.0):
    """
    Produit tensoriel GL sur [ax,bx] x [ay,by].
    Returns : xy (nx*ny, 2), w (nx*ny,) en NumPy.
    """
    x1, w1x = gauss_legendre(nx, ax, bx)
    y1, w1y = gauss_legendre(ny, ay, by)
    xx, yy = np.meshgrid(x1, y1, indexing='ij')
    wx, wy = np.meshgrid(w1x, w1y, indexing='ij')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    w = (wx * wy).ravel()
    return xy, w


# ============================================================
# Fonctions de test de Legendre (direction x)
# ============================================================
class LegendreTestFunctions:
    """
    v_k(x) = P_k(2x-1) - 1,  k = 1, ..., N_test
    => v_k(1) = 0,  v_k(0) = (-1)^k - 1
    => v_k'(x) = 2 P_k'(2x-1)
    """

    def __init__(self, n_test):
        self.n_test = n_test

    def eval_v(self, x01):
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        V = np.zeros((N, self.n_test))
        for k in range(self.n_test):
            c = np.zeros(k + 2); c[-1] = 1.0
            V[:, k] = leg.legval(xi, c) - 1.0
        return V

    def eval_dv(self, x01):
        xi = 2.0 * x01 - 1.0
        N = len(x01)
        dV = np.zeros((N, self.n_test))
        for k in range(self.n_test):
            c = np.zeros(k + 2); c[-1] = 1.0
            dc = leg.legder(c)
            dV[:, k] = 2.0 * leg.legval(xi, dc)
        return dV

    def v_at_zero(self):
        return np.array([(-1.0) ** (k + 1) - 1.0 for k in range(self.n_test)])


# ============================================================
# Fonctions de test sinusoidales (direction y)
# ============================================================
class SineTestFunctions:
    """
    w_j(y) = sin(j*pi*y),  j = 1, ..., N_test_y
    => w_j(0) = 0, w_j(1) = 0
    => w_j'(y) = j*pi*cos(j*pi*y)
    Orthogonalite : int_0^1 sin(j*pi*y)*sin(m*pi*y) dy = delta_{jm}/2
    """

    def __init__(self, n_test_y):
        self.n_test_y = n_test_y

    def eval_w(self, y01):
        """y01 : (M,) array dans [0,1]. Retourne (M, n_test_y)."""
        M = len(y01)
        W = np.zeros((M, self.n_test_y))
        for j in range(self.n_test_y):
            W[:, j] = np.sin((j + 1) * np.pi * y01)
        return W

    def eval_dw(self, y01):
        """Retourne (M, n_test_y)."""
        M = len(y01)
        dW = np.zeros((M, self.n_test_y))
        for j in range(self.n_test_y):
            dW[:, j] = (j + 1) * np.pi * np.cos((j + 1) * np.pi * y01)
        return dW


# ============================================================
# Differentiation automatique 2D
# ============================================================
def grad_net(u, xy):
    """
    Gradient du/dx, du/dy.
    u : (N, 1), xy : (N, 2) avec requires_grad=True.
    Retourne (u_x, u_y) chacun (N, 1).
    """
    g = torch.autograd.grad(u, xy, torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]


def laplacian_net(u, xy):
    """
    Laplacien u_xx + u_yy.
    u : (N, 1), xy : (N, 2) avec requires_grad=True.
    Retourne (lapl, u_x, u_y) pour reutilisation.
    """
    g = torch.autograd.grad(u, xy, torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
    u_x = g[:, 0:1]
    u_y = g[:, 1:2]

    g_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    u_xx = g_xx[:, 0:1]

    g_yy = torch.autograd.grad(u_y, xy, torch.ones_like(u_y),
                               create_graph=True, retain_graph=True)[0]
    u_yy = g_yy[:, 1:2]

    return u_xx + u_yy, u_x, u_y


# ============================================================
# Metriques d'erreur 2D
# ============================================================
def compute_errors_2d(net, u_exact_fn, u_dx_fn=None, u_dy_fn=None, n_eval=100):
    """
    Calcule L2, Linf, et optionnellement H1 sur [0,1]^2.
    n_eval : nombre de points par dimension (total = n_eval^2).
    """
    x1d = np.linspace(0, 1, n_eval)
    y1d = np.linspace(0, 1, n_eval)
    xx, yy = np.meshgrid(x1d, y1d, indexing='ij')
    u_ex = u_exact_fn(xx, yy)

    xy_flat = np.column_stack([xx.ravel(), yy.ravel()])
    xy_t = torch.tensor(xy_flat, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        u_pred_flat = net(xy_t).detach().cpu().numpy().flatten()
    u_pred = u_pred_flat.reshape(n_eval, n_eval)

    err = u_pred - u_ex
    dx = 1.0 / (n_eval - 1)

    l2 = np.sqrt(np.trapz(np.trapz(err ** 2, dx=dx, axis=1), dx=dx))
    linf = np.max(np.abs(err))

    result = {
        "xx": xx, "yy": yy, "u_exact": u_ex, "u_pred": u_pred,
        "error": err, "L2": l2, "Linf": linf,
    }

    if u_dx_fn is not None and u_dy_fn is not None:
        xy_t2 = torch.tensor(xy_flat, dtype=DTYPE, device=DEVICE)
        xy_t2.requires_grad_(True)
        u_p = net(xy_t2)
        g = torch.autograd.grad(u_p, xy_t2, torch.ones_like(u_p),
                                create_graph=False)[0]
        du_dx_pred = g[:, 0].detach().cpu().numpy().reshape(n_eval, n_eval)
        du_dy_pred = g[:, 1].detach().cpu().numpy().reshape(n_eval, n_eval)

        du_dx_ex = u_dx_fn(xx, yy)
        du_dy_ex = u_dy_fn(xx, yy)

        err_dx = du_dx_pred - du_dx_ex
        err_dy = du_dy_pred - du_dy_ex

        h1_semi = np.sqrt(
            np.trapz(np.trapz(err_dx ** 2 + err_dy ** 2, dx=dx, axis=1), dx=dx)
        )
        result["H1_semi"] = h1_semi
        result["H1"] = np.sqrt(l2 ** 2 + h1_semi ** 2)

    return result


# ============================================================
# Solveur FDM 2D de reference
# ============================================================
def solve_fdm_2d(f_fn, alpha, g0, N=100):
    """
    Resout -Delta u = f sur [0,1]^2 avec :
        u(1,y) = 0, u(x,0) = 0, u(x,1) = 0   (Dirichlet)
        u(0,y) = alpha * int_0^1 u(x,y) dx + g0  (integrale, pour chaque y)

    Stencil 5 points, systeme sparse.

    Returns : x1d (N+1,), y1d (N+1,), u_grid (N+1, N+1)
    """
    h = 1.0 / N
    x1d = np.linspace(0, 1, N + 1)
    y1d = np.linspace(0, 1, N + 1)

    n_total = (N + 1) ** 2

    def idx(i, j):
        return i * (N + 1) + j

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_total)

    for i in range(N + 1):
        for j in range(N + 1):
            k = idx(i, j)
            xi, yj = x1d[i], y1d[j]

            if j == 0 or j == N or i == N:
                # Dirichlet : u = 0 (bottom, top, right)
                rows.append(k); cols.append(k); vals.append(1.0)
                rhs[k] = 0.0

            elif i == 0:
                # Bord gauche : condition integrale
                # u_{0,j} = alpha * h * [u_{0,j}/2 + u_{1,j} + ... + u_{N-1,j} + u_{N,j}/2] + g0
                # (1 - alpha*h/2) * u_{0,j} - alpha*h * sum_{m=1}^{N-1} u_{m,j} = g0
                # (u_{N,j} = 0 donc le terme alpha*h/2*u_{N,j} disparait)
                rows.append(k); cols.append(k); vals.append(1.0 - alpha * h / 2.0)
                for m in range(1, N):
                    rows.append(k); cols.append(idx(m, j)); vals.append(-alpha * h)
                rhs[k] = g0

            else:
                # Interieur : stencil 5 points
                # 4*u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} = h^2*f
                rows.append(k); cols.append(k); vals.append(4.0)
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-1.0)
                rhs[k] = h ** 2 * f_fn(xi, yj)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_total, n_total))
    u_flat = spsolve(A, rhs)
    u_grid = u_flat.reshape(N + 1, N + 1)

    return x1d, y1d, u_grid
