"""
Utilitaires 2D non-lineaire : quadrature, fonctions de test,
derivees AD, metriques d'erreur, solveur FDM lineaire et Newton-Raphson.
"""

import numpy as np
import torch
from numpy.polynomial import legendre as leg
from scipy import sparse
from scipy.sparse.linalg import spsolve

DTYPE = torch.float64
DEVICE = "cpu"


# ============================================================
# Quadrature de Gauss-Legendre
# ============================================================
def gauss_legendre(n, a=0.0, b=1.0):
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def gauss_legendre_torch(n, a=0.0, b=1.0):
    x, w = gauss_legendre(n, a, b)
    return (
        torch.tensor(x, dtype=DTYPE).reshape(-1, 1),
        torch.tensor(w, dtype=DTYPE).reshape(-1, 1),
    )


def gauss_legendre_2d(nx, ny, ax=0.0, bx=1.0, ay=0.0, by=1.0):
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
    def __init__(self, n_test_y):
        self.n_test_y = n_test_y

    def eval_w(self, y01):
        M = len(y01)
        W = np.zeros((M, self.n_test_y))
        for j in range(self.n_test_y):
            W[:, j] = np.sin((j + 1) * np.pi * y01)
        return W

    def eval_dw(self, y01):
        M = len(y01)
        dW = np.zeros((M, self.n_test_y))
        for j in range(self.n_test_y):
            dW[:, j] = (j + 1) * np.pi * np.cos((j + 1) * np.pi * y01)
        return dW


# ============================================================
# Differentiation automatique 2D
# ============================================================
def grad_net(u, xy):
    g = torch.autograd.grad(u, xy, torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]


def div_flux(flux_x, flux_y, xy):
    """Calcule div(flux) = d(flux_x)/dx + d(flux_y)/dy via AD."""
    g_fx = torch.autograd.grad(flux_x, xy, torch.ones_like(flux_x),
                               create_graph=True, retain_graph=True)[0]
    g_fy = torch.autograd.grad(flux_y, xy, torch.ones_like(flux_y),
                               create_graph=True, retain_graph=True)[0]
    return g_fx[:, 0:1] + g_fy[:, 1:2]


# ============================================================
# Metriques d'erreur 2D
# ============================================================
def compute_errors_2d(net, u_exact_fn, u_dx_fn=None, u_dy_fn=None, n_eval=100):
    x1d = np.linspace(0, 1, n_eval)
    y1d = np.linspace(0, 1, n_eval)
    xx, yy = np.meshgrid(x1d, y1d, indexing='ij')
    u_ex = u_exact_fn(xx, yy)

    xy_flat = np.column_stack([xx.ravel(), yy.ravel()])
    xy_t = torch.tensor(xy_flat, dtype=DTYPE)

    with torch.no_grad():
        u_pred_flat = net(xy_t).cpu().numpy().flatten()
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
        xy_t2 = torch.tensor(xy_flat, dtype=DTYPE)
        xy_t2.requires_grad_(True)
        u_p = net(xy_t2)
        g = torch.autograd.grad(u_p, xy_t2, torch.ones_like(u_p),
                                create_graph=False)[0]
        du_dx_pred = g[:, 0].detach().numpy().reshape(n_eval, n_eval)
        du_dy_pred = g[:, 1].detach().numpy().reshape(n_eval, n_eval)
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
# Solveur FDM 2D lineaire (pour initialisation Newton)
# ============================================================
def solve_fdm_2d_linear(f_fn, alpha, g0, N=100):
    """Resout -Delta u = f sur [0,1]^2 (lineaire, pour init Newton)."""
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
            if j == 0 or j == N or i == N:
                rows.append(k); cols.append(k); vals.append(1.0)
                rhs[k] = 0.0
            elif i == 0:
                rows.append(k); cols.append(k); vals.append(1.0 - alpha * h / 2.0)
                for m in range(1, N):
                    rows.append(k); cols.append(idx(m, j)); vals.append(-alpha * h)
                rhs[k] = g0
            else:
                rows.append(k); cols.append(k); vals.append(4.0)
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-1.0)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-1.0)
                rhs[k] = h ** 2 * f_fn(x1d[i], y1d[j])

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_total, n_total))
    u_flat = spsolve(A, rhs)
    return x1d, y1d, u_flat.reshape(N + 1, N + 1)


# ============================================================
# Solveur FDM 2D non-lineaire (Newton-Raphson)
# ============================================================
def solve_fdm_2d_nonlinear(f_fn, k_fn, dk_fn, alpha, g0, N=100,
                           beta=1.0, tol=1e-10, max_iter=50, verbose=False):
    """
    Resout -div(k(u)*grad(u)) = f sur [0,1]^2 par Newton-Raphson.

    Discretisation :
    F_{i,j} = -[k_{i+1/2,j}*(u_{i+1,j}-u_{i,j}) - k_{i-1/2,j}*(u_{i,j}-u_{i-1,j})]/h^2
              -[k_{i,j+1/2}*(u_{i,j+1}-u_{i,j}) - k_{i,j-1/2}*(u_{i,j}-u_{i,j-1})]/h^2
              - f_{i,j} = 0
    """
    h = 1.0 / N
    x1d = np.linspace(0, 1, N + 1)
    y1d = np.linspace(0, 1, N + 1)
    n_total = (N + 1) ** 2

    def idx(i, j):
        return i * (N + 1) + j

    # Initialisation : solution FDM lineaire (beta=0)
    from exact_solution import f_source_np as f_src_beta
    f_lin = lambda xi, yj: f_src_beta(xi, yj, beta=0.0)
    _, _, u_init = solve_fdm_2d_linear(f_lin, alpha, g0, N=N)
    u = u_init.ravel().copy()

    for newton_it in range(max_iter):
        F = np.zeros(n_total)
        jac_rows, jac_cols, jac_vals = [], [], []

        for i in range(N + 1):
            for j in range(N + 1):
                kk = idx(i, j)

                if j == 0 or j == N or i == N:
                    # Dirichlet u = 0
                    F[kk] = u[kk]
                    jac_rows.append(kk); jac_cols.append(kk); jac_vals.append(1.0)

                elif i == 0:
                    # Integral BC : u_0j - alpha*trap(u[:,j]) - g0 = 0
                    trap = h * (u[idx(0, j)] / 2.0 + u[idx(N, j)] / 2.0)
                    for m in range(1, N):
                        trap += h * u[idx(m, j)]
                    F[kk] = u[kk] - alpha * trap - g0

                    jac_rows.append(kk); jac_cols.append(kk)
                    jac_vals.append(1.0 - alpha * h / 2.0)
                    for m in range(1, N):
                        jac_rows.append(kk); jac_cols.append(idx(m, j))
                        jac_vals.append(-alpha * h)

                else:
                    # Interieur : discretisation du flux non-lineaire
                    u_ij = u[kk]
                    u_ip = u[idx(i + 1, j)]
                    u_im = u[idx(i - 1, j)]
                    u_jp = u[idx(i, j + 1)]
                    u_jm = u[idx(i, j - 1)]

                    # Demi-points x
                    u_ip_mid = 0.5 * (u_ij + u_ip)
                    u_im_mid = 0.5 * (u_ij + u_im)
                    k_ip = k_fn(u_ip_mid, beta)
                    k_im = k_fn(u_im_mid, beta)

                    # Demi-points y
                    u_jp_mid = 0.5 * (u_ij + u_jp)
                    u_jm_mid = 0.5 * (u_ij + u_jm)
                    k_jp = k_fn(u_jp_mid, beta)
                    k_jm = k_fn(u_jm_mid, beta)

                    # Flux
                    flux_xp = k_ip * (u_ip - u_ij) / h
                    flux_xm = k_im * (u_ij - u_im) / h
                    flux_yp = k_jp * (u_jp - u_ij) / h
                    flux_ym = k_jm * (u_ij - u_jm) / h

                    F[kk] = -(flux_xp - flux_xm + flux_yp - flux_ym) / h \
                            - f_fn(x1d[i], y1d[j])

                    # Jacobien analytique
                    dk_ip = dk_fn(u_ip_mid, beta)
                    dk_im = dk_fn(u_im_mid, beta)
                    dk_jp = dk_fn(u_jp_mid, beta)
                    dk_jm = dk_fn(u_jm_mid, beta)

                    du_xr = u_ip - u_ij
                    du_xl = u_ij - u_im
                    du_yr = u_jp - u_ij
                    du_yl = u_ij - u_jm

                    # dF/du_{i,j}
                    dF_dij = (
                        -(0.5 * dk_ip * du_xr / h - k_ip / h) / h
                        + (0.5 * dk_im * du_xl / h + k_im / h) / h
                        - (0.5 * dk_jp * du_yr / h - k_jp / h) / h
                        + (0.5 * dk_jm * du_yl / h + k_jm / h) / h
                    )
                    jac_rows.append(kk); jac_cols.append(kk); jac_vals.append(dF_dij)

                    # dF/du_{i+1,j}
                    dF_dip = -(0.5 * dk_ip * du_xr / h + k_ip / h) / h
                    jac_rows.append(kk); jac_cols.append(idx(i + 1, j)); jac_vals.append(dF_dip)

                    # dF/du_{i-1,j}
                    dF_dim = (0.5 * dk_im * du_xl / h - k_im / h) / h
                    jac_rows.append(kk); jac_cols.append(idx(i - 1, j)); jac_vals.append(dF_dim)

                    # dF/du_{i,j+1}
                    dF_djp = -(0.5 * dk_jp * du_yr / h + k_jp / h) / h
                    jac_rows.append(kk); jac_cols.append(idx(i, j + 1)); jac_vals.append(dF_djp)

                    # dF/du_{i,j-1}
                    dF_djm = (0.5 * dk_jm * du_yl / h - k_jm / h) / h
                    jac_rows.append(kk); jac_cols.append(idx(i, j - 1)); jac_vals.append(dF_djm)

        J = sparse.csr_matrix((jac_vals, (jac_rows, jac_cols)), shape=(n_total, n_total))
        delta = spsolve(J, -F)
        u += delta

        res_norm = np.max(np.abs(F))
        if verbose:
            print(f"  Newton iter {newton_it+1}: ||F||_inf = {res_norm:.3e}, ||delta||_inf = {np.max(np.abs(delta)):.3e}")

        if res_norm < tol:
            break

    u_grid = u.reshape(N + 1, N + 1)
    return x1d, y1d, u_grid
