"""
PINN solver 2D — formulation forte, interface reutilisable.

    train_pinn(config, verbose) -> dict de resultats
"""

import time
import numpy as np
import torch
import torch.optim as optim

from network import MLP
from exact_solution import (
    ALPHA, G0, DTYPE,
    f_source_torch, u_exact_np, u_exact_dx_np, u_exact_dy_np,
)
from utils import (
    gauss_legendre_torch, laplacian_net, compute_errors_2d, DEVICE,
)

DEFAULT_CONFIG = {
    "n_hidden": 64,
    "n_layers": 5,
    "n_f_per_dim": 50,       # points de collocation par dimension (total = n_f^2)
    "n_b": 100,              # points par bord Dirichlet
    "n_q_x": 64,             # quadrature GL en x pour la BC integrale
    "n_q_y": 50,             # points y sur le bord gauche pour la BC integrale
    "w_f": 1.0,              # poids PDE
    "w_d": 50.0,             # poids Dirichlet (3 bords)
    "w_i": 50.0,             # poids integrale (bord gauche)
    "lr_adam": 1e-3,
    "n_adam": 15000,
    "n_lbfgs": 3000,
    "seed": 42,
}


def _make_points(cfg):
    """Cree les points de collocation et de bord."""
    nf = cfg["n_f_per_dim"]
    nb = cfg["n_b"]

    # Points interieurs : grille uniforme dans (0.01, 0.99)^2
    x1d = torch.linspace(0.01, 0.99, nf, dtype=DTYPE, device=DEVICE)
    y1d = torch.linspace(0.01, 0.99, nf, dtype=DTYPE, device=DEVICE)
    xx, yy = torch.meshgrid(x1d, y1d, indexing='ij')
    xy_f = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (nf^2, 2)
    xy_f.requires_grad_(True)

    # Bords Dirichlet (u=0)
    t = torch.linspace(0, 1, nb, dtype=DTYPE, device=DEVICE)
    xy_right = torch.stack([torch.ones(nb, dtype=DTYPE, device=DEVICE), t], dim=1)
    xy_bottom = torch.stack([t, torch.zeros(nb, dtype=DTYPE, device=DEVICE)], dim=1)
    xy_top = torch.stack([t, torch.ones(nb, dtype=DTYPE, device=DEVICE)], dim=1)

    # Bord gauche : points y pour la condition integrale (eviter les coins)
    y_left = torch.linspace(0.01, 0.99, cfg["n_q_y"], dtype=DTYPE, device=DEVICE)

    # Quadrature GL en x pour calculer int_0^1 u(x, y_j) dx
    x_quad, w_quad = gauss_legendre_torch(cfg["n_q_x"])

    return xy_f, xy_right, xy_bottom, xy_top, y_left, x_quad, w_quad


def _loss_pinn(net, xy_f, xy_right, xy_bottom, xy_top,
               y_left, x_quad, w_quad, cfg):
    """Calcule la perte PINN 2D."""
    # 1. Residu PDE : -Delta u - f = 0
    u = net(xy_f)
    lapl, _, _ = laplacian_net(u, xy_f)
    r_f = -lapl - f_source_torch(xy_f)
    l_pde = torch.mean(r_f ** 2)

    # 2. Dirichlet : u = 0 sur 3 bords
    l_right = torch.mean(net(xy_right) ** 2)
    l_bottom = torch.mean(net(xy_bottom) ** 2)
    l_top = torch.mean(net(xy_top) ** 2)
    l_D = l_right + l_bottom + l_top

    # 3. BC integrale sur bord gauche : u(0, y_j) = alpha * int_0^1 u(x, y_j) dx + g0
    n_y = len(y_left)
    n_qx = len(x_quad)

    # u(0, y_j)
    xy_0 = torch.stack([torch.zeros(n_y, dtype=DTYPE, device=DEVICE), y_left], dim=1)
    u_0 = net(xy_0)  # (n_y, 1)

    # int_0^1 u(x, y_j) dx pour chaque y_j
    y_rep = y_left.repeat_interleave(n_qx)       # (n_y*n_qx,)
    x_rep = x_quad.squeeze().repeat(n_y)          # (n_y*n_qx,)
    xy_quad = torch.stack([x_rep, y_rep], dim=1)  # (n_y*n_qx, 2)

    u_quad = net(xy_quad).reshape(n_y, n_qx)     # (n_y, n_qx)
    integrals = torch.sum(w_quad.squeeze() * u_quad, dim=1, keepdim=True)  # (n_y, 1)

    r_I = u_0 - ALPHA * integrals - G0
    l_I = torch.mean(r_I ** 2)

    loss = cfg["w_f"] * l_pde + cfg["w_d"] * l_D + cfg["w_i"] * l_I
    return loss, l_pde, l_D, l_I


def train_pinn(config=None, verbose=True):
    """
    Entraine un PINN 2D et retourne les resultats.

    Returns
    -------
    dict avec cles : 'net', 'history', 'errors', 'constraint_error', 'time', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    net = MLP(2, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE).to(DEVICE)
    xy_f, xy_right, xy_bottom, xy_top, y_left, x_quad, w_quad = _make_points(cfg)

    hist = {"loss": [], "loss_pde": [], "loss_D": [], "loss_I": []}

    if verbose:
        n_params = sum(p.numel() for p in net.parameters())
        nf = cfg["n_f_per_dim"]
        print(f"  PINN 2D | {cfg['n_hidden']}x{cfg['n_layers']} ({n_params} params) | "
              f"N_f={nf}x{nf}={nf**2} | seed={cfg['seed']}")

    t0 = time.time()

    # Phase Adam
    opt = optim.Adam(net.parameters(), lr=cfg["lr_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad()
        loss, lp, ld, li = _loss_pinn(
            net, xy_f, xy_right, xy_bottom, xy_top, y_left, x_quad, w_quad, cfg
        )
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        if verbose and (ep + 1) % (cfg["n_adam"] // 3) == 0:
            print(f"    Adam {ep+1:6d} | L={loss.item():.3e} PDE={lp.item():.3e} "
                  f"D={ld.item():.3e} I={li.item():.3e}")

    # Phase L-BFGS
    opt2 = optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                       history_size=50, line_search_fn="strong_wolfe")
    it = [0]

    def closure():
        opt2.zero_grad()
        loss, lp, ld, li = _loss_pinn(
            net, xy_f, xy_right, xy_bottom, xy_top, y_left, x_quad, w_quad, cfg
        )
        loss.backward()
        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        it[0] += 1
        if verbose and it[0] % (max(1, cfg["n_lbfgs"] // 3)) == 0:
            print(f"    LBFGS {it[0]:5d} | L={loss.item():.3e}")
        return loss

    for _ in range(cfg["n_lbfgs"]):
        opt2.step(closure)
        if it[0] >= cfg["n_lbfgs"]:
            break

    elapsed = time.time() - t0

    # Erreurs
    errors = compute_errors_2d(net, u_exact_np, u_exact_dx_np, u_exact_dy_np)

    # Erreur contrainte integrale (max sur y)
    with torch.no_grad():
        y_test = torch.linspace(0.01, 0.99, 50, dtype=DTYPE, device=DEVICE)
        n_y = len(y_test)
        n_qx = len(x_quad)
        xy_0 = torch.stack([torch.zeros(n_y, dtype=DTYPE, device=DEVICE), y_test], dim=1)
        u_0 = net(xy_0).squeeze()

        y_rep = y_test.repeat_interleave(n_qx)
        x_rep = x_quad.squeeze().repeat(n_y)
        xy_q = torch.stack([x_rep, y_rep], dim=1)
        u_q = net(xy_q).reshape(n_y, n_qx)
        ints = torch.sum(w_quad.squeeze() * u_q, dim=1)

        cstr = torch.max(torch.abs(u_0 - ALPHA * ints - G0)).item()

    if verbose:
        print(f"    => L2={errors['L2']:.3e} Linf={errors['Linf']:.3e} "
              f"H1={errors.get('H1', float('nan')):.3e} Cstr={cstr:.3e} "
              f"({elapsed:.1f}s)")

    return {
        "net": net, "history": hist, "errors": errors,
        "constraint_error": cstr, "time": elapsed, "config": cfg,
    }
