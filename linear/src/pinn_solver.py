"""
PINN solver — strong (collocation) formulation.

Solves the 1D steady-state heat conduction problem with a nonlocal integral
boundary condition using a Physics-Informed Neural Network.

The PDE residual is enforced point-wise at interior collocation points:

    r_f(x_i) = -u''_theta(x_i) - f(x_i)

The integral boundary condition is approximated by Gauss-Legendre quadrature:

    r_I = u_theta(0) - alpha * int_0^1 u_theta(x) dx - g0

The total loss is a weighted sum of the mean-squared PDE residual and the
squared boundary residuals:

    L = w_f * MSE(r_f) + w_d * r_D^2 + w_i * r_I^2

Two-phase optimisation strategy: Adam (first-order) followed by L-BFGS
(quasi-Newton) for fine convergence.

References
----------
    - Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
      neural networks. J. Comput. Phys., 378, 686-707.
    - Liu, D.C. & Nocedal, J. (1989). On the limited memory BFGS method for
      large scale optimization. Math. Program., 45, 503-528.

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import time
import numpy as np
import torch
import torch.optim as optim

from network import MLP
from exact_solution import ALPHA, G0, G1, f_source_torch, u_exact_np, u_exact_deriv_np
from utils import gauss_legendre_torch, diff, compute_errors, DTYPE, DEVICE


# -------------------------------------------------------------------
# Default hyper-parameters
# -------------------------------------------------------------------
DEFAULT_CONFIG = {
    "n_hidden": 32,         # neurons per hidden layer
    "n_layers": 4,          # number of hidden layers
    "n_f": 100,             # interior collocation points
    "n_q": 64,              # quadrature points for the integral BC
    "w_f": 1.0,             # weight — PDE residual
    "w_d": 50.0,            # weight — Dirichlet BC
    "w_i": 50.0,            # weight — integral BC
    "lr_adam": 1e-3,        # Adam learning rate
    "n_adam": 10_000,       # Adam iterations
    "n_lbfgs": 2_000,       # L-BFGS iterations
    "seed": 42,             # random seed
}


# -------------------------------------------------------------------
# Loss function
# -------------------------------------------------------------------
def _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg):
    """Compute the composite PINN loss.

    Returns
    -------
    loss_total, loss_pde, loss_dirichlet, loss_integral : Tensor
    """
    # PDE residual  -u'' - f = 0
    u = net(x_f)
    u_xx = diff(u, x_f, order=2)
    r_f = -u_xx - f_source_torch(x_f)
    l_pde = torch.mean(r_f ** 2)

    # Dirichlet BC  u(1) = g1
    r_D = net(x_1) - G1
    l_D = (r_D ** 2).squeeze()

    # Integral BC  u(0) = alpha * int u + g0
    u_0 = net(x_0)
    integral = torch.sum(w_q * net(x_q))
    r_I = (u_0 - ALPHA * integral - G0)
    l_I = (r_I ** 2).squeeze()

    loss = cfg["w_f"] * l_pde + cfg["w_d"] * l_D + cfg["w_i"] * l_I
    return loss, l_pde, l_D, l_I


# -------------------------------------------------------------------
# Training routine
# -------------------------------------------------------------------
def train_pinn(config=None, verbose=True):
    """Train a PINN and return the results.

    Parameters
    ----------
    config : dict or None
        Override any key in ``DEFAULT_CONFIG``.
    verbose : bool
        Print training progress to stdout.

    Returns
    -------
    dict
        'net'              — trained ``MLP`` instance
        'history'          — dict of loss lists
        'errors'           — dict of error metrics (L2, Linf, H1, ...)
        'constraint_error' — absolute violation of the integral BC
        'time'             — wall-clock training time (seconds)
        'config'           — effective configuration used
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Network
    net = MLP(1, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE).to(DEVICE)

    # Collocation and quadrature points
    x_f = torch.linspace(0.01, 0.99, cfg["n_f"], dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    x_f.requires_grad_(True)
    x_0 = torch.tensor([[0.0]], dtype=DTYPE, device=DEVICE)
    x_1 = torch.tensor([[1.0]], dtype=DTYPE, device=DEVICE)
    x_q, w_q = gauss_legendre_torch(cfg["n_q"])

    hist = {"loss": [], "loss_pde": [], "loss_D": [], "loss_I": []}

    if verbose:
        n_par = sum(p.numel() for p in net.parameters())
        print(f"  PINN | {cfg['n_hidden']}x{cfg['n_layers']} ({n_par} params) "
              f"| N_f={cfg['n_f']} | seed={cfg['seed']}")

    t0 = time.time()

    # Phase 1 — Adam
    opt = optim.Adam(net.parameters(), lr=cfg["lr_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad()
        loss, lp, ld, li = _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        if verbose and (ep + 1) % (cfg["n_adam"] // 3) == 0:
            print(f"    Adam {ep+1:6d} | L={loss.item():.3e}  "
                  f"PDE={lp.item():.3e}  D={ld.item():.3e}  I={li.item():.3e}")

    # Phase 2 — L-BFGS
    opt2 = optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    it = [0]

    def closure():
        opt2.zero_grad()
        loss, lp, ld, li = _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg)
        loss.backward()
        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        it[0] += 1
        if verbose and it[0] % max(1, cfg["n_lbfgs"] // 3) == 0:
            print(f"    LBFGS {it[0]:5d} | L={loss.item():.3e}")
        return loss

    for _ in range(cfg["n_lbfgs"]):
        opt2.step(closure)
        if it[0] >= cfg["n_lbfgs"]:
            break

    elapsed = time.time() - t0

    # Error metrics
    errors = compute_errors(net, u_exact_np, u_exact_deriv_np)
    with torch.no_grad():
        u0 = net(x_0).item()
        int_u = torch.sum(w_q * net(x_q)).item()
    cstr = abs(u0 - ALPHA * int_u - G0)

    if verbose:
        print(f"    => L2={errors['L2']:.3e}  Linf={errors['Linf']:.3e}  "
              f"H1={errors.get('H1', float('nan')):.3e}  "
              f"Cstr={cstr:.3e}  ({elapsed:.1f}s)")

    return {
        "net": net, "history": hist, "errors": errors,
        "constraint_error": cstr, "time": elapsed, "config": cfg,
    }


# -------------------------------------------------------------------
# Standalone execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    results = train_pinn(verbose=True)
    print(f"\nFinal L2 error: {results['errors']['L2']:.4e}")
