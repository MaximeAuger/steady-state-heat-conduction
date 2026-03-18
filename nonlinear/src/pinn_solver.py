"""
PINN solver for the nonlinear steady-state heat equation (strong form).

PDE:  -[k(u) u'(x)]' = f(x),   x in (0, 1)
      k(u) = 1 + beta * u^2

The PDE residual is computed via automatic differentiation on the flux:
    flux = k(u_theta) * u_theta'
    residual = -flux' - f(x)

This avoids manually expanding the chain rule and only requires first-order
AD applied twice (once for u', once for flux').

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import time
import numpy as np
import torch
import torch.optim as optim

from exact_solution import (
    ALPHA, G0, G1, DTYPE, BETA_DEFAULT,
    u_exact, u_exact_deriv, f_source_torch,
    k_conductivity, integral_u_exact,
)
from network import MLP
from utils import gauss_legendre, diff, compute_errors


# ===================================================================
# Default configuration
# ===================================================================

DEFAULT_CONFIG = {
    "n_hidden": 32,
    "n_layers": 4,
    "n_f": 200,          # interior collocation points
    "n_q": 64,           # quadrature points for integral BC
    "w_f": 1.0,          # PDE residual weight
    "w_d": 50.0,         # Dirichlet BC weight
    "w_i": 50.0,         # integral BC weight
    "lr_adam": 1e-3,
    "n_adam": 15_000,
    "n_lbfgs": 3_000,
    "seed": 42,
    "beta": BETA_DEFAULT, # nonlinearity strength
}


# ===================================================================
# Loss function
# ===================================================================

def _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg):
    """Compute the total PINN loss (strong-form residual).

    The key innovation for the nonlinear case is computing the residual
    via AD on the flux  phi = k(u)*u'  rather than expanding the chain
    rule manually.
    """
    beta = cfg["beta"]

    # --- PDE residual via flux differentiation ---
    u_f = net(x_f)                                      # (N_f, 1)
    u_x = diff(u_f, x_f, order=1)                       # u'
    flux = (1.0 + beta * u_f ** 2) * u_x                # k(u) * u'
    flux_x = diff(flux, x_f, order=1)                   # [k(u) u']'
    r_f = -flux_x - f_source_torch(x_f, beta=beta)      # residual
    l_pde = torch.mean(r_f ** 2)

    # --- Dirichlet BC: u(1) = g1 ---
    r_D = net(x_1) - G1
    l_D = (r_D ** 2).squeeze()

    # --- Integral BC: u(0) = alpha * int u dx + g0 ---
    u_0 = net(x_0)
    integral = torch.sum(w_q * net(x_q))
    r_I = u_0.squeeze() - ALPHA * integral - G0
    l_I = r_I ** 2

    loss = cfg["w_f"] * l_pde + cfg["w_d"] * l_D + cfg["w_i"] * l_I

    return loss, l_pde.item(), l_D.item(), l_I.item()


# ===================================================================
# Training procedure
# ===================================================================

def train_pinn(config=None, verbose=True):
    """Train a PINN on the nonlinear heat equation.

    Parameters
    ----------
    config : dict or None
        Overrides for DEFAULT_CONFIG.
    verbose : bool

    Returns
    -------
    dict with keys: 'net', 'history', 'errors', 'constraint_error', 'time', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    beta = cfg["beta"]
    if verbose:
        print(f"[PINN] Starting training (beta={beta})")

    # --- Network ---
    net = MLP(1, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE)

    # --- Collocation points ---
    x_f = torch.linspace(0.01, 0.99, cfg["n_f"], dtype=DTYPE).reshape(-1, 1)
    x_f.requires_grad_(True)
    x_0 = torch.tensor([[0.0]], dtype=DTYPE)
    x_1 = torch.tensor([[1.0]], dtype=DTYPE)

    # --- Quadrature for integral BC ---
    xq_np, wq_np = gauss_legendre(cfg["n_q"])
    x_q = torch.tensor(xq_np.reshape(-1, 1), dtype=DTYPE)
    w_q = torch.tensor(wq_np.reshape(-1, 1), dtype=DTYPE)

    # --- History ---
    hist = {"loss": [], "loss_pde": [], "loss_D": [], "loss_I": []}

    # ============================
    # Phase 1: Adam
    # ============================
    optimizer = optim.Adam(net.parameters(), lr=cfg["lr_adam"])
    t0 = time.time()

    for epoch in range(cfg["n_adam"]):
        optimizer.zero_grad()
        loss, lp, ld, li = _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg)
        loss.backward()
        optimizer.step()

        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp)
        hist["loss_D"].append(ld)
        hist["loss_I"].append(li)

        if verbose and (epoch + 1) % 5000 == 0:
            print(f"  Adam  {epoch+1:6d}/{cfg['n_adam']}  "
                  f"loss={loss.item():.3e}  pde={lp:.3e}  D={ld:.3e}  I={li:.3e}")

    # ============================
    # Phase 2: L-BFGS
    # ============================
    optimizer2 = optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    lbfgs_step = [0]

    def closure():
        optimizer2.zero_grad()
        loss, lp, ld, li = _loss_pinn(net, x_f, x_0, x_1, x_q, w_q, cfg)
        loss.backward()

        hist["loss"].append(loss.item())
        hist["loss_pde"].append(lp)
        hist["loss_D"].append(ld)
        hist["loss_I"].append(li)

        lbfgs_step[0] += 1
        if verbose and lbfgs_step[0] % 500 == 0:
            print(f"  LBFGS {lbfgs_step[0]:6d}/{cfg['n_lbfgs']}  "
                  f"loss={loss.item():.3e}  pde={lp:.3e}  D={ld:.3e}  I={li:.3e}")
        return loss

    for _ in range(cfg["n_lbfgs"]):
        optimizer2.step(closure)

    elapsed = time.time() - t0

    # --- Errors ---
    errors = compute_errors(net, u_exact, u_exact_deriv)

    # --- Integral constraint satisfaction ---
    with torch.no_grad():
        u0_val = net(x_0).item()
        int_val = torch.sum(w_q * net(x_q)).item()
    cstr = abs(u0_val - ALPHA * int_val - G0)

    if verbose:
        print(f"[PINN] Done in {elapsed:.1f}s")
        print(f"  L2 = {errors['L2']:.2e}   Linf = {errors['Linf']:.2e}")
        if "H1" in errors:
            print(f"  H1 = {errors['H1']:.2e}")
        print(f"  Integral BC error = {cstr:.2e}")

    return {
        "net": net,
        "history": hist,
        "errors": errors,
        "constraint_error": cstr,
        "time": elapsed,
        "config": cfg,
    }


# ===================================================================
# Quick test
# ===================================================================
if __name__ == "__main__":
    res = train_pinn({"n_adam": 2000, "n_lbfgs": 500}, verbose=True)
    print(f"\nFinal L2 error: {res['errors']['L2']:.2e}")
