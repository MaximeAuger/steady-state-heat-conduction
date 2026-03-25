"""
VPINN solver — weak (variational / Petrov-Galerkin) formulation.

Solves the 1D steady-state heat conduction problem with a nonlocal integral
boundary condition using a Variational Physics-Informed Neural Network.

Weak formulation
----------------
Find (u, lambda) in V x R such that

    int_0^1 u'(x) v'(x) dx  -  lambda * v(0)  =  int_0^1 f(x) v(x) dx
    u(0) = alpha * int_0^1 u(x) dx + g0        (integral BC)
    u(1) = g1                                    (Dirichlet BC)

where V = { v in H^1(0,1) : v(1) = 0 } and lambda ~ u'(0) is a scalar
Lagrange multiplier introduced by integration by parts.

Test functions are global shifted Legendre polynomials:

    v_k(x) = P_k(2x - 1) - 1,   k = 1, ..., N_test

All integrals are computed via a single Gauss-Legendre quadrature,
allowing a vectorised evaluation of all weak residuals through a single
forward pass of the neural network.

References
----------
    - Kharazmi, E., Zhang, Z. & Karniadakis, G.E. (2019). Variational
      physics-informed neural networks for solving partial differential
      equations. arXiv:1912.00873.
    - Kharazmi, E., Zhang, Z. & Karniadakis, G.E. (2021). hp-VPINNs:
      variational physics-informed neural networks with domain
      decomposition. CMAME, 374, 113547.

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import time
import numpy as np
import torch
import torch.optim as optim

from network import MLP
from exact_solution import (
    ALPHA, G0, G1,
    f_source_np, u_exact_np, u_exact_deriv_np,
)
from utils import (
    gauss_legendre, gauss_legendre_torch,
    LegendreTestFunctions, diff, compute_errors,
    DTYPE, DEVICE,
)


# -------------------------------------------------------------------
# Default hyper-parameters
# -------------------------------------------------------------------
DEFAULT_CONFIG = {
    "n_hidden": 32,         # neurons per hidden layer
    "n_layers": 4,          # number of hidden layers
    "n_test": 20,           # number of Legendre test functions
    "n_quad": 60,           # quadrature points for weak residuals
    "n_q_global": 64,       # quadrature points for the integral BC
    "w_w": 20.0,            # weight — weak residuals
    "w_d": 100.0,           # weight — Dirichlet BC
    "w_i": 100.0,           # weight — integral BC
    "lr_adam": 1e-3,        # Adam learning rate
    "n_adam": 15_000,       # Adam iterations
    "n_lbfgs": 3_000,       # L-BFGS iterations
    "seed": 42,             # random seed
}


# -------------------------------------------------------------------
# Pre-computed quadrature and test-function data
# -------------------------------------------------------------------
class VPINNData:
    """Pre-compute and store quadrature data and test-function evaluations.

    Because the Legendre test functions are global on [0, 1], a single set
    of quadrature points suffices for all residuals.  This means only ONE
    forward pass through the network is needed per loss evaluation.

    Attributes
    ----------
    x_q : Tensor (N_quad, 1)
        Quadrature nodes.
    w_q : Tensor (N_quad, 1)
        Quadrature weights.
    dV : Tensor (N_quad, n_test)
        Derivatives of test functions at quadrature nodes.
    int_f_v : Tensor (n_test,)
        Pre-computed integrals  int_0^1 f(x) v_k(x) dx.
    v_at_0 : Tensor (n_test,)
        Test-function values at x = 0.
    """

    def __init__(self, n_test: int, n_quad: int):
        tf = LegendreTestFunctions(n_test)
        x_np, w_np = gauss_legendre(n_quad, 0.0, 1.0)

        V = tf.eval_v(x_np)        # (N_quad, n_test)
        dV = tf.eval_dv(x_np)      # (N_quad, n_test)
        f_vals = f_source_np(x_np)  # (N_quad,)

        self.n_test = n_test
        self.x_q = torch.tensor(x_np, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        self.w_q = torch.tensor(w_np, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        self.dV = torch.tensor(dV, dtype=DTYPE, device=DEVICE)
        self.int_f_v = torch.tensor(
            np.sum(w_np[:, None] * f_vals[:, None] * V, axis=0), dtype=DTYPE,
            device=DEVICE,
        )
        self.v_at_0 = torch.tensor(tf.v_at_zero(), dtype=DTYPE, device=DEVICE)


# -------------------------------------------------------------------
# Loss function
# -------------------------------------------------------------------
def _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg):
    """Compute the composite VPINN loss (vectorised).

    Returns
    -------
    loss_total, loss_weak, loss_dirichlet, loss_integral : Tensor
    """
    # Weak residuals — single forward pass
    x_q = vd.x_q.clone().detach().requires_grad_(True)
    u_q = net(x_q)                                      # (N_quad, 1)
    du_q = diff(u_q, x_q, order=1)                      # (N_quad, 1)

    # int u'(x) v_k'(x) dx  for each k  (matrix product)
    weighted_du = vd.w_q * du_q                          # (N_quad, 1)
    int_du_dv = torch.sum(weighted_du * vd.dV, dim=0)    # (n_test,)

    # R_k = int u' v_k' dx  -  lambda * v_k(0)  -  int f v_k dx
    R = int_du_dv - lam * vd.v_at_0 - vd.int_f_v
    l_w = torch.mean(R ** 2)

    # Dirichlet BC
    r_D = net(x_1) - G1
    l_D = (r_D ** 2).squeeze()

    # Integral BC
    u_0 = net(x_0)
    integral = torch.sum(w_qg * net(x_qg))
    r_I = (u_0 - ALPHA * integral - G0)
    l_I = (r_I ** 2).squeeze()

    loss = cfg["w_w"] * l_w + cfg["w_d"] * l_D + cfg["w_i"] * l_I
    return loss, l_w, l_D, l_I


# -------------------------------------------------------------------
# Training routine
# -------------------------------------------------------------------
def train_vpinn(config=None, verbose=True):
    """Train a VPINN and return the results.

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
        'lam'              — learned Lagrange multiplier (approx. u'(0))
        'history'          — dict of loss lists
        'errors'           — error metrics (L2, Linf, H1, ...)
        'constraint_error' — absolute violation of the integral BC
        'time'             — wall-clock training time (seconds)
        'config'           — effective configuration used
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Network + Lagrange multiplier
    net = MLP(1, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE).to(DEVICE)
    lam = torch.nn.Parameter(torch.tensor([0.0], dtype=DTYPE, device=DEVICE))

    # Pre-computed data
    vd = VPINNData(cfg["n_test"], cfg["n_quad"])
    x_0 = torch.tensor([[0.0]], dtype=DTYPE, device=DEVICE)
    x_1 = torch.tensor([[1.0]], dtype=DTYPE, device=DEVICE)
    x_qg, w_qg = gauss_legendre_torch(cfg["n_q_global"])

    all_params = list(net.parameters()) + [lam]
    hist = {"loss": [], "loss_weak": [], "loss_D": [], "loss_I": []}

    if verbose:
        n_par = sum(p.numel() for p in net.parameters())
        print(f"  VPINN | {cfg['n_hidden']}x{cfg['n_layers']} ({n_par} params) "
              f"| N_test={cfg['n_test']}  N_quad={cfg['n_quad']} "
              f"| seed={cfg['seed']}")

    t0 = time.time()

    # Phase 1 — Adam
    opt = optim.Adam(all_params, lr=cfg["lr_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad()
        loss, lw, ld, li = _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        if verbose and (ep + 1) % (cfg["n_adam"] // 3) == 0:
            print(f"    Adam {ep+1:6d} | L={loss.item():.3e}  "
                  f"W={lw.item():.3e}  D={ld.item():.3e}  I={li.item():.3e}  "
                  f"lam={lam.item():.3f}")

    # Phase 2 — L-BFGS
    opt2 = optim.LBFGS(
        all_params, lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    it = [0]

    def closure():
        opt2.zero_grad()
        loss, lw, ld, li = _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg)
        loss.backward()
        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw.item())
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
        int_u = torch.sum(w_qg * net(x_qg)).item()
    cstr = abs(u0 - ALPHA * int_u - G0)

    lam_exact = u_exact_deriv_np(np.array([0.0]))[0]
    if verbose:
        print(f"    => L2={errors['L2']:.3e}  Linf={errors['Linf']:.3e}  "
              f"H1={errors.get('H1', float('nan')):.3e}  Cstr={cstr:.3e}  "
              f"lam={lam.item():.4f} (exact={lam_exact:.4f})  ({elapsed:.1f}s)")

    return {
        "net": net, "lam": lam.item(), "history": hist, "errors": errors,
        "constraint_error": cstr, "time": elapsed, "config": cfg,
    }


# -------------------------------------------------------------------
# Standalone execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    results = train_vpinn(verbose=True)
    print(f"\nFinal L2 error: {results['errors']['L2']:.4e}")
