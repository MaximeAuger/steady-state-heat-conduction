"""
VPINN solver for the nonlinear steady-state heat equation (weak form).

PDE:  -[k(u) u'(x)]' = f(x),   x in (0, 1)
      k(u) = 1 + beta * u^2

Weak formulation (Petrov-Galerkin):
    Find (u_theta, lambda) such that for every test function v_k in V:

        int_0^1 k(u_theta) u_theta' v_k' dx  -  lambda v_k(0)
            =  int_0^1 f(x) v_k(x) dx

    with  V = { v in H^1(0,1) : v(1) = 0 }
    and   lambda = k(u(0)) u'(0)   (boundary flux, Lagrange multiplier)

Test functions: v_k(x) = P_k(2x-1) - 1  (shifted Legendre polynomials).

Key difference from the linear case: the integrand k(u)*u'*v' is nonlinear
in u_theta and must be re-evaluated at every training step (not pre-computable).
However, int f*v_k is still pre-computable since f depends only on x.

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import time
import numpy as np
import torch
import torch.optim as optim

from exact_solution import (
    ALPHA, G0, G1, DTYPE, BETA_DEFAULT,
    u_exact, u_exact_deriv, f_source, integral_u_exact,
    k_conductivity,
)
from network import MLP
from utils import gauss_legendre, LegendreTestFunctions, diff, compute_errors


# ===================================================================
# Default configuration
# ===================================================================

DEFAULT_CONFIG = {
    "n_hidden": 32,
    "n_layers": 4,
    "n_test": 20,          # number of Legendre test functions
    "n_quad": 80,          # quadrature points for weak residuals
    "n_q_global": 64,      # quadrature points for integral BC
    "w_w": 20.0,           # weak residual weight
    "w_d": 100.0,          # Dirichlet BC weight
    "w_i": 100.0,          # integral BC weight
    "lr_adam": 1e-3,
    "n_adam": 15_000,
    "n_lbfgs": 3_000,
    "seed": 42,
    "beta": BETA_DEFAULT,
}


# ===================================================================
# Pre-computed data (quantities that do not change during training)
# ===================================================================

class VPINNData:
    """Pre-computed arrays for the VPINN loss evaluation.

    Attributes
    ----------
    x_q : Tensor (N_quad, 1)   quadrature nodes
    w_q : Tensor (N_quad, 1)   quadrature weights
    dV  : Tensor (N_quad, n_test)  test function derivatives at quad nodes
    int_f_v : Tensor (n_test,) pre-computed int f(x) v_k(x) dx
    v_at_0  : Tensor (n_test,) v_k(0)
    """

    def __init__(self, cfg):
        n_quad = cfg["n_quad"]
        n_test = cfg["n_test"]
        beta = cfg["beta"]

        xq, wq = gauss_legendre(n_quad)
        tf = LegendreTestFunctions(n_test)

        V = tf.eval_v(xq)        # (N_quad, n_test)
        dV = tf.eval_dv(xq)      # (N_quad, n_test)
        v0 = tf.v_at_zero()      # (n_test,)

        # Pre-compute int f(x) v_k(x) dx
        f_vals = f_source(xq, beta=beta)                  # (N_quad,)
        int_f_v = np.sum(wq[:, None] * f_vals[:, None] * V, axis=0)  # (n_test,)

        self.x_q = torch.tensor(xq.reshape(-1, 1), dtype=DTYPE)
        self.w_q = torch.tensor(wq.reshape(-1, 1), dtype=DTYPE)
        self.dV = torch.tensor(dV, dtype=DTYPE)
        self.int_f_v = torch.tensor(int_f_v, dtype=DTYPE)
        self.v_at_0 = torch.tensor(v0, dtype=DTYPE)


# ===================================================================
# Loss function
# ===================================================================

def _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg):
    """Compute the VPINN loss (weak-form residual).

    The nonlinear term k(u)*u' is computed at each evaluation because
    it depends on the current network prediction.
    """
    beta = cfg["beta"]

    # --- Weak residuals ---
    x_q = vd.x_q.clone().requires_grad_(True)
    u_q = net(x_q)                                      # (N_quad, 1)
    du_q = diff(u_q, x_q, order=1)                      # u' at quad points

    # Nonlinear flux: k(u) * u'
    k_u = 1.0 + beta * u_q ** 2                          # (N_quad, 1)
    weighted_flux = vd.w_q * k_u * du_q                   # w_i * k(u_i) * u'_i

    # Vectorised integration: int k(u)*u' * v_k' dx  for all k
    int_ku_dv = torch.sum(weighted_flux * vd.dV, dim=0)   # (n_test,)

    # Weak residual: R_k = int k(u)*u'*v_k' dx - lambda*v_k(0) - int f*v_k dx
    R = int_ku_dv - lam * vd.v_at_0 - vd.int_f_v         # (n_test,)
    l_w = torch.mean(R ** 2)

    # --- Dirichlet BC: u(1) = g1 ---
    r_D = net(x_1) - G1
    l_D = (r_D ** 2).squeeze()

    # --- Integral BC: u(0) = alpha * int u dx + g0 ---
    u_0 = net(x_0)
    integral = torch.sum(w_qg * net(x_qg))
    r_I = u_0.squeeze() - ALPHA * integral - G0
    l_I = r_I ** 2

    loss = cfg["w_w"] * l_w + cfg["w_d"] * l_D + cfg["w_i"] * l_I

    return loss, l_w.item(), l_D.item(), l_I.item()


# ===================================================================
# Training procedure
# ===================================================================

def train_vpinn(config=None, verbose=True):
    """Train a VPINN on the nonlinear heat equation.

    Parameters
    ----------
    config : dict or None
    verbose : bool

    Returns
    -------
    dict with keys: 'net', 'lam', 'history', 'errors',
                    'constraint_error', 'time', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    beta = cfg["beta"]
    if verbose:
        print(f"[VPINN] Starting training (beta={beta})")

    # --- Network + Lagrange multiplier ---
    net = MLP(1, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE)
    lam = torch.nn.Parameter(torch.tensor([0.0], dtype=DTYPE))

    # --- Pre-computed data ---
    vd = VPINNData(cfg)

    # --- Boundary / integral quadrature ---
    x_0 = torch.tensor([[0.0]], dtype=DTYPE)
    x_1 = torch.tensor([[1.0]], dtype=DTYPE)
    xqg_np, wqg_np = gauss_legendre(cfg["n_q_global"])
    x_qg = torch.tensor(xqg_np.reshape(-1, 1), dtype=DTYPE)
    w_qg = torch.tensor(wqg_np.reshape(-1, 1), dtype=DTYPE)

    # --- History ---
    hist = {"loss": [], "loss_weak": [], "loss_D": [], "loss_I": []}

    all_params = list(net.parameters()) + [lam]

    # ============================
    # Phase 1: Adam
    # ============================
    optimizer = optim.Adam(all_params, lr=cfg["lr_adam"])
    t0 = time.time()

    for epoch in range(cfg["n_adam"]):
        optimizer.zero_grad()
        loss, lw, ld, li = _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg)
        loss.backward()
        optimizer.step()

        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw)
        hist["loss_D"].append(ld)
        hist["loss_I"].append(li)

        if verbose and (epoch + 1) % 5000 == 0:
            print(f"  Adam  {epoch+1:6d}/{cfg['n_adam']}  "
                  f"loss={loss.item():.3e}  weak={lw:.3e}  "
                  f"D={ld:.3e}  I={li:.3e}  lam={lam.item():.4f}")

    # ============================
    # Phase 2: L-BFGS
    # ============================
    optimizer2 = optim.LBFGS(
        all_params, lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    lbfgs_step = [0]

    def closure():
        optimizer2.zero_grad()
        loss, lw, ld, li = _loss_vpinn(net, lam, vd, x_0, x_1, x_qg, w_qg, cfg)
        loss.backward()

        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw)
        hist["loss_D"].append(ld)
        hist["loss_I"].append(li)

        lbfgs_step[0] += 1
        if verbose and lbfgs_step[0] % 500 == 0:
            print(f"  LBFGS {lbfgs_step[0]:6d}/{cfg['n_lbfgs']}  "
                  f"loss={loss.item():.3e}  weak={lw:.3e}  "
                  f"D={ld:.3e}  I={li:.3e}  lam={lam.item():.4f}")
        return loss

    for _ in range(cfg["n_lbfgs"]):
        optimizer2.step(closure)

    elapsed = time.time() - t0

    # --- Errors ---
    errors = compute_errors(net, u_exact, u_exact_deriv)

    # --- Integral constraint satisfaction ---
    with torch.no_grad():
        u0_val = net(x_0).item()
        int_val = torch.sum(w_qg * net(x_qg)).item()
    cstr = abs(u0_val - ALPHA * int_val - G0)

    # --- Report lambda ---
    # The weak form residual uses: R = int k(u)u'v' - lam*v(0) - int f*v
    # Comparing with the IBP result: int k(u)u'v' + k(u(0))u'(0)v(0) = int f*v
    # we get:  lam_code = -k(u(0))*u'(0)   (sign from the integration by parts)
    u0_exact = u_exact(np.array([0.0]))[0]
    du0_exact = u_exact_deriv(np.array([0.0]))[0]
    k0_exact = k_conductivity(u0_exact, beta)
    lam_exact = -k0_exact * du0_exact  # negative sign from IBP convention

    if verbose:
        print(f"[VPINN] Done in {elapsed:.1f}s")
        print(f"  L2 = {errors['L2']:.2e}   Linf = {errors['Linf']:.2e}")
        if "H1" in errors:
            print(f"  H1 = {errors['H1']:.2e}")
        print(f"  Integral BC error = {cstr:.2e}")
        print(f"  lambda = {lam.item():.6f}  (exact: {lam_exact:.6f})")

    return {
        "net": net,
        "lam": lam.item(),
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
    from exact_solution import k_conductivity
    res = train_vpinn({"n_adam": 2000, "n_lbfgs": 500}, verbose=True)
    print(f"\nFinal L2 error: {res['errors']['L2']:.2e}")
