"""
VPINN solver 2D — formulation faible (Legendre x sinus), interface reutilisable.

Forme faible :
    int int [u_x * v_i' * w_j + u_y * v_i * w_j'] dxdy
    - v_i(0) * lam_j / 2
    - int int f * v_i * w_j dxdy = 0

Fonctions test :
    phi_{ij}(x,y) = v_i(x) * w_j(y)
    v_i(x) = P_i(2x-1) - 1  (Legendre, s'annule en x=1)
    w_j(y) = sin(j*pi*y)     (s'annule en y=0 et y=1)

Lambda :
    lam(y) = sum_j lam_j * sin(j*pi*y)  (Fourier-sinus)
    Decouplage par orthogonalite des sinus.

    train_vpinn(config, verbose) -> dict de resultats
"""

import time
import numpy as np
import torch
import torch.optim as optim

from network import MLP
from exact_solution import (
    ALPHA, G0, DTYPE, lambda_exact,
    f_source_np, u_exact_np, u_exact_dx_np, u_exact_dy_np,
)
from utils import (
    gauss_legendre, gauss_legendre_torch,
    LegendreTestFunctions, SineTestFunctions,
    grad_net, compute_errors_2d, DEVICE,
)

DEFAULT_CONFIG = {
    "n_hidden": 64,
    "n_layers": 5,
    "n_test_x": 15,         # fonctions de Legendre en x
    "n_test_y": 5,           # fonctions sinus en y
    "n_quad_x": 40,          # quadrature en x pour la forme faible
    "n_quad_y": 30,          # quadrature en y pour la forme faible
    "n_q_bc_x": 64,          # quadrature GL en x pour la BC integrale
    "n_q_bc_y": 50,          # points y sur le bord gauche
    "n_b": 100,              # points par bord Dirichlet
    "w_w": 20.0,             # poids residu faible
    "w_d": 100.0,            # poids Dirichlet
    "w_i": 100.0,            # poids BC integrale
    "lr_adam": 1e-3,
    "n_adam": 20000,
    "n_lbfgs": 5000,
    "seed": 42,
}


class VPINNData2D:
    """Pre-calcule quadrature, fonctions de test et integrales source pour le VPINN 2D."""

    def __init__(self, cfg):
        n_test_x = cfg["n_test_x"]
        n_test_y = cfg["n_test_y"]
        n_quad_x = cfg["n_quad_x"]
        n_quad_y = cfg["n_quad_y"]

        # Direction x : Legendre
        tf_x = LegendreTestFunctions(n_test_x)
        xq, wxq = gauss_legendre(n_quad_x, 0.0, 1.0)
        V_x = tf_x.eval_v(xq)        # (n_quad_x, n_test_x)
        dV_x = tf_x.eval_dv(xq)      # (n_quad_x, n_test_x)
        v_at_0 = tf_x.v_at_zero()    # (n_test_x,)

        # Direction y : Sinus
        tf_y = SineTestFunctions(n_test_y)
        yq, wyq = gauss_legendre(n_quad_y, 0.0, 1.0)
        W_y = tf_y.eval_w(yq)        # (n_quad_y, n_test_y)
        dW_y = tf_y.eval_dw(yq)      # (n_quad_y, n_test_y)

        # Grille 2D de quadrature
        xx, yy = np.meshgrid(xq, yq, indexing='ij')  # (n_quad_x, n_quad_y)
        xy_flat = np.column_stack([xx.ravel(), yy.ravel()])  # (NQ, 2)

        # Poids 2D
        w2d_np = np.outer(wxq, wyq)  # (n_quad_x, n_quad_y)

        # Pre-calcul int int f * v_i * w_j dxdy
        f_vals = f_source_np(xx, yy)  # (n_quad_x, n_quad_y)
        int_f_vw = np.einsum('pq,pi,qj,pq->ij', w2d_np, V_x, W_y, f_vals)
        # (n_test_x, n_test_y)

        # Stocker en tensors
        self.n_quad_x = n_quad_x
        self.n_quad_y = n_quad_y
        self.n_test_x = n_test_x
        self.n_test_y = n_test_y

        self.xy_quad = torch.tensor(xy_flat, dtype=DTYPE, device=DEVICE)
        self.wxq = torch.tensor(wxq, dtype=DTYPE, device=DEVICE)
        self.wyq = torch.tensor(wyq, dtype=DTYPE, device=DEVICE)
        self.V_x = torch.tensor(V_x, dtype=DTYPE, device=DEVICE)
        self.dV_x = torch.tensor(dV_x, dtype=DTYPE, device=DEVICE)
        self.W_y = torch.tensor(W_y, dtype=DTYPE, device=DEVICE)
        self.dW_y = torch.tensor(dW_y, dtype=DTYPE, device=DEVICE)
        self.v_at_0 = torch.tensor(v_at_0, dtype=DTYPE, device=DEVICE)
        self.int_f_vw = torch.tensor(int_f_vw, dtype=DTYPE, device=DEVICE)


def _loss_vpinn(net, lam, vd, xy_right, xy_bottom, xy_top,
                y_left, x_quad_bc, w_quad_bc, cfg):
    """Calcule la perte VPINN 2D (vectorisee)."""
    # --- Residus faibles ---
    xy_q = vd.xy_quad.clone().detach().requires_grad_(True)
    u_q = net(xy_q)  # (NQ, 1)

    # Gradients : du/dx, du/dy
    u_x, u_y = grad_net(u_q, xy_q)  # chacun (NQ, 1)

    # Reshape en grille 2D
    ux_2d = u_x.reshape(vd.n_quad_x, vd.n_quad_y)  # (p, q)
    uy_2d = u_y.reshape(vd.n_quad_x, vd.n_quad_y)

    # Poids 2D
    w2d = vd.wxq[:, None] * vd.wyq[None, :]  # (n_quad_x, n_quad_y)

    # Terme 1 : int int u_x * v_i'(x) * w_j(y) dxdy
    term1 = torch.einsum('pq,pq,pi,qj->ij', w2d, ux_2d, vd.dV_x, vd.W_y)

    # Terme 2 : int int u_y * v_i(x) * w_j'(y) dxdy
    term2 = torch.einsum('pq,pq,pi,qj->ij', w2d, uy_2d, vd.V_x, vd.dW_y)

    # Terme de bord : -v_i(0) * lam_j / 2
    boundary = vd.v_at_0[:, None] * lam[None, :] / 2.0  # (n_test_x, n_test_y)

    # Residu complet
    R = term1 + term2 - boundary - vd.int_f_vw  # (n_test_x, n_test_y)
    l_w = torch.mean(R ** 2)

    # --- Dirichlet : u = 0 sur 3 bords ---
    l_right = torch.mean(net(xy_right) ** 2)
    l_bottom = torch.mean(net(xy_bottom) ** 2)
    l_top = torch.mean(net(xy_top) ** 2)
    l_D = l_right + l_bottom + l_top

    # --- BC integrale sur bord gauche ---
    n_y = len(y_left)
    n_qx = len(x_quad_bc)

    xy_0 = torch.stack([torch.zeros(n_y, dtype=DTYPE, device=DEVICE), y_left], dim=1)
    u_0 = net(xy_0)  # (n_y, 1)

    y_rep = y_left.repeat_interleave(n_qx)
    x_rep = x_quad_bc.squeeze().repeat(n_y)
    xy_bc = torch.stack([x_rep, y_rep], dim=1)

    u_bc = net(xy_bc).reshape(n_y, n_qx)
    integrals = torch.sum(w_quad_bc.squeeze() * u_bc, dim=1, keepdim=True)

    r_I = u_0 - ALPHA * integrals - G0
    l_I = torch.mean(r_I ** 2)

    loss = cfg["w_w"] * l_w + cfg["w_d"] * l_D + cfg["w_i"] * l_I
    return loss, l_w, l_D, l_I


def train_vpinn(config=None, verbose=True):
    """
    Entraine un VPINN 2D et retourne les resultats.

    Returns
    -------
    dict avec cles : 'net', 'lam', 'history', 'errors', 'constraint_error', 'time', 'config'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    net = MLP(2, 1, cfg["n_hidden"], cfg["n_layers"]).to(DTYPE).to(DEVICE)
    lam = torch.nn.Parameter(torch.zeros(cfg["n_test_y"], dtype=DTYPE, device=DEVICE))

    vd = VPINNData2D(cfg)

    # Points de bord Dirichlet
    nb = cfg["n_b"]
    t = torch.linspace(0, 1, nb, dtype=DTYPE, device=DEVICE)
    xy_right = torch.stack([torch.ones(nb, dtype=DTYPE, device=DEVICE), t], dim=1)
    xy_bottom = torch.stack([t, torch.zeros(nb, dtype=DTYPE, device=DEVICE)], dim=1)
    xy_top = torch.stack([t, torch.ones(nb, dtype=DTYPE, device=DEVICE)], dim=1)

    # Bord gauche pour BC integrale
    y_left = torch.linspace(0.01, 0.99, cfg["n_q_bc_y"], dtype=DTYPE, device=DEVICE)
    x_quad_bc, w_quad_bc = gauss_legendre_torch(cfg["n_q_bc_x"])

    all_params = list(net.parameters()) + [lam]
    hist = {"loss": [], "loss_weak": [], "loss_D": [], "loss_I": []}

    if verbose:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  VPINN 2D | {cfg['n_hidden']}x{cfg['n_layers']} ({n_params} params) | "
              f"N_test={cfg['n_test_x']}x{cfg['n_test_y']} "
              f"N_quad={cfg['n_quad_x']}x{cfg['n_quad_y']} | seed={cfg['seed']}")

    t0 = time.time()

    # Phase Adam
    opt = optim.Adam(all_params, lr=cfg["lr_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad()
        loss, lw, ld, li = _loss_vpinn(
            net, lam, vd, xy_right, xy_bottom, xy_top,
            y_left, x_quad_bc, w_quad_bc, cfg
        )
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw.item())
        hist["loss_D"].append(ld.item())
        hist["loss_I"].append(li.item())
        if verbose and (ep + 1) % (cfg["n_adam"] // 3) == 0:
            lam_str = f"lam0={lam[0].item():.3f}"
            print(f"    Adam {ep+1:6d} | L={loss.item():.3e} W={lw.item():.3e} "
                  f"D={ld.item():.3e} I={li.item():.3e} {lam_str}")

    # Phase L-BFGS
    opt2 = optim.LBFGS(all_params, lr=1.0, max_iter=20,
                       history_size=50, line_search_fn="strong_wolfe")
    it = [0]

    def closure():
        opt2.zero_grad()
        loss, lw, ld, li = _loss_vpinn(
            net, lam, vd, xy_right, xy_bottom, xy_top,
            y_left, x_quad_bc, w_quad_bc, cfg
        )
        loss.backward()
        hist["loss"].append(loss.item())
        hist["loss_weak"].append(lw.item())
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
        n_qx = len(x_quad_bc)
        xy_0 = torch.stack([torch.zeros(n_y, dtype=DTYPE, device=DEVICE), y_test], dim=1)
        u_0 = net(xy_0).squeeze()

        y_rep = y_test.repeat_interleave(n_qx)
        x_rep = x_quad_bc.squeeze().repeat(n_y)
        xy_q = torch.stack([x_rep, y_rep], dim=1)
        u_q = net(xy_q).reshape(n_y, n_qx)
        ints = torch.sum(w_quad_bc.squeeze() * u_q, dim=1)

        cstr = torch.max(torch.abs(u_0 - ALPHA * ints - G0)).item()

    lam_ex = lambda_exact()
    lam_np = lam.detach().cpu().numpy()

    if verbose:
        print(f"    => L2={errors['L2']:.3e} Linf={errors['Linf']:.3e} "
              f"H1={errors.get('H1', float('nan')):.3e} Cstr={cstr:.3e} "
              f"lam0={lam_np[0]:.4f} (exact={lam_ex:.4f}) ({elapsed:.1f}s)")

    return {
        "net": net, "lam": lam_np.copy(), "history": hist, "errors": errors,
        "constraint_error": cstr, "time": elapsed, "config": cfg,
    }
