"""
========================================================================
  PINN vs VPINN — Conduction thermique 2D non-lineaire
  -div(k(u)*grad(u)) = f,  k(u) = 1 + beta*u^2
  Script complet : 5 etudes parametriques, 5 figures publication
========================================================================
"""

import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.8,
})

C_PINN = "#E74C3C"
C_VPINN = "#3498DB"
C_EXACT = "#2C3E50"
C_FDM = "#27AE60"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

from exact_solution import (
    verify, u_exact_np, f_source_np, ALPHA, G0, BETA_DEFAULT,
    lambda_exact, k_conductivity, dk_conductivity,
)
from utils import solve_fdm_2d_nonlinear, solve_fdm_2d_linear, DTYPE
from pinn_solver import train_pinn
from vpinn_solver import train_vpinn


# ============================================================
# ETUDE 1 : Comparaison principale (beta=1)
# ============================================================
def study_main_comparison():
    print("\n" + "=" * 70)
    print("  ETUDE 1 -- Comparaison principale PINN vs VPINN (beta=1)")
    print("=" * 70)
    res_pinn = train_pinn(verbose=True)
    res_vpinn = train_vpinn(verbose=True)

    # FDM reference
    print("  FDM reference (N=100)...")
    x1d, y1d, u_fdm = solve_fdm_2d_nonlinear(
        lambda x, y: f_source_np(x, y, beta=1.0),
        k_conductivity, dk_conductivity, ALPHA, G0, N=100, beta=1.0)
    xx, yy = np.meshgrid(x1d, y1d, indexing='ij')
    fdm_err = np.max(np.abs(u_fdm - u_exact_np(xx, yy)))
    print(f"  FDM N=100 Linf = {fdm_err:.4e}")

    return res_pinn, res_vpinn, {"x": x1d, "y": y1d, "u": u_fdm, "linf": fdm_err}


# ============================================================
# ETUDE 2 : Lineaire vs Non-lineaire
# ============================================================
def study_linear_vs_nonlinear():
    print("\n" + "=" * 70)
    print("  ETUDE 2 -- Lineaire (beta=0) vs Non-lineaire (beta=1)")
    print("=" * 70)

    cfg_base = {"n_adam": 10000, "n_lbfgs": 2000, "seed": 42}
    results = {}

    for beta in [0.0, 1.0]:
        label = f"beta={beta}"
        print(f"\n  --- {label} ---")
        cfg = {**cfg_base, "beta": beta}
        rp = train_pinn(config=cfg, verbose=True)
        rv = train_vpinn(config=cfg, verbose=True)
        results[beta] = {
            "pinn_L2": rp["errors"]["L2"], "vpinn_L2": rv["errors"]["L2"],
            "pinn_time": rp["time"], "vpinn_time": rv["time"],
            "pinn_hist": rp["history"]["loss"], "vpinn_hist": rv["history"]["loss"],
        }

    return results


# ============================================================
# ETUDE 3 : Force de non-linearite (beta sweep)
# ============================================================
def study_beta_sweep():
    print("\n" + "=" * 70)
    print("  ETUDE 3 -- Force de non-linearite (beta sweep)")
    print("=" * 70)

    betas = [0.0, 0.5, 1.0, 2.0, 5.0]
    cfg_base = {"n_adam": 10000, "n_lbfgs": 2000, "seed": 42}
    results = {}

    for beta in betas:
        print(f"\n  --- beta = {beta} ---")
        cfg = {**cfg_base, "beta": beta}
        rp = train_pinn(config=cfg, verbose=True)
        rv = train_vpinn(config=cfg, verbose=True)
        results[beta] = {
            "pinn_L2": rp["errors"]["L2"], "pinn_Linf": rp["errors"]["Linf"],
            "vpinn_L2": rv["errors"]["L2"], "vpinn_Linf": rv["errors"]["Linf"],
            "pinn_cstr": rp["constraint_error"], "vpinn_cstr": rv["constraint_error"],
            "vpinn_lam": rv["lam"],
        }

    return results


# ============================================================
# ETUDE 4 : Convergence VPINN vs N_test
# ============================================================
def study_ntest_convergence():
    print("\n" + "=" * 70)
    print("  ETUDE 4 -- Convergence VPINN vs N_test (beta=1)")
    print("=" * 70)

    ntest_pairs = [(3, 2), (5, 3), (10, 3), (15, 5), (20, 5), (25, 8)]
    cfg_base = {"n_adam": 10000, "n_lbfgs": 2000, "seed": 42, "beta": 1.0}
    results = {}

    for ntx, nty in ntest_pairs:
        label = f"{ntx}x{nty}"
        print(f"\n  --- N_test = {label} ---")
        cfg = {**cfg_base, "n_test_x": ntx, "n_test_y": nty,
               "n_quad_x": max(ntx + 10, 25), "n_quad_y": max(nty + 10, 15)}
        res = train_vpinn(config=cfg, verbose=True)
        results[label] = {
            "ntx": ntx, "nty": nty, "total": ntx * nty,
            "L2": res["errors"]["L2"], "Linf": res["errors"]["Linf"],
            "H1": res["errors"].get("H1", np.nan),
            "constraint": res["constraint_error"],
            "time": res["time"], "lam": res["lam"],
        }

    return results


# ============================================================
# ETUDE 5 : Robustesse multi-seeds (beta=1)
# ============================================================
def study_robustness():
    print("\n" + "=" * 70)
    print("  ETUDE 5 -- Robustesse (5 seeds, beta=1)")
    print("=" * 70)

    seeds = [42, 123, 256, 789, 1337]
    cfg_base = {"n_adam": 10000, "n_lbfgs": 2000, "beta": 1.0}
    pinn_results, vpinn_results = [], []

    for s in seeds:
        print(f"\n  --- Seed {s} ---")
        cfg = {**cfg_base, "seed": s}
        rp = train_pinn(config=cfg, verbose=False)
        rv = train_vpinn(config=cfg, verbose=False)
        pinn_results.append({
            "seed": s, "L2": rp["errors"]["L2"], "Linf": rp["errors"]["Linf"],
            "H1": rp["errors"].get("H1", np.nan),
            "constraint": rp["constraint_error"], "time": rp["time"],
        })
        vpinn_results.append({
            "seed": s, "L2": rv["errors"]["L2"], "Linf": rv["errors"]["Linf"],
            "H1": rv["errors"].get("H1", np.nan),
            "constraint": rv["constraint_error"], "time": rv["time"],
        })
        print(f"    PINN L2={rp['errors']['L2']:.3e}  VPINN L2={rv['errors']['L2']:.3e}")

    return pinn_results, vpinn_results


# ============================================================
# FIGURES
# ============================================================
def fig_main_comparison(res_pinn, res_vpinn, fdm_ref):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]
    hp = res_pinn["history"]
    hv = res_vpinn["history"]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    cf = ax.contourf(ep["xx"], ep["yy"], ep["u_exact"], levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("(a) Solution exacte")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 1])
    vmax = max(np.max(np.abs(ep["error"])), np.max(np.abs(ev["error"])))
    cf = ax.contourf(ep["xx"], ep["yy"], np.abs(ep["error"]), levels=30, cmap="hot_r", vmin=0, vmax=vmax)
    plt.colorbar(cf, ax=ax, shrink=0.8, format="%.1e")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"(b) |Erreur PINN| (L2={ep['L2']:.2e})")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 2])
    cf = ax.contourf(ev["xx"], ev["yy"], np.abs(ev["error"]), levels=30, cmap="hot_r", vmin=0, vmax=vmax)
    plt.colorbar(cf, ax=ax, shrink=0.8, format="%.1e")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"(c) |Erreur VPINN| (L2={ev['L2']:.2e})")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(hp["loss"], color=C_PINN, alpha=0.8, label="PINN")
    ax.semilogy(hv["loss"], color=C_VPINN, alpha=0.8, label="VPINN")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Perte totale")
    ax.set_title("(d) Convergence"); ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(hp["loss_pde"], label="PDE", alpha=0.6)
    ax.semilogy(hp["loss_D"], label="Dirichlet", alpha=0.6)
    ax.semilogy(hp["loss_I"], label="Integrale", alpha=0.6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Perte")
    ax.set_title("(e) PINN -- Detail"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lam_ex = lambda_exact(1.0)
    data = [
        ["Erreur L2", f"{ep['L2']:.2e}", f"{ev['L2']:.2e}"],
        ["Erreur Linf", f"{ep['Linf']:.2e}", f"{ev['Linf']:.2e}"],
        ["Erreur H1", f"{ep.get('H1',0):.2e}", f"{ev.get('H1',0):.2e}"],
        ["Cstr. int.", f"{res_pinn['constraint_error']:.2e}", f"{res_vpinn['constraint_error']:.2e}"],
        ["FDM Linf", f"{fdm_ref['linf']:.2e}", "--"],
        ["Temps (s)", f"{res_pinn['time']:.1f}", f"{res_vpinn['time']:.1f}"],
        ["Lambda_0", "--", f"{res_vpinn['lam'][0]:.4f} ({lam_ex[0]:.4f})"],
    ]
    tbl = ax.table(cellText=data, colLabels=["Metrique", "PINN", "VPINN"],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
    for j in range(3):
        tbl[0, j].set_facecolor("#34495E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("(f) Resume (beta=1)", pad=20)

    fig.savefig(os.path.join(RESULTS_DIR, "fig1_main_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig1_main_comparison.png")


def fig_linear_vs_nonlinear(lin_nl_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    betas = [0.0, 1.0]
    labels = ["beta=0 (lin)", "beta=1 (NL)"]
    x = np.arange(len(betas))

    ax = axes[0]
    p_l2 = [lin_nl_results[b]["pinn_L2"] for b in betas]
    v_l2 = [lin_nl_results[b]["vpinn_L2"] for b in betas]
    w = 0.35
    ax.bar(x - w/2, p_l2, w, color=C_PINN, label="PINN", alpha=0.8)
    ax.bar(x + w/2, v_l2, w, color=C_VPINN, label="VPINN", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Erreur L2"); ax.set_yscale("log")
    ax.set_title("(a) Erreur L2"); ax.legend()

    ax = axes[1]
    p_t = [lin_nl_results[b]["pinn_time"] for b in betas]
    v_t = [lin_nl_results[b]["vpinn_time"] for b in betas]
    ax.bar(x - w/2, p_t, w, color=C_PINN, label="PINN", alpha=0.8)
    ax.bar(x + w/2, v_t, w, color=C_VPINN, label="VPINN", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Temps (s)"); ax.set_title("(b) Temps"); ax.legend()

    ax = axes[2]
    for b in betas:
        ax.semilogy(lin_nl_results[b]["pinn_hist"], color=C_PINN,
                     alpha=0.5 + 0.3 * b, label=f"PINN b={b}")
        ax.semilogy(lin_nl_results[b]["vpinn_hist"], color=C_VPINN,
                     alpha=0.5 + 0.3 * b, label=f"VPINN b={b}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Perte")
    ax.set_title("(c) Convergence"); ax.legend(fontsize=7)

    fig.savefig(os.path.join(RESULTS_DIR, "fig2_linear_vs_nonlinear.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig2_linear_vs_nonlinear.png")


def fig_beta_sweep(beta_results):
    betas = sorted(beta_results.keys())
    p_l2 = [beta_results[b]["pinn_L2"] for b in betas]
    v_l2 = [beta_results[b]["vpinn_L2"] for b in betas]
    p_linf = [beta_results[b]["pinn_Linf"] for b in betas]
    v_linf = [beta_results[b]["vpinn_Linf"] for b in betas]
    p_cstr = [beta_results[b]["pinn_cstr"] for b in betas]
    v_cstr = [beta_results[b]["vpinn_cstr"] for b in betas]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.semilogy(betas, p_l2, "o-", color=C_PINN, lw=2, ms=8, label="PINN")
    ax.semilogy(betas, v_l2, "s-", color=C_VPINN, lw=2, ms=8, label="VPINN")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("Erreur L2")
    ax.set_title(r"(a) Erreur L2 vs $\beta$"); ax.legend()

    ax = axes[1]
    ax.semilogy(betas, p_linf, "o-", color=C_PINN, lw=2, ms=8, label="PINN")
    ax.semilogy(betas, v_linf, "s-", color=C_VPINN, lw=2, ms=8, label="VPINN")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("Erreur Linf")
    ax.set_title(r"(b) Erreur Linf vs $\beta$"); ax.legend()

    ax = axes[2]
    ax.semilogy(betas, p_cstr, "o-", color=C_PINN, lw=2, ms=8, label="PINN")
    ax.semilogy(betas, v_cstr, "s-", color=C_VPINN, lw=2, ms=8, label="VPINN")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("Erreur contrainte")
    ax.set_title(r"(c) Contrainte integrale vs $\beta$"); ax.legend()

    fig.savefig(os.path.join(RESULTS_DIR, "fig3_nonlinearity_strength.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig3_nonlinearity_strength.png")


def fig_ntest_convergence(ntest_results):
    labels = sorted(ntest_results.keys(), key=lambda k: ntest_results[k]["total"])
    totals = [ntest_results[k]["total"] for k in labels]
    l2s = [ntest_results[k]["L2"] for k in labels]
    linfs = [ntest_results[k]["Linf"] for k in labels]
    cstrs = [ntest_results[k]["constraint"] for k in labels]
    times = [ntest_results[k]["time"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.semilogy(totals, l2s, "o-", color=C_VPINN, lw=2, ms=7, label="L2")
    ax.semilogy(totals, linfs, "s--", color=C_PINN, lw=2, ms=7, label="Linf")
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total"); ax.set_ylabel("Erreur")
    ax.set_title(r"(a) VPINN NL vs $N_{\mathrm{test}}$"); ax.legend()
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (totals[i], l2s[i]), fontsize=7, ha='center', va='bottom')

    ax = axes[1]
    ax.semilogy(totals, cstrs, "D-", color="#8E44AD", lw=2, ms=7)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total"); ax.set_ylabel("Contrainte")
    ax.set_title("(b) Contrainte integrale")

    ax = axes[2]
    ax.plot(totals, times, "o-", color="#E67E22", lw=2, ms=7)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total"); ax.set_ylabel("Temps (s)")
    ax.set_title("(c) Cout computationnel")

    fig.savefig(os.path.join(RESULTS_DIR, "fig4_ntest_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig4_ntest_convergence.png")


def fig_robustness(pinn_rob, vpinn_rob):
    metrics = ["L2", "Linf", "H1", "constraint"]
    labels = ["Erreur L2", "Erreur Linf", "Erreur H1", "Cstr. int."]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        vals_p = [r[metric] for r in pinn_rob]
        vals_v = [r[metric] for r in vpinn_rob]
        x = np.array([0, 1])
        bp = ax.boxplot([vals_p, vals_v], positions=x, widths=0.35,
                        patch_artist=True, medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(C_PINN); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(C_VPINN); bp["boxes"][1].set_alpha(0.6)
        for i, vals in enumerate([vals_p, vals_v]):
            jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=25, zorder=5, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(["PINN", "VPINN"])
        ax.set_ylabel(label); ax.set_title(label); ax.set_yscale("log")

    fig.suptitle("Robustesse 5 seeds (beta=1, 2D NL)", fontsize=14, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig5_robustness.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig5_robustness.png")


# ============================================================
# Resume textuel
# ============================================================
def write_summary(res_pinn, res_vpinn, fdm_ref, lin_nl, beta_res, ntest_res,
                  pinn_rob, vpinn_rob, total_time):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]
    lines = []
    lines.append("=" * 70)
    lines.append("  RESUME -- PINN vs VPINN : Conduction 2D NL + Condition integrale")
    lines.append("=" * 70)
    lines.append("")
    lines.append("EDP : -div(k(u)*grad(u)) = f(x,y), k(u) = 1 + beta*u^2")
    lines.append("      u(0,y) = int u dx,  Dirichlet sur 3 bords")
    lines.append("Solution : u(x,y) = sin(pi*y) * [sin(pi*x) + (1-x)*4/pi]")
    lines.append("")

    lines.append("--- Comparaison principale (beta=1, seed=42) ---")
    lines.append(f"{'Metrique':<25} {'PINN':>12} {'VPINN':>12}")
    lines.append("-" * 51)
    lines.append(f"{'L2':<25} {ep['L2']:>12.4e} {ev['L2']:>12.4e}")
    lines.append(f"{'Linf':<25} {ep['Linf']:>12.4e} {ev['Linf']:>12.4e}")
    lines.append(f"{'H1':<25} {ep.get('H1',0):>12.4e} {ev.get('H1',0):>12.4e}")
    lines.append(f"{'Cstr':<25} {res_pinn['constraint_error']:>12.4e} {res_vpinn['constraint_error']:>12.4e}")
    lines.append(f"{'Temps (s)':<25} {res_pinn['time']:>12.1f} {res_vpinn['time']:>12.1f}")
    lines.append(f"{'FDM Linf':<25} {fdm_ref['linf']:>12.4e}")
    lam_ex = lambda_exact(1.0)
    lines.append(f"{'Lambda_0':<25} {'--':>12} {res_vpinn['lam'][0]:>12.4f} (exact={lam_ex[0]:.4f})")
    lines.append("")

    lines.append("--- Beta sweep ---")
    lines.append(f"{'beta':>6} {'PINN L2':>12} {'VPINN L2':>12} {'Ratio':>10}")
    for b in sorted(beta_res):
        r = beta_res[b]
        ratio = r["pinn_L2"] / max(r["vpinn_L2"], 1e-20)
        lines.append(f"{b:>6.1f} {r['pinn_L2']:>12.3e} {r['vpinn_L2']:>12.3e} {ratio:>10.1f}x")
    lines.append("")

    lines.append("--- Robustesse (5 seeds, beta=1) ---")
    for name, data in [("PINN", pinn_rob), ("VPINN", vpinn_rob)]:
        l2s = [r["L2"] for r in data]
        lines.append(f"  {name} L2: moy={np.mean(l2s):.3e} std={np.std(l2s):.3e}")
    lines.append("")
    lines.append(f"Temps total : {total_time:.1f} s")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(RESULTS_DIR, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> results_summary.txt")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_total = time.time()

    verify(beta=1.0)

    res_pinn, res_vpinn, fdm_ref = study_main_comparison()
    lin_nl = study_linear_vs_nonlinear()
    beta_res = study_beta_sweep()
    ntest_res = study_ntest_convergence()
    pinn_rob, vpinn_rob = study_robustness()

    pkl_path = os.path.join(RESULTS_DIR, "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "res_pinn": {k: v for k, v in res_pinn.items() if k != "net"},
            "res_vpinn": {k: v for k, v in res_vpinn.items() if k != "net"},
            "fdm_ref": fdm_ref, "lin_nl": lin_nl, "beta_res": beta_res,
            "ntest": ntest_res, "pinn_rob": pinn_rob, "vpinn_rob": vpinn_rob,
        }, f)

    print("\n" + "=" * 70)
    print("  GENERATION DES FIGURES")
    print("=" * 70)
    fig_main_comparison(res_pinn, res_vpinn, fdm_ref)
    fig_linear_vs_nonlinear(lin_nl)
    fig_beta_sweep(beta_res)
    fig_ntest_convergence(ntest_res)
    fig_robustness(pinn_rob, vpinn_rob)

    total_time = time.time() - t_total
    write_summary(res_pinn, res_vpinn, fdm_ref, lin_nl, beta_res, ntest_res,
                  pinn_rob, vpinn_rob, total_time)

    print(f"\n  Termine en {total_time:.0f}s. Figures dans {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
