"""
========================================================================
  PINN vs VPINN — Poisson 2D avec condition integrale non locale
  Script complet : entrainement, etudes parametriques, figures publication
========================================================================

Execution :  python run_all.py

Produit dans results/ :
    fig1_main_comparison.png    — Comparaison PINN vs VPINN
    fig2_fdm_validation.png     — Validation par differences finies
    fig3_ntest_convergence.png  — Convergence VPINN vs nb fonctions de test
    fig4_robustness.png         — Robustesse multi-seeds
    fig5_summary_table.png      — Tableau recapitulatif
    results_summary.txt         — Resume textuel complet
"""

import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Configuration globale des figures (style publication)
# ============================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})

# Palette coherente
C_PINN = "#E74C3C"
C_VPINN = "#3498DB"
C_EXACT = "#2C3E50"
C_FDM = "#27AE60"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

from exact_solution import verify, u_exact_np, f_source_np, ALPHA, G0, lambda_exact
from utils import solve_fdm_2d, gauss_legendre_torch, DTYPE
from pinn_solver import train_pinn
from vpinn_solver import train_vpinn


# ============================================================
# ETUDE 1 : Comparaison principale PINN vs VPINN
# ============================================================
def study_main_comparison():
    print("\n" + "=" * 70)
    print("  ETUDE 1 -- Comparaison principale PINN vs VPINN (2D)")
    print("=" * 70)

    res_pinn = train_pinn(verbose=True)
    res_vpinn = train_vpinn(verbose=True)

    return res_pinn, res_vpinn


# ============================================================
# ETUDE 2 : Validation FDM
# ============================================================
def study_fdm_validation():
    print("\n" + "=" * 70)
    print("  ETUDE 2 -- Validation par differences finies (FDM 2D)")
    print("=" * 70)

    results = {}
    for N in [20, 50, 100, 200]:
        x1d, y1d, u_fdm = solve_fdm_2d(f_source_np, ALPHA, G0, N=N)
        xx, yy = np.meshgrid(x1d, y1d, indexing='ij')
        u_ex = u_exact_np(xx, yy)
        err = np.max(np.abs(u_fdm - u_ex))
        h = 1.0 / N
        results[N] = {"x": x1d, "y": y1d, "u": u_fdm, "linf": err, "h": h}
        print(f"  FDM N={N:4d} | h={h:.1e} | Linf = {err:.3e}")

    return results


# ============================================================
# ETUDE 3 : Convergence VPINN vs N_test
# ============================================================
def study_ntest_convergence():
    print("\n" + "=" * 70)
    print("  ETUDE 3 -- Convergence VPINN vs N_test (2D)")
    print("=" * 70)

    ntest_pairs = [(3, 2), (5, 3), (10, 3), (15, 5), (20, 5), (25, 8)]
    results = {}

    base_cfg = {
        "n_adam": 10000,
        "n_lbfgs": 2000,
        "seed": 42,
    }

    for ntx, nty in ntest_pairs:
        label = f"{ntx}x{nty}"
        print(f"\n  --- N_test = {label} ---")
        cfg = {
            **base_cfg,
            "n_test_x": ntx, "n_test_y": nty,
            "n_quad_x": max(ntx + 10, 25),
            "n_quad_y": max(nty + 10, 15),
        }
        res = train_vpinn(config=cfg, verbose=True)
        results[label] = {
            "ntx": ntx, "nty": nty, "total": ntx * nty,
            "L2": res["errors"]["L2"],
            "Linf": res["errors"]["Linf"],
            "H1": res["errors"].get("H1", np.nan),
            "constraint": res["constraint_error"],
            "time": res["time"],
            "lam": res["lam"],
        }

    return results


# ============================================================
# ETUDE 4 : Robustesse multi-seeds
# ============================================================
def study_robustness():
    print("\n" + "=" * 70)
    print("  ETUDE 4 -- Robustesse (5 seeds, 2D)")
    print("=" * 70)

    seeds = [42, 123, 256, 789, 1337]

    base_cfg = {"n_adam": 10000, "n_lbfgs": 2000}

    pinn_results = []
    vpinn_results = []

    for s in seeds:
        print(f"\n  --- Seed {s} ---")
        cfg = {**base_cfg, "seed": s}

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

        print(f"    PINN  L2={rp['errors']['L2']:.3e}  VPINN L2={rv['errors']['L2']:.3e}")

    return pinn_results, vpinn_results


# ============================================================
# FIGURE 1 : Comparaison principale (contours 2D)
# ============================================================
def fig_main_comparison(res_pinn, res_vpinn):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]
    hp = res_pinn["history"]
    hv = res_vpinn["history"]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # 1a. Solution exacte (contour)
    ax = fig.add_subplot(gs[0, 0])
    cf = ax.contourf(ep["xx"], ep["yy"], ep["u_exact"], levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a) Solution exacte")
    ax.set_aspect("equal")

    # 1b. Erreur PINN
    ax = fig.add_subplot(gs[0, 1])
    err_p = np.abs(ep["error"])
    vmax = max(np.max(err_p), np.max(np.abs(ev["error"])))
    cf = ax.contourf(ep["xx"], ep["yy"], err_p, levels=30, cmap="hot_r", vmin=0, vmax=vmax)
    plt.colorbar(cf, ax=ax, shrink=0.8, format="%.1e")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"(b) |Erreur PINN| (L2={ep['L2']:.2e})")
    ax.set_aspect("equal")

    # 1c. Erreur VPINN
    ax = fig.add_subplot(gs[0, 2])
    err_v = np.abs(ev["error"])
    cf = ax.contourf(ev["xx"], ev["yy"], err_v, levels=30, cmap="hot_r", vmin=0, vmax=vmax)
    plt.colorbar(cf, ax=ax, shrink=0.8, format="%.1e")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"(c) |Erreur VPINN| (L2={ev['L2']:.2e})")
    ax.set_aspect("equal")

    # 1d. Convergence totale
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(hp["loss"], color=C_PINN, alpha=0.8, label="PINN")
    ax.semilogy(hv["loss"], color=C_VPINN, alpha=0.8, label="VPINN")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Perte totale")
    ax.set_title("(d) Convergence")
    ax.legend()

    # 1e. Detail PINN
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(hp["loss"], label="Total", alpha=0.8)
    ax.semilogy(hp["loss_pde"], label="PDE", alpha=0.6)
    ax.semilogy(hp["loss_D"], label="Dirichlet", alpha=0.6)
    ax.semilogy(hp["loss_I"], label="Integrale", alpha=0.6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Perte")
    ax.set_title("(e) PINN -- Detail des pertes")
    ax.legend(fontsize=8)

    # 1f. Tableau
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    data = [
        ["Erreur L2", f"{ep['L2']:.2e}", f"{ev['L2']:.2e}"],
        ["Erreur Linf", f"{ep['Linf']:.2e}", f"{ev['Linf']:.2e}"],
        ["Erreur H1", f"{ep.get('H1',0):.2e}", f"{ev.get('H1',0):.2e}"],
        ["Cstr. int.", f"{res_pinn['constraint_error']:.2e}",
                       f"{res_vpinn['constraint_error']:.2e}"],
        ["Temps (s)", f"{res_pinn['time']:.1f}", f"{res_vpinn['time']:.1f}"],
    ]
    if res_vpinn.get("lam") is not None:
        lam_ex = lambda_exact()
        data.append(["Lambda_0", "--", f"{res_vpinn['lam'][0]:.4f} ({lam_ex:.4f})"])

    tbl = ax.table(cellText=data, colLabels=["Metrique", "PINN", "VPINN"],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
    for j in range(3):
        tbl[0, j].set_facecolor("#34495E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("(f) Resume", pad=20)

    path = os.path.join(RESULTS_DIR, "fig1_main_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# FIGURE 2 : Validation FDM
# ============================================================
def fig_fdm_validation(res_pinn, res_vpinn, fdm_results):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 2a. Solution FDM (contour) — utiliser le FDM le plus fin
    N_max = max(fdm_results.keys())
    fdm_ref = fdm_results[N_max]
    xx, yy = np.meshgrid(fdm_ref["x"], fdm_ref["y"], indexing='ij')
    ax = axes[0]
    cf = ax.contourf(xx, yy, fdm_ref["u"], levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"(a) Solution FDM (N={N_max})")
    ax.set_aspect("equal")

    # 2b. Erreur FDM pour chaque N
    ax = axes[1]
    for N in sorted(fdm_results):
        fdm = fdm_results[N]
        xx_n, yy_n = np.meshgrid(fdm["x"], fdm["y"], indexing='ij')
        u_ex = u_exact_np(xx_n, yy_n)
        err_slice = np.abs(fdm["u"] - u_ex)[:, fdm["u"].shape[1] // 2]
        ax.semilogy(fdm["x"], err_slice, label=f"N={N}", alpha=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("Erreur absolue (y=0.5)")
    ax.set_title("(b) Erreur FDM (coupe y=0.5)")
    ax.legend(fontsize=8)

    # 2c. Convergence FDM vs h
    ax = axes[2]
    Ns = sorted(fdm_results.keys())
    hs = [fdm_results[N]["h"] for N in Ns]
    errs = [fdm_results[N]["linf"] for N in Ns]
    ax.loglog(hs, errs, "o-", color=C_FDM, lw=2, ms=8, label="FDM")
    h_ref = np.array([hs[0], hs[-1]])
    ax.loglog(h_ref, errs[0] * (h_ref / hs[0]) ** 2, "k--", alpha=0.5, label=r"$O(h^2)$")
    ax.axhline(ep["Linf"], color=C_PINN, ls="--", alpha=0.7, label="PINN Linf")
    ax.axhline(ev["Linf"], color=C_VPINN, ls="--", alpha=0.7, label="VPINN Linf")
    ax.set_xlabel("h (pas de maillage)"); ax.set_ylabel("Erreur Linf")
    ax.set_title("(c) Convergence FDM vs PINN/VPINN")
    ax.legend(fontsize=7)

    path = os.path.join(RESULTS_DIR, "fig2_fdm_validation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# FIGURE 3 : Convergence N_test
# ============================================================
def fig_ntest_convergence(ntest_results):
    labels = sorted(ntest_results.keys(), key=lambda k: ntest_results[k]["total"])
    totals = [ntest_results[k]["total"] for k in labels]
    l2s = [ntest_results[k]["L2"] for k in labels]
    linfs = [ntest_results[k]["Linf"] for k in labels]
    cstrs = [ntest_results[k]["constraint"] for k in labels]
    times = [ntest_results[k]["time"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 3a. Erreurs vs N_test total
    ax = axes[0]
    ax.semilogy(totals, l2s, "o-", color=C_VPINN, lw=2, ms=7, label="L2")
    ax.semilogy(totals, linfs, "s--", color=C_PINN, lw=2, ms=7, label="Linf")
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total ($N_x \times N_y$)")
    ax.set_ylabel("Erreur")
    ax.set_title(r"(a) Erreur VPINN vs $N_{\mathrm{test}}$")
    ax.legend()

    # Annotations
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (totals[i], l2s[i]), fontsize=7, ha='center', va='bottom')

    # 3b. Contrainte integrale
    ax = axes[1]
    ax.semilogy(totals, cstrs, "D-", color="#8E44AD", lw=2, ms=7)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Erreur contrainte integrale")
    ax.set_title("(b) Satisfaction contrainte integrale")

    # 3c. Temps
    ax = axes[2]
    ax.plot(totals, times, "o-", color="#E67E22", lw=2, ms=7)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Temps (s)")
    ax.set_title("(c) Cout computationnel")

    path = os.path.join(RESULTS_DIR, "fig3_ntest_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# FIGURE 4 : Robustesse multi-seeds
# ============================================================
def fig_robustness(pinn_rob, vpinn_rob):
    metrics = ["L2", "Linf", "H1", "constraint"]
    labels = ["Erreur L2", "Erreur Linf", "Erreur H1", "Cstr. int."]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        vals_p = [r[metric] for r in pinn_rob]
        vals_v = [r[metric] for r in vpinn_rob]

        x = np.array([0, 1])
        bp = ax.boxplot(
            [vals_p, vals_v],
            positions=x,
            widths=0.35,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor(C_PINN)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(C_VPINN)
        bp["boxes"][1].set_alpha(0.6)

        for i, vals in enumerate([vals_p, vals_v]):
            jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="black", s=25, zorder=5, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(["PINN", "VPINN"])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_yscale("log")

    fig.suptitle("Robustesse sur 5 seeds aleatoires (2D)", fontsize=14, y=1.02)

    path = os.path.join(RESULTS_DIR, "fig4_robustness.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# FIGURE 5 : Tableau recapitulatif
# ============================================================
def fig_summary_table(res_pinn, res_vpinn, pinn_rob, vpinn_rob, fdm_results):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]

    def mean_std(data, key):
        vals = [d[key] for d in data]
        return np.mean(vals), np.std(vals)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title(
        "Poisson 2D -- Condition integrale non locale\n"
        r"$-\Delta u = f$,  $u(0,y) = \int_0^1 u\, dx$,  Dirichlet sur 3 bords",
        fontsize=14, fontweight="bold", pad=20,
    )

    m_p_l2, s_p_l2 = mean_std(pinn_rob, "L2")
    m_v_l2, s_v_l2 = mean_std(vpinn_rob, "L2")
    m_p_linf, s_p_linf = mean_std(pinn_rob, "Linf")
    m_v_linf, s_v_linf = mean_std(vpinn_rob, "Linf")
    m_p_h1, s_p_h1 = mean_std(pinn_rob, "H1")
    m_v_h1, s_v_h1 = mean_std(vpinn_rob, "H1")
    m_p_c, s_p_c = mean_std(pinn_rob, "constraint")
    m_v_c, s_v_c = mean_std(vpinn_rob, "constraint")
    m_p_t, s_p_t = mean_std(pinn_rob, "time")
    m_v_t, s_v_t = mean_std(vpinn_rob, "time")

    N_fdm = max(fdm_results.keys())
    fdm_ref = fdm_results[N_fdm]["linf"]

    lam_ex = lambda_exact()
    lam_str = f"{res_vpinn['lam'][0]:.4f}" if res_vpinn.get("lam") is not None else "--"

    data = [
        ["Erreur L2 (best run)", f"{ep['L2']:.2e}", f"{ev['L2']:.2e}", "--"],
        ["Erreur L2 (moy +/- std)", f"{m_p_l2:.2e} +/- {s_p_l2:.2e}",
         f"{m_v_l2:.2e} +/- {s_v_l2:.2e}", "--"],
        ["Erreur Linf (best)", f"{ep['Linf']:.2e}", f"{ev['Linf']:.2e}",
         f"{fdm_ref:.2e}"],
        ["Erreur H1 (best)", f"{ep.get('H1',0):.2e}", f"{ev.get('H1',0):.2e}", "--"],
        ["Cstr. integrale (best)", f"{res_pinn['constraint_error']:.2e}",
         f"{res_vpinn['constraint_error']:.2e}", "--"],
        ["Lambda_0", "--", f"{lam_str} (exact={lam_ex:.4f})", "--"],
        ["Temps (moy +/- std)", f"{m_p_t:.1f} +/- {s_p_t:.1f}s",
         f"{m_v_t:.1f} +/- {s_v_t:.1f}s", "< 1s"],
    ]

    tbl = ax.table(
        cellText=data,
        colLabels=["Metrique", "PINN (fort)", "VPINN (faible)", f"FDM (N={N_fdm})"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)

    for j in range(4):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)
    for i in range(1, len(data) + 1):
        for j in range(4):
            if i % 2 == 0:
                tbl[i, j].set_facecolor("#ECF0F1")

    path = os.path.join(RESULTS_DIR, "fig5_summary_table.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================
# Resume textuel
# ============================================================
def write_summary(res_pinn, res_vpinn, fdm_results, ntest_results,
                  pinn_rob, vpinn_rob, total_time):
    ep = res_pinn["errors"]
    ev = res_vpinn["errors"]

    lines = []
    lines.append("=" * 70)
    lines.append("  RESUME -- PINN vs VPINN : Poisson 2D + Condition integrale")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Probleme : -Delta u(x,y) = f(x,y), (x,y) in (0,1)^2")
    lines.append("           u(0,y) = int_0^1 u(x,y) dx   (condition integrale)")
    lines.append("           u(1,y) = u(x,0) = u(x,1) = 0  (Dirichlet)")
    lines.append("Solution : u(x,y) = sin(pi*y) * [sin(pi*x) + (1-x)*4/pi]")
    lines.append("")

    lines.append("--- Comparaison principale (seed=42) ---")
    lines.append(f"{'Metrique':<30} {'PINN':>15} {'VPINN':>15}")
    lines.append("-" * 62)
    lines.append(f"{'Erreur L2':<30} {ep['L2']:>15.4e} {ev['L2']:>15.4e}")
    lines.append(f"{'Erreur Linf':<30} {ep['Linf']:>15.4e} {ev['Linf']:>15.4e}")
    lines.append(f"{'Erreur H1':<30} {ep.get('H1',0):>15.4e} {ev.get('H1',0):>15.4e}")
    lines.append(f"{'Cstr. integrale':<30} {res_pinn['constraint_error']:>15.4e} "
                 f"{res_vpinn['constraint_error']:>15.4e}")
    lines.append(f"{'Temps (s)':<30} {res_pinn['time']:>15.1f} {res_vpinn['time']:>15.1f}")
    if res_vpinn.get("lam") is not None:
        lam_ex = lambda_exact()
        lines.append(f"{'Lambda_0':<30} {'--':>15} {res_vpinn['lam'][0]:>15.4f}")
        lines.append(f"{'Lambda_0 exact':<30} {'--':>15} {lam_ex:>15.4f}")
    lines.append("")

    lines.append("--- Validation FDM ---")
    for N in sorted(fdm_results):
        lines.append(f"  FDM N={N:<5d} Linf = {fdm_results[N]['linf']:.4e}")
    lines.append("")

    lines.append("--- Convergence VPINN vs N_test ---")
    lines.append(f"{'N_test':>10} {'Total':>8} {'L2':>12} {'Linf':>12} {'Cstr.':>12}")
    for label in sorted(ntest_results.keys(), key=lambda k: ntest_results[k]["total"]):
        r = ntest_results[label]
        lines.append(f"{label:>10} {r['total']:>8d} {r['L2']:>12.3e} "
                     f"{r['Linf']:>12.3e} {r['constraint']:>12.3e}")
    lines.append("")

    lines.append("--- Robustesse (5 seeds) ---")
    for label_name, rob_data in [("PINN", pinn_rob), ("VPINN", vpinn_rob)]:
        l2s = [r["L2"] for r in rob_data]
        lines.append(f"  {label_name} L2 : moy={np.mean(l2s):.3e} std={np.std(l2s):.3e} "
                     f"min={np.min(l2s):.3e} max={np.max(l2s):.3e}")
    lines.append("")

    lines.append(f"Temps total d'execution : {total_time:.1f} s")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  -> {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_total = time.time()

    # Verification solution manufacturee
    verify()

    # Etudes
    res_pinn, res_vpinn = study_main_comparison()
    fdm_results = study_fdm_validation()
    ntest_results = study_ntest_convergence()
    pinn_rob, vpinn_rob = study_robustness()

    # Sauvegarde intermediaire
    pkl_path = os.path.join(RESULTS_DIR, "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "res_pinn": {k: v for k, v in res_pinn.items() if k != "net"},
            "res_vpinn": {k: v for k, v in res_vpinn.items() if k != "net"},
            "fdm": fdm_results,
            "ntest": ntest_results,
            "pinn_rob": pinn_rob,
            "vpinn_rob": vpinn_rob,
        }, f)
    print(f"\n  Resultats sauvegardes : {pkl_path}")

    # Figures
    print("\n" + "=" * 70)
    print("  GENERATION DES FIGURES")
    print("=" * 70)
    fig_main_comparison(res_pinn, res_vpinn)
    fig_fdm_validation(res_pinn, res_vpinn, fdm_results)
    fig_ntest_convergence(ntest_results)
    fig_robustness(pinn_rob, vpinn_rob)
    fig_summary_table(res_pinn, res_vpinn, pinn_rob, vpinn_rob, fdm_results)

    total_time = time.time() - t_total
    write_summary(res_pinn, res_vpinn, fdm_results, ntest_results,
                  pinn_rob, vpinn_rob, total_time)

    print(f"\n  Termine en {total_time:.0f}s. Figures dans {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
