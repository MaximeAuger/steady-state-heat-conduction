"""
========================================================================
  Analyse approfondie : 3 questions cles

  A) Crossover exact : beta fin entre 2 et 5 (VPINN 25x8 vs PINN)
  B) Push N_test a beta=5 : jusqu'ou peut-on aller ? (35x12, 40x15, 50x15)
  C) Sensibilite a la quadrature : 30x10 avec quadrature augmentee

  Produit :
    - fig8_crossover.png : point de croisement VPINN/PINN
    - fig9_push_ntest_beta5.png : N_test pousse a fond
    - fig10_quadrature_sensitivity.png : effet de la quadrature
    - study_deep_analysis_summary.txt
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
C_CROSS = "#8E44AD"
C_QUAD = "#27AE60"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

from vpinn_solver import train_vpinn
from pinn_solver import train_pinn
from exact_solution import BETA_DEFAULT


CFG_BASE = {
    "n_adam": 10000,
    "n_lbfgs": 2000,
    "seed": 42,
}


# ============================================================
# PARTIE A : Crossover exact VPINN vs PINN
# ============================================================
def study_crossover():
    """Beta fin entre 1.5 et 6.0, VPINN a 25x8 (meilleur compromis) vs PINN."""
    print("\n" + "=" * 70)
    print("  PARTIE A -- Crossover VPINN(25x8) vs PINN")
    print("=" * 70)

    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]

    results = {}
    for beta in betas:
        print(f"\n  beta = {beta}")

        # VPINN 25x8
        cfg_v = {**CFG_BASE, "beta": beta,
                 "n_test_x": 25, "n_test_y": 8,
                 "n_quad_x": 40, "n_quad_y": 20}
        rv = train_vpinn(config=cfg_v, verbose=False)

        # PINN
        cfg_p = {**CFG_BASE, "beta": beta}
        rp = train_pinn(config=cfg_p, verbose=False)

        results[beta] = {
            "vpinn_L2": rv["errors"]["L2"],
            "vpinn_Linf": rv["errors"]["Linf"],
            "vpinn_lam0": rv["lam"][0],
            "vpinn_time": rv["time"],
            "pinn_L2": rp["errors"]["L2"],
            "pinn_Linf": rp["errors"]["Linf"],
            "pinn_time": rp["time"],
        }
        ratio = rv["errors"]["L2"] / max(rp["errors"]["L2"], 1e-20)
        winner = "VPINN" if ratio < 1 else "PINN"
        print(f"    PINN L2={rp['errors']['L2']:.3e}  VPINN L2={rv['errors']['L2']:.3e}"
              f"  ratio={ratio:.2f}  -> {winner}")

    return results


# ============================================================
# PARTIE B : Push N_test a beta=5
# ============================================================
def study_push_ntest():
    """N_test pousse a fond pour beta=5."""
    print("\n" + "=" * 70)
    print("  PARTIE B -- Push N_test a beta=5")
    print("=" * 70)

    ntest_pairs = [
        (15, 5),    # 75 (reference)
        (20, 5),    # 100
        (25, 8),    # 200
        (30, 10),   # 300
        (35, 12),   # 420
        (40, 12),   # 480
        (40, 15),   # 600
        (50, 15),   # 750
    ]

    results = {}
    for ntx, nty in ntest_pairs:
        total = ntx * nty
        label = f"{ntx}x{nty}"
        print(f"\n  N_test = {label} ({total})")

        cfg = {
            **CFG_BASE,
            "beta": 5.0,
            "n_test_x": ntx,
            "n_test_y": nty,
            # Quadrature proportionnelle au nombre de fonctions test
            "n_quad_x": ntx + 20,
            "n_quad_y": nty + 15,
        }
        rv = train_vpinn(config=cfg, verbose=False)

        results[label] = {
            "ntx": ntx, "nty": nty, "total": total,
            "L2": rv["errors"]["L2"],
            "Linf": rv["errors"]["Linf"],
            "H1": rv["errors"].get("H1", np.nan),
            "constraint": rv["constraint_error"],
            "lam0": rv["lam"][0],
            "time": rv["time"],
        }
        print(f"    L2={rv['errors']['L2']:.3e}  Linf={rv['errors']['Linf']:.3e}"
              f"  lam0={rv['lam'][0]:.4f}  ({rv['time']:.1f}s)")

    # PINN reference
    rp = train_pinn(config={**CFG_BASE, "beta": 5.0}, verbose=False)
    pinn_ref = {"L2": rp["errors"]["L2"], "Linf": rp["errors"]["Linf"],
                "time": rp["time"]}
    print(f"\n  PINN ref: L2={rp['errors']['L2']:.3e}")

    return results, pinn_ref


# ============================================================
# PARTIE C : Sensibilite a la quadrature
# ============================================================
def study_quadrature():
    """Meme N_test (25x8 et 30x10) mais quadrature variable."""
    print("\n" + "=" * 70)
    print("  PARTIE C -- Sensibilite a la quadrature")
    print("=" * 70)

    configs = [
        # (n_test_x, n_test_y, n_quad_x, n_quad_y, label)
        # --- 25x8 ---
        (25, 8, 30, 15, "25x8 q=30x15"),
        (25, 8, 40, 20, "25x8 q=40x20"),
        (25, 8, 50, 30, "25x8 q=50x30"),
        (25, 8, 60, 40, "25x8 q=60x40"),
        (25, 8, 80, 50, "25x8 q=80x50"),
        # --- 30x10 ---
        (30, 10, 40, 20, "30x10 q=40x20"),
        (30, 10, 50, 25, "30x10 q=50x25"),
        (30, 10, 60, 35, "30x10 q=60x35"),
        (30, 10, 70, 40, "30x10 q=70x40"),
        (30, 10, 80, 50, "30x10 q=80x50"),
    ]

    # Tester a beta=1 (ou VPINN devrait bien marcher) et beta=5
    results = {}
    for beta in [1.0, 5.0]:
        print(f"\n  --- beta = {beta} ---")
        for ntx, nty, nqx, nqy, label in configs:
            key = (label, beta)
            print(f"    {label} beta={beta}")

            cfg = {
                **CFG_BASE,
                "beta": beta,
                "n_test_x": ntx,
                "n_test_y": nty,
                "n_quad_x": nqx,
                "n_quad_y": nqy,
            }
            rv = train_vpinn(config=cfg, verbose=False)

            results[key] = {
                "ntx": ntx, "nty": nty,
                "nqx": nqx, "nqy": nqy,
                "n_quad_total": nqx * nqy,
                "L2": rv["errors"]["L2"],
                "Linf": rv["errors"]["Linf"],
                "constraint": rv["constraint_error"],
                "time": rv["time"],
            }
            print(f"      L2={rv['errors']['L2']:.3e}  ({rv['time']:.1f}s)")

    return results


# ============================================================
# FIGURES
# ============================================================
def fig_crossover(cross_results):
    """Figure 8 : crossover VPINN/PINN."""
    betas = sorted(cross_results.keys())
    vpinn_l2 = [cross_results[b]["vpinn_L2"] for b in betas]
    pinn_l2 = [cross_results[b]["pinn_L2"] for b in betas]
    ratios = [v / max(p, 1e-20) for v, p in zip(vpinn_l2, pinn_l2)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) L2 vs beta
    ax = axes[0]
    ax.semilogy(betas, pinn_l2, 'D-', color=C_PINN, lw=2.5, ms=9, label="PINN")
    ax.semilogy(betas, vpinn_l2, 'o-', color=C_VPINN, lw=2.5, ms=9, label="VPINN (25x8)")

    # Trouver le crossover
    cross_beta = None
    for i in range(len(betas) - 1):
        if ratios[i] < 1 and ratios[i + 1] >= 1:
            # Interpolation log-lineaire
            b1, b2 = betas[i], betas[i + 1]
            r1, r2 = ratios[i], ratios[i + 1]
            cross_beta = b1 + (b2 - b1) * (1.0 - r1) / (r2 - r1)
            break

    if cross_beta is not None:
        ax.axvline(cross_beta, color=C_CROSS, ls='--', lw=2, alpha=0.7)
        ax.annotate(rf"$\beta^* \approx {cross_beta:.1f}$",
                    xy=(cross_beta, ax.get_ylim()[0]),
                    xytext=(cross_beta + 0.3, 3e-3),
                    fontsize=12, fontweight='bold', color=C_CROSS,
                    arrowprops=dict(arrowstyle='->', color=C_CROSS))

    # Zone coloriee
    ax.fill_between(betas, 1e-6, 1e0, where=[r < 1 for r in ratios],
                     color=C_VPINN, alpha=0.08, label="VPINN meilleur")
    ax.fill_between(betas, 1e-6, 1e0, where=[r >= 1 for r in ratios],
                     color=C_PINN, alpha=0.08, label="PINN meilleur")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Erreur L2")
    ax.set_title(r"(a) Crossover VPINN vs PINN")
    ax.legend(fontsize=8)

    # (b) Ratio VPINN/PINN
    ax = axes[1]
    ax.plot(betas, ratios, 'o-', color=C_CROSS, lw=2.5, ms=9)
    ax.axhline(1.0, color='black', ls='--', lw=1.5, alpha=0.5)
    ax.fill_between(betas, 0, 1, color=C_VPINN, alpha=0.1)
    ax.fill_between(betas, 1, max(ratios) * 1.2, color=C_PINN, alpha=0.1)

    if cross_beta is not None:
        ax.axvline(cross_beta, color=C_CROSS, ls='--', lw=2, alpha=0.7)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("L2(VPINN) / L2(PINN)")
    ax.set_title(r"(b) Ratio de precision")
    ax.set_ylim(0, min(max(ratios) * 1.3, 30))
    ax.text(1.0, 0.5, "VPINN\nmeilleur", fontsize=10, color=C_VPINN,
            fontweight='bold', ha='center')
    ax.text(5.0, max(ratios) * 0.8, "PINN\nmeilleur", fontsize=10,
            color=C_PINN, fontweight='bold', ha='center')

    # (c) Temps vs beta
    ax = axes[2]
    vpinn_t = [cross_results[b]["vpinn_time"] for b in betas]
    pinn_t = [cross_results[b]["pinn_time"] for b in betas]
    ax.plot(betas, pinn_t, 'D-', color=C_PINN, lw=2, ms=8, label="PINN")
    ax.plot(betas, vpinn_t, 'o-', color=C_VPINN, lw=2, ms=8, label="VPINN (25x8)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Temps (s)")
    ax.set_title("(c) Cout calcul vs beta")
    ax.legend()

    fig.suptitle(r"Crossover VPINN/PINN : ou est $\beta^*$ ?", fontsize=14, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig8_crossover.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig8_crossover.png")

    return cross_beta


def fig_push_ntest(push_results, pinn_ref):
    """Figure 9 : push N_test a beta=5."""
    labels = sorted(push_results.keys(),
                    key=lambda k: push_results[k]["total"])
    totals = [push_results[k]["total"] for k in labels]
    l2s = [push_results[k]["L2"] for k in labels]
    linfs = [push_results[k]["Linf"] for k in labels]
    cstrs = [push_results[k]["constraint"] for k in labels]
    times = [push_results[k]["time"] for k in labels]
    pinn_l2 = pinn_ref["L2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) L2 vs N_test
    ax = axes[0]
    ax.semilogy(totals, l2s, 'o-', color=C_VPINN, lw=2.5, ms=9, label="VPINN L2")
    ax.axhline(pinn_l2, color=C_PINN, ls='--', lw=2.5,
               label=f"PINN L2 ({pinn_l2:.2e})")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (totals[i], l2s[i]), fontsize=7,
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', rotation=30)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Erreur L2")
    ax.set_title(r"(a) VPINN L2 vs $N_{\mathrm{test}}$ ($\beta=5$)")
    ax.legend()

    # (b) Taux de convergence (log-log)
    ax = axes[1]
    log_t = np.log10(np.array(totals, dtype=float))
    log_l2 = np.log10(np.array(l2s, dtype=float))
    ax.plot(log_t, log_l2, 'o-', color=C_VPINN, lw=2.5, ms=9)

    # Fit lineaire sur les 5 derniers points
    if len(log_t) >= 4:
        coeffs = np.polyfit(log_t[-5:], log_l2[-5:], 1)
        x_fit = np.linspace(log_t[0], log_t[-1] + 0.3, 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=C_VPINN,
                alpha=0.5, lw=1.5, label=f"pente = {coeffs[0]:.2f}")
        # Extrapolation : quand atteint-on le PINN ?
        log_pinn = np.log10(pinn_l2)
        ntest_needed = 10 ** ((log_pinn - coeffs[1]) / coeffs[0])
        ax.axhline(log_pinn, color=C_PINN, ls='--', lw=1.5, alpha=0.5)
        if ntest_needed > 0 and ntest_needed < 1e6:
            ax.annotate(f"PINN atteint a\nN_test ~ {ntest_needed:.0f}",
                        xy=(np.log10(ntest_needed), log_pinn),
                        xytext=(log_t[-1] - 0.5, log_pinn + 0.3),
                        fontsize=9, color=C_PINN,
                        arrowprops=dict(arrowstyle='->', color=C_PINN))

    ax.set_xlabel(r"$\log_{10}(N_{\mathrm{test}})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{L2})$")
    ax.set_title("(b) Taux de convergence (log-log)")
    ax.legend()

    # (c) Temps vs N_test
    ax = axes[2]
    ax.plot(totals, times, 'o-', color="#E67E22", lw=2.5, ms=9)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Temps (s)")
    ax.set_title(r"(c) Cout calcul ($\beta=5$)")

    # Temps par point de N_test
    ax2 = ax.twinx()
    efficiency = [l2 * t for l2, t in zip(l2s, times)]
    ax2.plot(totals, efficiency, 's--', color=C_CROSS, lw=1.5, ms=6, alpha=0.6)
    ax2.set_ylabel("L2 x Temps (efficacite)", color=C_CROSS)
    ax2.tick_params(axis='y', labelcolor=C_CROSS)

    fig.suptitle(r"Push $N_{\mathrm{test}}$ a $\beta=5$ : ou est la limite ?",
                 fontsize=14, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig9_push_ntest_beta5.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig9_push_ntest_beta5.png")


def fig_quadrature(quad_results):
    """Figure 10 : sensibilite a la quadrature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for col, beta in enumerate([1.0, 5.0]):
        ax = axes[col]

        # Extraire 25x8
        data_25 = [(k, v) for (k, b), v in quad_results.items()
                    if b == beta and k.startswith("25x8")]
        data_25.sort(key=lambda x: x[1]["n_quad_total"])

        data_30 = [(k, v) for (k, b), v in quad_results.items()
                    if b == beta and k.startswith("30x10")]
        data_30.sort(key=lambda x: x[1]["n_quad_total"])

        if data_25:
            nq = [v["n_quad_total"] for _, v in data_25]
            l2 = [v["L2"] for _, v in data_25]
            ax.semilogy(nq, l2, 'o-', color=C_VPINN, lw=2, ms=8,
                        label="N_test=25x8")

        if data_30:
            nq = [v["n_quad_total"] for _, v in data_30]
            l2 = [v["L2"] for _, v in data_30]
            ax.semilogy(nq, l2, 's-', color=C_QUAD, lw=2, ms=8,
                        label="N_test=30x10")

        ax.set_xlabel("Points de quadrature (total)")
        ax.set_ylabel("Erreur L2")
        ax.set_title(rf"$\beta = {beta}$")
        ax.legend()

    fig.suptitle("Sensibilite a la quadrature : l'erreur vient-elle de l'integration ?",
                 fontsize=13, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig10_quadrature_sensitivity.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig10_quadrature_sensitivity.png")


# ============================================================
# Resume texte
# ============================================================
def write_summary(cross_results, cross_beta, push_results, pinn_ref,
                  quad_results, total_time):
    lines = []
    lines.append("=" * 80)
    lines.append("  ANALYSE APPROFONDIE : 3 questions cles")
    lines.append("=" * 80)

    # --- A : Crossover ---
    lines.append("")
    lines.append("--- A) CROSSOVER VPINN(25x8) vs PINN ---")
    lines.append(f"{'beta':>6} {'PINN L2':>11} {'VPINN L2':>11} {'Ratio':>8} {'Gagnant':>10}")
    lines.append("-" * 52)
    for beta in sorted(cross_results.keys()):
        r = cross_results[beta]
        ratio = r["vpinn_L2"] / max(r["pinn_L2"], 1e-20)
        winner = "VPINN" if ratio < 1 else "PINN"
        lines.append(f"{beta:>6.1f} {r['pinn_L2']:>11.3e} {r['vpinn_L2']:>11.3e}"
                      f" {ratio:>8.2f} {winner:>10}")
    if cross_beta is not None:
        lines.append(f"\n  => Crossover a beta* ~ {cross_beta:.2f}")
    else:
        lines.append("\n  => Pas de crossover detecte dans la plage testee")

    # --- B : Push N_test ---
    lines.append("")
    lines.append("--- B) PUSH N_test a beta=5 ---")
    labels = sorted(push_results.keys(),
                    key=lambda k: push_results[k]["total"])
    lines.append(f"{'N_test':>12} {'Total':>6} {'L2':>11} {'Linf':>11} {'Temps':>8}")
    lines.append("-" * 54)
    for lbl in labels:
        r = push_results[lbl]
        lines.append(f"{lbl:>12} {r['total']:>6} {r['L2']:>11.3e}"
                      f" {r['Linf']:>11.3e} {r['time']:>7.1f}s")
    lines.append(f"\n  PINN ref: L2={pinn_ref['L2']:.3e}")

    l2_first = push_results[labels[0]]["L2"]
    l2_last = push_results[labels[-1]]["L2"]
    total_last = push_results[labels[-1]]["total"]
    lines.append(f"  Amelioration {labels[0]} -> {labels[-1]}: x{l2_first/l2_last:.1f}")
    lines.append(f"  Ratio final VPINN/PINN: {l2_last/pinn_ref['L2']:.1f}x")

    # Taux de convergence
    totals = [push_results[k]["total"] for k in labels]
    l2s = [push_results[k]["L2"] for k in labels]
    if len(totals) >= 4:
        log_t = np.log10(np.array(totals[-5:], dtype=float))
        log_l2 = np.log10(np.array(l2s[-5:], dtype=float))
        coeffs = np.polyfit(log_t, log_l2, 1)
        lines.append(f"  Taux de convergence (pente log-log): {coeffs[0]:.2f}")
        log_pinn = np.log10(pinn_ref["L2"])
        ntest_needed = 10 ** ((log_pinn - coeffs[1]) / coeffs[0])
        if 0 < ntest_needed < 1e6:
            lines.append(f"  N_test extrapole pour atteindre PINN: ~{ntest_needed:.0f}")

    # --- C : Quadrature ---
    lines.append("")
    lines.append("--- C) SENSIBILITE A LA QUADRATURE ---")
    for beta in [1.0, 5.0]:
        lines.append(f"\n  beta = {beta}:")
        for prefix in ["25x8", "30x10"]:
            data = [(k, v) for (k, b), v in quad_results.items()
                    if b == beta and k.startswith(prefix)]
            data.sort(key=lambda x: x[1]["n_quad_total"])
            if data:
                l2_min = min(v["L2"] for _, v in data)
                l2_max = max(v["L2"] for _, v in data)
                lines.append(f"    {prefix}: L2 varie de {l2_min:.3e} a {l2_max:.3e}"
                              f" (facteur {l2_max/l2_min:.1f}x)")
                best = min(data, key=lambda x: x[1]["L2"])
                lines.append(f"      Meilleur: {best[0]} -> L2={best[1]['L2']:.3e}")

    lines.append("")
    lines.append(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} min)")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(RESULTS_DIR, "study_deep_analysis_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> study_deep_analysis_summary.txt")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_total = time.time()

    cross_results = study_crossover()
    push_results, pinn_ref = study_push_ntest()
    quad_results = study_quadrature()

    # Sauvegarde
    pkl_path = os.path.join(RESULTS_DIR, "deep_analysis_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "crossover": cross_results,
            "push_ntest": push_results,
            "pinn_ref": pinn_ref,
            "quadrature": quad_results,
        }, f)
    print(f"  -> deep_analysis_results.pkl")

    # Figures
    print("\n" + "=" * 70)
    print("  GENERATION DES FIGURES")
    print("=" * 70)
    cross_beta = fig_crossover(cross_results)
    fig_push_ntest(push_results, pinn_ref)
    fig_quadrature(quad_results)

    # Resume
    total_time = time.time() - t_total
    write_summary(cross_results, cross_beta, push_results, pinn_ref,
                  quad_results, total_time)

    print(f"\n  Termine en {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
