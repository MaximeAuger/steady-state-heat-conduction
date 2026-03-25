"""
========================================================================
  Etude complementaire : Carte de precision VPINN (N_test x beta)

  Question : peut-on sauver le VPINN a beta=5 en augmentant N_test ?

  Produit :
    - Heatmap N_test x beta (erreur L2)
    - Courbes de coupe : L2 vs N_test pour chaque beta
    - Courbe iso-beta : L2 vs beta pour chaque N_test
    - Tableau recapitulatif texte
========================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.8,
})

C_PINN = "#E74C3C"
C_VPINN = "#3498DB"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

from vpinn_solver import train_vpinn
from pinn_solver import train_pinn
from exact_solution import BETA_DEFAULT


# ============================================================
# Configuration du sweep
# ============================================================
# N_test pairs: (n_test_x, n_test_y) -> label, N_total
NTEST_PAIRS = [
    (5, 3),    # 15
    (10, 3),   # 30
    (15, 5),   # 75
    (20, 5),   # 100
    (25, 8),   # 200
    (30, 10),  # 300
]

BETAS = [0.0, 0.5, 1.0, 2.0, 5.0]

# Config allégée pour le sweep (10k Adam + 2k LBFGS par run)
CFG_BASE = {
    "n_adam": 10000,
    "n_lbfgs": 2000,
    "seed": 42,
}


def ntest_label(ntx, nty):
    return f"{ntx}x{nty}"


# ============================================================
# Sweep principal
# ============================================================
def run_sweep():
    """Lance le sweep N_test x beta et retourne les resultats."""
    print("\n" + "=" * 70)
    print("  SWEEP N_test x beta -- VPINN 2D NL")
    print("=" * 70)

    n_runs = len(NTEST_PAIRS) * len(BETAS)
    print(f"  {len(NTEST_PAIRS)} configs N_test x {len(BETAS)} betas = {n_runs} runs VPINN")
    print(f"  Estimation : ~{n_runs * 3:.0f} min (a ~3 min/run)\n")

    results = {}
    run_idx = 0

    for ntx, nty in NTEST_PAIRS:
        for beta in BETAS:
            run_idx += 1
            label = ntest_label(ntx, nty)
            print(f"  [{run_idx:2d}/{n_runs}] N_test={label}, beta={beta}")

            cfg = {
                **CFG_BASE,
                "beta": beta,
                "n_test_x": ntx,
                "n_test_y": nty,
                # Quadrature suffisante pour le nombre de fonctions test
                "n_quad_x": max(ntx + 15, 30),
                "n_quad_y": max(nty + 10, 20),
            }

            t0 = time.time()
            res = train_vpinn(config=cfg, verbose=False)
            elapsed = time.time() - t0

            key = (label, beta)
            results[key] = {
                "ntx": ntx,
                "nty": nty,
                "n_total": ntx * nty,
                "beta": beta,
                "L2": res["errors"]["L2"],
                "Linf": res["errors"]["Linf"],
                "H1": res["errors"].get("H1", np.nan),
                "constraint": res["constraint_error"],
                "lam0": res["lam"][0],
                "time": elapsed,
            }

            print(f"           L2={res['errors']['L2']:.3e}  "
                  f"Linf={res['errors']['Linf']:.3e}  "
                  f"lam0={res['lam'][0]:.4f}  ({elapsed:.1f}s)")

    # Aussi lancer PINN comme référence pour chaque beta
    print("\n  --- PINN reference pour chaque beta ---")
    pinn_ref = {}
    for beta in BETAS:
        cfg_pinn = {**CFG_BASE, "beta": beta}
        rp = train_pinn(config=cfg_pinn, verbose=False)
        pinn_ref[beta] = {
            "L2": rp["errors"]["L2"],
            "Linf": rp["errors"]["Linf"],
            "time": rp["time"],
        }
        print(f"    PINN beta={beta}: L2={rp['errors']['L2']:.3e}")

    return results, pinn_ref


# ============================================================
# Figure 1 : Heatmap N_test x beta
# ============================================================
def fig_heatmap(results, pinn_ref):
    """Heatmap de l'erreur L2 VPINN en fonction de N_test et beta."""
    labels = [ntest_label(ntx, nty) for ntx, nty in NTEST_PAIRS]
    totals = [ntx * nty for ntx, nty in NTEST_PAIRS]

    # Construire la matrice
    n_ntest = len(NTEST_PAIRS)
    n_beta = len(BETAS)
    Z = np.zeros((n_ntest, n_beta))

    for i, (ntx, nty) in enumerate(NTEST_PAIRS):
        for j, beta in enumerate(BETAS):
            key = (ntest_label(ntx, nty), beta)
            Z[i, j] = results[key]["L2"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # --- (a) Heatmap ---
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Z, aspect='auto', cmap='RdYlGn_r', norm=LogNorm(),
                   origin='lower')
    ax.set_xticks(range(n_beta))
    ax.set_xticklabels([f"{b}" for b in BETAS])
    ax.set_yticks(range(n_ntest))
    ax.set_yticklabels([f"{l}\n({t})" for l, t in zip(labels, totals)])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$N_{\mathrm{test}}$ (total)")
    ax.set_title(r"(a) Erreur L2 VPINN : $N_{\mathrm{test}} \times \beta$")

    # Annoter chaque cellule
    for i in range(n_ntest):
        for j in range(n_beta):
            val = Z[i, j]
            color = "white" if val > 5e-3 else "black"
            ax.text(j, i, f"{val:.1e}", ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    cb = plt.colorbar(im, ax=ax, shrink=0.8, label="Erreur L2")

    # --- (b) Coupes : L2 vs N_test pour chaque beta ---
    ax = fig.add_subplot(gs[0, 1])
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_beta))

    for j, beta in enumerate(BETAS):
        l2_vals = [Z[i, j] for i in range(n_ntest)]
        ax.semilogy(totals, l2_vals, 'o-', color=colors[j], lw=2, ms=7,
                    label=rf"$\beta={beta}$")
        # Ligne PINN reference
        if j == 0:
            ax.axhline(pinn_ref[beta]["L2"], color=colors[j], ls='--', alpha=0.4)

    # Bande PINN pour comparaison
    pinn_min = min(pinn_ref[b]["L2"] for b in BETAS)
    pinn_max = max(pinn_ref[b]["L2"] for b in BETAS)
    ax.axhspan(pinn_min, pinn_max, color=C_PINN, alpha=0.1, label="PINN range")

    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Erreur L2")
    ax.set_title(r"(b) VPINN L2 vs $N_{\mathrm{test}}$ (par $\beta$)")
    ax.legend(fontsize=8, ncol=2)

    # --- (c) Coupes : L2 vs beta pour chaque N_test ---
    ax = fig.add_subplot(gs[1, 0])
    colors2 = plt.cm.viridis(np.linspace(0.1, 0.9, n_ntest))

    for i, (ntx, nty) in enumerate(NTEST_PAIRS):
        l2_vals = [Z[i, j] for j in range(n_beta)]
        label = f"{ntest_label(ntx, nty)} ({ntx*nty})"
        ax.semilogy(BETAS, l2_vals, 's-', color=colors2[i], lw=2, ms=7,
                    label=label)

    # PINN reference
    pinn_l2 = [pinn_ref[b]["L2"] for b in BETAS]
    ax.semilogy(BETAS, pinn_l2, 'D--', color=C_PINN, lw=2.5, ms=9,
                label="PINN", zorder=10)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Erreur L2")
    ax.set_title(r"(c) L2 vs $\beta$ (par $N_{\mathrm{test}}$)")
    ax.legend(fontsize=7, ncol=2)

    # --- (d) Ratio VPINN/PINN ---
    ax = fig.add_subplot(gs[1, 1])

    for i, (ntx, nty) in enumerate(NTEST_PAIRS):
        ratios = []
        for j, beta in enumerate(BETAS):
            vpinn_l2 = Z[i, j]
            pinn_l2_val = pinn_ref[beta]["L2"]
            ratios.append(vpinn_l2 / max(pinn_l2_val, 1e-20))
        label = f"{ntest_label(ntx, nty)} ({ntx*nty})"
        ax.semilogy(BETAS, ratios, 's-', color=colors2[i], lw=2, ms=7,
                    label=label)

    ax.axhline(1.0, color='black', ls='--', lw=1.5, alpha=0.5, label="VPINN = PINN")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Ratio L2(VPINN) / L2(PINN)")
    ax.set_title(r"(d) Ratio de precision VPINN/PINN")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(r"Carte de precision VPINN : peut-on sauver le VPINN a fort $\beta$ ?",
                 fontsize=14, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig6_ntest_beta_heatmap.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig6_ntest_beta_heatmap.png")


# ============================================================
# Figure 2 : Focus beta=5 (la question cle)
# ============================================================
def fig_focus_beta5(results, pinn_ref):
    """Focus sur beta=5 : evolution de l'erreur avec N_test."""
    labels = [ntest_label(ntx, nty) for ntx, nty in NTEST_PAIRS]
    totals = [ntx * nty for ntx, nty in NTEST_PAIRS]

    beta5_l2 = []
    beta5_linf = []
    beta5_cstr = []
    beta5_times = []
    beta5_lam0 = []

    for ntx, nty in NTEST_PAIRS:
        key = (ntest_label(ntx, nty), 5.0)
        r = results[key]
        beta5_l2.append(r["L2"])
        beta5_linf.append(r["Linf"])
        beta5_cstr.append(r["constraint"])
        beta5_times.append(r["time"])
        beta5_lam0.append(r["lam0"])

    pinn_l2_b5 = pinn_ref[5.0]["L2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) L2 et Linf vs N_test a beta=5
    ax = axes[0]
    ax.semilogy(totals, beta5_l2, 'o-', color=C_VPINN, lw=2.5, ms=9,
                label="VPINN L2")
    ax.semilogy(totals, beta5_linf, 's--', color=C_VPINN, lw=1.5, ms=7,
                alpha=0.6, label="VPINN Linf")
    ax.axhline(pinn_l2_b5, color=C_PINN, ls='--', lw=2, label=f"PINN L2 ({pinn_l2_b5:.2e})")

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (totals[i], beta5_l2[i]), fontsize=7,
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')

    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Erreur")
    ax.set_title(r"(a) Erreur vs $N_{\mathrm{test}}$ ($\beta=5$)")
    ax.legend()

    # (b) Contrainte integrale vs N_test
    ax = axes[1]
    ax.semilogy(totals, beta5_cstr, 'D-', color="#8E44AD", lw=2, ms=8)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel("Erreur contrainte")
    ax.set_title(r"(b) Contrainte integrale ($\beta=5$)")

    # (c) Improvement factor
    ax = axes[2]
    # Normaliser par la valeur a N_test le plus petit
    baseline = beta5_l2[0]
    improvement = [baseline / l2 for l2 in beta5_l2]
    ax.plot(totals, improvement, 'o-', color="#E67E22", lw=2.5, ms=9)
    ax.set_xlabel(r"$N_{\mathrm{test}}$ total")
    ax.set_ylabel(f"Facteur d'amelioration (vs {labels[0]})")
    ax.set_title(r"(c) Gain par rapport a $N_{\mathrm{test}}$ min")

    # Annoter le facteur final
    ax.annotate(f"x{improvement[-1]:.1f}", (totals[-1], improvement[-1]),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black'))

    fig.suptitle(r"Focus $\beta=5$ : le VPINN peut-il etre sauve ?",
                 fontsize=14, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig7_focus_beta5.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig7_focus_beta5.png")


# ============================================================
# Resume texte
# ============================================================
def write_summary(results, pinn_ref, total_time):
    labels = [ntest_label(ntx, nty) for ntx, nty in NTEST_PAIRS]
    totals = [ntx * nty for ntx, nty in NTEST_PAIRS]

    lines = []
    lines.append("=" * 80)
    lines.append("  ETUDE COMPLEMENTAIRE : Carte de precision VPINN (N_test x beta)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Question : peut-on sauver le VPINN a beta=5 en augmentant N_test ?")
    lines.append("")

    # Tableau complet
    lines.append("--- Erreur L2 VPINN ---")
    header = f"{'N_test':>12}" + "".join(f"  beta={b:>3}" for b in BETAS)
    lines.append(header)
    lines.append("-" * len(header))

    for i, (ntx, nty) in enumerate(NTEST_PAIRS):
        label = f"{ntest_label(ntx, nty)} ({totals[i]:>3d})"
        vals = ""
        for beta in BETAS:
            key = (ntest_label(ntx, nty), beta)
            vals += f"  {results[key]['L2']:>9.2e}"
        lines.append(f"{label:>12}{vals}")

    lines.append("")
    lines.append("--- PINN reference ---")
    for beta in BETAS:
        lines.append(f"  beta={beta}: L2={pinn_ref[beta]['L2']:.3e}")

    lines.append("")

    # Focus beta=5
    lines.append("--- Focus beta=5 ---")
    l2_first = results[(ntest_label(*NTEST_PAIRS[0]), 5.0)]["L2"]
    l2_last = results[(ntest_label(*NTEST_PAIRS[-1]), 5.0)]["L2"]
    pinn_l2_5 = pinn_ref[5.0]["L2"]
    lines.append(f"  N_test min ({labels[0]}):  L2 = {l2_first:.3e}")
    lines.append(f"  N_test max ({labels[-1]}): L2 = {l2_last:.3e}")
    lines.append(f"  Amelioration :           x{l2_first/l2_last:.1f}")
    lines.append(f"  PINN reference :         L2 = {pinn_l2_5:.3e}")
    lines.append(f"  Ratio final VPINN/PINN : {l2_last/pinn_l2_5:.1f}x")
    lines.append("")

    # Conclusion
    vpinn_beats_pinn = l2_last < pinn_l2_5
    if vpinn_beats_pinn:
        lines.append("  CONCLUSION : OUI, le VPINN peut battre le PINN a beta=5")
        lines.append("  avec suffisamment de fonctions test.")
    else:
        ratio = l2_last / pinn_l2_5
        if ratio < 5:
            lines.append(f"  CONCLUSION : Le VPINN se rapproche du PINN (ratio {ratio:.1f}x)")
            lines.append("  mais ne le depasse pas encore a beta=5.")
            lines.append("  Plus de fonctions test pourraient combler l'ecart.")
        else:
            lines.append(f"  CONCLUSION : Le VPINN reste {ratio:.0f}x moins precis que le PINN")
            lines.append("  a beta=5, meme avec N_test=300.")
            lines.append("  La formulation forte (PINN) est nettement superieure pour")
            lines.append("  les non-linearites fortes en 2D.")

    lines.append("")
    lines.append(f"Temps total : {total_time:.1f} s ({total_time/60:.1f} min)")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(RESULTS_DIR, "study_ntest_beta_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> study_ntest_beta_summary.txt")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_total = time.time()

    # Sweep
    results, pinn_ref = run_sweep()

    # Sauvegarder les resultats bruts
    pkl_path = os.path.join(RESULTS_DIR, "ntest_beta_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"results": results, "pinn_ref": pinn_ref}, f)
    print(f"  -> ntest_beta_results.pkl")

    # Figures
    print("\n" + "=" * 70)
    print("  GENERATION DES FIGURES")
    print("=" * 70)
    fig_heatmap(results, pinn_ref)
    fig_focus_beta5(results, pinn_ref)

    # Resume
    total_time = time.time() - t_total
    write_summary(results, pinn_ref, total_time)

    print(f"\n  Termine en {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
