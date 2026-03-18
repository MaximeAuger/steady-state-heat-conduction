"""
Master script: run all studies for the nonlinear PINN vs VPINN benchmark.

Studies
-------
1. Main comparison   — PINN vs VPINN vs FDM  (beta=1)
2. Linear vs NL      — compare training difficulty (beta=0 vs beta=1)
3. Nonlinearity      — effect of beta on accuracy
4. N_test convergence — VPINN accuracy vs number of test functions
5. Robustness        — multiple random seeds

Generates 5 publication-quality figures in ../results/

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exact_solution import (
    u_exact, u_exact_deriv, f_source, integral_u_exact,
    ALPHA, G0, G1, BETA_DEFAULT,
    k_conductivity, dk_conductivity,
)
from pinn_solver import train_pinn
from vpinn_solver import train_vpinn
from utils import solve_fdm_nonlinear, solve_fdm_linear, compute_errors

# ===================================================================
# Paths
# ===================================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================================================================
# Matplotlib styling
# ===================================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})

ALL_RESULTS = {}


# ===================================================================
# Study 1: Main comparison  (beta = 1)
# ===================================================================

def study_main_comparison():
    print("\n" + "=" * 70)
    print("STUDY 1: Main comparison  PINN vs VPINN vs FDM  (beta = 1)")
    print("=" * 70)

    res_pinn = train_pinn(verbose=True)
    res_vpinn = train_vpinn(verbose=True)

    # FDM reference
    print("\n[FDM] Nonlinear Newton-Raphson (N=2000)...")
    x_fdm, u_fdm = solve_fdm_nonlinear(
        f_source, k_conductivity, dk_conductivity,
        ALPHA, G0, G1, N=2000, verbose=True, beta=BETA_DEFAULT,
    )
    u_ex_fdm = u_exact(x_fdm)
    fdm_L2 = np.sqrt(np.trapz((u_fdm - u_ex_fdm) ** 2, x_fdm))

    print(f"\n[FDM] L2 = {fdm_L2:.2e}")

    ALL_RESULTS["study1"] = {
        "pinn": res_pinn,
        "vpinn": res_vpinn,
        "fdm": {"x": x_fdm, "u": u_fdm, "L2": fdm_L2},
    }
    return res_pinn, res_vpinn


def plot_fig1():
    """Figure 1: Main comparison — solutions, errors, convergence."""
    d = ALL_RESULTS["study1"]
    rp, rv = d["pinn"], d["vpinn"]
    ep, ev = rp["errors"], rv["errors"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # (0,0) Solutions
    ax = axes[0, 0]
    ax.plot(ep["x"], ep["u_exact"], "k-", label="Exact", linewidth=2)
    ax.plot(ep["x"], ep["u_pred"], "r--", label="PINN")
    ax.plot(ev["x"], ev["u_pred"], "b-.", label="VPINN")
    ax.plot(d["fdm"]["x"], d["fdm"]["u"], "g:", label="FDM", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Solutions")
    ax.legend()

    # (0,1) Pointwise errors
    ax = axes[0, 1]
    ax.semilogy(ep["x"], np.abs(ep["error"]), "r-", label=f"PINN (L2={ep['L2']:.2e})")
    ax.semilogy(ev["x"], np.abs(ev["error"]), "b-", label=f"VPINN (L2={ev['L2']:.2e})")
    u_fdm_interp = np.interp(ep["x"], d["fdm"]["x"], d["fdm"]["u"])
    ax.semilogy(ep["x"], np.abs(u_fdm_interp - ep["u_exact"]), "g--",
                label=f"FDM (L2={d['fdm']['L2']:.2e})", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("|error|")
    ax.set_title("Pointwise error")
    ax.legend()

    # (0,2) Loss convergence
    ax = axes[0, 2]
    ax.semilogy(rp["history"]["loss"], "r-", alpha=0.6, label="PINN")
    ax.semilogy(rv["history"]["loss"], "b-", alpha=0.6, label="VPINN")
    ax.axvline(rp["config"]["n_adam"], color="gray", ls=":", alpha=0.5, label="Adam→LBFGS")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")
    ax.set_title("Convergence")
    ax.legend()

    # (1,0) PINN loss breakdown
    ax = axes[1, 0]
    ax.semilogy(rp["history"]["loss_pde"], "r-", alpha=0.5, label="PDE")
    ax.semilogy(rp["history"]["loss_D"], "g-", alpha=0.5, label="Dirichlet")
    ax.semilogy(rp["history"]["loss_I"], "b-", alpha=0.5, label="Integral BC")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss component")
    ax.set_title("PINN loss breakdown")
    ax.legend()

    # (1,1) VPINN loss breakdown
    ax = axes[1, 1]
    ax.semilogy(rv["history"]["loss_weak"], "r-", alpha=0.5, label="Weak residual")
    ax.semilogy(rv["history"]["loss_D"], "g-", alpha=0.5, label="Dirichlet")
    ax.semilogy(rv["history"]["loss_I"], "b-", alpha=0.5, label="Integral BC")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss component")
    ax.set_title("VPINN loss breakdown")
    ax.legend()

    # (1,2) Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Method", "L2", "Linf", "Constraint", "Time (s)"],
        ["PINN", f"{ep['L2']:.2e}", f"{ep['Linf']:.2e}",
         f"{rp['constraint_error']:.2e}", f"{rp['time']:.0f}"],
        ["VPINN", f"{ev['L2']:.2e}", f"{ev['Linf']:.2e}",
         f"{rv['constraint_error']:.2e}", f"{rv['time']:.0f}"],
        ["FDM", f"{d['fdm']['L2']:.2e}", "—", "—", "—"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for j in range(5):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary", pad=20)

    fig.suptitle("Nonlinear Heat Conduction: PINN vs VPINN (k(u)=1+u²)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_main_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("[Fig 1] Saved: fig1_main_comparison.png")


# ===================================================================
# Study 2: Linear vs Nonlinear comparison
# ===================================================================

def study_linear_vs_nonlinear():
    print("\n" + "=" * 70)
    print("STUDY 2: Linear (beta=0) vs Nonlinear (beta=1)")
    print("=" * 70)

    # Linear (beta=0)
    print("\n--- Linear (beta = 0) ---")
    rp_lin = train_pinn({"beta": 0.0}, verbose=True)
    rv_lin = train_vpinn({"beta": 0.0}, verbose=True)

    # Nonlinear already computed in study 1
    rp_nl = ALL_RESULTS["study1"]["pinn"]
    rv_nl = ALL_RESULTS["study1"]["vpinn"]

    ALL_RESULTS["study2"] = {
        "pinn_lin": rp_lin, "vpinn_lin": rv_lin,
        "pinn_nl": rp_nl, "vpinn_nl": rv_nl,
    }


def plot_fig2():
    """Figure 2: Linear vs Nonlinear — side by side comparison."""
    d = ALL_RESULTS["study2"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (0) L2 error comparison (bar chart)
    ax = axes[0]
    methods = ["PINN\nβ=0", "VPINN\nβ=0", "PINN\nβ=1", "VPINN\nβ=1"]
    l2_vals = [
        d["pinn_lin"]["errors"]["L2"],
        d["vpinn_lin"]["errors"]["L2"],
        d["pinn_nl"]["errors"]["L2"],
        d["vpinn_nl"]["errors"]["L2"],
    ]
    colors = ["#FF6B6B", "#4ECDC4", "#C0392B", "#1ABC9C"]
    bars = ax.bar(methods, l2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("L2 error")
    ax.set_yscale("log")
    ax.set_title("Accuracy comparison")
    for bar, val in zip(bars, l2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                f"{val:.1e}", ha="center", va="bottom", fontsize=8)

    # (1) Training time comparison
    ax = axes[1]
    times = [
        d["pinn_lin"]["time"], d["vpinn_lin"]["time"],
        d["pinn_nl"]["time"], d["vpinn_nl"]["time"],
    ]
    bars = ax.bar(methods, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Training time (s)")
    ax.set_title("Computational cost")
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=8)

    # (2) Loss convergence overlay
    ax = axes[2]
    ax.semilogy(d["pinn_lin"]["history"]["loss"], "r-", alpha=0.4, label="PINN β=0")
    ax.semilogy(d["vpinn_lin"]["history"]["loss"], "b-", alpha=0.4, label="VPINN β=0")
    ax.semilogy(d["pinn_nl"]["history"]["loss"], "r-", alpha=0.8, linewidth=2, label="PINN β=1")
    ax.semilogy(d["vpinn_nl"]["history"]["loss"], "b-", alpha=0.8, linewidth=2, label="VPINN β=1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")
    ax.set_title("Convergence: linear vs nonlinear")
    ax.legend()

    fig.suptitle("Linear vs Nonlinear Problem Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_linear_vs_nonlinear.png"), bbox_inches="tight")
    plt.close(fig)
    print("[Fig 2] Saved: fig2_linear_vs_nonlinear.png")


# ===================================================================
# Study 3: Effect of nonlinearity strength beta
# ===================================================================

def study_nonlinearity_strength():
    print("\n" + "=" * 70)
    print("STUDY 3: Effect of nonlinearity strength beta")
    print("=" * 70)

    betas = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = []

    for beta in betas:
        print(f"\n--- beta = {beta} ---")
        rp = train_pinn({"beta": beta, "n_adam": 15000, "n_lbfgs": 3000}, verbose=True)
        rv = train_vpinn({"beta": beta, "n_adam": 15000, "n_lbfgs": 3000}, verbose=True)
        results.append({"beta": beta, "pinn": rp, "vpinn": rv})

    ALL_RESULTS["study3"] = results


def plot_fig3():
    """Figure 3: Effect of beta on accuracy."""
    results = ALL_RESULTS["study3"]

    betas = [r["beta"] for r in results]
    pinn_l2 = [r["pinn"]["errors"]["L2"] for r in results]
    vpinn_l2 = [r["vpinn"]["errors"]["L2"] for r in results]
    pinn_linf = [r["pinn"]["errors"]["Linf"] for r in results]
    vpinn_linf = [r["vpinn"]["errors"]["Linf"] for r in results]
    pinn_cstr = [r["pinn"]["constraint_error"] for r in results]
    vpinn_cstr = [r["vpinn"]["constraint_error"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (0) L2 error vs beta
    ax = axes[0]
    ax.semilogy(betas, pinn_l2, "rs-", label="PINN", markersize=8)
    ax.semilogy(betas, vpinn_l2, "bo-", label="VPINN", markersize=8)
    ax.set_xlabel("β (nonlinearity strength)")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error vs β")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1) Linf error vs beta
    ax = axes[1]
    ax.semilogy(betas, pinn_linf, "rs-", label="PINN", markersize=8)
    ax.semilogy(betas, vpinn_linf, "bo-", label="VPINN", markersize=8)
    ax.set_xlabel("β (nonlinearity strength)")
    ax.set_ylabel("L∞ error")
    ax.set_title("L∞ error vs β")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Constraint error vs beta
    ax = axes[2]
    ax.semilogy(betas, pinn_cstr, "rs-", label="PINN", markersize=8)
    ax.semilogy(betas, vpinn_cstr, "bo-", label="VPINN", markersize=8)
    ax.set_xlabel("β (nonlinearity strength)")
    ax.set_ylabel("Integral BC error")
    ax.set_title("Constraint satisfaction vs β")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Nonlinearity Strength k(u) = 1 + βu²",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_nonlinearity_strength.png"), bbox_inches="tight")
    plt.close(fig)
    print("[Fig 3] Saved: fig3_nonlinearity_strength.png")


# ===================================================================
# Study 4: VPINN convergence with N_test
# ===================================================================

def study_ntest_convergence():
    print("\n" + "=" * 70)
    print("STUDY 4: VPINN convergence with N_test")
    print("=" * 70)

    n_tests = [3, 5, 10, 15, 20, 30]
    results = []

    for nt in n_tests:
        print(f"\n--- N_test = {nt} ---")
        rv = train_vpinn({"n_test": nt}, verbose=True)
        results.append({"n_test": nt, "vpinn": rv})

    ALL_RESULTS["study4"] = results


def plot_fig4():
    """Figure 4: VPINN convergence vs number of test functions."""
    results = ALL_RESULTS["study4"]

    n_tests = [r["n_test"] for r in results]
    l2 = [r["vpinn"]["errors"]["L2"] for r in results]
    linf = [r["vpinn"]["errors"]["Linf"] for r in results]
    cstr = [r["vpinn"]["constraint_error"] for r in results]
    times = [r["vpinn"]["time"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (0) L2 and Linf
    ax = axes[0]
    ax.semilogy(n_tests, l2, "bo-", label="L2", markersize=8)
    ax.semilogy(n_tests, linf, "rs-", label="L∞", markersize=8)
    ax.set_xlabel("Number of test functions")
    ax.set_ylabel("Error")
    ax.set_title("VPINN accuracy vs N_test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1) Constraint error
    ax = axes[1]
    ax.semilogy(n_tests, cstr, "go-", markersize=8)
    ax.set_xlabel("Number of test functions")
    ax.set_ylabel("Integral BC error")
    ax.set_title("Constraint satisfaction vs N_test")
    ax.grid(True, alpha=0.3)

    # (2) Training time
    ax = axes[2]
    ax.plot(n_tests, times, "ko-", markersize=8)
    ax.set_xlabel("Number of test functions")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Cost vs N_test")
    ax.grid(True, alpha=0.3)

    fig.suptitle("VPINN Convergence: Number of Test Functions (Nonlinear, β=1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "fig4_ntest_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("[Fig 4] Saved: fig4_ntest_convergence.png")


# ===================================================================
# Study 5: Robustness (multiple seeds)
# ===================================================================

def study_robustness():
    print("\n" + "=" * 70)
    print("STUDY 5: Robustness analysis (5 random seeds)")
    print("=" * 70)

    seeds = [42, 123, 256, 789, 1337]
    pinn_runs = []
    vpinn_runs = []

    for s in seeds:
        print(f"\n--- seed = {s} ---")
        rp = train_pinn({"seed": s}, verbose=True)
        rv = train_vpinn({"seed": s}, verbose=True)
        pinn_runs.append(rp)
        vpinn_runs.append(rv)

    ALL_RESULTS["study5"] = {"seeds": seeds, "pinn": pinn_runs, "vpinn": vpinn_runs}


def plot_fig5():
    """Figure 5: Robustness — boxplots over multiple seeds."""
    d = ALL_RESULTS["study5"]
    pinn_runs = d["pinn"]
    vpinn_runs = d["vpinn"]

    def extract(runs, key):
        return [r["errors"][key] for r in runs]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics = [("L2", "L2 error"), ("Linf", "L∞ error"),
               ("H1", "H1 error"), (None, "Integral BC error")]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        if key is not None:
            pv = extract(pinn_runs, key)
            vv = extract(vpinn_runs, key)
        else:
            pv = [r["constraint_error"] for r in pinn_runs]
            vv = [r["constraint_error"] for r in vpinn_runs]

        bp = ax.boxplot([pv, vv], labels=["PINN", "VPINN"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#FF6B6B")
        bp["boxes"][1].set_facecolor("#4ECDC4")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate mean
        for i, vals in enumerate([pv, vv]):
            mean_val = np.mean(vals)
            ax.annotate(f"μ={mean_val:.1e}", xy=(i + 1, mean_val),
                        fontsize=7, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5))

    fig.suptitle("Robustness Analysis (5 seeds, Nonlinear β=1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(RESULTS_DIR, "fig5_robustness.png"), bbox_inches="tight")
    plt.close(fig)
    print("[Fig 5] Saved: fig5_robustness.png")


# ===================================================================
# Text summary
# ===================================================================

def write_summary():
    """Write a text summary of all results."""
    lines = ["=" * 70, "NONLINEAR BENCHMARK RESULTS SUMMARY", "=" * 70, ""]
    lines.append(f"PDE: -[k(u) u']' = f(x),  k(u) = 1 + beta*u^2")
    lines.append(f"Manufactured solution: u(x) = sin(pi*x) + (1-x)*4/pi")
    lines.append("")

    # Study 1
    d1 = ALL_RESULTS["study1"]
    lines.append("--- Study 1: Main Comparison (beta=1) ---")
    lines.append(f"  PINN  L2 = {d1['pinn']['errors']['L2']:.2e}  "
                 f"Linf = {d1['pinn']['errors']['Linf']:.2e}  "
                 f"time = {d1['pinn']['time']:.0f}s")
    lines.append(f"  VPINN L2 = {d1['vpinn']['errors']['L2']:.2e}  "
                 f"Linf = {d1['vpinn']['errors']['Linf']:.2e}  "
                 f"time = {d1['vpinn']['time']:.0f}s")
    lines.append(f"  FDM   L2 = {d1['fdm']['L2']:.2e}")
    lines.append("")

    # Study 2
    d2 = ALL_RESULTS["study2"]
    lines.append("--- Study 2: Linear vs Nonlinear ---")
    for tag, label in [("pinn_lin", "PINN β=0"), ("vpinn_lin", "VPINN β=0"),
                       ("pinn_nl", "PINN β=1"), ("vpinn_nl", "VPINN β=1")]:
        r = d2[tag]
        lines.append(f"  {label:12s}  L2 = {r['errors']['L2']:.2e}  time = {r['time']:.0f}s")
    lines.append("")

    # Study 3
    lines.append("--- Study 3: Nonlinearity Strength ---")
    for r in ALL_RESULTS["study3"]:
        lines.append(f"  beta={r['beta']:.1f}  "
                     f"PINN L2={r['pinn']['errors']['L2']:.2e}  "
                     f"VPINN L2={r['vpinn']['errors']['L2']:.2e}")
    lines.append("")

    # Study 4
    lines.append("--- Study 4: VPINN N_test Convergence ---")
    for r in ALL_RESULTS["study4"]:
        lines.append(f"  N_test={r['n_test']:3d}  "
                     f"L2={r['vpinn']['errors']['L2']:.2e}  "
                     f"time={r['vpinn']['time']:.0f}s")
    lines.append("")

    # Study 5
    d5 = ALL_RESULTS["study5"]
    pinn_l2 = [r["errors"]["L2"] for r in d5["pinn"]]
    vpinn_l2 = [r["errors"]["L2"] for r in d5["vpinn"]]
    lines.append("--- Study 5: Robustness (5 seeds) ---")
    lines.append(f"  PINN  L2: mean={np.mean(pinn_l2):.2e}  "
                 f"std={np.std(pinn_l2):.2e}  "
                 f"min={np.min(pinn_l2):.2e}  max={np.max(pinn_l2):.2e}")
    lines.append(f"  VPINN L2: mean={np.mean(vpinn_l2):.2e}  "
                 f"std={np.std(vpinn_l2):.2e}  "
                 f"min={np.min(vpinn_l2):.2e}  max={np.max(vpinn_l2):.2e}")

    txt = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\n[Summary] Saved: {path}")
    print(txt)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    t_total = time.time()

    # Run all studies
    study_main_comparison()
    plot_fig1()

    study_linear_vs_nonlinear()
    plot_fig2()

    study_nonlinearity_strength()
    plot_fig3()

    study_ntest_convergence()
    plot_fig4()

    study_robustness()
    plot_fig5()

    # Save raw results
    pkl_path = os.path.join(RESULTS_DIR, "all_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(ALL_RESULTS, fh)
    print(f"\n[Pickle] Saved: {pkl_path}")

    write_summary()

    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {time.time() - t_total:.0f}s")
    print(f"{'=' * 70}")
