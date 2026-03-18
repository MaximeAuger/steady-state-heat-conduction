"""
Master script — run all studies and generate publication-quality figures.

This script performs four numerical studies comparing PINN (strong form)
and VPINN (weak form) on the 1D heat conduction problem with a nonlocal
integral boundary condition, then generates five figures and a text summary.

Studies
-------
    1. Main comparison      — PINN vs VPINN (baseline)
    2. FDM validation       — independent finite-difference reference
    3. N_test convergence   — VPINN accuracy vs. number of test functions
    4. Robustness           — statistical analysis over 5 random seeds

Outputs  (written to ../figures/)
-------
    fig1_main_comparison.png    — side-by-side PINN vs VPINN
    fig2_fdm_validation.png     — comparison with finite differences
    fig3_ntest_convergence.png  — VPINN convergence study
    fig4_robustness.png         — box-plots over multiple seeds
    fig5_summary_table.png      — consolidated results table
    results_summary.txt         — plain-text summary

Usage
-----
    cd src/
    python run_all.py

Author: Maxime Auger
        Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import os
import sys
import time
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for batch runs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Figure style ──────────────────────────────────────────────────
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

C_PINN  = "#E74C3C"    # red
C_VPINN = "#3498DB"     # blue
C_EXACT = "#2C3E50"     # dark slate
C_FDM   = "#27AE60"     # green

# ── Paths ─────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "figures")

from exact_solution import (
    verify_manufactured_solution, u_exact_np, u_exact_deriv_np,
    f_source_np, ALPHA, G0, G1,
)
from utils import solve_fdm, gauss_legendre_torch, DTYPE
from pinn_solver import train_pinn
from vpinn_solver import train_vpinn


# ==================================================================
# Study 1 — Main comparison
# ==================================================================
def study_main_comparison():
    print("\n" + "=" * 70)
    print("  STUDY 1 — Main comparison: PINN vs VPINN")
    print("=" * 70)
    return train_pinn(verbose=True), train_vpinn(verbose=True)


# ==================================================================
# Study 2 — FDM validation
# ==================================================================
def study_fdm_validation():
    print("\n" + "=" * 70)
    print("  STUDY 2 — Finite-difference reference (FDM)")
    print("=" * 70)
    results = {}
    for N in [100, 500, 2000]:
        x, u = solve_fdm(f_source_np, ALPHA, G0, G1, N=N)
        err = np.max(np.abs(u - u_exact_np(x)))
        results[N] = {"x": x, "u": u, "linf": err, "h": 1.0 / N}
        print(f"  N={N:5d}  h={1.0/N:.1e}  Linf={err:.3e}")
    return results


# ==================================================================
# Study 3 — VPINN convergence vs N_test
# ==================================================================
def study_ntest_convergence():
    print("\n" + "=" * 70)
    print("  STUDY 3 — VPINN convergence vs N_test")
    print("=" * 70)
    ntest_values = [3, 5, 10, 15, 20, 30]
    base = {"n_adam": 8_000, "n_lbfgs": 1_500, "seed": 42}
    results = {}
    for nt in ntest_values:
        print(f"\n  --- N_test = {nt} ---")
        cfg = {**base, "n_test": nt, "n_quad": max(nt + 10, 30)}
        r = train_vpinn(config=cfg, verbose=True)
        results[nt] = {
            "L2": r["errors"]["L2"], "Linf": r["errors"]["Linf"],
            "H1": r["errors"].get("H1", np.nan),
            "constraint": r["constraint_error"], "time": r["time"],
        }
    return results


# ==================================================================
# Study 4 — Robustness (multiple seeds)
# ==================================================================
def study_robustness():
    print("\n" + "=" * 70)
    print("  STUDY 4 — Robustness (5 random seeds)")
    print("=" * 70)
    seeds = [42, 123, 256, 789, 1337]
    base = {"n_adam": 8_000, "n_lbfgs": 1_500}
    pinn_res, vpinn_res = [], []
    for s in seeds:
        print(f"\n  --- seed = {s} ---")
        cfg = {**base, "seed": s}
        rp = train_pinn(config=cfg, verbose=False)
        rv = train_vpinn(config=cfg, verbose=False)
        pinn_res.append({
            "seed": s, "L2": rp["errors"]["L2"],
            "Linf": rp["errors"]["Linf"],
            "H1": rp["errors"].get("H1", np.nan),
            "constraint": rp["constraint_error"], "time": rp["time"],
        })
        vpinn_res.append({
            "seed": s, "L2": rv["errors"]["L2"],
            "Linf": rv["errors"]["Linf"],
            "H1": rv["errors"].get("H1", np.nan),
            "constraint": rv["constraint_error"], "time": rv["time"],
        })
        print(f"    PINN L2={rp['errors']['L2']:.3e}   "
              f"VPINN L2={rv['errors']['L2']:.3e}")
    return pinn_res, vpinn_res


# ==================================================================
# Figure 1 — Main comparison
# ==================================================================
def fig_main(rp, rv):
    ep, ev = rp["errors"], rv["errors"]
    hp, hv = rp["history"], rv["history"]

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # (a) Solutions
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep["x"], ep["u_exact"], color=C_EXACT, lw=2.5,
            label=r"$u_{\mathrm{exact}}$")
    ax.plot(ep["x"], ep["u_pred"], "--", color=C_PINN, lw=2, label="PINN")
    ax.plot(ev["x"], ev["u_pred"], ":",  color=C_VPINN, lw=2.2, label="VPINN")
    ax.set(xlabel="x", ylabel="u(x)", title="(a) Solutions"); ax.legend()

    # (b) Point-wise error
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(ep["x"], np.abs(ep["error"]), color=C_PINN, label="PINN")
    ax.semilogy(ev["x"], np.abs(ev["error"]), color=C_VPINN, label="VPINN")
    ax.set(xlabel="x", ylabel=r"$|u_\theta - u_{\mathrm{exact}}|$",
           title="(b) Absolute error"); ax.legend()

    # (c) Total loss
    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(hp["loss"], color=C_PINN, alpha=0.8, label="PINN")
    ax.semilogy(hv["loss"], color=C_VPINN, alpha=0.8, label="VPINN")
    ax.set(xlabel="Iteration", ylabel="Total loss",
           title="(c) Convergence"); ax.legend()

    # (d) PINN loss breakdown
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(hp["loss"], label="Total", alpha=0.8)
    ax.semilogy(hp["loss_pde"], label="PDE", alpha=0.6)
    ax.semilogy(hp["loss_D"], label="Dirichlet", alpha=0.6)
    ax.semilogy(hp["loss_I"], label="Integral", alpha=0.6)
    ax.set(xlabel="Iteration", ylabel="Loss",
           title="(d) PINN — loss breakdown"); ax.legend(fontsize=8)

    # (e) VPINN loss breakdown
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(hv["loss"], label="Total", alpha=0.8)
    ax.semilogy(hv["loss_weak"], label="Weak", alpha=0.6)
    ax.semilogy(hv["loss_D"], label="Dirichlet", alpha=0.6)
    ax.semilogy(hv["loss_I"], label="Integral", alpha=0.6)
    ax.set(xlabel="Iteration", ylabel="Loss",
           title="(e) VPINN — loss breakdown"); ax.legend(fontsize=8)

    # (f) Summary table
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    data = [
        ["L2 error",   f"{ep['L2']:.2e}",   f"{ev['L2']:.2e}"],
        ["Linf error", f"{ep['Linf']:.2e}", f"{ev['Linf']:.2e}"],
        ["H1 error",   f"{ep.get('H1',0):.2e}", f"{ev.get('H1',0):.2e}"],
        ["Integral BC", f"{rp['constraint_error']:.2e}",
                        f"{rv['constraint_error']:.2e}"],
        ["Time (s)",    f"{rp['time']:.1f}",  f"{rv['time']:.1f}"],
    ]
    t = ax.table(cellText=data, colLabels=["Metric", "PINN", "VPINN"],
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.2, 1.6)
    for j in range(3):
        t[0, j].set_facecolor("#34495E")
        t[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("(f) Summary", pad=20)

    fig.savefig(os.path.join(FIGURES_DIR, "fig1_main_comparison.png"),
                bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Figure 2 — FDM validation
# ==================================================================
def fig_fdm(rp, rv, fdm):
    ep, ev = rp["errors"], rv["errors"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(ep["x"], ep["u_exact"], color=C_EXACT, lw=2.5,
            label=r"$u_{\mathrm{exact}}$")
    ax.plot(ep["x"], ep["u_pred"], "--", color=C_PINN, lw=1.8, label="PINN")
    ax.plot(ev["x"], ev["u_pred"], ":",  color=C_VPINN, lw=2, label="VPINN")
    d = fdm[2000]
    ax.plot(d["x"][::20], d["u"][::20], "s", color=C_FDM, ms=4,
            label="FDM (N=2000)")
    ax.set(xlabel="x", ylabel="u(x)", title="(a) Four methods"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(ep["x"], np.abs(ep["error"]), color=C_PINN, label="PINN")
    ax.semilogy(ev["x"], np.abs(ev["error"]), color=C_VPINN, label="VPINN")
    for N, data in fdm.items():
        ax.semilogy(data["x"], np.abs(data["u"] - u_exact_np(data["x"])),
                    "--", alpha=0.7, label=f"FDM N={N}")
    ax.set(xlabel="x", ylabel="Absolute error", title="(b) Errors")
    ax.legend(fontsize=7)

    ax = axes[2]
    hs   = [fdm[N]["h"]    for N in sorted(fdm)]
    errs = [fdm[N]["linf"] for N in sorted(fdm)]
    ax.loglog(hs, errs, "o-", color=C_FDM, lw=2, ms=8, label="FDM")
    h_ref = np.array([hs[0], hs[-1]])
    ax.loglog(h_ref, errs[0] * (h_ref / hs[0]) ** 2, "k--", alpha=0.5,
              label=r"$O(h^2)$")
    ax.axhline(ep["Linf"], color=C_PINN, ls="--", alpha=0.7, label="PINN")
    ax.axhline(ev["Linf"], color=C_VPINN, ls="--", alpha=0.7, label="VPINN")
    ax.set(xlabel="h", ylabel="Linf error",
           title="(c) FDM convergence vs PINN/VPINN"); ax.legend(fontsize=7)

    fig.savefig(os.path.join(FIGURES_DIR, "fig2_fdm_validation.png"),
                bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Figure 3 — N_test convergence
# ==================================================================
def fig_ntest(ntest_res):
    nts = sorted(ntest_res.keys())
    l2   = [ntest_res[n]["L2"]         for n in nts]
    linf = [ntest_res[n]["Linf"]       for n in nts]
    h1   = [ntest_res[n]["H1"]         for n in nts]
    cstr = [ntest_res[n]["constraint"] for n in nts]
    ts   = [ntest_res[n]["time"]       for n in nts]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.semilogy(nts, l2,   "o-",  color=C_VPINN, ms=7, label="L2")
    ax.semilogy(nts, linf, "s--", color=C_PINN,  ms=7, label="Linf")
    ax.semilogy(nts, h1,   "^:",  color=C_FDM,   ms=7, label="H1")
    ax.set(xlabel=r"$N_{\mathrm{test}}$", ylabel="Error",
           title=r"(a) VPINN error vs $N_{\mathrm{test}}$"); ax.legend()

    ax = axes[1]
    ax.semilogy(nts, cstr, "D-", color="#8E44AD", ms=7)
    ax.set(xlabel=r"$N_{\mathrm{test}}$", ylabel="Integral BC error",
           title="(b) Integral constraint satisfaction")

    ax = axes[2]
    ax.plot(nts, ts, "o-", color="#E67E22", lw=2, ms=7)
    ax.set(xlabel=r"$N_{\mathrm{test}}$", ylabel="Time (s)",
           title="(c) Computational cost")

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_ntest_convergence.png"),
                bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Figure 4 — Robustness
# ==================================================================
def fig_robustness(pr, vr):
    metrics = ["L2", "Linf", "H1", "constraint"]
    labels  = ["L2 error", "Linf error", "H1 error", "Integral BC"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for idx, (m, lab) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        vp = [r[m] for r in pr]
        vv = [r[m] for r in vr]
        bp = ax.boxplot([vp, vv], positions=[0, 1], widths=0.35,
                        patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(C_PINN);  bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(C_VPINN); bp["boxes"][1].set_alpha(0.6)
        for i, vals in enumerate([vp, vv]):
            jit = np.random.RandomState(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jit, vals,
                       color="black", s=25, zorder=5, alpha=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PINN", "VPINN"])
        ax.set(ylabel=lab, title=lab); ax.set_yscale("log")

    fig.suptitle("Robustness over 5 random seeds", fontsize=14, y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_robustness.png"),
                bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Figure 5 — Summary table
# ==================================================================
def fig_summary(rp, rv, pr, vr, fdm):
    ep, ev = rp["errors"], rv["errors"]

    def ms(data, key):
        v = [d[key] for d in data]
        return np.mean(v), np.std(v)

    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
    ax.set_title(
        "Steady-state heat conduction 1D — Nonlocal integral BC\n"
        r"$-u''(x)=f(x)$,  $u(0)=\int_0^1 u\,dx$,  $u(1)=0$",
        fontsize=14, fontweight="bold", pad=20)

    mp_l2, sp_l2 = ms(pr, "L2");       mv_l2, sv_l2 = ms(vr, "L2")
    mp_li, sp_li = ms(pr, "Linf");     mv_li, sv_li = ms(vr, "Linf")
    mp_h1, sp_h1 = ms(pr, "H1");       mv_h1, sv_h1 = ms(vr, "H1")
    mp_c,  sp_c  = ms(pr, "constraint"); mv_c, sv_c = ms(vr, "constraint")
    mp_t,  sp_t  = ms(pr, "time");     mv_t, sv_t = ms(vr, "time")

    data = [
        ["L2 (best run)",
         f"{ep['L2']:.2e}", f"{ev['L2']:.2e}", "—"],
        [u"L2 (mean \u00b1 std)",
         f"{mp_l2:.2e} \u00b1 {sp_l2:.2e}",
         f"{mv_l2:.2e} \u00b1 {sv_l2:.2e}", "—"],
        ["Linf (best)",
         f"{ep['Linf']:.2e}", f"{ev['Linf']:.2e}",
         f"{fdm[2000]['linf']:.2e}"],
        [u"Linf (mean \u00b1 std)",
         f"{mp_li:.2e} \u00b1 {sp_li:.2e}",
         f"{mv_li:.2e} \u00b1 {sv_li:.2e}", "—"],
        ["H1 (best)",
         f"{ep.get('H1',0):.2e}", f"{ev.get('H1',0):.2e}", "—"],
        ["Integral BC (best)",
         f"{rp['constraint_error']:.2e}",
         f"{rv['constraint_error']:.2e}", "—"],
        [u"Time (mean \u00b1 std)",
         f"{mp_t:.1f} \u00b1 {sp_t:.1f}s",
         f"{mv_t:.1f} \u00b1 {sv_t:.1f}s", "< 0.01s"],
    ]
    t = ax.table(cellText=data,
                 colLabels=["Metric", "PINN (strong)", "VPINN (weak)",
                            "FDM (N=2000)"],
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 1.8)
    for j in range(4):
        t[0, j].set_facecolor("#2C3E50")
        t[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)
    for i in range(1, len(data) + 1):
        for j in range(4):
            if i % 2 == 0:
                t[i, j].set_facecolor("#ECF0F1")

    fig.savefig(os.path.join(FIGURES_DIR, "fig5_summary_table.png"),
                bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Text summary
# ==================================================================
def write_summary(rp, rv, fdm, ntest, pr, vr, total):
    ep, ev = rp["errors"], rv["errors"]
    lines = [
        "=" * 70,
        "  SUMMARY — PINN vs VPINN: 1D heat conduction + integral BC",
        "=" * 70, "",
        "Problem:  -u''(x) = f(x),  x in (0,1)",
        "          u(0) = int_0^1 u(x) dx     (integral BC)",
        "          u(1) = 0                    (Dirichlet BC)",
        "Solution: u(x) = sin(pi*x) + (1-x)*4/pi", "",
        "--- Main comparison (seed=42) ---",
        f"{'Metric':<30} {'PINN':>15} {'VPINN':>15}",
        "-" * 62,
        f"{'L2 error':<30} {ep['L2']:>15.4e} {ev['L2']:>15.4e}",
        f"{'Linf error':<30} {ep['Linf']:>15.4e} {ev['Linf']:>15.4e}",
        f"{'H1 error':<30} {ep.get('H1',0):>15.4e} {ev.get('H1',0):>15.4e}",
        f"{'Integral BC':<30} {rp['constraint_error']:>15.4e} "
        f"{rv['constraint_error']:>15.4e}",
        f"{'Time (s)':<30} {rp['time']:>15.1f} {rv['time']:>15.1f}", "",
        "--- FDM validation ---",
    ]
    for N in sorted(fdm):
        lines.append(f"  N={N:<5d}  Linf = {fdm[N]['linf']:.4e}")
    lines += ["", "--- VPINN convergence vs N_test ---",
              f"{'N_test':>8} {'L2':>12} {'Linf':>12} {'H1':>12}"]
    for nt in sorted(ntest):
        r = ntest[nt]
        lines.append(f"{nt:>8d} {r['L2']:>12.3e} {r['Linf']:>12.3e} "
                     f"{r['H1']:>12.3e}")
    lines += ["", "--- Robustness (5 seeds) ---"]
    for lab, d in [("PINN", pr), ("VPINN", vr)]:
        v = [r["L2"] for r in d]
        lines.append(f"  {lab} L2: mean={np.mean(v):.3e}  std={np.std(v):.3e}  "
                     f"min={np.min(v):.3e}  max={np.max(v):.3e}")
    lines += ["", f"Total wall-clock time: {total:.1f} s", "=" * 70]

    txt = "\n".join(lines)
    print("\n" + txt)
    path = os.path.join(FIGURES_DIR, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


# ==================================================================
# Main entry point
# ==================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    t0 = time.time()

    verify_manufactured_solution()

    rp, rv = study_main_comparison()
    fdm    = study_fdm_validation()
    ntest  = study_ntest_convergence()
    pr, vr = study_robustness()

    # Save raw results
    pkl = os.path.join(FIGURES_DIR, "all_results.pkl")
    with open(pkl, "wb") as f:
        pickle.dump({
            "pinn": {k: v for k, v in rp.items() if k != "net"},
            "vpinn": {k: v for k, v in rv.items() if k != "net"},
            "fdm": fdm, "ntest": ntest, "pinn_rob": pr, "vpinn_rob": vr,
        }, f)

    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)
    fig_main(rp, rv);         print("  [done] fig1_main_comparison.png")
    fig_fdm(rp, rv, fdm);    print("  [done] fig2_fdm_validation.png")
    fig_ntest(ntest);         print("  [done] fig3_ntest_convergence.png")
    fig_robustness(pr, vr);   print("  [done] fig4_robustness.png")
    fig_summary(rp, rv, pr, vr, fdm); print("  [done] fig5_summary_table.png")

    total = time.time() - t0
    write_summary(rp, rv, fdm, ntest, pr, vr, total)
    print(f"\n  All done in {total:.0f}s.  Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
