#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two 2D maps (simulation vs experiment) stored as CSV with columns:
  - delta_x (or similar), delta_y (or similar), counts (or similar)

Outputs:
  - compare_marginal_dx.png
  - compare_marginal_dy.png
  - compare_scatter_prob.png
  - compare_second_moment_ellipses.png
  - compare_ratio_map.png

And prints a console report with all key metrics:
  totals, peaks, centroids, covariance, sigmas, anisotropy, Pearson r, TV, JS divergence.
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ Column detection ------------------------

def _normalize_colname(c: str) -> str:
    return "".join(ch.lower() for ch in str(c).strip())

def autodetect_columns(df: pd.DataFrame, xcol=None, ycol=None, vcol=None):
    cols = list(df.columns)
    cols_norm = {_normalize_colname(c): c for c in cols}

    def pick(candidates):
        for cand in candidates:
            key = _normalize_colname(cand)
            if key in cols_norm:
                return cols_norm[key]
        # try "contains"
        for cand in candidates:
            key = _normalize_colname(cand)
            for k, orig in cols_norm.items():
                if key in k:
                    return orig
        return None

    if xcol is None:
        xcol = pick([
            "delta_x", "dx", "deltax", "Δx", "delx", "x", "ix", "i_x"
        ])
    if ycol is None:
        ycol = pick([
            "delta_y", "dy", "deltay", "Δy", "dely", "y", "iy", "i_y"
        ])
    if vcol is None:
        vcol = pick([
            "counts", "count", "coincidencias", "coincidences", "numero_de_coincidencias",
            "n", "value", "entries"
        ])

    missing = [name for name, col in [("xcol", xcol), ("ycol", ycol), ("vcol", vcol)] if col is None]
    if missing:
        raise ValueError(
            f"No pude autodetectar columnas {missing}. Columnas disponibles: {cols}\n"
            f"Sugerencia: ejecuta con --xcol --ycol --vcol."
        )
    return xcol, ycol, vcol


# ------------------------ Grid building & alignment ------------------------

def df_to_pivot_grid(df: pd.DataFrame, xcol: str, ycol: str, vcol: str) -> pd.DataFrame:
    # Ensure numeric x/y
    d = df[[xcol, ycol, vcol]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol, vcol])

    grid = pd.pivot_table(
        d, index=ycol, columns=xcol, values=vcol,
        aggfunc="sum", fill_value=0.0
    )

    # sort indices/columns numerically
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid

def align_grids(g_sim: pd.DataFrame, g_exp: pd.DataFrame):
    xs = np.array(sorted(set(g_sim.columns.astype(float)).union(set(g_exp.columns.astype(float)))), dtype=float)
    ys = np.array(sorted(set(g_sim.index.astype(float)).union(set(g_exp.index.astype(float)))), dtype=float)

    g_sim_al = g_sim.reindex(index=ys, columns=xs, fill_value=0.0)
    g_exp_al = g_exp.reindex(index=ys, columns=xs, fill_value=0.0)
    return xs, ys, g_sim_al.to_numpy(dtype=float), g_exp_al.to_numpy(dtype=float)


# ------------------------ Metrics ------------------------

def safe_normalize(grid: np.ndarray) -> np.ndarray:
    total = float(grid.sum())
    if total <= 0:
        raise ValueError("El mapa tiene suma total <= 0, no se puede normalizar.")
    return grid / total

def moments(xs: np.ndarray, ys: np.ndarray, grid_counts: np.ndarray):
    # Use probability map
    P = safe_normalize(grid_counts)
    X, Y = np.meshgrid(xs, ys)

    mx = float((P * X).sum())
    my = float((P * Y).sum())

    cx = float((P * (X - mx) ** 2).sum())
    cy = float((P * (Y - my) ** 2).sum())
    cxy = float((P * (X - mx) * (Y - my)).sum())

    cov = np.array([[cx, cxy], [cxy, cy]], dtype=float)

    # Eigen decomposition for principal axes
    w, v = np.linalg.eigh(cov)  # ascending
    idx = np.argsort(w)[::-1]   # descending
    w = w[idx]
    v = v[:, idx]

    std_major = float(math.sqrt(max(w[0], 0.0)))
    std_minor = float(math.sqrt(max(w[1], 0.0)))

    # angle of major axis eigenvector
    angle_deg = float(np.degrees(np.arctan2(v[1, 0], v[0, 0])))

    std_x = float(math.sqrt(max(cx, 0.0)))
    std_y = float(math.sqrt(max(cy, 0.0)))

    anisotropy = float(std_major / std_minor) if std_minor > 0 else float("inf")

    return {
        "mean": (mx, my),
        "cov": cov,
        "std_x": std_x,
        "std_y": std_y,
        "std_major": std_major,
        "std_minor": std_minor,
        "angle_deg": angle_deg,
        "anisotropy": anisotropy,
    }

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def total_variation(Pa: np.ndarray, Pb: np.ndarray) -> float:
    return 0.5 * float(np.abs(Pa - Pb).sum())

def js_divergence_bits(Pa: np.ndarray, Pb: np.ndarray, eps: float = 1e-12) -> float:
    a = Pa.ravel().astype(float) + eps
    b = Pb.ravel().astype(float) + eps
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    js = 0.5 * (np.sum(a * np.log(a / m)) + np.sum(b * np.log(b / m)))
    return float(js / np.log(2.0))


# ------------------------ Plots ------------------------

def ellipse_points(mean, cov, nsigma=1.0, n=400):
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    t = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2, n)

    scale = np.diag(nsigma * np.sqrt(np.maximum(w, 0.0)))
    pts = (v @ scale @ circle).T + np.array(mean, dtype=float)
    return pts

def save_marginals(xs, ys, P_sim, P_exp, outdir):
    mx_sim = P_sim.sum(axis=0)
    mx_exp = P_exp.sum(axis=0)
    my_sim = P_sim.sum(axis=1)
    my_exp = P_exp.sum(axis=1)

    # Δx marginal
    plt.figure(figsize=(7, 5))
    plt.plot(xs, mx_sim, label="Simulation (normalized)")
    plt.plot(xs, mx_exp, label="Experiment (normalized)")
    plt.xlabel("Δx (bar-index difference)")
    plt.ylabel("Probability per Δx bin")
    plt.title("Marginal distribution along Δx")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "compare_marginal_dx.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # Δy marginal
    plt.figure(figsize=(7, 5))
    plt.plot(ys, my_sim, label="Simulation (normalized)")
    plt.plot(ys, my_exp, label="Experiment (normalized)")
    plt.xlabel("Δy (bar-index difference)")
    plt.ylabel("Probability per Δy bin")
    plt.title("Marginal distribution along Δy")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "compare_marginal_dy.png")
    plt.savefig(p, dpi=200)
    plt.close()

def save_scatter(P_sim, P_exp, pearson_r, outdir):
    ps = P_sim.ravel()
    pe = P_exp.ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(ps, pe, s=10, alpha=0.6)
    plt.xlabel("Simulation probability per bin")
    plt.ylabel("Experiment probability per bin")
    plt.title(f"Bin-by-bin comparison (Pearson r = {pearson_r:.3f})")
    mx = max(float(ps.max()), float(pe.max()))
    plt.plot([0, mx], [0, mx], linestyle="--", label="y = x")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "compare_scatter_prob.png")
    plt.savefig(p, dpi=200)
    plt.close()

def save_ellipses(mom_sim, mom_exp, outdir):
    pts_s = ellipse_points(mom_sim["mean"], mom_sim["cov"], nsigma=1.0)
    pts_e = ellipse_points(mom_exp["mean"], mom_exp["cov"], nsigma=1.0)

    plt.figure(figsize=(6, 6))
    plt.plot(pts_s[:, 0], pts_s[:, 1], label="Simulation (1σ ellipse)")
    plt.plot(pts_e[:, 0], pts_e[:, 1], label="Experiment (1σ ellipse)")
    plt.scatter([mom_sim["mean"][0]], [mom_sim["mean"][1]], s=30, label="Sim centroid")
    plt.scatter([mom_exp["mean"][0]], [mom_exp["mean"][1]], s=30, label="Exp centroid")
    plt.xlabel("Δx (bar-index difference)")
    plt.ylabel("Δy (bar-index difference)")
    plt.title("Shape comparison via 2nd moments")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "compare_second_moment_ellipses.png")
    plt.savefig(p, dpi=200)
    plt.close()

def save_ratio_map(xs, ys, P_sim, P_exp, outdir, eps=1e-12):
    ratio = (P_exp + eps) / (P_sim + eps)

    plt.figure(figsize=(6, 5.5))
    plt.imshow(
        ratio,
        origin="lower",
        extent=[xs.min() - 0.5, xs.max() + 0.5, ys.min() - 0.5, ys.max() + 0.5],
        aspect="equal",
    )
    plt.colorbar(label="(Experiment / Simulation) probability ratio")
    plt.xlabel("Δx (bar-index difference)")
    plt.ylabel("Δy (bar-index difference)")
    plt.title("2D ratio map after normalization")
    plt.tight_layout()
    p = os.path.join(outdir, "compare_ratio_map.png")
    plt.savefig(p, dpi=200)
    plt.close()


# ------------------------ Reporting ------------------------

def print_report(name_sim, name_exp, xs, ys, G_sim, G_exp, mom_sim, mom_exp, pearson_r, tv, js_bits):
    sim_total = float(G_sim.sum())
    exp_total = float(G_exp.sum())
    sim_peak = float(G_sim.max())
    exp_peak = float(G_exp.max())

    def fmt_cov(C):
        return (f"[[{C[0,0]:.6f}, {C[0,1]:.6f}],\n"
                f" [{C[1,0]:.6f}, {C[1,1]:.6f}]]")

    print("\n" + "=" * 80)
    print("2D MAP COMPARISON REPORT (shape-only via normalization)")
    print("=" * 80)

    print(f"\nGrid:")
    print(f"  Δx bins: {len(xs)}   range: [{xs.min():.3f}, {xs.max():.3f}]")
    print(f"  Δy bins: {len(ys)}   range: [{ys.min():.3f}, {ys.max():.3f}]")
    print(f"  Grid shape (rows=Δy, cols=Δx): {G_sim.shape}")

    print(f"\nRaw counts (NOT normalized):")
    print(f"  {name_sim}: total = {sim_total:.0f}   peak = {sim_peak:.0f}")
    print(f"  {name_exp}: total = {exp_total:.0f}   peak = {exp_peak:.0f}")
    if exp_total > 0:
        print(f"  Total ratio (sim/exp) = {sim_total/exp_total:.3f}")
    if exp_peak > 0:
        print(f"  Peak ratio  (sim/exp) = {sim_peak/exp_peak:.3f}")

    print(f"\nSecond-moment shape metrics (computed on normalized P = N / sum(N)):")

    print(f"\n  {name_sim}:")
    print(f"    centroid (μx, μy) = ({mom_sim['mean'][0]:.6f}, {mom_sim['mean'][1]:.6f})")
    print(f"    σx = {mom_sim['std_x']:.6f}   σy = {mom_sim['std_y']:.6f}")
    print(f"    σ_major = {mom_sim['std_major']:.6f}   σ_minor = {mom_sim['std_minor']:.6f}")
    print(f"    anisotropy (σ_major/σ_minor) = {mom_sim['anisotropy']:.6f}")
    print(f"    major-axis angle [deg] = {mom_sim['angle_deg']:.6f}")
    print(f"    covariance =\n{fmt_cov(mom_sim['cov'])}")

    print(f"\n  {name_exp}:")
    print(f"    centroid (μx, μy) = ({mom_exp['mean'][0]:.6f}, {mom_exp['mean'][1]:.6f})")
    print(f"    σx = {mom_exp['std_x']:.6f}   σy = {mom_exp['std_y']:.6f}")
    print(f"    σ_major = {mom_exp['std_major']:.6f}   σ_minor = {mom_exp['std_minor']:.6f}")
    print(f"    anisotropy (σ_major/σ_minor) = {mom_exp['anisotropy']:.6f}")
    print(f"    major-axis angle [deg] = {mom_exp['angle_deg']:.6f}")
    print(f"    covariance =\n{fmt_cov(mom_exp['cov'])}")

    print(f"\nGlobal similarity metrics (bin-by-bin, on normalized maps):")
    print(f"  Pearson correlation r = {pearson_r:.6f}   (1=identical up to linear scaling)")
    print(f"  Total variation TV    = {tv:.6f}         (0=identical, 1=maximally different)")
    print(f"  JS divergence (bits)  = {js_bits:.6f}    (0=identical)")

    print("\n" + "=" * 80)
    print("End of report.")
    print("=" * 80 + "\n")


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare two 2D coincidence maps (sim vs exp) from CSV.")
    ap.add_argument("--sim", required=True, help="Simulation CSV path")
    ap.add_argument("--exp", required=True, help="Experiment CSV path")
    ap.add_argument("--outdir", default=".", help="Output directory for PNG figures")
    ap.add_argument("--xcol", default=None, help="Name of Δx column (overrides autodetect)")
    ap.add_argument("--ycol", default=None, help="Name of Δy column (overrides autodetect)")
    ap.add_argument("--vcol", default=None, help="Name of counts column (overrides autodetect)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_sim = pd.read_csv(args.sim)
    df_exp = pd.read_csv(args.exp)

    # Detect columns (or use user-specified)
    xcol_s, ycol_s, vcol_s = autodetect_columns(df_sim, args.xcol, args.ycol, args.vcol)
    xcol_e, ycol_e, vcol_e = autodetect_columns(df_exp, args.xcol, args.ycol, args.vcol)

    # Build pivot grids
    gsim = df_to_pivot_grid(df_sim, xcol_s, ycol_s, vcol_s)
    gexp = df_to_pivot_grid(df_exp, xcol_e, ycol_e, vcol_e)

    # Align onto common (Δx, Δy) support
    xs, ys, Gs, Ge = align_grids(gsim, gexp)

    # Normalized probability maps
    Ps = safe_normalize(Gs)
    Pe = safe_normalize(Ge)

    # Metrics
    mom_s = moments(xs, ys, Gs)
    mom_e = moments(xs, ys, Ge)
    r = pearson_corr(Ps, Pe)
    tv = total_variation(Ps, Pe)
    jsb = js_divergence_bits(Ps, Pe, eps=1e-12)

    # Plots
    save_marginals(xs, ys, Ps, Pe, args.outdir)
    save_scatter(Ps, Pe, r, args.outdir)
    save_ellipses(mom_s, mom_e, args.outdir)
    save_ratio_map(xs, ys, Ps, Pe, args.outdir)

    # Report
    print("\nDetected columns:")
    print(f"  SIM: x={xcol_s}  y={ycol_s}  v={vcol_s}")
    print(f"  EXP: x={xcol_e}  y={ycol_e}  v={vcol_e}")

    print_report(
        name_sim="Simulation",
        name_exp="Experiment",
        xs=xs, ys=ys,
        G_sim=Gs, G_exp=Ge,
        mom_sim=mom_s, mom_exp=mom_e,
        pearson_r=r, tv=tv, js_bits=jsb
    )

    print("Saved figures in:", os.path.abspath(args.outdir))
    for fn in [
        "compare_marginal_dx.png",
        "compare_marginal_dy.png",
        "compare_scatter_prob.png",
        "compare_second_moment_ellipses.png",
        "compare_ratio_map.png",
    ]:
        print("  -", os.path.join(os.path.abspath(args.outdir), fn))


if __name__ == "__main__":
    main()
