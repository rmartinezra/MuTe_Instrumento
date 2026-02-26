#!/usr/bin/env python3
"""02_mute_corrections_v2.py

Forward model for a 2-panel MuTe-like hodoscope (two 15x15 pixel planes; pixel pitch a=4 cm;
separation d=30 cm). We predict *absolute* counts in each discrete direction (Δx,Δy) using:

  N_pred(Δx,Δy) = Δt_det * I(θ) * T_geom(Δx,Δy) * ε(θ)

with intensity per unit *perpendicular* area:
  I(θ) = dN / (dA_perp dΩ dt)

Geometric term per discrete (u,v) cell:
  u = s_x/d,  v = s_y/d,  with s_x=Δx a, s_y=Δy a
  cosθ = 1/sqrt(1+u^2+v^2)

  ΔΩ_cell ≈ cos^3θ * (Δu Δv),  with Δu=Δv=a/d   (uniform u,v grid)
  A_ov = max(0, L-|s_x|) * max(0, L-|s_y|), L = N a

  T_geom = (A_ov * cosθ) * ΔΩ_cell     [m^2 sr] if A_ov in m^2

Notes:
- Here we use ΔΩ_cell approximation; you can swap it for an exact solid-angle
  from the acceptance PDF (Van Oosterom–Strackee) or MC ray-tracing.
- Default ε(θ)=1 (ideal). You can add a phenomenological ε(θ)=cos^kθ if desired.

Inputs:
  data/mapa_pixeles_delta_xy.csv        columns: delta_x, delta_y, counts
  data/arti_Itheta_from_mu_bins.csv     output of 01_arti_histograms_v2.py

Outputs (data/):
  mute_pred_counts_abs.csv
  mute_efficiency_proxy.csv

Outputs (figs/):
  mute_counts_measured.png
  mute_counts_predicted.png
  mute_efficiency_proxy.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def build_grid(df_counts, dx_min, dx_max, dy_min, dy_max):
    nx = dx_max - dx_min + 1
    ny = dy_max - dy_min + 1
    grid = np.zeros((ny, nx), dtype=float)  # rows=dy, cols=dx
    for _, r in df_counts.iterrows():
        dx, dy, c = int(r["delta_x"]), int(r["delta_y"]), float(r["counts"])
        if dx_min <= dx <= dx_max and dy_min <= dy <= dy_max:
            grid[dy - dy_min, dx - dx_min] = c
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="data/mapa_pixeles_delta_xy.csv")
    ap.add_argument("--arti_I", default="data/arti_Itheta_from_mu_bins.csv")
    ap.add_argument("--dt_det", type=float, default=12*3600.0, help="Detector exposure [s] (default 12h).")
    ap.add_argument("--N", type=int, default=15, help="Pixels per side (15).")
    ap.add_argument("--a_cm", type=float, default=4.0, help="Pixel pitch [cm].")
    ap.add_argument("--d_cm", type=float, default=30.0, help="Panel separation [cm].")
    ap.add_argument("--eps_model", choices=["ideal", "cosk"], default="ideal")
    ap.add_argument("--k", type=float, default=0.0, help="If eps_model=cosk, ε(θ)=cos^k θ.")
    ap.add_argument("--outdir", default=".", help="Project root containing data/ and figs/.")
    args = ap.parse_args()

    ROOT = Path(args.outdir).resolve()
    DATA = ROOT / "data"
    FIGS = ROOT / "figs"
    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / args.counts)
    dfI = pd.read_csv(ROOT / args.arti_I)

    # Intensity curve I(θ) from ARTI (μ bins) -> interpolate in θ
    th_I = dfI["theta_center_deg"].to_numpy()
    I = dfI["I_m2_sr_s"].to_numpy()
    # Clean
    mI = np.isfinite(th_I) & np.isfinite(I) & (I > 0)
    th_I = th_I[mI]
    I = I[mI]
    # Sort by theta
    order = np.argsort(th_I)
    th_I, I = th_I[order], I[order]

    def I_of_theta(theta_deg):
        return np.interp(theta_deg, th_I, I, left=np.nan, right=np.nan)

    # Geometry
    N = int(args.N)
    a = args.a_cm / 100.0  # m
    d = args.d_cm / 100.0  # m
    L = N * a              # m
    du = a / d
    dv = a / d

    # Determine allowed Δx,Δy: for two N×N planes, deltas in [-(N-1), +(N-1)]
    dx_min, dx_max = -(N-1), (N-1)
    dy_min, dy_max = -(N-1), (N-1)

    # Build measured grid (missing entries -> 0)
    meas = build_grid(df, dx_min, dx_max, dy_min, dy_max)

    # Compute predicted counts
    dx_vals = np.arange(dx_min, dx_max + 1)
    dy_vals = np.arange(dy_min, dy_max + 1)

    pred = np.full_like(meas, np.nan, dtype=float)
    Tgeom = np.full_like(meas, np.nan, dtype=float)
    theta_grid = np.full_like(meas, np.nan, dtype=float)

    for j, dy in enumerate(dy_vals):
        for i, dx in enumerate(dx_vals):
            sx = dx * a
            sy = dy * a

            ax = max(0.0, L - abs(sx))
            ay = max(0.0, L - abs(sy))
            Aov = ax * ay  # m^2

            if Aov <= 0:
                pred[j, i] = 0.0
                Tgeom[j, i] = 0.0
                theta_grid[j, i] = np.nan
                continue

            u = sx / d
            v = sy / d
            cos_th = 1.0 / np.sqrt(1.0 + u*u + v*v)
            th = np.rad2deg(np.arccos(np.clip(cos_th, 0, 1)))
            theta_grid[j, i] = th

            dOmega_cell = (cos_th**3) * (du * dv)  # sr (approx)
            T = (Aov * cos_th) * dOmega_cell       # m^2 sr
            Tgeom[j, i] = T

            I_th = I_of_theta(th)  # m^-2 sr^-1 s^-1
            if not np.isfinite(I_th):
                pred[j, i] = np.nan
                continue

            if args.eps_model == "ideal":
                eps = 1.0
            else:
                eps = cos_th**args.k

            pred[j, i] = args.dt_det * I_th * T * eps

    # Efficiency proxy from data (idealized): ε_proxy = N_meas / N_pred
    eps_proxy = np.full_like(meas, np.nan, dtype=float)
    mgood = np.isfinite(pred) & (pred > 0)
    eps_proxy[mgood] = meas[mgood] / pred[mgood]

    # Save per-(dx,dy) table
    rows = []
    for j, dy in enumerate(dy_vals):
        for i, dx in enumerate(dx_vals):
            rows.append({
                "delta_x": int(dx),
                "delta_y": int(dy),
                "theta_deg": float(theta_grid[j, i]) if np.isfinite(theta_grid[j, i]) else np.nan,
                "counts_meas": float(meas[j, i]),
                "T_geom_m2sr": float(Tgeom[j, i]),
                "counts_pred": float(pred[j, i]) if np.isfinite(pred[j, i]) else np.nan,
                "eps_proxy": float(eps_proxy[j, i]) if np.isfinite(eps_proxy[j, i]) else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "mute_pred_counts_abs.csv", index=False)

    pd.DataFrame(rows)[["delta_x","delta_y","eps_proxy"]].to_csv(DATA / "mute_efficiency_proxy.csv", index=False)

    # Plot helpers
    def imshow_grid(arr, title, fname, log=False):
        plt.figure(figsize=(6.5, 5.6))
        A = arr.copy()
        if log:
            A = np.where(A > 0, A, np.nan)
            plt.imshow(np.log10(A), origin="lower", aspect="equal",
                       extent=[dx_min-0.5, dx_max+0.5, dy_min-0.5, dy_max+0.5])
            plt.colorbar(label="log10(value)")
        else:
            plt.imshow(A, origin="lower", aspect="equal",
                       extent=[dx_min-0.5, dx_max+0.5, dy_min-0.5, dy_max+0.5])
            plt.colorbar(label="value")
        plt.xlabel(r"$\Delta x$ (pixels)")
        plt.ylabel(r"$\Delta y$ (pixels)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGS / fname, dpi=200)
        plt.close()

    imshow_grid(meas, "Measured counts (input map)", "mute_counts_measured.png", log=True)
    imshow_grid(pred, f"Predicted counts (Δt={args.dt_det/3600:.1f} h)", "mute_counts_predicted.png", log=True)
    imshow_grid(eps_proxy, "Efficiency proxy: ε = N_meas / N_pred", "mute_efficiency_proxy.png", log=False)

if __name__ == "__main__":
    main()
