#!/usr/bin/env python3
"""02_mute_forward_mc.py

Robust forward model (NO small-cell approximations) for a 2-plane MuTe-like telescope.

Goal
----
Predict *absolute* counts per discrete direction cell (Δx,Δy) using:
  N_pred(Δx,Δy) = Δt_det * ∫ I(θ) dT_cell(θ,φ)

with intensity defined per unit *perpendicular* area:
  I(θ) = dN / (dA_perp dΩ dt)   [m^-2 sr^-1 s^-1]

and the purely geometric measure:
  dT = (cosθ dA_top) dΩ   subject to:
    (1) ray crosses top plane within the active square
    (2) ray intersects the bottom plane within its active square
    (3) the pair of hit pixels implies a unique (Δx,Δy) cell.

Method (Monte Carlo)
--------------------
We sample rays uniformly in:
  - top-plane position (x0,y0) over the active square (side L=N*a)
  - direction uniformly in solid angle within θ ∈ [0, θ_max] and φ ∈ [0, 2π)

For each ray we compute bottom intersection and discrete pixel indices,
then accumulate two cell tables:

  T_geom_cell = ∫ (cosθ dA) dΩ            [m^2 sr]
  W_cell      = ∫ I(θ) (cosθ dA) dΩ       [s^-1]   (rate per cell)

Then:
  N_pred_cell = Δt_det * W_cell

Inputs
------
- data/arti_Itheta_from_mu_bins.csv  (from 01_arti_histograms_v2.py)
  columns: theta_center_deg, I_m2_sr_s
- data/mapa_pixeles_delta_xy.csv     (optional; for epsilon proxy)
  columns: delta_x, delta_y, counts

Outputs (data/)
---------------
- acceptance_cell_Tgeom_m2sr.csv
- acceptance_cell_rate_s.csv
- mute_pred_counts_abs_mc.csv
- mute_efficiency_proxy_mc.csv (if counts map provided)

Outputs (figs/)
---------------
- mute_counts_predicted_mc.png
- mute_efficiency_proxy_mc.png (if counts map provided)

Notes
-----
- This is purely geometric + ARTI I(θ). No ΔΩ≈cos^3θΔuΔv approximation.
- If later you have I(θ,φ), replace I(theta) by I(theta,phi) in the weight.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def build_grid_from_table(df, dx_min, dx_max, dy_min, dy_max, value_col):
    nx = dx_max - dx_min + 1
    ny = dy_max - dy_min + 1
    grid = np.full((ny, nx), np.nan, dtype=float)
    for _, r in df.iterrows():
        dx, dy = int(r["delta_x"]), int(r["delta_y"])
        if dx_min <= dx <= dx_max and dy_min <= dy <= dy_max:
            grid[dy - dy_min, dx - dx_min] = float(r[value_col])
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arti_I", default="data/arti_Itheta_from_mu_bins.csv")
    ap.add_argument("--counts", default="data/mapa_pixeles_delta_xy.csv",
                    help="Optional measured counts map to compute epsilon proxy.")
    ap.add_argument("--no-counts", action="store_true", help="Skip reading measured counts.")
    ap.add_argument("--dt_det", type=float, default=12*3600.0, help="Detector exposure [s].")
    ap.add_argument("--N", type=int, default=15, help="Pixels per side.")
    ap.add_argument("--a_cm", type=float, default=4.0, help="Pixel pitch [cm].")
    ap.add_argument("--d_cm", type=float, default=30.0, help="Plane separation [cm].")
    ap.add_argument("--theta_max", type=float, default=80.0, help="Max zenith for MC sampling [deg].")
    ap.add_argument("--nsamp", type=int, default=3_000_000, help="Total MC rays.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=".", help="Project root containing data/ and figs/.")
    args = ap.parse_args()

    ROOT = Path(args.outdir).resolve()
    DATA = ROOT / "data"
    FIGS = ROOT / "figs"
    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    # --- Load I(theta)
    dfI = pd.read_csv(ROOT / args.arti_I)
    th = dfI["theta_center_deg"].to_numpy()
    I  = dfI["I_m2_sr_s"].to_numpy()
    m = np.isfinite(th) & np.isfinite(I) & (I > 0)
    th, I = th[m], I[m]
    order = np.argsort(th)
    th, I = th[order], I[order]

    def I_of_theta(theta_deg):
        return np.interp(theta_deg, th, I, left=np.nan, right=np.nan)

    # --- Geometry (meters)
    N = int(args.N)
    a = args.a_cm / 100.0
    d = args.d_cm / 100.0
    L = N * a
    halfL = 0.5 * L

    # Allowed deltas
    dx_min, dx_max = -(N-1), (N-1)
    dy_min, dy_max = -(N-1), (N-1)
    nx = dx_max - dx_min + 1
    ny = dy_max - dy_min + 1

    # Accumulators
    Tgeom = np.zeros((ny, nx), dtype=np.float64)  # [m^2 sr]
    Wrate = np.zeros((ny, nx), dtype=np.float64)  # [s^-1]

    rng = np.random.default_rng(args.seed)

    # --- Sampling ranges
    theta_max = np.deg2rad(args.theta_max)
    mu_min = float(np.cos(theta_max))
    mu_max = 1.0
    Omega = 2.0 * np.pi * (mu_max - mu_min)  # sr
    A_top = L * L                             # m^2

    # Each sample represents dA dΩ:
    wAOm = (A_top * Omega) / args.nsamp        # m^2 sr per ray

    # Chunked vectorization
    chunk = 250_000
    remaining = int(args.nsamp)

    while remaining > 0:
        n = min(chunk, remaining)
        remaining -= n

        x0 = rng.uniform(-halfL, halfL, size=n)
        y0 = rng.uniform(-halfL, halfL, size=n)

        phi = rng.uniform(0.0, 2.0*np.pi, size=n)
        mu  = rng.uniform(mu_min, mu_max, size=n)
        # tanθ = sqrt(1-μ^2)/μ
        sinth = np.sqrt(np.clip(1.0 - mu*mu, 0.0, 1.0))
        tanth = np.divide(sinth, mu, out=np.zeros_like(sinth), where=mu>0)

        sx = d * tanth * np.cos(phi)
        sy = d * tanth * np.sin(phi)

        xb = x0 + sx
        yb = y0 + sy

        hit = (np.abs(xb) <= halfL) & (np.abs(yb) <= halfL)
        if not np.any(hit):
            continue

        x0h, y0h = x0[hit], y0[hit]
        xbh, ybh = xb[hit], yb[hit]
        muh = mu[hit]

        ix0 = np.floor((x0h + halfL) / a).astype(int)
        iy0 = np.floor((y0h + halfL) / a).astype(int)
        ixb = np.floor((xbh + halfL) / a).astype(int)
        iyb = np.floor((ybh + halfL) / a).astype(int)

        inside = (ix0>=0)&(ix0<N)&(iy0>=0)&(iy0<N)&(ixb>=0)&(ixb<N)&(iyb>=0)&(iyb<N)
        if not np.any(inside):
            continue

        ix0, iy0, ixb, iyb, muh = ix0[inside], iy0[inside], ixb[inside], iyb[inside], muh[inside]

        dx = ixb - ix0
        dy = iyb - iy0

        ii = dx - dx_min
        jj = dy - dy_min

        valid = (ii>=0)&(ii<nx)&(jj>=0)&(jj<ny)
        if not np.any(valid):
            continue

        ii = ii[valid]
        jj = jj[valid]
        muvv = muh[valid]

        dT = muvv * wAOm  # [m^2 sr] per ray

        theta_deg = np.rad2deg(np.arccos(np.clip(muvv, 0.0, 1.0)))
        Ivals = I_of_theta(theta_deg)
        goodI = np.isfinite(Ivals) & (Ivals > 0)
        if not np.any(goodI):
            continue

        ii2 = ii[goodI]
        jj2 = jj[goodI]
        dT2 = dT[goodI]
        I2  = Ivals[goodI]

        np.add.at(Tgeom, (jj2, ii2), dT2)
        np.add.at(Wrate, (jj2, ii2), I2 * dT2)

    # --- Output tables
    rows = []
    for dy in range(dy_min, dy_max+1):
        for dx in range(dx_min, dx_max+1):
            ii = dx - dx_min
            jj = dy - dy_min
            rows.append({
                "delta_x": int(dx),
                "delta_y": int(dy),
                "T_geom_m2sr": float(Tgeom[jj, ii]),
                "rate_pred_s": float(Wrate[jj, ii]),
                "counts_pred": float(Wrate[jj, ii] * args.dt_det),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA / "mute_pred_counts_abs_mc.csv", index=False)
    df_out[["delta_x","delta_y","T_geom_m2sr"]].to_csv(DATA / "acceptance_cell_Tgeom_m2sr.csv", index=False)
    df_out[["delta_x","delta_y","rate_pred_s"]].to_csv(DATA / "acceptance_cell_rate_s.csv", index=False)

    # --- Plot predicted counts
    pred_grid = build_grid_from_table(df_out, dx_min, dx_max, dy_min, dy_max, "counts_pred")

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

    imshow_grid(pred_grid, f"Predicted counts (MC geom, Δt={args.dt_det/3600:.1f} h)", "mute_counts_predicted_mc.png", log=True)

    # --- Optional epsilon proxy
    if not args.no_counts:
        dfm = pd.read_csv(ROOT / args.counts)
        # ensure columns
        if "counts" in dfm.columns and "counts_meas" not in dfm.columns:
            dfm = dfm.rename(columns={"counts":"counts_meas"})
        meas_grid = build_grid_from_table(dfm, dx_min, dx_max, dy_min, dy_max, "counts_meas")
        eps = np.divide(meas_grid, pred_grid, out=np.full_like(meas_grid, np.nan), where=(pred_grid>0))
        eps_rows = []
        for dy in range(dy_min, dy_max+1):
            for dx in range(dx_min, dx_max+1):
                ii = dx - dx_min
                jj = dy - dy_min
                eps_rows.append({
                    "delta_x": int(dx),
                    "delta_y": int(dy),
                    "eps_proxy": float(eps[jj, ii]) if np.isfinite(eps[jj, ii]) else np.nan
                })
        pd.DataFrame(eps_rows).to_csv(DATA / "mute_efficiency_proxy_mc.csv", index=False)
        imshow_grid(eps, "Efficiency proxy (MC geom): ε = N_meas / N_pred", "mute_efficiency_proxy_mc.png", log=False)

    meta = {
        "method": "MC over top-plane position and direction uniform in solid angle",
        "A_top_m2": A_top,
        "theta_max_deg": float(args.theta_max),
        "Omega_sr": Omega,
        "nsamp": int(args.nsamp),
        "dt_det_s": float(args.dt_det),
        "geometry": {"N": N, "a_cm": float(args.a_cm), "d_cm": float(args.d_cm)},
        "I_definition": "perpendicular area (dA_perp = cosθ dA_horiz)"
    }
    (DATA / "mute_forward_mc_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done. Wrote:", DATA / "mute_pred_counts_abs_mc.csv")

if __name__ == "__main__":
    main()
