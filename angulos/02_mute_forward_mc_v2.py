#!/usr/bin/env python3
"""02_mute_forward_mc_v2.py

Robust (no small-cell approximations) Monte-Carlo forward model for a 2-plane MuTe-like telescope,
plus optional *inference* of the effective intensity used by a Geant4 injection from the measured
(Δx,Δy) count map.

Key definitions
--------------
Intensity per unit perpendicular area:
  I(θ) = dN / (dA_perp dΩ dt)   [m^-2 sr^-1 s^-1]
with dA_perp = cosθ dA_horiz.

Geometric acceptance per cell:
  T_geom(Δx,Δy) = ∫_cell (cosθ dA) dΩ      [m^2 sr]

Forward prediction (absolute):
  N_pred(Δx,Δy) = Δt_det * ∫_cell I(θ) (cosθ dA) dΩ
                = Δt_det * W(Δx,Δy)        where W has units s^-1

Monte-Carlo method (ray tracing)
--------------------------------
Sample uniformly:
  - (x0,y0) on top active square (side L=N*a)
  - direction uniform in solid angle: φ uniform, μ=cosθ uniform in [cosθ_max, 1]

Propagate to bottom plane, require intersection within active square.
Determine top/bottom pixel indices and define cell (Δx,Δy) = (ix_b-ix_0, iy_b-iy_0).

Accumulate per cell:
  - T_geom = Σ (μ * wAΩ)
  - W      = Σ (I(θ) * μ * wAΩ)
  - <θ>_T  = (Σ θ * μ * wAΩ) / T_geom   (diagnostic: typical θ per cell, weighted by geometric measure)

Additionally, if a measured map N_meas(Δx,Δy) is provided, we can infer an *effective* intensity
shape used by the Geant4 injection:
  I_eff_cell ≈ N_meas / (Δt_det * T_geom)
and then bin I_eff_cell versus <θ>_T to get I_eff(θ).

Why this matters
----------------
If Geant4 was injected with an angular distribution different from ARTI (e.g. restricted θ range,
cos^k, fixed spectrum, etc.), then N_meas will not match ARTI-based N_pred even with perfect geometry.
The I_eff diagnostic lets you see what flux your Geant4 sample effectively corresponds to.

Inputs
------
- data/arti_Itheta_from_mu_bins.csv  (from 01_arti_histograms_v2.py)  [optional if --infer-only]
- data/mapa_pixeles_delta_xy.csv     (optional)

Outputs (data/)
---------------
- mute_pred_counts_abs_mc.csv
- acceptance_cell_Tgeom_m2sr.csv
- acceptance_cell_theta_mean_deg.csv
- mute_efficiency_proxy_mc.csv (if counts map provided)
- inferred_Ieff_from_meas.csv  (if counts map provided)

Outputs (figs/)
---------------
- mute_counts_predicted_mc.png
- mute_efficiency_proxy_mc.png (if counts map provided)
- inferred_Ieff_vs_theta.png   (if counts map provided)

"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def build_grid_from_table(df, dx_min, dx_max, dy_min, dy_max, value_col, fill_value=np.nan):
    nx = dx_max - dx_min + 1
    ny = dy_max - dy_min + 1
    grid = np.full((ny, nx), fill_value, dtype=float)
    for _, r in df.iterrows():
        dx, dy = int(r["delta_x"]), int(r["delta_y"])
        if dx_min <= dx <= dx_max and dy_min <= dy <= dy_max:
            grid[dy - dy_min, dx - dx_min] = float(r[value_col])
    return grid

def imshow_grid(arr, dx_min, dx_max, dy_min, dy_max, title, fname, outdir, log=False):
    import matplotlib.pyplot as plt
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
    plt.savefig(Path(outdir) / fname, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arti_I", default="data/arti_Itheta_from_mu_bins.csv",
                    help="ARTI-derived I(theta) table. Not required if --infer-only.")
    ap.add_argument("--counts", default="data/mapa_pixeles_delta_xy.csv",
                    help="Measured/simulated (Δx,Δy) counts map.")
    ap.add_argument("--no-counts", action="store_true", help="Skip reading measured counts.")
    ap.add_argument("--infer-only", action="store_true",
                    help="Do NOT compute ARTI-based prediction; only compute T_geom and infer I_eff from counts.")
    ap.add_argument("--dt_det", type=float, default=12*3600.0, help="Detector exposure [s].")
    ap.add_argument("--N", type=int, default=15, help="Pixels per side.")
    ap.add_argument("--a_cm", type=float, default=4.0, help="Pixel pitch [cm].")
    ap.add_argument("--d_cm", type=float, default=30.0, help="Plane separation [cm].")
    ap.add_argument("--theta_max", type=float, default=80.0, help="Max zenith for MC sampling [deg].")
    ap.add_argument("--nsamp", type=int, default=3_000_000, help="Total MC rays.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--theta_bins", type=float, default=2.0, help="Bin width for inferred I_eff(θ) [deg].")
    ap.add_argument("--outdir", default=".", help="Project root containing data/ and figs/.")
    args = ap.parse_args()

    ROOT = Path(args.outdir).resolve()
    DATA = ROOT / "data"
    FIGS = ROOT / "figs"
    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    # --- Load I(theta) if needed
    if not args.infer_only:
        dfI = pd.read_csv(ROOT / args.arti_I)
        th = dfI["theta_center_deg"].to_numpy()
        I  = dfI["I_m2_sr_s"].to_numpy()
        m = np.isfinite(th) & np.isfinite(I) & (I > 0)
        th, I = th[m], I[m]
        order = np.argsort(th)
        th, I = th[order], I[order]

        def I_of_theta(theta_deg):
            return np.interp(theta_deg, th, I, left=np.nan, right=np.nan)
    else:
        def I_of_theta(theta_deg):
            return np.full_like(np.array(theta_deg, dtype=float), np.nan, dtype=float)

    # --- Geometry (meters)
    N = int(args.N)
    a = args.a_cm / 100.0
    d = args.d_cm / 100.0
    L = N * a
    halfL = 0.5 * L

    dx_min, dx_max = -(N-1), (N-1)
    dy_min, dy_max = -(N-1), (N-1)
    nx = dx_max - dx_min + 1
    ny = dy_max - dy_min + 1

    # Accumulators
    Tgeom = np.zeros((ny, nx), dtype=np.float64)       # Σ μ wAΩ
    Wrate = np.zeros((ny, nx), dtype=np.float64)       # Σ I(θ) μ wAΩ
    Th_m1 = np.zeros((ny, nx), dtype=np.float64)       # Σ θ * μ wAΩ  (θ in deg)

    rng = np.random.default_rng(args.seed)

    theta_max = np.deg2rad(args.theta_max)
    mu_min = float(np.cos(theta_max))
    mu_max = 1.0
    Omega = 2.0 * np.pi * (mu_max - mu_min)  # sr
    A_top = L * L                             # m^2
    wAOm = (A_top * Omega) / args.nsamp        # m^2 sr per ray

    chunk = 250_000
    remaining = int(args.nsamp)

    while remaining > 0:
        n = min(chunk, remaining)
        remaining -= n

        x0 = rng.uniform(-halfL, halfL, size=n)
        y0 = rng.uniform(-halfL, halfL, size=n)

        phi = rng.uniform(0.0, 2.0*np.pi, size=n)
        mu  = rng.uniform(mu_min, mu_max, size=n)

        sinth = np.sqrt(np.clip(1.0 - mu*mu, 0.0, 1.0))
        tanth = np.divide(sinth, mu, out=np.zeros_like(sinth), where=mu>0)

        xb = x0 + d * tanth * np.cos(phi)
        yb = y0 + d * tanth * np.sin(phi)

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

        dT = muvv * wAOm
        theta_deg = np.rad2deg(np.arccos(np.clip(muvv, 0.0, 1.0)))

        np.add.at(Tgeom, (jj, ii), dT)
        np.add.at(Th_m1, (jj, ii), theta_deg * dT)

        if not args.infer_only:
            Ivals = I_of_theta(theta_deg)
            goodI = np.isfinite(Ivals) & (Ivals > 0)
            if np.any(goodI):
                np.add.at(Wrate, (jj[goodI], ii[goodI]), Ivals[goodI] * dT[goodI])

    # Mean theta per cell (weighted by geometric measure)
    theta_mean = np.divide(Th_m1, Tgeom, out=np.full_like(Tgeom, np.nan), where=Tgeom>0)

    # Write core tables
    rows = []
    for dy in range(dy_min, dy_max+1):
        for dx in range(dx_min, dx_max+1):
            ii = dx - dx_min
            jj = dy - dy_min
            rows.append({
                "delta_x": int(dx),
                "delta_y": int(dy),
                "T_geom_m2sr": float(Tgeom[jj, ii]),
                "theta_mean_deg": float(theta_mean[jj, ii]) if np.isfinite(theta_mean[jj, ii]) else np.nan,
                "rate_pred_s": float(Wrate[jj, ii]) if not args.infer_only else np.nan,
                "counts_pred": float(Wrate[jj, ii] * args.dt_det) if not args.infer_only else np.nan,
            })
    df_out = pd.DataFrame(rows)

    df_out[["delta_x","delta_y","T_geom_m2sr"]].to_csv(DATA / "acceptance_cell_Tgeom_m2sr.csv", index=False)
    df_out[["delta_x","delta_y","theta_mean_deg"]].to_csv(DATA / "acceptance_cell_theta_mean_deg.csv", index=False)

    if not args.infer_only:
        df_out.to_csv(DATA / "mute_pred_counts_abs_mc.csv", index=False)
        pred_grid = build_grid_from_table(df_out, dx_min, dx_max, dy_min, dy_max, "counts_pred", fill_value=0.0)
        imshow_grid(pred_grid, dx_min, dx_max, dy_min, dy_max,
                    f"Predicted counts (MC geom, Δt={args.dt_det/3600:.1f} h)",
                    "mute_counts_predicted_mc.png", FIGS, log=True)

    # Optional: compare to measured and infer I_eff
    if not args.no_counts:
        dfm = pd.read_csv(ROOT / args.counts)
        if "counts" in dfm.columns and "counts_meas" not in dfm.columns:
            dfm = dfm.rename(columns={"counts":"counts_meas"})
        meas_grid = build_grid_from_table(dfm, dx_min, dx_max, dy_min, dy_max, "counts_meas", fill_value=0.0)

        # Efficiency proxy if prediction exists
        if not args.infer_only:
            pred_grid = build_grid_from_table(df_out, dx_min, dx_max, dy_min, dy_max, "counts_pred", fill_value=0.0)
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
            imshow_grid(eps, dx_min, dx_max, dy_min, dy_max,
                        "Efficiency proxy (MC geom): ε = N_meas / N_pred",
                        "mute_efficiency_proxy_mc.png", FIGS, log=False)

        # Infer I_eff per cell
        Ieff_cell = np.divide(meas_grid, (args.dt_det * Tgeom),
                              out=np.full_like(meas_grid, np.nan), where=(Tgeom>0))

        # Build a long table and bin vs theta_mean
        theta_cell = theta_mean
        valid = np.isfinite(Ieff_cell) & np.isfinite(theta_cell) & (Ieff_cell > 0)
        tvals = theta_cell[valid]
        ivals = Ieff_cell[valid]

        bw = float(args.theta_bins)
        edges = np.arange(0, args.theta_max + bw, bw)
        idx = np.digitize(tvals, edges) - 1
        nb = len(edges) - 1

        out_rows = []
        for k in range(nb):
            sel = idx == k
            if not np.any(sel):
                continue
            tmid = 0.5 * (edges[k] + edges[k+1])
            # use median for robustness (cells have different solid angles)
            out_rows.append({
                "theta_center_deg": tmid,
                "Ieff_m2_sr_s_median": float(np.median(ivals[sel])),
                "Ieff_m2_sr_s_mean": float(np.mean(ivals[sel])),
                "n_cells": int(sel.sum())
            })
        dfIeff = pd.DataFrame(out_rows)
        dfIeff.to_csv(DATA / "inferred_Ieff_from_meas.csv", index=False)

        # Plot inferred Ieff
        if len(dfIeff) > 0:
            plt.figure(figsize=(8, 4.8))
            plt.plot(dfIeff["theta_center_deg"], dfIeff["Ieff_m2_sr_s_median"])
            plt.yscale("log")
            plt.xlabel(r"Zenith angle $\theta$ (deg)")
            plt.ylabel(r"$I_{\rm eff}(\theta)$  [m$^{-2}$ sr$^{-1}$ s$^{-1}$]")
            plt.title("Effective intensity inferred from measured map and MC geometry")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGS / "inferred_Ieff_vs_theta.png", dpi=200)
            plt.close()

    meta = {
        "method": "MC over top-plane position and direction uniform in solid angle",
        "A_top_m2": float(A_top),
        "theta_max_deg": float(args.theta_max),
        "Omega_sr": float(Omega),
        "nsamp": int(args.nsamp),
        "dt_det_s": float(args.dt_det),
        "geometry": {"N": N, "a_cm": float(args.a_cm), "d_cm": float(args.d_cm)},
        "I_definition": "perpendicular area (dA_perp = cosθ dA_horiz)"
    }
    (DATA / "mute_forward_mc_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.")

if __name__ == "__main__":
    main()
