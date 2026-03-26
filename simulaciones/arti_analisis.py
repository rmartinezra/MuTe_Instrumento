#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_theta_intensity_shw.py

Lee un archivo .shw, calcula theta_x, theta_y y theta a partir de px, py, pz,
y ajusta la distribución angular con un modelo I(theta) = I0 * cos(theta)^n.

Supuesto físico:
- La simulación representa partículas que cruzan una superficie horizontal
  de área A = 1 m^2 durante T = 3600 s.
- En ese caso:
      dN = I(theta) * cos(theta) * dA * dt * dOmega
  donde I(theta) tiene unidades:
      counts / (s * m^2 * sr)

Por tanto:
1) Conteos por anillo:
   N_i = A*T * ∫_ring I0 cos^n(theta) cos(theta) dOmega

2) Intensidad angular promedio por anillo:
   Ibar_i = N_i / (A*T*∫_ring cos(theta) dOmega)

Uso:
    python3 fit_theta_intensity_shw.py 1horamu.shw

Opcional:
    python3 fit_theta_intensity_shw.py 1horamu.shw --outdir salida --bin-width 1.0 --theta-max 75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# Lectura
# ============================================================
def read_shw(filepath: str | Path) -> pd.DataFrame:
    cols = [
        "corsika_id",
        "px", "py", "pz",
        "x", "y", "z",
        "shower_id", "prm_id",
        "prm_energy", "prm_theta", "prm_phi",
    ]

    df = pd.read_csv(
        filepath,
        comment="#",
        sep=r"\s+",
        names=cols,
        dtype={"corsika_id": str, "shower_id": str, "prm_id": str},
        engine="python",
    )

    df = df.dropna(subset=["px", "py", "pz"]).copy()
    return df


# ============================================================
# Ángulos
# ============================================================
def compute_angles(df: pd.DataFrame) -> pd.DataFrame:
    px = df["px"].to_numpy(dtype=float)
    py = df["py"].to_numpy(dtype=float)
    pz = df["pz"].to_numpy(dtype=float)

    # Convención consistente con tu pipeline
    theta_x_deg = np.degrees(np.arctan2(px, pz))
    theta_y_deg = np.degrees(np.arctan2(py, pz))

    # Ángulo polar respecto al eje +z
    theta_deg = np.degrees(np.arctan2(np.sqrt(px**2 + py**2), pz))

    out = df.copy()
    out["theta_x_deg"] = theta_x_deg
    out["theta_y_deg"] = theta_y_deg
    out["theta_deg"] = theta_deg
    return out


# ============================================================
# Histogramas angulares
# ============================================================
def build_ring_data(theta_deg, bin_width_deg=1.0, theta_max_deg=75.0):
    edges = np.arange(0.0, theta_max_deg + bin_width_deg, bin_width_deg)
    counts, _ = np.histogram(theta_deg, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    t0 = np.radians(edges[:-1])
    t1 = np.radians(edges[1:])

    # Solido angular del anillo
    domega = 2.0 * np.pi * (np.cos(t0) - np.cos(t1))

    # Factor exacto del área proyectada: ∫ cos(theta) dOmega en el anillo
    proj_solid = np.pi * (np.sin(t1)**2 - np.sin(t0)**2)

    # Poisson
    sigma_counts = np.sqrt(np.maximum(counts, 1.0))

    return {
        "edges_deg": edges,
        "centers_deg": centers,
        "counts": counts,
        "sigma_counts": sigma_counts,
        "domega_sr": domega,
        "proj_solid_sr": proj_solid,
    }


# ============================================================
# Modelos
# ============================================================
def ring_counts_model(theta_center_deg, I0, n, bin_width_deg, area_m2, time_s):
    """
    Modelo exacto integrado por bin para conteos por anillo.

    N_i = A*T*2pi*I0/(n+2) * [cos(theta0)^(n+2) - cos(theta1)^(n+2)]
    """
    t0 = np.radians(theta_center_deg - 0.5 * bin_width_deg)
    t1 = np.radians(theta_center_deg + 0.5 * bin_width_deg)

    return (
        area_m2 * time_s * 2.0 * np.pi * I0 / (n + 2.0)
        * (np.cos(t0)**(n + 2.0) - np.cos(t1)**(n + 2.0))
    )


def intensity_avg_model(theta_center_deg, I0, n, bin_width_deg):
    """
    Intensidad angular promedio en el bin:

    Ibar_i = N_i / (A*T*∫ cos(theta)dOmega)

    con
    N_i = A*T*2pi*I0/(n+2) * [cos^(n+2)(t0)-cos^(n+2)(t1)]

    entonces
    Ibar_i = [2*I0/(n+2)] * [cos^(n+2)(t0)-cos^(n+2)(t1)] / [sin^2(t1)-sin^2(t0)]
    """
    t0 = np.radians(theta_center_deg - 0.5 * bin_width_deg)
    t1 = np.radians(theta_center_deg + 0.5 * bin_width_deg)

    num = 2.0 * I0 * (np.cos(t0)**(n + 2.0) - np.cos(t1)**(n + 2.0))
    den = (n + 2.0) * (np.sin(t1)**2 - np.sin(t0)**2)
    return num / den


def reduced_chi2(y, yfit, sigma, n_params):
    chi2 = np.sum(((y - yfit) / sigma) ** 2)
    ndof = max(len(y) - n_params, 1)
    return chi2, ndof, chi2 / ndof


# ============================================================
# Plots base
# ============================================================
def save_xy_plots(df: pd.DataFrame, outdir: Path):
    theta_x = df["theta_x_deg"].to_numpy()
    theta_y = df["theta_y_deg"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(theta_x, bins=181, range=(-90, 90))
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel("Counts")
    plt.title(r"Injected angular distribution of $\theta_x$")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "hist_theta_x.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(theta_y, bins=181, range=(-90, 90))
    plt.xlabel(r"$\theta_y$ [deg]")
    plt.ylabel("Counts")
    plt.title(r"Injected angular distribution of $\theta_y$")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "hist_theta_y.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.hist2d(theta_x, theta_y, bins=181, range=[[-90, 90], [-90, 90]])
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Injected angular distribution in $(\theta_x,\theta_y)$")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.savefig(outdir / "hist2d_theta_xy.png", dpi=220)
    plt.close()


# ============================================================
# Ajuste de conteos por anillo
# ============================================================
def fit_ring_counts(ring, area_m2, time_s, bin_width_deg):
    centers = ring["centers_deg"]
    counts = ring["counts"]
    sigma_counts = ring["sigma_counts"]

    # Evitar bins casi vacíos en el ajuste
    mask = counts > 10

    def model(x, I0, n):
        return ring_counts_model(x, I0, n, bin_width_deg, area_m2, time_s)

    p0 = [counts.max() / (area_m2 * time_s), 2.3]
    bounds = ([0.0, 0.0], [np.inf, 20.0])

    popt, pcov = curve_fit(
        model,
        centers[mask],
        counts[mask],
        p0=p0,
        sigma=sigma_counts[mask],
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )

    I0, n = popt
    I0_err, n_err = np.sqrt(np.diag(pcov))

    yfit = model(centers[mask], I0, n)
    chi2, ndof, redchi2 = reduced_chi2(counts[mask], yfit, sigma_counts[mask], 2)

    return {
        "mask": mask,
        "I0": I0,
        "I0_err": I0_err,
        "n": n,
        "n_err": n_err,
        "chi2": chi2,
        "ndof": ndof,
        "redchi2": redchi2,
        "yfit_masked": yfit,
    }


# ============================================================
# Ajuste de intensidad angular
# ============================================================
def fit_intensity(ring, area_m2, time_s, bin_width_deg):
    centers = ring["centers_deg"]
    counts = ring["counts"]
    sigma_counts = ring["sigma_counts"]
    proj_solid = ring["proj_solid_sr"]

    # Intensidad angular promedio por anillo
    intensity = counts / (area_m2 * time_s * proj_solid)
    sigma_intensity = sigma_counts / (area_m2 * time_s * proj_solid)

    mask = counts > 10

    def model(x, I0, n):
        return intensity_avg_model(x, I0, n, bin_width_deg)

    p0 = [np.nanmax(intensity[mask]), 2.3]
    bounds = ([0.0, 0.0], [np.inf, 20.0])

    popt, pcov = curve_fit(
        model,
        centers[mask],
        intensity[mask],
        p0=p0,
        sigma=sigma_intensity[mask],
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )

    I0, n = popt
    I0_err, n_err = np.sqrt(np.diag(pcov))

    yfit = model(centers[mask], I0, n)
    chi2, ndof, redchi2 = reduced_chi2(
        intensity[mask], yfit, sigma_intensity[mask], 2
    )

    return {
        "mask": mask,
        "intensity": intensity,
        "sigma_intensity": sigma_intensity,
        "I0": I0,
        "I0_err": I0_err,
        "n": n,
        "n_err": n_err,
        "chi2": chi2,
        "ndof": ndof,
        "redchi2": redchi2,
        "yfit_masked": yfit,
    }


# ============================================================
# Guardado de tablas y figuras
# ============================================================
def save_outputs(df, ring, fit_counts, fit_int, outdir, area_m2, time_s, bin_width_deg):
    centers = ring["centers_deg"]
    edges = ring["edges_deg"]
    counts = ring["counts"]
    sigma_counts = ring["sigma_counts"]
    domega = ring["domega_sr"]
    proj_solid = ring["proj_solid_sr"]

    intensity = fit_int["intensity"]
    sigma_intensity = fit_int["sigma_intensity"]

    # CSV de ángulos evento a evento
    event_out = df[[
        "corsika_id", "px", "py", "pz", "theta_x_deg", "theta_y_deg", "theta_deg"
    ]].copy()
    event_out.to_csv(outdir / "theta_eventos.csv", index=False)

    # CSV por bin
    bins_out = pd.DataFrame({
        "theta_min_deg": edges[:-1],
        "theta_max_deg": edges[1:],
        "theta_center_deg": centers,
        "counts": counts,
        "sigma_counts": sigma_counts,
        "domega_sr": domega,
        "proj_solid_sr": proj_solid,
        "intensity_avg_counts_per_s_m2_sr": intensity,
        "sigma_intensity_avg_counts_per_s_m2_sr": sigma_intensity,
    })
    bins_out.to_csv(outdir / "theta_bins.csv", index=False)

    # Resumen de ajustes
    summary = pd.DataFrame({
        "fit_type": ["ring_counts", "angular_intensity"],
        "area_m2": [area_m2, area_m2],
        "time_s": [time_s, time_s],
        "bin_width_deg": [bin_width_deg, bin_width_deg],
        "I0_counts_per_s_m2_sr": [fit_counts["I0"], fit_int["I0"]],
        "I0_err": [fit_counts["I0_err"], fit_int["I0_err"]],
        "n": [fit_counts["n"], fit_int["n"]],
        "n_err_stat": [fit_counts["n_err"], fit_int["n_err"]],
        "chi2": [fit_counts["chi2"], fit_int["chi2"]],
        "ndof": [fit_counts["ndof"], fit_int["ndof"]],
        "reduced_chi2": [fit_counts["redchi2"], fit_int["redchi2"]],
    })
    summary.to_csv(outdir / "fit_summary.csv", index=False)

    # Plot conteos por anillo
    plt.figure(figsize=(8, 5.5))
    plt.errorbar(
        centers, counts, yerr=sigma_counts,
        fmt="o", markersize=3, linewidth=0.8, capsize=2, label="Simulation"
    )

    x_dense = np.linspace(edges[0] + 0.5 * bin_width_deg, edges[-1] - 0.5 * bin_width_deg, 500)
    y_dense = ring_counts_model(
        x_dense,
        fit_counts["I0"],
        fit_counts["n"],
        bin_width_deg,
        area_m2,
        time_s
    )

    plt.plot(
        x_dense, y_dense,
        linewidth=2,
        label=(
            rf"Fit: $I(\theta)=I_0\cos^n\theta$, "
            rf"$n={fit_counts['n']:.3f}\pm{fit_counts['n_err']:.3f}$"
        )
    )

    plt.xlabel(r"$\theta$ [deg]")
    plt.ylabel("Counts per ring")
    plt.title(r"Ring counts vs $\theta$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_ring_counts_theta.png", dpi=220)
    plt.close()

    # Plot intensidad angular
    plt.figure(figsize=(8, 5.5))
    plt.errorbar(
        centers, intensity, yerr=sigma_intensity,
        fmt="o", markersize=3, linewidth=0.8, capsize=2, label="Simulation"
    )

    y_dense_int = intensity_avg_model(
        x_dense,
        fit_int["I0"],
        fit_int["n"],
        bin_width_deg
    )

    plt.plot(
        x_dense, y_dense_int,
        linewidth=2,
        label=(
            rf"Fit: $I(\theta)=I_0\cos^n\theta$, "
            rf"$n={fit_int['n']:.3f}\pm{fit_int['n_err']:.3f}$"
        )
    )

    plt.xlabel(r"$\theta$ [deg]")
    plt.ylabel(r"Angular intensity [counts s$^{-1}$ m$^{-2}$ sr$^{-1}$]")
    plt.title(r"Angular intensity vs $\theta$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_intensity_theta.png", dpi=220)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_shw", help="Archivo .shw")
    parser.add_argument("--outdir", default="theta_fit_output", help="Directorio de salida")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Ancho de bin en grados")
    parser.add_argument("--theta-max", type=float, default=75.0, help="Theta máximo para el fit")
    parser.add_argument("--area-m2", type=float, default=1.0, help="Área horizontal en m^2")
    parser.add_argument("--time-s", type=float, default=3600.0, help="Tiempo total en s")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_shw(args.input_shw)
    df = compute_angles(df)

    # Para este análisis físico usamos solo partículas con pz > 0
    # y theta dentro del rango [0, theta_max]
    df = df[df["pz"] > 0].copy()
    df = df[(df["theta_deg"] >= 0.0) & (df["theta_deg"] <= args.theta_max)].copy()

    if len(df) == 0:
        raise RuntimeError("No quedaron eventos válidos tras aplicar pz>0 y el corte en theta.")

    # Plots base
    save_xy_plots(df, outdir)

    # Ring data
    ring = build_ring_data(
        df["theta_deg"].to_numpy(),
        bin_width_deg=args.bin_width,
        theta_max_deg=args.theta_max,
    )

    # Fits
    fit_counts = fit_ring_counts(
        ring,
        area_m2=args.area_m2,
        time_s=args.time_s,
        bin_width_deg=args.bin_width,
    )

    fit_int = fit_intensity(
        ring,
        area_m2=args.area_m2,
        time_s=args.time_s,
        bin_width_deg=args.bin_width,
    )

    # Outputs
    save_outputs(
        df=df,
        ring=ring,
        fit_counts=fit_counts,
        fit_int=fit_int,
        outdir=outdir,
        area_m2=args.area_m2,
        time_s=args.time_s,
        bin_width_deg=args.bin_width,
    )

    print("\n=== RESULTS ===")
    print(f"Events used: {len(df)}")
    print("\nRing counts fit:")
    print(f"  I0 = {fit_counts['I0']:.6e} ± {fit_counts['I0_err']:.6e} counts s^-1 m^-2 sr^-1")
    print(f"  n  = {fit_counts['n']:.6f} ± {fit_counts['n_err']:.6f}")
    print(f"  chi2/ndof = {fit_counts['chi2']:.3f}/{fit_counts['ndof']} = {fit_counts['redchi2']:.3f}")

    print("\nAngular intensity fit:")
    print(f"  I0 = {fit_int['I0']:.6e} ± {fit_int['I0_err']:.6e} counts s^-1 m^-2 sr^-1")
    print(f"  n  = {fit_int['n']:.6f} ± {fit_int['n_err']:.6f}")
    print(f"  chi2/ndof = {fit_int['chi2']:.3f}/{fit_int['ndof']} = {fit_int['redchi2']:.3f}")

    print(f"\nOutput directory: {outdir}")


if __name__ == "__main__":
    main()
