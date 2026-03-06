#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrección angular (GEANT4 + ARTI) con:
  1) medición cruda N(θ)
  2) corrección geométrica N/G
  3) corrección geométrica + eficiencia 2D (suavizada y ANCLADA por bin de θ):
        N / (G * eps_final)

Incluye:
  - Ajuste de K en: N(θ) ≈ K * G(θ) * cos^n(θ)     (solo para comparar el crudo con el modelo)
  - Mapas 2D: N, G, ΔΩ, A_overlap, eps_raw, eps_smooth, eps_final, N/G, N/(G*eps_final)
  - Curvas 1D vs θ (hasta theta_max_plot): crudo, geom, geom+2Dfinal; con barras Poisson
  - Heatmap FINAL de "tasa" angular corregida:
        rate(θx,θy) = N / (G * eps_final * exposure_s)
    (si exposure_s corresponde al tiempo simulado; por defecto 3600 s)

Notas importantes:
  - eps_raw se define usando el flujo ARTI: Φ_in(θ) ∝ cos^n(θ) (n=2.25 por defecto)
  - eps_smooth es un suavizado 2D ponderado (por N) de eps_raw
  - eps_final aplica un "anclaje radial por bin de θ" para conservar la normalización 1D:
        sum_{pix en bin θ} G*eps_final  =  sum_{pix en bin θ} G*eps_raw
    Esto evita que el suavizado distorsione la curva 1D y te acerca mejor a cos^n.

Uso:
  python3 correcion_calculo_anchored.py mapa_pixeles_delta_xy_sim.csv --outdir out --distance-cm 30 --n-art 2.25

Requiere: numpy, pandas, matplotlib
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# -------------------------
# Suavizado 2D ponderado (sin scipy)
# -------------------------
def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return k


def convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            sl = padded[i : i + 2 * pad + 1, :]
            out[i, :] = np.sum(sl * kernel[:, None], axis=0)
        return out
    elif axis == 1:
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(arr, dtype=float)
        for j in range(arr.shape[1]):
            sl = padded[:, j : j + 2 * pad + 1]
            out[:, j] = np.sum(sl * kernel[None, :], axis=1)
        return out
    else:
        raise ValueError("axis must be 0 or 1")


def gaussian_smooth_weighted(value: np.ndarray, weight: np.ndarray, sigma_pix: float) -> np.ndarray:
    """
    Suavizado gaussiano separable ponderado:
      out = smooth(value*weight)/smooth(weight)
    NaN se maneja con weight=0.
    """
    k = gaussian_kernel1d(sigma_pix)
    num = np.nan_to_num(value * weight, nan=0.0)
    w0 = np.nan_to_num(weight, nan=0.0)

    num_s = convolve1d_reflect(num, k, axis=0)
    num_s = convolve1d_reflect(num_s, k, axis=1)

    w_s = convolve1d_reflect(w0, k, axis=0)
    w_s = convolve1d_reflect(w_s, k, axis=1)

    return np.divide(num_s, w_s, out=np.full_like(num_s, np.nan), where=w_s > 0)


# -------------------------
# Geometría
# -------------------------
def build_grids(max_delta: int, pitch_cm: float, distance_cm: float, nbars: int, n_art: float):
    """
    Construye grids completos para deltas en [-max_delta..max_delta].
    """
    L = nbars * pitch_cm
    factor = pitch_cm / distance_cm
    du = dv = factor

    deltas = np.arange(-max_delta, max_delta + 1)
    DX, DY = np.meshgrid(deltas, deltas)  # rows=dy, cols=dx

    u = DX * factor
    v = DY * factor

    theta_deg = np.degrees(np.arctan(np.sqrt(u * u + v * v)))
    cos_theta = 1.0 / np.sqrt(1.0 + u * u + v * v)

    dOmega = (du * dv) / np.power(1.0 + u * u + v * v, 1.5)

    A_overlap = np.clip(L - np.abs(DX) * pitch_cm, 0, None) * np.clip(L - np.abs(DY) * pitch_cm, 0, None)
    G = A_overlap * cos_theta * dOmega  # [cm^2 sr]

    phi = np.cos(np.radians(theta_deg)) ** n_art  # ARTI relativo (normaliza a 1 en 0°)

    # Edges en θx,θy para pcolormesh (como en angulo.py)
    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)
    theta_edges = np.degrees(np.arctan(delta_edges * factor))
    TXe, TYe = np.meshgrid(theta_edges, theta_edges)

    return {
        "L": L,
        "factor": factor,
        "du": du,
        "dv": dv,
        "deltas": deltas,
        "DX": DX,
        "DY": DY,
        "theta_deg": theta_deg,
        "cos_theta": cos_theta,
        "dOmega": dOmega,
        "A_overlap": A_overlap,
        "G": G,
        "phi": phi,
        "TXe": TXe,
        "TYe": TYe,
    }


def fill_counts_grid(df: pd.DataFrame, deltas: np.ndarray) -> np.ndarray:
    idx_map = {int(v): i for i, v in enumerate(deltas)}
    N_grid = np.zeros((len(deltas), len(deltas)), dtype=float)
    for dx, dy, c in zip(df["delta_x"].astype(int), df["delta_y"].astype(int), df["counts"].astype(float)):
        if dx in idx_map and dy in idx_map:
            N_grid[idx_map[dy], idx_map[dx]] = c
    return N_grid


# -------------------------
# Binning en θ
# -------------------------
def theta_binning_with_index(theta_grid: np.ndarray, weights_grid: np.ndarray, binw_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve:
      centers: centros de bin
      idx_grid: índice de bin por pixel (misma forma que theta_grid)
      binned: suma de weights por bin
      edges: bordes
    """
    th = theta_grid.ravel()
    ww = weights_grid.ravel()

    th_max = float(np.nanmax(th))
    edges = np.arange(0.0, np.ceil(th_max / binw_deg) * binw_deg + binw_deg, binw_deg)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(th, edges) - 1
    valid = (idx >= 0) & (idx < len(centers))
    binned = np.bincount(idx[valid], weights=ww[valid], minlength=len(centers))

    return centers, idx.reshape(theta_grid.shape), binned, edges


def fit_K(centers: np.ndarray, N_bin: np.ndarray, G_bin: np.ndarray, n_art: float, theta_max_fit: float, min_counts_fit: float) -> float:
    """
    Ajusta K en N(θ) ≈ K * G(θ) * cos^n(θ) usando bins con estadística.
    """
    phi_bin = np.cos(np.radians(centers)) ** n_art
    m = (centers <= theta_max_fit) & (N_bin >= min_counts_fit) & (G_bin > 0) & (phi_bin > 0)
    x = G_bin[m] * phi_bin[m]
    y = N_bin[m]
    w = 1.0 / np.maximum(y, 1.0)  # Poisson-ish
    return float(np.sum(w * x * y) / np.sum(w * x * x))


def first_valid_index(arr: np.ndarray) -> Optional[int]:
    for i, v in enumerate(arr):
        if np.isfinite(v) and v > 0:
            return i
    return None


def prop_norm_error(arr: np.ndarray, sig: np.ndarray, i0: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    a0 = arr[i0]
    s0 = sig[i0]
    ok = np.isfinite(arr) & (arr > 0) & np.isfinite(sig) & (sig >= 0) & np.isfinite(a0) & (a0 > 0) & np.isfinite(s0)
    out[ok] = (arr[ok] / a0) * np.sqrt((sig[ok] / arr[ok]) ** 2 + (s0 / a0) ** 2)
    return out


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sim_csv", help="CSV: delta_x, delta_y, counts")
    p.add_argument("--outdir", default="out_paso1_2D_anchored", help="Directorio de salida")
    p.add_argument("--pitch-cm", type=float, default=4.0)
    p.add_argument("--distance-cm", type=float, default=30.0)
    p.add_argument("--nbars", type=int, default=15)
    p.add_argument("--max-delta", type=int, default=14)

    p.add_argument("--n-art", type=float, default=2.25, help="ARTI: Φ(θ) ∝ cos^n θ")
    p.add_argument("--theta-bin-deg", type=float, default=2.0)
    p.add_argument("--theta-max-plot", type=float, default=60.0)
    p.add_argument("--theta-max-fit", type=float, default=60.0)
    p.add_argument("--min-counts-fit", type=float, default=50.0)

    p.add_argument("--sigma-smooth-pix", type=float, default=1.2, help="Sigma del suavizado 2D (en pixeles Δ)")
    p.add_argument("--eps-clip-min", type=float, default=0.2)
    p.add_argument("--eps-clip-max", type=float, default=3.0)

    p.add_argument("--g-threshold-frac", type=float, default=0.02,
                   help="Máscara FOV: aprende eps solo donde G > frac*max(G)")
    p.add_argument("--anchor-scale-min", type=float, default=0.5, help="Clipping del reescalado por bin θ (min)")
    p.add_argument("--anchor-scale-max", type=float, default=2.0, help="Clipping del reescalado por bin θ (max)")

    p.add_argument("--exposure-s", type=float, default=3600.0,
                   help="Tiempo de exposición de la simulación [s] (para tasas). Default: 3600.")
    p.add_argument("--export-theta-csv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim = pd.read_csv(args.sim_csv)
    for col in ("delta_x", "delta_y", "counts"):
        if col not in sim.columns:
            raise ValueError(f"Falta columna '{col}' en {args.sim_csv}")

    geom = build_grids(args.max_delta, args.pitch_cm, args.distance_cm, args.nbars, args.n_art)
    N_grid = fill_counts_grid(sim, geom["deltas"])

    # --- 1D binning para K ---
    centers, idx_grid, N_bin, edges = theta_binning_with_index(geom["theta_deg"], N_grid, args.theta_bin_deg)
    _, _, G_bin, _ = theta_binning_with_index(geom["theta_deg"], geom["G"], args.theta_bin_deg)

    K = fit_K(centers, N_bin, G_bin, args.n_art, args.theta_max_fit, args.min_counts_fit)
    print(f"K (normalización global) = {K:.6g}")

    # --- eps_raw = N / (K G cos^n) ---
    N_pred_grid = K * geom["G"] * geom["phi"]
    eps_raw = np.divide(
        N_grid, N_pred_grid,
        out=np.full_like(N_grid, np.nan),
        where=(N_pred_grid > 0) & (N_grid > 0)
    )

    # --- Máscara de FOV para aprender eps (evita bordes con G muy pequeño) ---
    G_max = float(np.nanmax(geom["G"]))
    g_thr = args.g_threshold_frac * G_max
    mask_fov = (geom["G"] > g_thr)

    # --- Relleno de NaNs con promedio por anillos r^2 (solo dentro de FOV) ---
    r2 = geom["DX"] * geom["DX"] + geom["DY"] * geom["DY"]
    eps_fill = eps_raw.copy()
    for r2_val in np.unique(r2):
        m = (r2 == r2_val) & np.isfinite(eps_raw) & mask_fov
        if np.any(m):
            ww = np.maximum(N_grid[m], 1.0)
            mu = float(np.sum(eps_raw[m] * ww) / np.sum(ww))
            eps_fill[(r2 == r2_val) & ~np.isfinite(eps_fill) & mask_fov] = mu

    # --- Suavizado 2D ponderado por N (solo donde hay datos) ---
    w_eps = np.where(np.isfinite(eps_fill) & mask_fov, np.maximum(N_grid, 1.0), 0.0)
    eps_smooth = gaussian_smooth_weighted(eps_fill, w_eps, args.sigma_smooth_pix)
    eps_smooth = np.clip(eps_smooth, args.eps_clip_min, args.eps_clip_max)

    # --- ANCLAJE por bin de θ para preservar la 1D ---
    # Queremos: sum_{bin θ} G*eps_final  = sum_{bin θ} G*eps_raw (usando solo pix válidos)
    eps_final = eps_smooth.copy()

    for b in range(len(centers)):
        m = (idx_grid == b) & mask_fov & np.isfinite(eps_raw)
        if not np.any(m):
            continue
        S_raw = float(np.sum(geom["G"][m] * eps_raw[m]))
        S_smo = float(np.sum(geom["G"][m] * eps_final[m]))
        if S_smo <= 0:
            continue
        scale = S_raw / S_smo
        scale = float(np.clip(scale, args.anchor_scale_min, args.anchor_scale_max))
        eps_final[idx_grid == b] *= scale

    eps_final = np.clip(eps_final, args.eps_clip_min, args.eps_clip_max)

    # -----------------------------
    # Mapas 2D: flujos / tasas
    # -----------------------------
    F_geom_grid = np.divide(N_grid, geom["G"], out=np.full_like(N_grid, np.nan), where=(geom["G"] > 0) & (N_grid > 0))
    F_eff_grid = np.divide(
        N_grid, geom["G"] * eps_final,
        out=np.full_like(N_grid, np.nan),
        where=(geom["G"] > 0) & (eps_final > 0) & (N_grid > 0)
    )

    # Tasa angular (por segundo) corregida final:
    rate_eff_grid = F_eff_grid / float(args.exposure_s)  # [1/(cm^2 sr s)] si N corresponde a exposure_s

    # -----------------------------
    # Curvas 1D hasta theta_max_plot
    # -----------------------------
    theta_max_plot = args.theta_max_plot
    sel = centers <= theta_max_plot

    # Crudo: N(θ)
    Nbin = N_bin.copy()
    sigN = np.sqrt(np.maximum(Nbin, 0.0))
    i0 = first_valid_index(Nbin)
    if i0 is None:
        raise RuntimeError("No hay bins válidos para normalizar.")
    N_rel = Nbin / Nbin[i0]
    sigN_rel = prop_norm_error(Nbin, sigN, i0)

    # Modelo crudo: K G cos^n
    phi_bin = np.cos(np.radians(centers)) ** args.n_art
    Npred = K * G_bin * phi_bin
    Npred_rel = Npred / Npred[i0] if (np.isfinite(Npred[i0]) and Npred[i0] > 0) else Npred

    # Geométrica: N/G
    Fgeom = np.divide(Nbin, G_bin, out=np.full_like(Nbin, np.nan), where=G_bin > 0)
    sigFgeom = np.divide(sigN, G_bin, out=np.full_like(sigN, np.nan), where=G_bin > 0)
    Fgeom_rel = Fgeom / Fgeom[i0]
    sigFgeom_rel = prop_norm_error(Fgeom, sigFgeom, i0)

    # Geom + 2D final: N/(G*eps_final)
    _, _, Geff_bin, _ = theta_binning_with_index(geom["theta_deg"], geom["G"] * eps_final, args.theta_bin_deg)
    Feff = np.divide(Nbin, Geff_bin, out=np.full_like(Nbin, np.nan), where=Geff_bin > 0)
    sigFeff = np.divide(sigN, Geff_bin, out=np.full_like(sigN, np.nan), where=Geff_bin > 0)
    Feff_rel = Feff / Feff[i0]
    sigFeff_rel = prop_norm_error(Feff, sigFeff, i0)

    # ARTI esperado relativo
    Fart_rel = phi_bin / phi_bin[i0] if (np.isfinite(phi_bin[i0]) and phi_bin[i0] > 0) else phi_bin

    # -----------------------------
    # FIGURAS 1D (crudo / geom / geom+2Dfinal / comparación)
    # -----------------------------
    # 1) Crudo vs modelo
    m = sel & (Nbin > 0) & np.isfinite(N_rel) & np.isfinite(sigN_rel) & np.isfinite(Npred_rel) & (Npred_rel > 0)
    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(centers[m], N_rel[m], yerr=sigN_rel[m], fmt="o", markersize=3, capsize=2,
                 label="Medición cruda: N(θ) (norm., Poisson)")
    plt.plot(centers[m], Npred_rel[m], "-", label=fr"Modelo: $K\,G(\theta)\cos^{{{args.n_art:.2f}}}\theta$ (norm.)")
    plt.yscale("log")
    plt.xlim(0, theta_max_plot)
    plt.xlabel(r"Ángulo cenital $|\theta|$ [deg]")
    plt.ylabel("Distribución cruda (normalizada)")
    plt.title(f"Crudo vs modelo (θ≤{theta_max_plot:g}°)")
    plt.legend()
    savefig(outdir / "1D_crudo_vs_modelo.png")

    # 2) Corrección geométrica: N/G vs cos^n
    m2 = sel & np.isfinite(Fgeom_rel) & (Fgeom_rel > 0) & np.isfinite(sigFgeom_rel) & np.isfinite(Fart_rel) & (Fart_rel > 0)
    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(centers[m2], Fgeom_rel[m2], yerr=sigFgeom_rel[m2], fmt="o", markersize=3, capsize=2,
                 label=r"Geométrica: $(N/G)(\theta)$ (norm., Poisson)")
    plt.plot(centers[m2], Fart_rel[m2], "-", label=fr"ARTI esperado: $\cos^{{{args.n_art:.2f}}}\theta$ (norm.)")
    plt.yscale("log")
    plt.xlim(0, theta_max_plot)
    plt.xlabel(r"Ángulo cenital $|\theta|$ [deg]")
    plt.ylabel("Distribución corregida geométricamente (norm.)")
    plt.title(f"Corrección geométrica (θ≤{theta_max_plot:g}°)")
    plt.legend()
    savefig(outdir / "1D_geom_vs_ARTI.png")

    # 3) Corrección geom + 2D final: N/(G eps_final) vs cos^n
    m3 = sel & np.isfinite(Feff_rel) & (Feff_rel > 0) & np.isfinite(sigFeff_rel) & np.isfinite(Fart_rel) & (Fart_rel > 0)
    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(centers[m3], Feff_rel[m3], yerr=sigFeff_rel[m3], fmt="o", markersize=3, capsize=2,
                 label=r"Geom.+2D final: $N/(G\,\varepsilon_{final})(\theta)$ (norm., Poisson)")
    plt.plot(centers[m3], Fart_rel[m3], "-", label=fr"ARTI esperado: $\cos^{{{args.n_art:.2f}}}\theta$ (norm.)")
    plt.yscale("log")
    plt.xlim(0, theta_max_plot)
    plt.xlabel(r"Ángulo cenital $|\theta|$ [deg]")
    plt.ylabel("Distribución corregida (norm.)")
    plt.title(f"Corrección geométrica + eficiencia 2D FINAL (θ≤{theta_max_plot:g}°)")
    plt.legend()
    savefig(outdir / "1D_geom2Dfinal_vs_ARTI.png")

    # 4) Comparación: geom vs geom+2Dfinal vs ARTI
    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(centers[m2], Fgeom_rel[m2], yerr=sigFgeom_rel[m2], fmt="o", markersize=3, capsize=2,
                 label=r"Solo geom.: $(N/G)(\theta)$ (norm.)")
    plt.errorbar(centers[m3], Feff_rel[m3], yerr=sigFeff_rel[m3], fmt="o", markersize=3, capsize=2,
                 label=r"Geom.+2D final: $N/(G\varepsilon)(\theta)$ (norm.)")
    plt.plot(centers[m2], Fart_rel[m2], "-", label=fr"ARTI: $\cos^{{{args.n_art:.2f}}}\theta$ (norm.)")
    plt.yscale("log")
    plt.xlim(0, theta_max_plot)
    plt.xlabel(r"Ángulo cenital $|\theta|$ [deg]")
    plt.ylabel("Distribución corregida (norm.)")
    plt.title(f"Comparación de correcciones (θ≤{theta_max_plot:g}°)")
    plt.legend()
    savefig(outdir / "1D_compare_geom_vs_2Dfinal_vs_ARTI.png")

    # -----------------------------
    # FIGURAS 2D (incluye heatmap final de tasas)
    # -----------------------------
    TXe, TYe = geom["TXe"], geom["TYe"]

    # A_overlap
    plt.figure(figsize=(7.0, 6.0))
    Aplot = np.ma.masked_less_equal(geom["A_overlap"], 0)
    plt.pcolormesh(TXe, TYe, Aplot, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Mapa $A_{\rm overlap}$ [cm$^2$]")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$A_{\rm overlap}$ [cm$^2$]")
    savefig(outdir / "2D_map_Aoverlap.png")

    # dOmega
    plt.figure(figsize=(7.0, 6.0))
    dOplot = np.ma.masked_less_equal(geom["dOmega"], 0)
    plt.pcolormesh(TXe, TYe, dOplot, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Mapa $\Delta\Omega$ [sr]")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$\Delta\Omega$ [sr]")
    savefig(outdir / "2D_map_dOmega.png")

    # G
    plt.figure(figsize=(7.0, 6.0))
    Gplot = np.ma.masked_less_equal(geom["G"], 0)
    plt.pcolormesh(TXe, TYe, Gplot, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Mapa $G_{\rm geom}(\theta_x,\theta_y)$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$G$ [cm$^2$ sr]")
    savefig(outdir / "2D_map_Ggeom.png")

    # N
    plt.figure(figsize=(7.0, 6.0))
    Nplot = np.ma.masked_less_equal(N_grid, 0)
    plt.pcolormesh(TXe, TYe, Nplot, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title("GEANT4: mapa de cuentas N(θx,θy)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label="Cuentas")
    savefig(outdir / "2D_map_counts.png")

    # N/G
    plt.figure(figsize=(7.0, 6.0))
    Fplot = np.ma.masked_where(~np.isfinite(F_geom_grid) | (F_geom_grid <= 0), F_geom_grid)
    plt.pcolormesh(TXe, TYe, Fplot, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Proxy flujo: $N/G_{\rm geom}$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$N/G$ [1/(cm$^2$ sr)]")
    savefig(outdir / "2D_map_fluxproxy_N_over_G.png")

    # eps_raw
    plt.figure(figsize=(7.0, 6.0))
    Eraw = np.ma.masked_where(~np.isfinite(eps_raw) | (eps_raw <= 0), eps_raw)
    plt.pcolormesh(TXe, TYe, Eraw, shading="auto", norm=LogNorm(vmin=0.3, vmax=3.0))
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"$\varepsilon_{\rm raw} = N/(K\,G\,\cos^{n}\theta)$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$\varepsilon$ (adim.)")
    savefig(outdir / "2D_map_eps_raw.png")

    # eps_smooth
    plt.figure(figsize=(7.0, 6.0))
    Es = np.ma.masked_where(~np.isfinite(eps_smooth) | (eps_smooth <= 0), eps_smooth)
    plt.pcolormesh(TXe, TYe, Es, shading="auto", norm=LogNorm(vmin=0.3, vmax=3.0))
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"$\varepsilon_{\rm smooth}$ (suavizado 2D ponderado)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$\varepsilon$ (adim.)")
    savefig(outdir / "2D_map_eps_smooth.png")

    # eps_final
    plt.figure(figsize=(7.0, 6.0))
    Ef = np.ma.masked_where(~np.isfinite(eps_final) | (eps_final <= 0), eps_final)
    plt.pcolormesh(TXe, TYe, Ef, shading="auto", norm=LogNorm(vmin=0.3, vmax=3.0))
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"$\varepsilon_{\rm final}$ (suavizado + anclaje por bin de θ)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$\varepsilon$ (adim.)")
    savefig(outdir / "2D_map_eps_final.png")

    # N/(G eps_final)
    plt.figure(figsize=(7.0, 6.0))
    F2 = np.ma.masked_where(~np.isfinite(F_eff_grid) | (F_eff_grid <= 0), F_eff_grid)
    plt.pcolormesh(TXe, TYe, F2, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Proxy flujo corregido: $N/(G_{\rm geom}\,\varepsilon_{\rm final})$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06, label=r"$N/(G\varepsilon)$ [1/(cm$^2$ sr)]")
    savefig(outdir / "2D_map_fluxproxy_after_2Dfinal.png")

    # HEATMAP FINAL: tasa angular corregida (por segundo)
    plt.figure(figsize=(7.0, 6.0))
    R = np.ma.masked_where(~np.isfinite(rate_eff_grid) | (rate_eff_grid <= 0), rate_eff_grid)
    plt.pcolormesh(TXe, TYe, R, shading="auto", norm=LogNorm())
    plt.xlabel(r"$\theta_x$ [deg]")
    plt.ylabel(r"$\theta_y$ [deg]")
    plt.title(r"Tasa angular corregida final: $N/(G\,\varepsilon_{\rm final}\,T)$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(orientation="horizontal", pad=0.08, fraction=0.06,
                 label=r"Tasa [1/(cm$^2$ sr s)]")
    savefig(outdir / "2D_heatmap_rate_corrected_final.png")

    # -----------------------------
    # Export CSV 1D (opcional)
    # -----------------------------
    if args.export_theta_csv:
        df_out = pd.DataFrame({
            "theta_center_deg": centers,
            "N_counts": Nbin,
            "G_geom_cm2sr": G_bin,
            "Geff_cm2sr": Geff_bin,
            "N_rel": N_rel,
            "N_rel_sigma": sigN_rel,
            "Fgeom_rel": Fgeom_rel,
            "Fgeom_rel_sigma": sigFgeom_rel,
            "Feff_rel": Feff_rel,
            "Feff_rel_sigma": sigFeff_rel,
            "arti_cosn_rel": Fart_rel,
        })
        df_out.to_csv(outdir / "theta_summary.csv", index=False)

    print(f"Listo. Salidas en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
