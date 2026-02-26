#!/usr/bin/env python3
"""
Compare geometric acceptance AΩ(θ,φ) for a 2-plane square telescope.

Two conventions:
  (1) Horizontal-area convention: A_eff,h(θ,φ) = A_overlap * cosθ
  (2) Perpendicular-area convention: A_eff,⊥(θ,φ) = A_overlap

We compute AΩ = A_eff * ΔΩ per (θ,φ) bin, using:
  - Monte Carlo ray-tracing over the top plane
  - Analytic overlap-area formula

Also outputs:
  - Ratio (with cosθ)/(no cosθ) integrated over φ vs θ
  - Ratio map in (θ,φ)

Outputs (figs/):
  acceptance_AOmega_theta_4curves.png
  acceptance_ratio_MC_over_analytic_withcos.png
  acceptance_ratio_MC_over_analytic_nocos.png
  acceptance_withcos_over_nocos_vs_theta.png
  acceptance_withcos_over_nocos_map.png

Outputs (data/):
  acceptance_tables_with_without_cos.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[0]
FIGS = ROOT / "figs"
DATA = ROOT / "data"
FIGS.mkdir(exist_ok=True)

# --- Geometry (MuTe-like) ---
pitch_cm = 4.0
Npix = 15
L_cm = Npix * pitch_cm
halfL = L_cm / 2
d_cm = 30.0
A_top = L_cm * L_cm

# --- Binning ---
theta_edges = np.deg2rad(np.arange(0, 80 + 2, 2))        # 2° bins
phi_edges   = np.deg2rad(np.arange(-180, 180 + 5, 5))    # 5° bins
theta_cent = 0.5 * (theta_edges[:-1] + theta_edges[1:])
phi_cent   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

# --- Monte Carlo sample points (top plane) ---
n_points = 60000
rng = np.random.default_rng(12345)
x0 = rng.uniform(-halfL, halfL, size=n_points)
y0 = rng.uniform(-halfL, halfL, size=n_points)
dA = A_top / n_points

# Containers: MC and analytic, with and without cos factor
acc_mc_withcos = np.zeros((len(theta_cent), len(phi_cent)), dtype=float)
acc_mc_nocos   = np.zeros_like(acc_mc_withcos)
acc_an_withcos = np.zeros_like(acc_mc_withcos)
acc_an_nocos   = np.zeros_like(acc_mc_withcos)

# --- Compute per-bin AΩ ---
for i, th in enumerate(theta_cent):
    sinth = math.sin(th)
    costh = math.cos(th)
    tanth = sinth / (costh + 1e-15)
    dth = theta_edges[i + 1] - theta_edges[i]

    for j, ph in enumerate(phi_cent):
        dph = phi_edges[j + 1] - phi_edges[j]

        sx = d_cm * tanth * math.cos(ph)
        sy = d_cm * tanth * math.sin(ph)

        # ΔΩ for this bin
        dOmega = sinth * dth * dph

        # ---- MC ray tracing ----
        xb = x0 + sx
        yb = y0 + sy
        hit = (np.abs(xb) <= halfL) & (np.abs(yb) <= halfL)

        Aeff_nocos = hit.sum() * dA              # perpendicular-area convention
        Aeff_withcos = Aeff_nocos * costh        # horizontal-area convention

        acc_mc_nocos[i, j]   = Aeff_nocos * dOmega
        acc_mc_withcos[i, j] = Aeff_withcos * dOmega

        # ---- Analytic overlap ----
        Aov = (L_cm - abs(sx)) * (L_cm - abs(sy))
        if Aov < 0:
            Aov = 0.0

        Aeff_an_nocos = Aov
        Aeff_an_withcos = Aov * costh

        acc_an_nocos[i, j]   = Aeff_an_nocos * dOmega
        acc_an_withcos[i, j] = Aeff_an_withcos * dOmega

# --- Integrate over phi to get AΩ(θ) ---
AO_mc_withcos = np.nansum(acc_mc_withcos, axis=1)
AO_an_withcos = np.nansum(acc_an_withcos, axis=1)
AO_mc_nocos   = np.nansum(acc_mc_nocos, axis=1)
AO_an_nocos   = np.nansum(acc_an_nocos, axis=1)

# --- Plot 4 curves in one figure ---
plt.figure(figsize=(8.5, 5.0))
plt.plot(np.rad2deg(theta_cent), AO_mc_withcos, "o-", label="MC (with cosθ)")
plt.plot(np.rad2deg(theta_cent), AO_an_withcos, "o-", label="Analytic (with cosθ)")
plt.plot(np.rad2deg(theta_cent), AO_mc_nocos,   "o-", label="MC (no cosθ)")
plt.plot(np.rad2deg(theta_cent), AO_an_nocos,   "o-", label="Analytic (no cosθ)")
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel(r"$A\Omega(\theta)$ [cm$^2$ sr] (integrated over $\phi$)")
plt.title("Geometric acceptance AΩ(θ): analytic vs MC, with/without cosθ convention")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "acceptance_AOmega_theta_4curves.png", dpi=200)
plt.close()

# --- Ratio maps MC/Analytic ---
ratio_withcos = np.divide(acc_mc_withcos, acc_an_withcos,
                          out=np.full_like(acc_mc_withcos, np.nan),
                          where=acc_an_withcos > 0)
ratio_nocos = np.divide(acc_mc_nocos, acc_an_nocos,
                        out=np.full_like(acc_mc_nocos, np.nan),
                        where=acc_an_nocos > 0)

def save_ratio_map(arr, fname, title, vmin=0.8, vmax=1.2):
    plt.figure(figsize=(9.2, 4.8))
    plt.imshow(arr, origin="lower",
               extent=[np.rad2deg(phi_edges[0]), np.rad2deg(phi_edges[-1]),
                       np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1])],
               aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel(r"Azimuth $\phi$ [deg]")
    plt.ylabel(r"Zenith $\theta$ [deg]")
    plt.title(title)
    plt.colorbar(label="Ratio")
    plt.tight_layout()
    plt.savefig(FIGS / fname, dpi=200)
    plt.close()

save_ratio_map(ratio_withcos, "acceptance_ratio_MC_over_analytic_withcos.png",
               "Acceptance ratio MC/Analytic (with cosθ convention)")
save_ratio_map(ratio_nocos, "acceptance_ratio_MC_over_analytic_nocos.png",
               "Acceptance ratio MC/Analytic (no cosθ convention)")

# --- NEW: (with cos)/(no cos) diagnostics ---
# Integrated over phi vs theta
ratio_cos_over_nocos_mc = np.divide(AO_mc_withcos, AO_mc_nocos,
                                    out=np.full_like(AO_mc_withcos, np.nan),
                                    where=AO_mc_nocos > 0)
ratio_cos_over_nocos_an = np.divide(AO_an_withcos, AO_an_nocos,
                                    out=np.full_like(AO_an_withcos, np.nan),
                                    where=AO_an_nocos > 0)

plt.figure(figsize=(8.5, 5.0))
plt.plot(np.rad2deg(theta_cent), ratio_cos_over_nocos_mc, "o-", label="MC: (with cosθ)/(no cosθ)")
plt.plot(np.rad2deg(theta_cent), ratio_cos_over_nocos_an, "o-", label="Analytic: (with cosθ)/(no cosθ)")
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel(r"Ratio")
plt.title(r"Effect of cosθ convention: $A\Omega_{\rm withcos}/A\Omega_{\rm nocos}$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "acceptance_withcos_over_nocos_vs_theta.png", dpi=200)
plt.close()

# Map in (theta,phi): using MC (you can also do analytic similarly)
ratio_map_cos_over_nocos = np.divide(acc_mc_withcos, acc_mc_nocos,
                                     out=np.full_like(acc_mc_withcos, np.nan),
                                     where=acc_mc_nocos > 0)

save_ratio_map(ratio_map_cos_over_nocos,
               "acceptance_withcos_over_nocos_map.png",
               r"MC map: $A\Omega_{\rm withcos}/A\Omega_{\rm nocos}$ per (θ,φ) bin",
               vmin=0.0, vmax=1.0)

# --- Save table ---
rows = []
for i, th in enumerate(theta_cent):
    for j, ph in enumerate(phi_cent):
        rows.append({
            "theta_deg": float(np.rad2deg(th)),
            "phi_deg": float(np.rad2deg(ph)),

            "AOmega_MC_withcos_cm2sr": float(acc_mc_withcos[i, j]),
            "AOmega_analytic_withcos_cm2sr": float(acc_an_withcos[i, j]),
            "ratio_MC_over_analytic_withcos": float(ratio_withcos[i, j]) if np.isfinite(ratio_withcos[i, j]) else np.nan,

            "AOmega_MC_nocos_cm2sr": float(acc_mc_nocos[i, j]),
            "AOmega_analytic_nocos_cm2sr": float(acc_an_nocos[i, j]),
            "ratio_MC_over_analytic_nocos": float(ratio_nocos[i, j]) if np.isfinite(ratio_nocos[i, j]) else np.nan,

            "ratio_withcos_over_nocos_MC": float(ratio_map_cos_over_nocos[i, j]) if np.isfinite(ratio_map_cos_over_nocos[i, j]) else np.nan,
        })

pd.DataFrame(rows).to_csv(DATA / "acceptance_tables_with_without_cos.csv", index=False)

print("Done.")
print("Saved:")
print(" - figs/acceptance_AOmega_theta_4curves.png")
print(" - figs/acceptance_ratio_MC_over_analytic_withcos.png")
print(" - figs/acceptance_ratio_MC_over_analytic_nocos.png")
print(" - figs/acceptance_withcos_over_nocos_vs_theta.png")
print(" - figs/acceptance_withcos_over_nocos_map.png")
print(" - data/acceptance_tables_with_without_cos.csv")
