#!/usr/bin/env python3
"""
Geometric acceptance AΩ(θ,φ) for a 2-plane square telescope:
1) Monte Carlo ray-tracing over the top plane
2) Analytic overlap-area approximation

Geometry:
  Npix=15, pitch=4 cm => L=60 cm
  separation d=30 cm

Outputs (figs/):
  acceptance_AOmega_theta_MC_vs_analytic.png
  acceptance_ratio_MC_over_analytic.png

Output (data/):
  acceptance_tables_theta_phi.csv

Convention note:
  We include a cosθ factor in A_eff ("horizontal area" convention).
  If you prefer acceptance for intensity defined per unit area perpendicular to direction,
  remove the cosθ factor in Aeff_horiz.
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

a_cm, d_cm, Npix = 4.0, 30.0, 15
L_cm = Npix * a_cm
halfL = L_cm / 2
A_top = L_cm * L_cm

theta_edges = np.deg2rad(np.arange(0, 80 + 2, 2))       # 2° bins
phi_edges   = np.deg2rad(np.arange(-180, 180 + 5, 5))   # 5° bins
theta_cent = 0.5 * (theta_edges[:-1] + theta_edges[1:])
phi_cent   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

# MC samples on top plane
n_points = 40000
rng = np.random.default_rng(12345)
x0 = rng.uniform(-halfL, halfL, size=n_points)
y0 = rng.uniform(-halfL, halfL, size=n_points)
dA = A_top / n_points

acc_mc = np.zeros((len(theta_cent), len(phi_cent)))
for i, th in enumerate(theta_cent):
    sinth = math.sin(th)
    costh = math.cos(th)
    tanth = sinth / (costh + 1e-15)
    dth = theta_edges[i + 1] - theta_edges[i]
    for j, ph in enumerate(phi_cent):
        dph = phi_edges[j + 1] - phi_edges[j]
        sx = d_cm * tanth * math.cos(ph)
        sy = d_cm * tanth * math.sin(ph)

        xb = x0 + sx
        yb = y0 + sy
        hit = (np.abs(xb) <= halfL) & (np.abs(yb) <= halfL)

        # horizontal-area convention:
        Aeff = hit.sum() * dA * costh

        dOmega = sinth * dth * dph
        acc_mc[i, j] = Aeff * dOmega

# Analytic overlap
acc_an = np.zeros_like(acc_mc)
for i, th in enumerate(theta_cent):
    sinth = np.sin(th)
    costh = np.cos(th)
    tanth = sinth / (costh + 1e-15)
    dth = theta_edges[i + 1] - theta_edges[i]
    for j, ph in enumerate(phi_cent):
        dph = phi_edges[j + 1] - phi_edges[j]
        sx = d_cm * tanth * np.cos(ph)
        sy = d_cm * tanth * np.sin(ph)

        Aov = (L_cm - abs(sx)) * (L_cm - abs(sy))
        if Aov < 0:
            Aov = 0.0

        Aeff = Aov * costh
        dOmega = sinth * dth * dph
        acc_an[i, j] = Aeff * dOmega

ratio = np.divide(acc_mc, acc_an, out=np.full_like(acc_mc, np.nan), where=acc_an > 0)

# Integrate over phi
AO_mc = np.nansum(acc_mc, axis=1)
AO_an = np.nansum(acc_an, axis=1)

plt.figure(figsize=(8, 4.8))
plt.plot(np.rad2deg(theta_cent), AO_mc, "o-", label="MC ray-tracing")
plt.plot(np.rad2deg(theta_cent), AO_an, "o-", label="Analytic overlap")
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel(r"$A\Omega(\theta)$ [cm$^2$ sr] (integrated over $\phi$)")
plt.title("Geometric acceptance for 2-plane square telescope")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "acceptance_AOmega_theta_MC_vs_analytic.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 4.6))
plt.imshow(ratio, origin="lower",
           extent=[np.rad2deg(phi_edges[0]), np.rad2deg(phi_edges[-1]),
                   np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1])],
           aspect="auto", vmin=0.8, vmax=1.2)
plt.xlabel(r"Azimuth $\phi$ [deg]")
plt.ylabel(r"Zenith $\theta$ [deg]")
plt.title("Acceptance ratio: MC / analytic overlap (AΩ per bin)")
plt.colorbar(label="MC/Analytic")
plt.tight_layout()
plt.savefig(FIGS / "acceptance_ratio_MC_over_analytic.png", dpi=200)
plt.close()

# Save table
rows = []
for i, th in enumerate(theta_cent):
    for j, ph in enumerate(phi_cent):
        rows.append({
            "theta_deg": float(np.rad2deg(th)),
            "phi_deg": float(np.rad2deg(ph)),
            "AOmega_MC_cm2sr": float(acc_mc[i, j]),
            "AOmega_analytic_cm2sr": float(acc_an[i, j]),
            "ratio_MC_over_analytic": float(ratio[i, j]) if np.isfinite(ratio[i, j]) else np.nan
        })
pd.DataFrame(rows).to_csv(DATA / "acceptance_tables_theta_phi.csv", index=False)

print("Done.")
